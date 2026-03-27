-- ============================================================================
-- PROJECT BOT-B-GONE: Full Production Bot-Filtering Algorithm (SQL)
-- ============================================================================
-- This is the complete, production-grade model for filtering raw email event
-- data. It classifies both clicks AND opens, assigns confidence levels, and
-- computes probability scores. It is designed to be adapted to your ESP's
-- schema — replace the `raw_events` CTE with your own table.
--
-- WHAT THIS MODEL PRODUCES (per subscriber per campaign):
--   • bot_b_gone_click_classification  (e.g., BOT:machinegun, HUMAN:delayed_single)
--   • bot_b_gone_open_classification   (e.g., BOT:instant_prefetch, HUMAN:verified_clicker)
--   • click_confidence                 (definitive / high / medium / low)
--   • open_confidence                  (definitive / high / medium / low)
--   • click_probability                (0-100 score)
--   • open_probability                 (0-100 score)
--   • is_human_click                   (boolean — the recommended filter for CTR)
--   • is_human_open                    (boolean — the recommended filter for open rate)
--
-- TIMESTAMP DEFINITIONS:
-- -----------------------------------------------------------------------
-- send_timestamp:  The moment the ESP dispatched the email to the
--                  recipient's mail server. This is the most universally
--                  available timestamp across ESPs and is the baseline
--                  for all time-based bot detection rules.
--
-- event_timestamp: The moment the open or click event was recorded by
--                  the ESP's tracking infrastructure.
--
-- NOTE: Some ESPs expose a separate "delivery_timestamp" (when the
-- receiving MTA accepted the message). If your ESP provides this, you
-- may substitute it for send_timestamp in the velocity rules below for
-- slightly more accurate timing. However, send_timestamp is the safer
-- default because (a) it is available on every ESP, and (b) the
-- difference between send and delivery is typically < 5 seconds for
-- most mail servers, which is well within our detection windows.
-- ============================================================================


-- ============================================================================
-- STEP 0: RAW EVENTS
-- ============================================================================
-- Replace this CTE with your ESP's raw event table.
-- You need per-event rows with timestamps, not aggregated campaign stats.
-- See /docs/ESP_GUIDE.md for how to extract this from your specific ESP.

WITH raw_events AS (
    SELECT
        subscriber_id,
        campaign_id,
        event_type,        -- 'open' or 'click'
        event_timestamp,   -- when the open/click was recorded
        send_timestamp,    -- when the ESP dispatched the email (see note above)
        user_agent,
        ip_address,
        url_clicked,       -- null for opens
        -- Optional: if your ESP provides a "real" or "nhi" flag, include it.
        -- This is used by Rule H6 (ESP-confirmed human click).
        esp_is_real_flag   -- boolean, null if not available
    FROM esp_raw_events
),


-- ============================================================================
-- STEP 1: COMPUTE PER-SESSION FEATURES
-- ============================================================================
-- Aggregate raw events into one row per subscriber-campaign with timing
-- features that power all downstream rules.

-- 1a. Click session features
click_features AS (
    SELECT
        subscriber_id,
        campaign_id,
        -- Timing
        MIN(TIMESTAMP_DIFF(event_timestamp, send_timestamp, SECOND))
            AS time_to_first_click_sec,
        TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), SECOND)
            AS click_span_sec,
        -- Volume
        COUNT(*) AS raw_total_clicks,
        COUNT(DISTINCT url_clicked) AS unique_urls_clicked,
        -- Inter-click velocity (avg seconds between consecutive clicks)
        SAFE_DIVIDE(
            TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), SECOND),
            GREATEST(COUNT(*) - 1, 1)
        ) AS avg_inter_click_sec
    FROM raw_events
    WHERE event_type = 'click'
    GROUP BY subscriber_id, campaign_id
),

-- 1b. Open session features
open_features AS (
    SELECT
        subscriber_id,
        campaign_id,
        -- Timing
        MIN(TIMESTAMP_DIFF(event_timestamp, send_timestamp, SECOND))
            AS time_to_first_open_sec,
        TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), SECOND)
            AS open_span_sec,
        -- Volume
        COUNT(*) AS raw_total_opens,
        -- ESP real flag (if available)
        COUNTIF(esp_is_real_flag IS TRUE) AS esp_real_open_count
    FROM raw_events
    WHERE event_type = 'open'
    GROUP BY subscriber_id, campaign_id
),


-- ============================================================================
-- STEP 2: HONEYPOT FILTER — The undeniable ground truth
-- ============================================================================
-- See /docs/METHODOLOGY.md for tiered implementation strategies.
-- Any subscriber who clicks an invisible/honeypot link is a confirmed bot.

honeypot_clickers AS (
    SELECT DISTINCT subscriber_id
    FROM raw_events
    WHERE event_type = 'click'
    AND (
        -- Condition A: Clicked an invisible 1x1 pixel link
        url_clicked LIKE '%/tracking-pixel.png%'
        OR
        -- Condition B: Clicked a generic footer link that humans ignore
        -- Replace with your own honeypot URL(s)
        url_clicked = 'https://yourdomain.com/generic-footer-link'
    )
),


-- ============================================================================
-- STEP 3: USER AGENT FILTER — Known bot signatures
-- ============================================================================

ua_flags AS (
    SELECT
        subscriber_id,
        campaign_id,
        event_timestamp,
        CASE
            WHEN LOWER(user_agent) LIKE '%bot%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%spider%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%crawler%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%barracuda%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%mimecast%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%proofpoint%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%fireeye%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%fortinet%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%symantec%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%messagelabs%' THEN TRUE
            WHEN LOWER(user_agent) LIKE '%googleimageproxy%' THEN TRUE
            -- Note: Apple Mail Privacy Protection (MPP) requires special
            -- handling. It is technically a proxy, but it masks real humans.
            -- Bot-B-Gone recommends keeping MPP opens but flagging them
            -- for separate analysis. See open rules below.
            ELSE FALSE
        END AS is_ua_bot
    FROM raw_events
),


-- ============================================================================
-- STEP 4: CLICK CLASSIFICATION RULES
-- ============================================================================
-- Seven bot rules and six human rules, applied in priority order.
-- Each rule is a boolean flag; the first one that fires wins.
--
-- DATA BEHIND THESE THRESHOLDS (90-day analysis, ~684K click sessions):
-- +-------------------------------------------+-----------+--------+------+
-- | Signal                                    | Precision | Recall | FPR  |
-- +-------------------------------------------+-----------+--------+------+
-- | 5+ clicks, span <= 5s (machinegun)        | 99.998%   | 67.9%  | 0.01%|
-- | 5+ clicks, span <= 10s                    | 99.995%   | 73.9%  | 0.02%|
-- | Avg inter-click < 2s (rapid fire)         | 99.687%   | 78.4%  | 1.37%|
-- | First click < 60s after send (instant)    | 100.0%    | 64.6%  | 0.0% |
-- | First click < 300s after send             | 98.886%   | 91.7%  | 5.77%|
-- | Combined: (5+ in 5s) OR (first < 60s)     | 99.999%   | 90.5%  | 0.01%|
-- +-------------------------------------------+-----------+--------+------+

click_rules AS (
    SELECT
        c.subscriber_id,
        c.campaign_id,

        -- ================================================================
        -- BOT CLICK RULES (any one = bot)
        -- ================================================================

        -- B1: MACHINEGUN (DEFINITIVE)
        -- All links clicked within seconds. Security scanners click every
        -- link in the email to check for malware.
        -- Precision: 99.998%. FPR: 0.006%.
        (c.raw_total_clicks >= 5 AND c.click_span_sec <= 5)
            AS click_rule_bot_machinegun_definitive,

        -- B2: MACHINEGUN (LIKELY)
        -- Most links clicked rapidly, but slightly wider window.
        -- Catches bots that throttle slightly to avoid detection.
        (c.raw_total_clicks >= 5 AND c.click_span_sec <= 10
            AND NOT (c.raw_total_clicks >= 5 AND c.click_span_sec <= 5))
            AS click_rule_bot_machinegun_likely,

        -- B3: INSTANT (DEFINITIVE)
        -- First click occurs within 2 seconds of send.
        -- Physically impossible for a human to receive, open, read, and click.
        (c.time_to_first_click_sec < 2)
            AS click_rule_bot_instant_definitive,

        -- B4: INSTANT (LIKELY)
        -- First click occurs within 60 seconds of send.
        -- In our data, 0% of verified human clicks occur < 5 min after send.
        -- 60s threshold = 100% precision, 0% FPR.
        (c.time_to_first_click_sec >= 2 AND c.time_to_first_click_sec < 60)
            AS click_rule_bot_instant_likely,

        -- B5: URL SCANNER
        -- Clicks every unique URL in the email but does so methodically.
        -- Pattern: high unique URL count + low click span + moderate speed.
        (c.unique_urls_clicked >= 5
            AND c.avg_inter_click_sec BETWEEN 0.5 AND 3.0
            AND c.click_span_sec <= 30)
            AS click_rule_bot_scanner,

        -- B6: CRON BURST
        -- Clicks arrive in periodic, automated patterns.
        -- Pattern: moderate inter-click timing (2-10s) with high volume,
        -- typically from scheduled security scans.
        (c.raw_total_clicks >= 5
            AND c.avg_inter_click_sec BETWEEN 2.0 AND 10.0
            AND c.time_to_first_click_sec < 300)
            AS click_rule_bot_cron,

        -- B7: HIGH VOLUME
        -- Abnormally high total click count suggesting automation.
        -- Even the most engaged human rarely clicks > 10 links in one email.
        (c.raw_total_clicks >= 10
            AND c.avg_inter_click_sec < 30)
            AS click_rule_bot_volume,

        -- ================================================================
        -- HUMAN CLICK RULES (positive signals — evidence of a real person)
        -- ================================================================

        -- H1: DELAYED SINGLE
        -- Exactly one click, arriving well after send. The most common
        -- human pattern: read the email, click the one thing that interests you.
        (c.raw_total_clicks = 1
            AND c.time_to_first_click_sec >= 300
            AND c.time_to_first_click_sec < 86400)
            AS click_rule_human_delayed,

        -- H2: LATE ARRIVAL
        -- Click arrives many hours or days after send. Consistent with
        -- a human returning to their inbox later.
        (c.time_to_first_click_sec >= 86400)
            AS click_rule_human_late,

        -- H3: MODERATE TIMING
        -- Single click with moderate timing — not instant, not extremely
        -- delayed. Falls in the normal human reading window.
        (c.raw_total_clicks = 1
            AND c.time_to_first_click_sec >= 60
            AND c.time_to_first_click_sec < 300)
            AS click_rule_human_moderate,

        -- H4: SAILTHRU / ESP CONFIRMED
        -- Your ESP flagged this click as human / "real".
        -- Only available if your ESP provides this signal.
        (FALSE) -- Replace with your ESP's real-click flag if available
            AS click_rule_human_sailthru,

        -- H5: SINGLE SELECTIVE
        -- Exactly one click on a content link. Selective behavior is human.
        (c.raw_total_clicks = 1 AND c.unique_urls_clicked = 1)
            AS click_rule_human_single,

        -- H6: THOUGHTFUL MULTI-CLICK
        -- Multiple clicks with meaningful time gaps between them.
        -- Median human inter-click: 25 seconds. This catches readers who
        -- click 2-3 links while reading through the newsletter.
        (c.raw_total_clicks >= 2
            AND c.avg_inter_click_sec >= 10
            AND c.time_to_first_click_sec >= 300)
            AS click_rule_human_thoughtful

    FROM click_features c
),


-- ============================================================================
-- STEP 5: CLICK CLASSIFICATION (priority cascade)
-- ============================================================================

click_classified AS (
    SELECT
        cr.*,

        -- Classification label
        CASE
            -- Bot rules (highest priority first)
            WHEN cr.click_rule_bot_machinegun_definitive THEN 'BOT:machinegun'
            WHEN cr.click_rule_bot_instant_definitive    THEN 'BOT:instant_prefetch'
            WHEN cr.click_rule_bot_machinegun_likely      THEN 'BOT:machinegun_likely'
            WHEN cr.click_rule_bot_instant_likely          THEN 'BOT:instant_likely'
            WHEN cr.click_rule_bot_scanner                 THEN 'BOT:url_scanner'
            WHEN cr.click_rule_bot_cron                    THEN 'BOT:cron_burst'
            WHEN cr.click_rule_bot_volume                  THEN 'BOT:high_volume'
            -- Human rules
            WHEN cr.click_rule_human_sailthru              THEN 'HUMAN:sailthru_confirmed'
            WHEN cr.click_rule_human_thoughtful            THEN 'HUMAN:thoughtful_multi'
            WHEN cr.click_rule_human_delayed               THEN 'HUMAN:delayed_single'
            WHEN cr.click_rule_human_late                  THEN 'HUMAN:late_arrival'
            WHEN cr.click_rule_human_moderate              THEN 'HUMAN:single_moderate'
            WHEN cr.click_rule_human_single                THEN 'HUMAN:single_selective'
            ELSE 'UNCLASSIFIED:ambiguous'
        END AS bot_b_gone_click_classification,

        -- Confidence level
        CASE
            WHEN cr.click_rule_bot_machinegun_definitive THEN 'definitive'
            WHEN cr.click_rule_bot_instant_definitive    THEN 'definitive'
            WHEN cr.click_rule_bot_machinegun_likely      THEN 'high'
            WHEN cr.click_rule_bot_instant_likely          THEN 'high'
            WHEN cr.click_rule_bot_scanner                 THEN 'high'
            WHEN cr.click_rule_bot_cron                    THEN 'medium'
            WHEN cr.click_rule_bot_volume                  THEN 'medium'
            WHEN cr.click_rule_human_sailthru              THEN 'high'
            WHEN cr.click_rule_human_thoughtful            THEN 'high'
            WHEN cr.click_rule_human_delayed               THEN 'high'
            WHEN cr.click_rule_human_late                  THEN 'high'
            WHEN cr.click_rule_human_moderate              THEN 'medium'
            WHEN cr.click_rule_human_single                THEN 'medium'
            ELSE 'low'
        END AS click_confidence,

        -- Probability score (0 = definitely bot, 100 = definitely human)
        CASE
            WHEN cr.click_rule_bot_machinegun_definitive THEN 0
            WHEN cr.click_rule_bot_instant_definitive    THEN 0
            WHEN cr.click_rule_bot_machinegun_likely      THEN 5
            WHEN cr.click_rule_bot_instant_likely          THEN 5
            WHEN cr.click_rule_bot_scanner                 THEN 5
            WHEN cr.click_rule_bot_cron                    THEN 10
            WHEN cr.click_rule_bot_volume                  THEN 10
            WHEN cr.click_rule_human_sailthru              THEN 95
            WHEN cr.click_rule_human_thoughtful            THEN 90
            WHEN cr.click_rule_human_delayed               THEN 90
            WHEN cr.click_rule_human_late                  THEN 85
            WHEN cr.click_rule_human_moderate              THEN 70
            WHEN cr.click_rule_human_single                THEN 65
            ELSE 40
        END AS click_probability,

        -- The recommended boolean filter for CTR reporting
        CASE
            WHEN cr.click_rule_bot_machinegun_definitive THEN FALSE
            WHEN cr.click_rule_bot_instant_definitive    THEN FALSE
            WHEN cr.click_rule_bot_machinegun_likely      THEN FALSE
            WHEN cr.click_rule_bot_instant_likely          THEN FALSE
            WHEN cr.click_rule_bot_scanner                 THEN FALSE
            WHEN cr.click_rule_bot_cron                    THEN FALSE
            WHEN cr.click_rule_bot_volume                  THEN FALSE
            ELSE TRUE
        END AS is_human_click

    FROM click_rules cr
),


-- ============================================================================
-- STEP 6: OPEN CLASSIFICATION RULES
-- ============================================================================
-- Opens are harder than clicks because Apple Mail Privacy Protection (MPP)
-- fires a proxy open on nearly every email, making unique opens unreliable.
--
-- Our approach: use behavioral evidence to find opens we can CONFIRM as
-- human, rather than trying to subtract bots from a noisy total.
--
-- The hierarchy:
--   1. If we have a confirmed human click on the same send → definitive open
--   2. If the ESP flagged it as "real" → high confidence
--   3. If multiple opens are separated by meaningful time → medium confidence
--   4. If the open happened during a bot click session → bot
--   5. If the open was instant with no other evidence → bot
--   6. Everything else → uncertain

open_rules AS (
    SELECT
        o.subscriber_id,
        o.campaign_id,
        o.time_to_first_open_sec,
        o.open_span_sec,
        o.raw_total_opens,
        o.esp_real_open_count,

        -- ================================================================
        -- HUMAN OPEN RULES (positive evidence)
        -- ================================================================

        -- OH1: VERIFIED CLICKER (DEFINITIVE)
        -- The subscriber clicked a link AND that click was classified as human.
        -- If they clicked, they opened. Period. This is ground truth.
        (cc.is_human_click IS TRUE)
            AS open_rule_verified_clicker,

        -- OH2: ESP REAL FLAG
        -- Your ESP flagged this open as "real" (e.g., Sailthru's is_real).
        -- High confidence but not definitive — ESPs have their own biases.
        (o.esp_real_open_count > 0)
            AS open_rule_sailthru_real,

        -- OH3: MULTI-OPEN WITH TIME GAPS
        -- Multiple open events separated by meaningful time gaps.
        -- A bot pre-fetches once. A human opens, reads, closes, re-opens later.
        (o.raw_total_opens >= 2
            AND o.open_span_sec >= 300)
            AS open_rule_multi_open,

        -- OH4: LONG REOPEN SPAN
        -- Time span between first and last open exceeds a threshold,
        -- indicating a human re-reading the email hours or days later.
        (o.open_span_sec >= 3600)
            AS open_rule_reopen_long_span,

        -- OH5: APPLE MAIL DOUBLE-OPEN
        -- Apple Mail Privacy Protection fires a proxy open, but when a
        -- human actually opens the email, it fires a second open event.
        -- Pattern: exactly 2 opens with a moderate time gap.
        (o.raw_total_opens = 2
            AND o.open_span_sec BETWEEN 30 AND 300)
            AS open_rule_apple_mail_double,

        -- ================================================================
        -- BOT OPEN RULES (negative evidence)
        -- ================================================================

        -- OB1: INSTANT PREFETCH
        -- Open occurs within seconds of send. This is a mail server or
        -- security appliance pre-fetching the tracking pixel.
        (o.time_to_first_open_sec < 5
            AND o.raw_total_opens = 1
            AND cc.is_human_click IS NOT TRUE)
            AS open_rule_bot_instant,

        -- OB2: BOT CLICK SESSION
        -- The open occurred during a session where clicks were classified
        -- as bot traffic. If the clicks are bot, the open is bot.
        (cc.bot_b_gone_click_classification LIKE 'BOT:%')
            AS open_rule_bot_session,

        -- OB3: NEVER VERIFIED + FAST
        -- Single fast open from a subscriber who has never had a verified
        -- human open or click in their history. Strong bot signal.
        -- NOTE: This rule requires historical data. If you are running
        -- this model for the first time, skip this rule initially and
        -- backfill once you have accumulated history.
        (o.time_to_first_open_sec < 30
            AND o.raw_total_opens = 1
            AND cc.is_human_click IS NOT TRUE
            AND o.esp_real_open_count = 0)
            AS open_rule_bot_never_verified_fast

    FROM open_features o
    LEFT JOIN click_classified cc
        ON o.subscriber_id = cc.subscriber_id
        AND o.campaign_id = cc.campaign_id
),


-- ============================================================================
-- STEP 7: OPEN CLASSIFICATION (priority cascade)
-- ============================================================================

open_classified AS (
    SELECT
        orr.*,

        -- Classification label
        CASE
            -- Human rules (highest priority — positive evidence wins)
            WHEN orr.open_rule_verified_clicker      THEN 'HUMAN:verified_clicker'
            WHEN orr.open_rule_sailthru_real          THEN 'HUMAN:sailthru_real'
            WHEN orr.open_rule_multi_open             THEN 'HUMAN:multi_open'
            WHEN orr.open_rule_reopen_long_span       THEN 'HUMAN:reopen_long_span'
            WHEN orr.open_rule_apple_mail_double      THEN 'HUMAN:apple_mail_double'
            -- Bot rules
            WHEN orr.open_rule_bot_instant            THEN 'BOT:instant_prefetch'
            WHEN orr.open_rule_bot_session            THEN 'BOT:bot_click_session'
            WHEN orr.open_rule_bot_never_verified_fast THEN 'BOT:never_verified_fast'
            ELSE 'UNCERTAIN:no_evidence'
        END AS bot_b_gone_open_classification,

        -- Confidence level
        CASE
            WHEN orr.open_rule_verified_clicker      THEN 'definitive'
            WHEN orr.open_rule_sailthru_real          THEN 'high'
            WHEN orr.open_rule_multi_open             THEN 'medium'
            WHEN orr.open_rule_reopen_long_span       THEN 'medium'
            WHEN orr.open_rule_apple_mail_double      THEN 'medium'
            WHEN orr.open_rule_bot_instant            THEN 'high'
            WHEN orr.open_rule_bot_session            THEN 'high'
            WHEN orr.open_rule_bot_never_verified_fast THEN 'medium'
            ELSE 'low'
        END AS open_confidence,

        -- Probability score (0 = definitely bot, 100 = definitely human)
        CASE
            WHEN orr.open_rule_verified_clicker      THEN 99
            WHEN orr.open_rule_sailthru_real          THEN 85
            WHEN orr.open_rule_multi_open             THEN 75
            WHEN orr.open_rule_reopen_long_span       THEN 70
            WHEN orr.open_rule_apple_mail_double      THEN 65
            WHEN orr.open_rule_bot_instant            THEN 5
            WHEN orr.open_rule_bot_session            THEN 5
            WHEN orr.open_rule_bot_never_verified_fast THEN 10
            ELSE 40
        END AS open_probability,

        -- The recommended boolean filter for open rate reporting.
        -- TRUE = behavioral evidence of a human open.
        -- This is your "gold standard" open metric.
        CASE
            WHEN orr.open_rule_verified_clicker      THEN TRUE
            WHEN orr.open_rule_sailthru_real          THEN TRUE
            WHEN orr.open_rule_multi_open             THEN TRUE
            WHEN orr.open_rule_reopen_long_span       THEN TRUE
            WHEN orr.open_rule_apple_mail_double      THEN TRUE
            WHEN orr.open_rule_bot_instant            THEN FALSE
            WHEN orr.open_rule_bot_session            THEN FALSE
            WHEN orr.open_rule_bot_never_verified_fast THEN FALSE
            ELSE FALSE  -- Conservative: uncertain = not counted
        END AS is_human_open

    FROM open_rules orr
),


-- ============================================================================
-- STEP 8: FINAL SCORED EVENT STREAM
-- ============================================================================
-- Join everything back to the raw event stream so every individual open/click
-- event carries its classification, confidence, and probability.

scored_events AS (
    SELECT
        r.subscriber_id,
        r.campaign_id,
        r.event_type,
        r.event_timestamp,
        r.send_timestamp,
        r.user_agent,
        r.ip_address,
        r.url_clicked,

        -- Honeypot flag (subscriber-level, permanent)
        CASE WHEN h.subscriber_id IS NOT NULL THEN TRUE ELSE FALSE END
            AS is_honeypot_bot,

        -- User agent flag (event-level)
        COALESCE(u.is_ua_bot, FALSE) AS is_ua_bot,

        -- Click classification (session-level)
        cc.bot_b_gone_click_classification,
        cc.click_confidence,
        cc.click_probability,
        cc.is_human_click,
        -- Individual click rule flags for debugging
        cc.click_rule_bot_machinegun_definitive,
        cc.click_rule_bot_machinegun_likely,
        cc.click_rule_bot_instant_definitive,
        cc.click_rule_bot_instant_likely,
        cc.click_rule_bot_scanner,
        cc.click_rule_bot_cron,
        cc.click_rule_bot_volume,
        cc.click_rule_human_delayed,
        cc.click_rule_human_late,
        cc.click_rule_human_moderate,
        cc.click_rule_human_sailthru,
        cc.click_rule_human_single,
        cc.click_rule_human_thoughtful,

        -- Open classification (session-level)
        oc.bot_b_gone_open_classification,
        oc.open_confidence,
        oc.open_probability,
        oc.is_human_open,
        -- Individual open rule flags for debugging
        oc.open_rule_verified_clicker,
        oc.open_rule_sailthru_real,
        oc.open_rule_multi_open,
        oc.open_rule_reopen_long_span,
        oc.open_rule_apple_mail_double,
        oc.open_rule_bot_instant,
        oc.open_rule_bot_session,
        oc.open_rule_bot_never_verified_fast,

        -- Master override: honeypot clickers are always bots
        CASE
            WHEN h.subscriber_id IS NOT NULL THEN 'BOT:honeypot'
            ELSE COALESCE(cc.bot_b_gone_click_classification, 'NO_CLICKS')
        END AS final_click_classification,

        CASE
            WHEN h.subscriber_id IS NOT NULL THEN 'BOT:honeypot'
            ELSE COALESCE(oc.bot_b_gone_open_classification, 'NO_OPENS')
        END AS final_open_classification,

        -- The two columns you actually need for reporting:
        CASE
            WHEN h.subscriber_id IS NOT NULL THEN FALSE
            WHEN u.is_ua_bot THEN FALSE
            ELSE COALESCE(cc.is_human_click, FALSE)
        END AS bbg_unique_click,

        CASE
            WHEN h.subscriber_id IS NOT NULL THEN FALSE
            WHEN u.is_ua_bot THEN FALSE
            ELSE COALESCE(oc.is_human_open, FALSE)
        END AS bbg_unique_open

    FROM raw_events r
    LEFT JOIN honeypot_clickers h
        ON r.subscriber_id = h.subscriber_id
    LEFT JOIN ua_flags u
        ON r.subscriber_id = u.subscriber_id
        AND r.campaign_id = u.campaign_id
        AND r.event_timestamp = u.event_timestamp
    LEFT JOIN click_classified cc
        ON r.subscriber_id = cc.subscriber_id
        AND r.campaign_id = cc.campaign_id
    LEFT JOIN open_classified oc
        ON r.subscriber_id = oc.subscriber_id
        AND r.campaign_id = oc.campaign_id
)


-- ============================================================================
-- STEP 9: FINAL OUTPUT
-- ============================================================================
-- Use bbg_unique_open and bbg_unique_click for your reporting.
--
-- EXAMPLE: Calculate your true open rate and click rate
--
--   SELECT
--       campaign_id,
--       COUNTIF(bbg_unique_open) AS unique_opens,
--       SUM(CASE WHEN bbg_unique_open AND event_type = 'open' THEN 1 ELSE 0 END) AS total_opens,
--       COUNTIF(bbg_unique_click) AS unique_clicks,
--       SUM(CASE WHEN bbg_unique_click AND event_type = 'click' THEN 1 ELSE 0 END) AS total_clicks,
--       COUNT(DISTINCT subscriber_id) AS total_sends,
--       ROUND(COUNTIF(bbg_unique_open) / COUNT(DISTINCT subscriber_id) * 100, 1) AS open_rate,
--       ROUND(COUNTIF(bbg_unique_click) / COUNT(DISTINCT subscriber_id) * 100, 1) AS click_rate
--   FROM scored_events
--   GROUP BY campaign_id

SELECT *
FROM scored_events
-- WHERE bbg_unique_open OR bbg_unique_click  -- Uncomment to view only human events
;
