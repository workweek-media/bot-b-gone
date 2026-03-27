-- ============================================================================
-- PROJECT BOT-B-GONE: Reference Bot-Filtering Algorithm (SQL)
-- ============================================================================
-- This is a foundational model for filtering raw email event data.
-- It is designed to be adapted to your specific ESP's schema.
-- 
-- Core logic:
-- 1. Identify known bot user agents
-- 2. Identify impossible timing (velocity)
-- 3. Identify honeypot clickers
-- 4. Filter the raw event stream
-- ============================================================================

WITH raw_events AS (
    -- Replace this with your ESP's raw event table
    SELECT 
        subscriber_id,
        campaign_id,
        event_type, -- 'open' or 'click'
        event_timestamp,
        delivery_timestamp,
        user_agent,
        ip_address,
        url_clicked -- null for opens
    FROM esp_raw_events
),

-- 1. VELOCITY FILTER: Bots operate on machine time
velocity_flags AS (
    SELECT 
        subscriber_id,
        campaign_id,
        event_timestamp,
        -- Flag events that occur within 3 seconds of delivery
        CASE 
            WHEN TIMESTAMP_DIFF(event_timestamp, delivery_timestamp, SECOND) < 3 THEN TRUE 
            ELSE FALSE 
        END as is_velocity_bot
    FROM raw_events
),

-- 2. USER AGENT FILTER: Known bot signatures
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
            WHEN LOWER(user_agent) LIKE '%googleimageproxy%' THEN TRUE
            -- Note: Apple Mail Privacy Protection (MPP) requires special handling.
            -- It is technically a proxy, but it masks real humans. 
            -- Bot-B-Gone recommends keeping MPP opens but flagging them for separate analysis.
            ELSE FALSE
        END as is_ua_bot
    FROM raw_events
),

-- 3. HONEYPOT FILTER: The undeniable ground truth
honeypot_clickers AS (
    SELECT DISTINCT subscriber_id
    FROM raw_events
    WHERE event_type = 'click'
    AND (
        -- Condition A: Clicked an invisible 1x1 pixel link
        url_clicked LIKE '%/tracking-pixel.png%'
        OR 
        -- Condition B: Clicked a generic footer link that humans ignore
        url_clicked = 'https://yourdomain.com/generic-footer-link'
    )
),

-- 4. AGGREGATE FLAGS
scored_events AS (
    SELECT 
        r.*,
        v.is_velocity_bot,
        u.is_ua_bot,
        CASE WHEN h.subscriber_id IS NOT NULL THEN TRUE ELSE FALSE END as is_honeypot_bot,
        
        -- Final Bot-B-Gone Classification
        CASE 
            WHEN v.is_velocity_bot THEN 'BOT_VELOCITY'
            WHEN u.is_ua_bot THEN 'BOT_USER_AGENT'
            WHEN h.subscriber_id IS NOT NULL THEN 'BOT_HONEYPOT'
            ELSE 'HUMAN_ESTIMATE'
        END as bot_b_gone_classification
        
    FROM raw_events r
    LEFT JOIN velocity_flags v USING (subscriber_id, campaign_id, event_timestamp)
    LEFT JOIN ua_flags u USING (subscriber_id, campaign_id, event_timestamp)
    LEFT JOIN honeypot_clickers h USING (subscriber_id)
)

-- 5. FINAL OUTPUT: The Cleaned Event Stream
SELECT *
FROM scored_events
-- WHERE bot_b_gone_classification = 'HUMAN_ESTIMATE' -- Uncomment to view only clean data
;
