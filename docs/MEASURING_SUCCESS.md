# Measuring Success: The Confidence Scorecard

The goal of Bot-B-Gone is not just to build a better filter, but to create a **shared reporting standard** for the publishing industry. 

When a publisher tells an advertiser, "We have a 25% open rate," that number is meaningless without knowing the filter strictness. We need a nutrition label for email metrics.

## 1. The Four Output Metrics

After running the Bot-B-Gone filter against your raw ESP data, you get four clean metrics per campaign. These are the numbers you should report.

| Metric | What It Means | How to Calculate |
|:---|:---|:---|
| **Unique Opens** | Count of subscribers confirmed as human openers | `COUNTIF(bbg_unique_open)` |
| **Total Opens** | Total human open events (includes re-reads) | `SUM(CASE WHEN bbg_unique_open AND event_type = 'open' THEN 1 ELSE 0 END)` |
| **Unique Clicks** | Count of subscribers confirmed as human clickers | `COUNTIF(bbg_unique_click)` |
| **Total Clicks** | Total human click events (includes multi-clicks) | `SUM(CASE WHEN bbg_unique_click AND event_type = 'click' THEN 1 ELSE 0 END)` |

From these, derive your standard rates:

```sql
SELECT
    campaign_id,
    COUNTIF(bbg_unique_open) AS unique_opens,
    COUNTIF(bbg_unique_click) AS unique_clicks,
    COUNT(DISTINCT subscriber_id) AS total_sends,
    ROUND(COUNTIF(bbg_unique_open) / COUNT(DISTINCT subscriber_id) * 100, 1) AS open_rate,
    ROUND(COUNTIF(bbg_unique_click) / COUNT(DISTINCT subscriber_id) * 100, 1) AS click_rate,
    ROUND(COUNTIF(bbg_unique_click) / NULLIF(COUNTIF(bbg_unique_open), 0) * 100, 1) AS ctor
FROM scored_events
GROUP BY campaign_id
```

## 2. Understanding the Classification System

Every subscriber-campaign pair receives two classifications: one for clicks and one for opens. Each classification includes a label, a confidence level, and a probability score.

### Click Classifications

The model applies 7 bot rules and 6 human rules in a priority cascade. The first rule that fires wins.

| Classification | Type | Confidence | Probability |
|:---|:---|:---|:---|
| `BOT:machinegun` | Bot | Definitive | 0 |
| `BOT:instant_prefetch` | Bot | Definitive | 0 |
| `BOT:machinegun_likely` | Bot | High | 5 |
| `BOT:instant_likely` | Bot | High | 5 |
| `BOT:url_scanner` | Bot | High | 5 |
| `BOT:cron_burst` | Bot | Medium | 10 |
| `BOT:high_volume` | Bot | Medium | 10 |
| `HUMAN:sailthru_confirmed` | Human | High | 95 |
| `HUMAN:thoughtful_multi` | Human | High | 90 |
| `HUMAN:delayed_single` | Human | High | 90 |
| `HUMAN:late_arrival` | Human | High | 85 |
| `HUMAN:single_moderate` | Human | Medium | 70 |
| `HUMAN:single_selective` | Human | Medium | 65 |
| `UNCLASSIFIED:ambiguous` | Unknown | Low | 40 |

*(See `/charts/click_rule_breakdown.png` for a visual breakdown of what each rule catches)*

### Open Classifications

Opens are harder than clicks because Apple Mail Privacy Protection fires proxy opens. The model uses behavioral evidence to confirm human opens rather than trying to subtract bots.

| Classification | Type | Confidence | Probability |
|:---|:---|:---|:---|
| `HUMAN:verified_clicker` | Human | Definitive | 99 |
| `HUMAN:sailthru_real` | Human | High | 85 |
| `HUMAN:multi_open` | Human | Medium | 75 |
| `HUMAN:reopen_long_span` | Human | Medium | 70 |
| `HUMAN:apple_mail_double` | Human | Medium | 65 |
| `BOT:instant_prefetch` | Bot | High | 5 |
| `BOT:bot_click_session` | Bot | High | 5 |
| `BOT:never_verified_fast` | Bot | Medium | 10 |
| `UNCERTAIN:no_evidence` | Unknown | Low | 40 |

*(See `/charts/open_rule_breakdown.png` for a visual breakdown of open classifications)*

## 3. Using Probability Scores to Tune Your Filter

The binary `bbg_unique_open` and `bbg_unique_click` columns use our recommended thresholds. But every publisher's tolerance is different. The probability scores (0–100) let you set your own.

| Threshold | What You Get | Best For |
|:---|:---|:---|
| `probability >= 85` | Only high-confidence humans | Media kits, guaranteed CPM campaigns |
| `probability >= 65` | Balanced — includes moderate-confidence humans | Internal analytics, editorial decisions |
| `probability >= 40` | Generous — includes uncertain events | Total reach estimates, longitudinal analysis |
| `probability >= 0` | Everything (no filtering) | Debugging, comparing to ESP raw numbers |

**Example: Custom strict open rate**
```sql
SELECT
    campaign_id,
    COUNTIF(open_probability >= 85) AS strict_unique_opens,
    COUNT(DISTINCT subscriber_id) AS total_sends,
    ROUND(COUNTIF(open_probability >= 85) / COUNT(DISTINCT subscriber_id) * 100, 1) AS strict_open_rate
FROM scored_events
GROUP BY campaign_id
```

*(See `/charts/probability_distribution.png` for the bimodal distribution of scores)*

## 4. The Confidence Scorecard

We propose that publishers adopt the **Confidence Scorecard** when reporting metrics to advertisers. This framework forces transparency about the False Positive (FP) and False Negative (FN) tradeoff.

### The Three Tiers of Confidence

**Tier 1: Sell Against It (70%+ Confidence)**
* **Profile:** Strict filtering. High false negative rate (you are missing real readers), but near-zero false positive rate (you are counting almost no bots).
* **Use Case:** Media kits, rate cards, guaranteed CPM campaigns.
* **The Pitch:** "Our reported open rate is 18%. We know the actual number is higher because of Apple MPP, but we have filtered out all bot noise. Every open you pay for is a guaranteed human."

**Tier 2: Directional Use (50% - 70% Confidence)**
* **Profile:** Moderate filtering. Balances FP and FN to find the mathematical "best estimate" of the true audience size.
* **Use Case:** Internal analytics, comparing newsletter performance, editorial decisions.
* **The Pitch:** "This is our best estimate of our true reach, but it contains a margin of error we aren't comfortable charging advertisers for."

**Tier 3: Vanity Metric (Below 30% Confidence)**
* **Profile:** Loose or no filtering. This is what most ESP dashboards report by default.
* **Use Case:** None. 
* **The Pitch:** "This number looks great, but it is heavily contaminated by security scanners."

*(See `/charts/confidence_decay.png` and `/charts/the_tradeoff.png` for the visual curves)*

## 5. How to Report Your Scorecard

When sharing data publicly or with partners, use this standard format:

```text
Newsletter: [Name]
Reporting Period: [Q1 2026]
List Size: [50,000 subscribers]

ESP-Reported Open Rate:           [42%]
Bot-B-Gone Filtered Open Rate:    [18%]    (bbg_unique_open)
Bot-B-Gone Filtered Click Rate:   [1.1%]   (bbg_unique_click)
Implied CTOR:                     [6.1%]

Bot-B-Gone Confidence Score:      [72%]    (Tier 1)
Estimated False Positive Rate:    [<5%]
Estimated False Negative Rate:    [~25%]
Probability Threshold Used:       [65]     (default)
```

### Example: The Honest Media Kit

> *"Our ESP reports a 45% open rate. However, using the Bot-B-Gone Framework, we apply a strict bot filter to ensure advertiser ROI. Our **Guaranteed Human Open Rate is 18%** (Tier 1 Confidence, <5% False Positive Rate). This yields a highly engaged 6.5% CTOR."*

By adopting this standard, publishers turn honesty into a competitive advantage. Advertisers will quickly learn to ask: *"Is your 40% open rate a Tier 1 or a Tier 3?"*

## 6. How to Measure Your Own False Positive & False Negative Rates

You do not need to trust our numbers. Here is how to measure your own FP and FN rates using your data.

### Step 1: Establish Ground Truth with Honeypots

Deploy a honeypot link (see `/docs/METHODOLOGY.md`, Section 1). Any subscriber who clicks it is a **confirmed bot**. Any subscriber who clicks a content link and *does not* click the honeypot is a **confirmed human**.

### Step 2: Run the Filter

Apply the Bot-B-Gone SQL model to your raw event data. This produces `bbg_unique_click` and `bbg_unique_open` for every subscriber-campaign pair.

### Step 3: Compare Against Ground Truth

```sql
-- False Positive Rate: What % of honeypot-confirmed bots did we MISS?
SELECT
    ROUND(
        COUNTIF(bbg_unique_click = TRUE AND is_honeypot_bot = TRUE)
        / NULLIF(COUNTIF(is_honeypot_bot = TRUE), 0) * 100, 2
    ) AS false_positive_rate_pct

FROM scored_events
WHERE event_type = 'click';

-- False Negative Rate: What % of confirmed humans did we FLAG as bot?
SELECT
    ROUND(
        COUNTIF(bbg_unique_click = FALSE AND is_honeypot_bot = FALSE
                AND bot_b_gone_click_classification != 'NO_CLICKS')
        / NULLIF(COUNTIF(is_honeypot_bot = FALSE
                AND bot_b_gone_click_classification != 'NO_CLICKS'), 0) * 100, 2
    ) AS false_negative_rate_pct

FROM scored_events
WHERE event_type = 'click';
```

### Step 4: Plot Your Own Tradeoff Curve

Sweep across probability thresholds and measure FP/FN at each level:

```sql
SELECT
    threshold,
    COUNTIF(click_probability >= threshold AND is_honeypot_bot = TRUE)
        / NULLIF(COUNTIF(is_honeypot_bot = TRUE), 0) * 100 AS fp_rate,
    COUNTIF(click_probability < threshold AND is_honeypot_bot = FALSE)
        / NULLIF(COUNTIF(is_honeypot_bot = FALSE), 0) * 100 AS fn_rate
FROM scored_events
CROSS JOIN UNNEST(GENERATE_ARRAY(0, 100, 5)) AS threshold
WHERE bot_b_gone_click_classification != 'NO_CLICKS'
GROUP BY threshold
ORDER BY threshold
```

Feed the output into the chart generator (`/charts/generate_charts.py`) or your own visualization tool to produce your own tradeoff curve.

*(See `/charts/precision_recall.png` for how each rule performs on precision vs. recall)*

## 7. Chart Reference

All charts are generated by `/charts/generate_charts.py`. Run `python3 generate_charts.py` to regenerate.

| Chart | File | What It Shows |
|:---|:---|:---|
| FP/FN Tradeoff | `the_tradeoff.png` | How false positives and false negatives trade off as you loosen the filter |
| Confidence Decay | `confidence_decay.png` | How confidence drops as reported open rate increases |
| CTOR Math Check | `ctor_math.png` | What CTOR your open rate implies at various click rates |
| Pinocchio Scorecard | `scorecard.png` | Example of the standardized reporting format |
| Bot Timeline | `bot_timeline.png` | Bot contamination growth over time (2019–2026) |
| Click Timing | `click_timing_distribution.png` | When bots click vs. humans (time after send) |
| Inter-Click Velocity | `interclick_velocity.png` | Bot vs. human inter-click speed distributions |
| Click Rule Breakdown | `click_rule_breakdown.png` | What % of bot sessions each click rule catches |
| Open Rule Breakdown | `open_rule_breakdown.png` | How opens are classified across all rules |
| Precision vs. Recall | `precision_recall.png` | Every bot click rule plotted by precision and recall |
| Probability Distribution | `probability_distribution.png` | Bimodal distribution of click and open probability scores |
