# Bot-B-Gone 🤥

**An open-source framework for filtering bot noise and finding the true human open rate of your email newsletter.**

> *"Your open rate is a lie. A structural distortion built into every major ESP. And almost nobody in this industry will talk about it plainly."* — [Read the full manifesto in A Media Operator](#)

## The Problem

More than half of all email opens are bots. Up to 97% of clicks are non-human. Security scanners, link prefetchers, and corporate mail proxies are firing pixels and following links before a person ever touches the email.

Every Email Service Provider (ESP) has a bot filter. But ESPs are incentivized to show you a number that feels good, not a number that is perfectly accurate. High open rates keep customers happy. 

If you are a publisher or platform selling advertising against those numbers, you are flying on instruments you didn't build, can't inspect, and don't control.

## The Solution

Bot-B-Gone is a shared industry framework for taking back control of your data. It provides:

1. **The Methodology:** How to use active honeypots, click timing, velocity signals, and behavioral patterns to find your true baseline.
2. **The Algorithm:** A production-grade SQL model that classifies every click and open event using 13 click rules (7 bot + 6 human) and 8 open rules (3 bot + 5 human), with confidence levels and probability scores.
3. **The ESP Guide:** Exactly what raw data to demand from Sailthru, SendGrid, Mailchimp, beehiiv, and others.
4. **The Standard:** A "Confidence Scorecard" for publishers to transparently report their metrics to advertisers.
5. **The Measurement Kit:** SQL queries to calculate your own FP/FN rates, plus 11 data-backed charts you can regenerate.

## Repository Structure

* `/model/bot_b_gone_filter.sql` - **The full production algorithm.** 13 click rules, 8 open rules, confidence scoring, probability scores. Drop it into your data warehouse and adapt to your ESP schema.
* `/docs/METHODOLOGY.md` - The core concepts: honeypots, click timing signals, velocity analysis, open classification, and the complete rule reference with data tables.
* `/docs/MEASURING_SUCCESS.md` - The four output metrics, probability score tuning, the Confidence Scorecard, and SQL queries to measure your own FP/FN rates.
* `/docs/ESP_GUIDE.md` - How to extract the necessary raw event data from major ESPs.
* `/docs/THE_ESP_CHALLENGE.md` - The open challenge to ESPs.
* `/charts/generate_charts.py` - Generates all 11 visualizations. Run `python3 generate_charts.py` to regenerate.
* `/charts/` - 11 pre-generated charts: timing distributions, rule breakdowns, precision/recall, probability distributions, tradeoff curves, and the Pinocchio Scorecard.

## Quick Start

```sql
-- 1. Adapt the raw_events CTE in /model/bot_b_gone_filter.sql to your ESP schema
-- 2. Run the query against your data warehouse
-- 3. Use the two output columns for reporting:
SELECT
    campaign_id,
    COUNTIF(bbg_unique_open) AS unique_opens,
    COUNTIF(bbg_unique_click) AS unique_clicks,
    COUNT(DISTINCT subscriber_id) AS total_sends,
    ROUND(COUNTIF(bbg_unique_open) / COUNT(DISTINCT subscriber_id) * 100, 1) AS open_rate,
    ROUND(COUNTIF(bbg_unique_click) / COUNT(DISTINCT subscriber_id) * 100, 1) AS click_rate
FROM scored_events
GROUP BY campaign_id
```

## Why Open Source?

Display advertising went through this exact reckoning 15 years ago with impression fraud. The response was to create shared standards (like the IAB). Email needs the same thing. 

There is nothing proprietary about pattern matching on user agents, IP behavior, timing signatures, and click cadences. The only reason these models are black boxes is because transparency would expose how much noise each platform lets through.

## How to Contribute

If you are a publisher, send this to your data team. 
If you are an engineer, **please star the repo, fork it, and contribute.** Add the patterns you're seeing from your ESP. Help us build the shared standard this industry desperately needs.

---
*Bot-B-Gone was initiated to help publishers build trust with advertisers through radical data transparency.*

## The ESP Challenge

We don't want to fight ESPs. We want to align incentives. We are challenging every major ESP to run the Bot-B-Gone Framework against their own data and publish the results. 

[Read the full ESP Challenge here.](/docs/THE_ESP_CHALLENGE.md)
