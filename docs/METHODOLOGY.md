# The Bot-B-Gone Methodology

To find the true human open rate of an email newsletter, you cannot rely on circular logic (using opens to validate opens). You must anchor your model on hard behavioral data.

## 1. The Honeypot Breakthrough

The foundation of this methodology is the **Active Honeypot**. 

Bots click links indiscriminately to check for malware. Humans click links contextually because they are interested in the content. By analyzing the *types* of links clicked, you can establish an undeniable ground truth.

**The Rule:** Any subscriber who clicks an invisible link, or who *only* clicks generic footer links without ever clicking contextual editorial links, is a bot. Period. 

By filtering out these honeypot clickers, you establish your **Hard Floor**—the absolute minimum number of verified human readers.

### Tiered Implementation Strategies

Depending on your technical resources, you can implement honeypots at three different levels:

**Tier 1: The Invisible Pixel (Best)**
* **How:** Place an invisible 1x1 pixel link in your email template, hidden via CSS so humans cannot physically click it.
* **Why:** This is the purest signal. 100% of clicks on this link are bots. 

**Tier 2: The Link-Type Proxy (Good)**
* **How:** Categorize all links in your newsletter as either "Content" (editorial links, ads) or "Generic" (homepage, social icons, unsubscribe, privacy policy). 
* **Why:** If a subscriber has 15 lifetime clicks, and all 15 are on your Twitter icon or homepage link, they are a bot. Humans don't read a newsletter just to click the homepage link every day.

**Tier 3: The Footer Trap (Minimum Viable)**
* **How:** If you cannot modify your template or categorize links at scale, simply place a plain-text URL to your homepage buried deep in the footer.
* **Why:** It requires zero engineering. You just pull a report of everyone who clicked that specific URL and cross-reference it against your active reader list.

## 2. Click Timing & Velocity Signals

Beyond honeypots, the *timing* of click events is one of the strongest bot detection signals available. Bots operate on machine time. Humans do not.

We validated the following signals against 90 days of production data covering ~684,000 click sessions (567K bot, 102K verified human, 16K ambiguous). The results are unambiguous.

### Signal A: Time-to-First-Click After Send

Enterprise security scanners (Barracuda, Mimecast, Proofpoint, etc.) intercept inbound email and pre-click every link to check for malware. This happens *before* the email ever reaches a human inbox.

| Time After Send | Bot Sessions | Human Sessions |
|:---|:---|:---|
| < 2 seconds | 252 | 0 |
| 2–10 seconds | 61,899 | 0 |
| 10–60 seconds | 304,339 | 0 |
| 1–5 minutes | 153,616 | 0 |
| 5–60 minutes | 33,736 | 22,039 |
| 1 hour+ | 13,126 | 73,653 |

**The pattern:** 87% of bot clicks happen between 10 seconds and 5 minutes after the send. **Zero** verified human clicks occur within the first 5 minutes. The earliest human clicks begin trickling in between 5 and 60 minutes, with 72% happening over an hour later.

**Recommended threshold:** Flag any click session where the first click occurs < 60 seconds after `send_timestamp`. This achieves **100% precision** (zero false positives) and catches 64.6% of bots. You can widen to 300 seconds (5 minutes) for 91.7% recall at a 5.8% false positive rate.

**A note on "delivery time" vs. "send time":** The reference SQL uses `send_timestamp` (when the ESP dispatched the email) rather than "delivery time" because (a) send time is universally available across ESPs, and (b) the difference between send and delivery is typically < 5 seconds. If your ESP provides a true delivery timestamp, you may substitute it for slightly tighter accuracy.

### Signal B: Click Clustering ("Machinegun" Detection)

Security bots do not click one link at a time. They click every link in the email in rapid succession. This produces a distinctive "machinegun" pattern: many clicks compressed into a tiny time window.

| Metric | Bot (Machinegun) | Human (Thoughtful Multi-Click) |
|:---|:---|:---|
| Median inter-click interval | 0.72 seconds | 25 seconds |
| Median click span (first to last) | 4.1 seconds | 36 seconds |
| Mean total clicks per session | 11.4 | 2.4 |

**The "5 clicks in 5 seconds" rule:** Out of 567K bot click sessions, 385,230 (67.9%) exhibited 5 or more clicks within a 5-second window. Out of 101,553 verified human sessions, only **6** did this. That is a **99.998% precision** rate with a false positive rate of 0.006%.

For context, the 6 human "false positives" were all sessions with exactly 5 clicks on only 1–2 unique URLs, arriving hours after the send — likely rapid re-clicks on a slow-loading link, not bot behavior. They were classified as human by other strong signals (Sailthru confirmation, late arrival time).

### Signal C: Inter-Click Velocity

For sessions with 2 or more clicks, the average time between consecutive clicks is a powerful discriminator even when the total click count is below the machinegun threshold.

| Inter-Click Interval | Predominantly | Notes |
|:---|:---|:---|
| < 0.5 seconds | Bot | Physically impossible for a human |
| 0.5–2 seconds | Bot (99.7% precision) | Catches 78% of bots, 1.4% FPR |
| 2–10 seconds | Mixed | Ambiguous zone; combine with other signals |
| 10+ seconds | Human | Consistent with reading, scrolling, deciding |

**Recommended threshold:** Flag multi-click sessions where the average inter-click interval is < 2 seconds. This catches 78.4% of bots with 99.7% precision.

### Combined Click Signal Performance

The individual signals are strong, but combining them covers more of the bot population without sacrificing precision:

| Rule Combination | Precision | Recall | FPR |
|:---|:---|:---|:---|
| (5+ clicks in 5s) OR (first click < 60s) | 99.999% | 90.5% | 0.01% |
| Any single rule fires | 99.7%+ | 90%+ | < 1.4% |

## 3. The Complete Click Rule Reference

The SQL model (`/model/bot_b_gone_filter.sql`) applies 7 bot rules and 6 human rules in a priority cascade. Each click session is classified by the first rule that fires.

### Bot Click Rules

| Rule | Code | What It Detects | Confidence |
|:---|:---|:---|:---|
| B1 | `click_rule_bot_machinegun_definitive` | 5+ clicks within 5 seconds | Definitive |
| B2 | `click_rule_bot_machinegun_likely` | 5+ clicks within 10 seconds (but not 5s) | High |
| B3 | `click_rule_bot_instant_definitive` | First click < 2 seconds after send | Definitive |
| B4 | `click_rule_bot_instant_likely` | First click 2–60 seconds after send | High |
| B5 | `click_rule_bot_scanner` | 5+ unique URLs, inter-click 0.5–3s, span < 30s | High |
| B6 | `click_rule_bot_cron` | 5+ clicks, inter-click 2–10s, first click < 5 min | Medium |
| B7 | `click_rule_bot_volume` | 10+ total clicks with avg inter-click < 30s | Medium |

### Human Click Rules

| Rule | Code | What It Detects | Confidence |
|:---|:---|:---|:---|
| H1 | `click_rule_human_delayed` | Single click, 5 min – 24 hr after send | High |
| H2 | `click_rule_human_late` | Any click arriving 24+ hours after send | High |
| H3 | `click_rule_human_moderate` | Single click, 1–5 min after send | Medium |
| H4 | `click_rule_human_sailthru` | ESP flagged click as "real" / human | High |
| H5 | `click_rule_human_single` | Exactly 1 click on 1 URL | Medium |
| H6 | `click_rule_human_thoughtful` | 2+ clicks, avg inter-click 10s+, first click 5 min+ | High |

## 4. Open Classification

Opens are fundamentally harder than clicks because Apple Mail Privacy Protection (MPP) fires a proxy open on nearly every email, making raw unique opens unreliable. Our approach inverts the problem: instead of trying to subtract bots from a noisy total, we look for behavioral evidence that **confirms** a human opened the email.

### Human Open Rules

| Rule | Code | What It Detects | Confidence |
|:---|:---|:---|:---|
| OH1 | `open_rule_verified_clicker` | Subscriber had a human-classified click on the same send | Definitive |
| OH2 | `open_rule_sailthru_real` | ESP flagged the open as "real" | High |
| OH3 | `open_rule_multi_open` | 2+ opens separated by 5+ minutes | Medium |
| OH4 | `open_rule_reopen_long_span` | First-to-last open span exceeds 1 hour | Medium |
| OH5 | `open_rule_apple_mail_double` | Exactly 2 opens, 30s–5 min apart (MPP + real open) | Medium |

### Bot Open Rules

| Rule | Code | What It Detects | Confidence |
|:---|:---|:---|:---|
| OB1 | `open_rule_bot_instant` | Single open < 5 seconds after send, no human click | High |
| OB2 | `open_rule_bot_session` | Open occurred during a bot-classified click session | High |
| OB3 | `open_rule_bot_never_verified_fast` | Fast single open, no ESP real flag, no human click history | Medium |

### The "Uncertain" Category

If none of the above rules fire, the open is classified as `UNCERTAIN:no_evidence`. This typically means a single open event with no corroborating signal — it *could* be a real human (especially an Apple Mail user), but we cannot confirm it.

The conservative approach (recommended for advertiser-facing metrics) is to exclude uncertain opens. The generous approach (useful for total reach estimates) is to include them. The SQL model provides both:
* `is_human_open = TRUE` → conservative (only confirmed human opens)
* `open_probability >= 40` → generous (includes uncertain opens)

## 5. Confidence Scoring & Probability

Every classification carries two additional signals:

**Confidence Level** (`definitive`, `high`, `medium`, `low`): How certain we are about the classification. Use this to tune your filter strictness. For advertiser-facing metrics, you may want to only count `definitive` and `high` confidence human events.

**Probability Score** (0–100): A numeric score where 0 = definitely bot and 100 = definitely human. This allows you to set your own threshold rather than using our binary classification. For example:
* `probability >= 85` → strict (only high-confidence humans)
* `probability >= 65` → balanced (includes moderate-confidence humans)
* `probability >= 40` → generous (includes uncertain events)

## 6. The False Positive vs. False Negative Tradeoff

Every bot-filtering model makes two types of errors:
* **False Positives (FP):** A bot counted as a real human (inflates your rate).
* **False Negatives (FN):** A real human flagged as a bot (deflates your rate).

As you loosen your filter to capture more humans, you inevitably let in more bots. There is a mathematical crossover point where every additional point of open rate you claim adds more bots to your metrics than humans.

*(See `/charts/the_tradeoff.png` for the visual curve)*

## 7. The CTOR Math Check

For B2B publishers sending long-form editorial content, the expected Click-to-Open Rate (CTOR) is between **4% and 7%**. 

If your ESP reports a 40% open rate, but your unique click rate is 1%, your implied CTOR is 2.5%. This is below the floor for long-form media. It means your open rate is mathematically inflated by bot noise.

**The Formula:**
`Implied CTOR = Unique Click Rate / Reported Unique Open Rate`

If your implied CTOR falls below your format's benchmark, you need to tighten your bot filter.

## 8. The Re-Read Multiplier

Because Apple Mail Privacy Protection (MPP) fires a proxy open immediately upon delivery, *unique* opens are fundamentally broken. However, *total* opens still carry massive signal.

When a real human opens your email on their phone, and then opens it again later on their desktop, that generates multiple open events. 

By isolating the cohort of verified human readers (via honeypot clicks), you can calculate your **Re-Read Multiplier**—the average number of total opens generated by a single human reader. For high-quality B2B content, this multiplier often approaches 2.0x. This is a critical metric for proving audience gravity to advertisers.
