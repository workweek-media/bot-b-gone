# V2 Architecture — How It Works

## The Core Insight

Email bot detection has two fundamentally different problems:

**Opens** are hard because Apple Mail Privacy Protection (MPP) pre-fetches every tracking pixel. ~57% of all opens are Apple Mail, and most are automated. You can't subtract bots from a noisy total — you must find behavioral evidence that a human actually read the email.

**Clicks** are easier because click patterns are highly deterministic. Bots click every link in <5 seconds. Humans click 1-2 links minutes or hours later. But the tricky cases are corporate security scanners that create phantom clicks from real subscribers.

## Open Classification (4 Layers)

### Layer 1: MPP Stripping

Every Apple Mail open is classified as BOT unless it has a confirming signal:
- **Click confirmation**: The subscriber also clicked a link in the same blast
- **Circadian confirmation**: The subscriber is a known human (from Layer 3) AND the open occurred during peak reading hours (7am-12pm, 5pm-8pm)

Why these hours? Human email reading follows a circadian pattern — morning commute, lunch break, evening wind-down. Bot pre-fetches fire uniformly across all hours or within seconds of delivery.

**Impact**: This single layer moves the open rate from ~46% to ~30%. It's the biggest lever.

### Layer 2: Scanner Session Detection

Corporate email scanners (Barracuda, Mimecast, Proofpoint) open every email within seconds of delivery to scan for malware. These create real open events that look like engagement but aren't.

**Rule**: If an open fires within 10 seconds of send AND the subscriber has a history of bot-like behavior (from the GMM clustering) AND there's no follow-up click → BOT.

### Layer 3: Per-Open ML Model

A Random Forest classifier trained on your own data using the GMM subscriber labels as weak supervision. It uses 10 features:

| Feature | Why It Matters |
|---------|---------------|
| `seconds_after_send` | Bots fire within seconds; humans open minutes to hours later |
| `device_is_apple_mail` | Apple MPP creates fake opens |
| `device_is_gmail` | Gmail has its own proxy pre-fetching |
| `device_is_bot` | Known bot user agents (IE, "Other", Windows Tablet) |
| `device_is_iphone` | Mobile opens are more likely human |
| `device_is_android` | Mobile opens are more likely human |
| `had_click` | Click confirms the open is real |
| `hour_of_open` | Humans read on circadian patterns |
| `is_weekday` | Weekday vs weekend patterns differ |
| `click_within_5s_of_open` | Click immediately after open = bot scanner |

The model handles three subscriber categories from the GMM:
- **V7-Human**: Trusted as human → open is human
- **V7-Bot + model recovery**: If the ML model disagrees with the GMM label, the open is recovered as human
- **Unknown**: ML model decides

### Layer 4: Community Override

Subscribers with ground-truth human signals are always classified as human:
- Paid subscribers / members
- Event registrants or attendees
- Community contributors
- Anyone with a verified purchase

**Why this matters**: 13% of verified humans have bot-like email patterns because their corporate scanners create the bot activity. The community override prevents false positives on your most valuable subscribers.

## Click Classification (3 Steps)

### Step 1: Known Human (Never Filtered)

These clicks bypass all models — they are protected:

| Signal | Threshold | Rationale |
|--------|-----------|-----------|
| Community verified | Always | Ground truth |
| Single content click, no social, >60s | Always | Strongest human pattern (20% of all clicks) |
| Non-bot subscriber with open, >15s | Always | Normal subscriber behavior |
| Selective bot-sub, single content click | >60s, click rate <10% | Real human behind corporate scanner |
| Late deliberate click | >1 hour, has open | Nobody reads email at 3am then clicks 2 hours later by accident |
| Super fan | 2+ clicks, span >30min | Humans browse over time, bots click everything instantly |

### Step 2: Known Bot (Always Filtered)

| Signal | Threshold | Rationale |
|--------|-----------|-----------|
| Fast multi-click with social link | 3+ clicks in <60s, includes social | Scanners click every link including footer icons |
| All URL types clicked fast | 4+ types in <5min | No human clicks content, social, unsub, AND house ads in 5 min |
| Social-only clicker | All clicks are social footer links | 99.5% of social footer clicks are bots |
| Social bot subscriber | 80%+ historical social clicks | Subscriber-level honeypot signal |
| Early phantom | No open, <5min after send | Scanner pre-clicking without rendering |
| Instant click | <5s after send | Physically impossible for human |

### Step 3: Uncertain Pool (ML + Empirical Grid)

Clicks that aren't clearly human or bot are scored using two blended signals:

**ML Model (50% weight)**: Random Forest with 14 click features, trained on your data.

**Empirical Click Grid (50% weight)**: A universal lookup table mapping (click_count × time_after_send) to bot probability. Built from millions of click events:

```
             <5s   <10s   <30s    <1m    <5m   <30m    <1h    <1d   >1d
1 click      87%    89%    78%    65%    51%    46%    47%    49%   50%
2 clicks     98%    97%    97%    98%    80%    56%    53%    54%   53%
3 clicks     99%   100%    99%    98%    92%    76%    72%    68%   70%
6-10 clicks  99%    99%   100%   100%   100%    98%    91%    77%   76%
10+ clicks   99%    99%    99%   100%    99%    97%    95%    93%   88%
```

(Values are bot probability. A 1-click event <5s after send is 87% likely bot.)

For subscribers flagged as bots by the GMM, the grid gets 70% weight (their ML model predictions are less reliable).

**Decision threshold**: Combined score < 0.40 → human.

## Total Opens

Once you have unique human openers, total opens is simple:

```
total_human_opens = unique_human_openers × raw_opens_per_opener
```

For confirmed human openers, all their raw open events are human. The `opens-per-opener` ratio is a content quality metric — a 2.5x ratio means your audience re-reads content across multiple sessions and devices.

## The Social Honeypot Signal

This is the most powerful click signal we discovered. Every newsletter has social footer links (LinkedIn, Twitter, Instagram). These are quasi-honeypots:

- **99.5% of social footer link clicks are bots**
- Subscribers who ONLY click social links across multiple blasts are definitively bots
- Subscribers with >80% of their historical clicks on social links are bots

To build this signal from your own data:
1. Parse the URLs from your click data
2. Classify each URL as `social`, `content`, `unsub`, or `house_ad`
3. For each subscriber, compute `social_clicks / total_clicks`
4. Subscribers with ratio > 0.80 → social bot

## The Subscriber Click Profile

A subscriber who clicks in 2% of the emails they receive is a real human who clicks selectively. A subscriber who clicks in 60% of emails is almost certainly a corporate scanner.

To build this signal:
1. For each subscriber, count: total blasts received, total blasts clicked
2. `click_rate = blasts_clicked / blasts_received`
3. Subscribers with click_rate < 10% are "selective" — their deliberate clicks should be protected
4. Subscribers with click_rate > 30% are "habitual" — likely scanners

This profile is critical for the "real human behind a corporate scanner" case. A bank employee's corporate scanner creates bot-like patterns, but they personally click 1 link in 5% of newsletters they receive. That 5% click rate tells you they're human.
