# Measuring Success: The Confidence Scorecard

The goal of Bot-B-Gone is not just to build a better filter, but to create a **shared reporting standard** for the publishing industry. 

When a publisher tells an advertiser, "We have a 25% open rate," that number is meaningless without knowing the filter strictness. We need a nutrition label for email metrics.

## The Confidence Scorecard

We propose that publishers adopt the **Confidence Scorecard** when reporting metrics to advertisers. This framework forces transparency about the False Positive (FP) and False Negative (FN) tradeoff.

### The Three Tiers of Confidence

**🟢 Tier 1: Sell Against It (70%+ Confidence)**
* **Profile:** Strict filtering. High false negative rate (you are missing real readers), but near-zero false positive rate (you are counting almost no bots).
* **Use Case:** Media kits, rate cards, guaranteed CPM campaigns.
* **The Pitch:** "Our reported open rate is 18%. We know the actual number is higher because of Apple MPP, but we have filtered out all bot noise. Every open you pay for is a guaranteed human."

**🟡 Tier 2: Directional Use (50% - 70% Confidence)**
* **Profile:** Moderate filtering. Balances FP and FN to find the mathematical "best estimate" of the true audience size.
* **Use Case:** Internal analytics, comparing newsletter performance, editorial decisions.
* **The Pitch:** "This is our best estimate of our true reach, but it contains a margin of error we aren't comfortable charging advertisers for."

**🔴 Tier 3: Vanity Metric (Below 30% Confidence)**
* **Profile:** Loose or no filtering. This is what most ESP dashboards report by default.
* **Use Case:** None. 
* **The Pitch:** "This number looks great, but it is heavily contaminated by security scanners."

## How to Report Your Scorecard

When sharing data publicly or with partners, use this standard format:

```text
Newsletter: [Name]
Reported Unique Open Rate: [XX]%
Implied CTOR: [X.X]%

Bot-B-Gone Confidence Score: [XX]% (Tier [1/2/3])
Estimated False Positive Rate: [XX]%
Estimated False Negative Rate: [XX]%
```

### Example: The Honest Media Kit

> *"Our ESP reports a 45% open rate. However, using the Bot-B-Gone Framework, we apply a strict bot filter to ensure advertiser ROI. Our **Guaranteed Human Open Rate is 18%** (Tier 1 Confidence, <5% False Positive Rate). This yields a highly engaged 6.5% CTOR."*

By adopting this standard, publishers turn honesty into a competitive advantage. Advertisers will quickly learn to ask: *"Is your 40% open rate a Tier 1 or a Tier 3?"*
