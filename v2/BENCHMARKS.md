# Industry Benchmarks & Validation

Use these benchmarks to validate your bot detection model. If your numbers are outside these ranges, something is likely misconfigured.

## Open Rate Benchmarks

| Publisher Type | Raw OR (ESP reported) | True Human OR | Bot Open % |
|---|---|---|---|
| B2B Newsletter (long-form) | 40-55% | 18-28% | 40-60% |
| B2C Newsletter | 35-50% | 20-35% | 30-50% |
| Media/News | 30-45% | 15-25% | 40-60% |
| E-commerce | 25-40% | 15-25% | 30-50% |

Source: Omeda reports ~22% for B2B after proper filtering. Industry consensus from Validity, SparkPost, and deliverability experts is that 40-60% of email opens are non-human.

## Click Rate Benchmarks

| Publisher Type | Raw CTR | True Human CTR | Bot Click % |
|---|---|---|---|
| B2B Newsletter | 2-5% | 0.8-2.0% | 50-80% |
| B2C Newsletter | 1-3% | 0.5-1.5% | 40-70% |

## CTOR (Click-to-Open Rate) — The Math Check

CTOR = Unique Clicks / Unique Opens

| Content Type | Expected CTOR |
|---|---|
| B2B long-form editorial | 4-7% |
| B2C promotional | 2-5% |
| News digest | 3-6% |

**If your CTOR is below the floor for your format, your open rate is inflated.** A B2B publisher reporting 45% open rate and 1% click rate has a 2.2% CTOR — well below the 4-7% benchmark. The real open rate is closer to 14-25%.

## Opens-Per-Opener (Content Quality)

| Content Quality | Opens/Opener |
|---|---|
| Average newsletter | 1.5-2.0x |
| Good engagement | 2.0-2.5x |
| Exceptional (viral/must-read) | 3.0-5.0x |

This metric measures re-reads across devices and sessions. A 2.5x means your average reader comes back 1-2 additional times.

## Validation Checklist

After running the model, check these:

- [ ] **Unique open rate between 18-30%** for B2B (if higher, increase MPP stripping aggressiveness)
- [ ] **Bot open % between 40-60%** (if lower, you're not catching Apple MPP)
- [ ] **CTOR between 4-7%** for B2B (if lower, opens are still inflated)
- [ ] **Verified humans pass at >97%** (paying subscribers, event attendees should almost never be filtered)
- [ ] **Social footer links are >95% bot** (if not, check your URL classification)
- [ ] **Single content clicks >60s are >80% human** (if not, check your human rules)
- [ ] **Opens-per-opener between 2.0-3.0x** (if below 1.5x, you may be over-filtering re-reads)

## FP/FN Rate Targets

| Metric | Conservative | Balanced | Aggressive |
|---|---|---|---|
| Open FP (killing real reads) | <2% | <3% | <5% |
| Open FN (keeping bot opens) | <5% | <5% | <10% |
| Click FN (keeping bot clicks) | <1% | <2% | <5% |
| Click FP (killing real clicks) | <15% | <25% | <35% |

"Conservative" prioritizes never losing a real engagement (ideal for advertiser-facing metrics).
"Aggressive" prioritizes catching every bot (ideal for internal analytics).
