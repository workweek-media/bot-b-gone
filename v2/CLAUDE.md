# Bot-B-Gone V2 — Start Here

You are a data analyst or engineer at an email publisher. You want to know your real human open and click rates. This directory gives you everything you need to build a production bot detection model for your newsletter, validated against industry benchmarks.

## What You'll Build

A four-layer cascade that classifies every email open and click as human or bot:

1. **Apple MPP Stripping** — Removes fake opens from Apple Mail Privacy Protection
2. **Scanner Detection** — Catches corporate security scanners (Barracuda, Mimecast, etc.)
3. **ML Classification** — Random Forest models trained on YOUR data
4. **Ground Truth Override** — Verified humans (paying subscribers, event attendees) never filtered

## Expected Results

| Metric | Before | After | Industry Benchmark |
|--------|--------|-------|--------------------|
| Unique open rate | 40-55% | 20-28% | Omeda B2B: ~22% |
| Bot open % | 0-5% | 40-55% | Industry consensus: 40-60% |
| Click false negative | High | < 2% | Almost no bot clicks kept |
| Click false positive | Unknown | < 25% | Rarely kills real clicks |
| Opens-per-opener | Unknown | 2.0-3.0x | Content quality signal |

## Step-by-Step Setup

### Step 1: Get Your Raw Data

You need per-subscriber, per-blast event data from your ESP. Required columns:

```
subscriber_id    — unique identifier per subscriber
blast_id         — unique identifier per email send/campaign
send_timestamp   — when the email was sent
open_timestamp   — when the subscriber opened (null if no open)
click_timestamp  — when the subscriber clicked (null if no click)
device           — device/client name (e.g., "Apple Mail", "Gmail", "Outlook")
```

Strongly recommended:
```
click_urls       — URLs clicked (space-separated or array)
click_timestamps — per-click timestamps (for multi-click analysis)
user_agent       — full user agent string
```

See `/docs/ESP_GUIDE.md` for extraction instructions per ESP.

### Step 2: Build Subscriber Profiles

Run `build_profiles.py` on your historical data (90+ days recommended):

```bash
python3 v2/build_profiles.py --input your_events.csv --output profiles/
```

This creates:
- `subscriber_classifications.csv` — GMM clustering into Bot / Human / Scanner-Affected
- `click_profiles.json` — per-subscriber click rate (what fraction of emails they click)
- `social_signal.json` — per-subscriber social footer click fraction (honeypot signal)

### Step 3: Train Your Models

```bash
python3 v2/train_models.py --events your_events.csv --profiles profiles/
```

This trains two Random Forest classifiers on YOUR data:
- Open model (10 features, expects ~96% AUC)
- Click model (14 features, expects ~99% AUC)

The models use your subscriber GMM labels as weak supervision, then the cascade logic handles the rest.

### Step 4: Run the Classifier

```bash
# Single blast:
python3 v2/classify.py --blast blast_12345.csv --profiles profiles/ --models models/

# All blasts:
python3 v2/classify.py --input-dir blasts/ --profiles profiles/ --models models/ --output results.csv
```

### Step 5: Validate

Compare your results against the industry benchmarks in `/v2/BENCHMARKS.md`. Key checks:
- Is your unique open rate between 18-30%? (If higher, you're not filtering enough)
- Is your CTOR between 4-7%? (If lower, your opens are inflated)
- Do verified humans (paying subscribers) pass at >97%? (If not, your FP rate is too high)

## File Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | This file — start here |
| `classify.py` | Production classifier — processes blast CSVs |
| `build_profiles.py` | Builds subscriber behavioral profiles from historical data |
| `train_models.py` | Trains open + click ML models on your data |
| `BENCHMARKS.md` | Industry benchmark data and validation methodology |
| `ARCHITECTURE.md` | Detailed explanation of every layer, rule, and threshold |
| `reference_data/click_grid.json` | Universal click probability grid (safe to use as-is) |
| `reference_data/thresholds.json` | Recommended thresholds with explanations |
