# Bot-B-Gone ML — Autoresearch Agent Instructions

## Objective

You are optimizing a machine learning model that classifies email engagement events as **human** or **bot**. Your single goal is to **maximize the K-S (Kolmogorov-Smirnov) score** on the validation set while keeping the **False Positive Rate (FPR) below 1%**.

The K-S statistic measures the maximum separation between the predicted probability distributions for bots vs. humans. A perfect score is 100.

## Ground Truth

The training data uses two anchors of absolute certainty:

- **HUMAN (label=1):** Sailthru real opens. The ESP's own "real" flag. It undercounts humans (low recall), but when it says "real," it IS real (high precision). This is the floor.
- **BOT (label=0):** Machinegun definitive bots. These clicked 5+ unique links within 5 seconds — the honeypot equivalent. Zero ambiguity.

Everything in between (the "uncertain gap" — 7.4M events) has no label. The model's job is to score those events on a 0-1 probability scale.

## Files

| File | Role | Can you modify it? |
|---|---|---|
| `prepare.py` | Immutable evaluation harness. Loads data, splits, computes K-S and all metrics. | **NO** |
| `train.py` | Model training script. Feature engineering, hyperparameters, model architecture. | **YES — this is your playground** |
| `data/gold_standard.csv` | Training data: 1.87M rows, 19 raw features, binary label. | **NO** |
| `data/feature_columns.txt` | List of the 19 raw feature names. | **NO** |
| `results.tsv` | Experiment log. Each run appends a row. | Read-only (auto-populated) |

## What You Can Change in train.py

1. **Feature engineering** — The `engineer_features()` function. Add log transforms, interaction terms, polynomial features, binning, etc.
2. **Hyperparameters** — The `MODEL_PARAMS` dict. Tune depth, learning rate, estimators, regularization, etc.
3. **Model architecture** — Replace XGBoost with LightGBM, CatBoost, a neural net, or an ensemble.
4. **Threshold** — The `THRESHOLD` variable. Tune the classification cutoff.
5. **Class weights** — `scale_pos_weight` to handle the 4.8:1 class imbalance.
6. **Feature selection** — Drop or combine features.

## Raw Features Available (19 columns)

### Click Timing
- `time_to_first_click_sec`: Seconds from send to first click (-1 if no click)
- `avg_inter_click_sec`: Average seconds between consecutive clicks (-1 if <2 clicks)
- `click_span_sec`: Seconds between first and last click (-1 if <2 clicks)

### Click Volume
- `raw_total_clicks`: Total click events
- `nhi_clicks`: Non-human-interaction clicks (Sailthru's own bot flag)
- `unique_urls_clicked`: Distinct URLs clicked

### Open Timing
- `time_to_first_open_sec`: Seconds from send to first open (-1 if no open)
- `open_span_sec`: Seconds between first and last open (-1 if <2 opens)
- `first_nhi_open_sec`: Seconds to first NHI-flagged open (-1 if none)

### Open Volume
- `raw_total_opens`: Total open events
- `nhi_opens`: Non-human-interaction opens

### User History
- `user_historical_open_rate`: This user's lifetime open rate (0-1)
- `user_lifetime_verified_opens`: Total verified opens across all sends

### Derived Ratios
- `clicks_per_second`: raw_total_clicks / click_span_sec
- `url_diversity_ratio`: unique_urls_clicked / raw_total_clicks
- `nhi_open_ratio`: nhi_opens / raw_total_opens
- `nhi_click_ratio`: nhi_clicks / raw_total_clicks
- `has_any_clicks`: Binary flag
- `has_any_opens`: Binary flag

## Key Behavioral Insights (from our analysis)

Use these to guide your feature engineering:

1. **Time-to-first-open is bimodal.** 85% of bots open within 60 seconds. 87% of humans open after 1 minute. The 10-30 second window is 50% bots.
2. **Bots fire exactly 1 open.** 85.5% of bots have exactly 1 open per send. 40% of humans re-open emails (2+ opens).
3. **User history is the strongest signal.** 72% of confirmed humans have a 75-100% historical open rate. Only 4% of bots do.
4. **NHI click ratio is nearly perfect.** 99.9% for bots vs. 38.8% for humans.
5. **Inter-click velocity:** Bot median = 0.13 seconds. Human median = 1.0 second.
6. **The uncertain gap's clickers look bot-like.** 97.3% NHI click ratio, median first click at 38s.

## How to Run

```bash
cd /path/to/bot-b-gone/ml
python3 train.py
```

This will:
1. Load data via `prepare.py`
2. Split into train/val/test (60/20/20)
3. Engineer features
4. Train the model
5. Evaluate on validation and test sets
6. Print K-S score, AUC-ROC, FPR, FNR, confusion matrix
7. Log results to `results.tsv`
8. Save model to `model.json`

## Scoring

Your experiments are ranked by:
1. **Primary:** Validation K-S score (higher is better, max 100)
2. **Constraint:** Validation FPR must be < 1% (bots misclassified as human)
3. **Secondary:** Test K-S score (generalization check)

## Current Baseline

The baseline XGBoost achieves K-S = 100.0 on both validation and test. This is because the two ground truth groups (Sailthru real vs. machinegun bots) are extremely well-separated. The real challenge is what happens when this model scores the 7.4M uncertain events — does the probability distribution look reasonable, or does it collapse?

Focus your experiments on:
- **Calibration:** Are the probability scores meaningful in the 0.3-0.7 range?
- **Robustness:** Does the model still work if you remove the top feature?
- **Generalization:** Train on 60 days, test on the last 30 days (temporal split).
