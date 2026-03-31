# Bot-B-Gone ML — Autoresearch Agent Instructions

## Objective

You are optimizing a machine learning model that classifies email engagement events as **human** or **bot** on a continuous 0.0-1.0 probability scale. Your goal is to **maximize the composite score** on the validation set.

The composite score balances four objectives:
- **K-S separation (30%):** Can the model still perfectly separate definitive bots from definitive humans?
- **MSE on soft labels (30%):** Can the model learn the full spectrum of rule-based confidence tiers?
- **Calibration (20%):** Are the predicted probabilities meaningful (not just 0 or 1)?
- **Spread (20%):** Does the model produce a distribution across the 0-1 range, not a binary collapse?

## The Problem We're Solving

The previous binary model (trained on only the extremes) achieved K-S = 100 but collapsed to binary predictions on the uncertain middle — 97.6% scored as "definitely human" and 2.4% as "definitely bot" with nothing in between. That's useless for the 7.4M events in the gray zone.

We now train on **soft labels** derived from our rule-based system, which assigns continuous confidence scores:
- 0.00 = definitive bot (machinegun clicker / honeypot)
- 0.25 = medium-confidence bot (single bot rule triggered)
- 0.50 = ambiguous (no rules triggered)
- 0.70 = weak human signal (one human rule)
- 1.00 = definitive human (verified clicker)

## Files

| File | Role | Can you modify it? |
|---|---|---|
| `prepare.py` | Immutable evaluation harness. Loads data, splits, computes composite score. | **NO** |
| `train.py` | Model training script. Feature engineering, hyperparameters, model architecture. | **YES — this is your playground** |
| `data/soft_labeled.csv` | Training data: 2M rows, 19 raw features, soft_label (0-1), hard_label (0/1/NULL). | **NO** |
| `data/feature_columns.txt` | List of the 19 raw feature names. | **NO** |
| `results.tsv` | Experiment log. Each run appends a row. | Read-only (auto-populated) |

## What You Can Change in train.py

1. **Feature engineering** — The `engineer_features()` function.
2. **Hyperparameters** — The `MODEL_PARAMS` dict.
3. **Model architecture** — Replace XGBoost with LightGBM, CatBoost, neural net, ensemble.
4. **Objective function** — Custom loss functions that better handle soft labels.
5. **Post-processing** — Calibration layers (Platt scaling, isotonic regression).
6. **Sample weighting** — Weight the hard-labeled extremes differently from the ambiguous middle.

## Training Data Distribution

| Soft Label Range | Count | % | Meaning |
|---|---|---|---|
| 0.00-0.05 | 77,834 | 3.9% | Definitive bot |
| 0.05-0.15 | 35,767 | 1.8% | High-confidence bot |
| 0.15-0.25 | 75,216 | 3.8% | Medium bot |
| 0.25-0.35 | 8,138 | 0.4% | Weak bot |
| 0.45-0.55 | 1,095,019 | 54.8% | AMBIGUOUS (no rules triggered) |
| 0.65-0.75 | 487,842 | 24.4% | Weak human |
| 0.75-0.85 | 157,383 | 7.9% | Moderate human |
| 0.85-0.95 | 61,698 | 3.1% | Strong human |
| 0.95-1.00 | 1,103 | 0.1% | Definitive human |

Key challenge: 54.8% of events are ambiguous at 0.50. The model needs to learn to spread these based on behavioral signals.

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
- `clicks_per_second`, `url_diversity_ratio`, `nhi_open_ratio`, `nhi_click_ratio`
- `has_any_clicks`, `has_any_opens`

## Key Behavioral Insights

1. **User history is the strongest signal.** 72% of confirmed humans have 75-100% historical open rate. Only 4% of bots do.
2. **Time-to-first-open is bimodal.** 85% of bots open within 60s. 87% of humans open after 1min.
3. **Bots fire exactly 1 open.** 85.5% of bots have 1 open. 40% of humans re-open.
4. **NHI click ratio is nearly perfect.** 99.9% for bots vs 38.8% for humans.
5. **The 54.8% ambiguous events** have NO rule signals — the model must use raw behavioral features to score them.

## How to Run

```bash
cd /path/to/bot-b-gone/ml
python3 train.py > run.log 2>&1
grep "COMPOSITE SCORE\|val_composite\|Validation Composite" run.log
```

## Scoring

Experiments are ranked by validation composite score (higher is better, max 100):
- **Primary:** `composite_score` on validation set
- **Constraint:** K-S on hard labels must stay > 90 (don't sacrifice discrimination)
- **Secondary:** Test composite score (generalization check)

## Experiment Ideas to Try

1. **Sample weighting:** Weight the 0.50 ambiguous events lower, extremes higher
2. **Custom loss:** Huber loss or quantile regression instead of squared error
3. **Isotonic calibration:** Post-hoc calibration layer on top of XGBoost
4. **Two-stage model:** First classify bot/human/uncertain, then regress within each group
5. **Feature interactions:** Multiply user_historical_open_rate × time_to_first_open_sec
6. **Deeper trees:** max_depth=8-10 to capture more complex patterns
7. **LightGBM:** Often better calibrated than XGBoost out of the box
8. **Neural net:** Small MLP might handle the continuous target better

## The Experiment Loop

LOOP FOREVER:

1. Look at the current composite score
2. Modify `train.py` with an experimental idea
3. Git commit with a descriptive message
4. Run: `python3 train.py > run.log 2>&1`
5. Read results: `grep "Composite\|composite" run.log`
6. If composite improved → keep commit
7. If composite is equal or worse → git reset
8. Log to results.tsv
9. NEVER STOP — run until the human interrupts you
