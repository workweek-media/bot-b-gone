# Bot-B-Gone ML — Autoresearch Agent Instructions

## Objective

You are optimizing a machine learning model that classifies email engagement events as **human** or **bot** on a continuous 0.0-1.0 probability scale. Your goal is to **maximize the composite score** on the validation set.

The composite score balances four objectives:
- **K-S separation (30%):** Can the model still perfectly separate definitive bots from definitive humans?
- **MSE on soft labels (30%):** Can the model learn the full spectrum of rule-based confidence tiers?
- **Calibration (20%):** Are the predicted probabilities meaningful (not just 0 or 1)?
- **Spread (20%):** Does the model produce a distribution across the 0-1 range, not a binary collapse?

## CRITICAL RULE: One Change at a Time

Follow Karpathy's core principle: **complexify only one at a time.**

1. Change **ONE thing**
2. Measure
3. Understand **WHY** it moved (or didn't)
4. If it helped → **keep it and compound the next change on top**
5. If it didn't → revert, understand why, try a different single change
6. Each step should be a building block that compounds on the last

**DO NOT** change multiple things at once. **DO NOT** shotgun experiments. Each experiment must have:
- One stated hypothesis
- One change to test it
- Full analysis of what moved and why before the next experiment

## Current Best: 94.00 / 100 (Experiment 40)

| Metric | Value |
|---|---|
| Composite | **94.00** |
| K-S | 99.81 |
| MSE | 0.001716 |
| ECE | 0.025633 |
| Spread | 84.10 |

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
| `data/soft_labeled.csv` | Training data: 4.37M rows, 19 raw features, soft_label (0-1), hard_label (0/1/NULL). Post-July 2025 (after Sailthru algorithm change). | **NO** |
| `data/feature_columns.txt` | List of the 19 raw feature names. | **NO** |
| `results.tsv` | Experiment log. Each run appends a row. | Read-only (auto-populated) |

## Training Data Distribution (4.37M rows, 8 months post-July 2025)

| Soft Label Range | Count | % | Meaning |
|---|---|---|---|
| 0.00-0.05 | 175K | 4.0% | Definitive bot |
| 0.05-0.15 | 68K | 1.6% | High-confidence bot |
| 0.15-0.25 | 324K | 7.4% | Medium bot |
| 0.45-0.55 | 2.36M | **54.0%** | AMBIGUOUS (no rules triggered) |
| 0.65-0.75 | 944K | 21.6% | Weak human |
| 0.75-0.85 | 358K | 8.2% | Moderate human |
| 0.85-0.95 | 131K | 3.0% | Strong human |

Gold standard: 179K definitive bots, 554K definitive humans.

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
5. **The 54.0% ambiguous events** have NO rule signals — the model must use raw behavioral features to score them.

## What Has Been Tried (45 Experiments)

### What WORKED (keep these in the current best):
| Change | Impact | Why |
|---|---|---|
| LightGBM instead of XGBoost | +0.5 composite | Better native calibration |
| 8-signal label spreading (vs 3) | +0.8 composite | Richer humanness signal for the 54% ambiguous events |
| Asymmetric spreading (bot 1.3x, human 0.8x) | +0.14 composite | Bot signals are cleaner, so push them harder |
| spread_amount = 0.42 | Optimal | Binary searched from 0.15 to 0.50 |
| 1200 boosting rounds | +0.01 | Marginal but consistent |
| 4.37M rows (9mo data, post-July) | +0.29 composite | More edge cases in gray zone |
| Rich feature engineering (43 features from 19 raw) | +0.5 composite | Log transforms, ratios, interactions |

### What DIDN'T work:
| Change | Result | Why It Failed |
|---|---|---|
| Isotonic calibration | -2.0 composite | Destroyed spread — collapsed predictions to narrow range |
| Custom confidence penalty objective | -0.5 to -1.5 | Always hurt MSE more than it helped spread |
| Self-training / bootstrapping | Neutral | Model converged to same predictions each round |
| XGBoost + LightGBM ensemble | Neutral | Models agree too much (mean disagreement 0.0007) |
| Huber loss | -1.5 composite | Worse on every metric |
| DART boosting | Too slow | 8+ minutes per run, killed |
| Two-model blend (soft + hard) | -12 composite | Hard model pushes gold standard away from soft labels |
| hist_rate weight 25% → 40% | -2.3 composite | Labels pushed too extreme (std 0.086 vs 0.052), MSE 4x worse |
| Sigmoid spreading | -0.8 composite | Same tradeoff — spread up but MSE 2x worse |
| Feature interactions (cross-products) | -0.3 composite | Added noise |
| Asymmetric 1.5/0.7 (too aggressive) | -0.1 composite | Spread improved but MSE hurt |

### What is SATURATED (don't bother):
| Hyperparameter | Tested Values | Conclusion |
|---|---|---|
| learning_rate | 0.02, 0.03 | Identical at 1200 rounds |
| min_child_samples | 10, 50 | Identical |
| feature_fraction | 0.5, 0.8 | Identical |
| spread_amount | 0.35, 0.40, 0.42, 0.44, 0.45, 0.50 | 0.42 is optimal |
| num_boost_round | 800, 1200 | 1200 marginally better |

### The Core Tension

**Spread and MSE are in direct opposition.** Any change that pushes predictions further from 0.50 (improving spread) also pushes them further from the soft labels (hurting MSE). The composite weights MSE at 0.3 and spread at 0.2, so MSE always wins.

To break past 94.00, you need to find changes that improve spread WITHOUT hurting MSE. This likely means:
1. Better feature engineering that gives the model more information to be confidently right
2. Better label spreading that is more accurate (not just wider)
3. A fundamentally different approach to the 54% ambiguous events

## Unexplored Directions (Ranked by Expected Impact)

1. **Cross-send features**: Subscriber behavior across multiple sends (repeat offender score). Requires data pipeline change.
2. **Domain-type feature**: Corporate vs freemail email domain. Omeda flagged this as important. Can be derived from existing data.
3. **Neural network (MLP)**: Small feedforward net might capture non-linear interactions trees miss. Needs careful regularization.
4. **Quantile regression**: Train multiple models at different quantiles, ensemble the predictions.
5. **Feature ablation study**: Systematically remove features one at a time to find which ones are noise.
6. **Time-based features**: Day of week, hour of day when the event occurred. Bots may cluster at certain times.
7. **PU-learning**: Positive-Unlabeled learning framework that treats the 54% ambiguous as unlabeled instead of 0.50.

## How to Run

```bash
cd /path/to/bot-b-gone/ml
python3 train.py > run.log 2>&1
grep "COMPOSITE SCORE\|Validation Composite\|Test Composite" run.log
```

## The Experiment Loop

LOOP FOREVER:

1. Read the current composite score and experiment history
2. Form ONE hypothesis about what will improve the score
3. Make ONE change to `train.py` to test it
4. Git commit with a descriptive message: `git add ml/train.py && git commit -m "expN: ONE change - description"`
5. Run: `python3 train.py > run.log 2>&1`
6. Read results: `grep "Composite\|composite" run.log`
7. Analyze: WHY did it move? What metric changed? Does the histogram look better?
8. If composite improved → keep commit, compound next change on top
9. If composite is equal or worse → `git checkout HEAD~1 -- ml/train.py` to revert
10. Log to results.tsv
11. NEVER STOP — run until the human interrupts you
