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

| Metric | Value | Component Score | Max Possible |
|---|---|---|---|
| Composite | **94.00** | — | 100.00 |
| K-S | 99.78 | 29.93 | 30.00 |
| MSE | 0.001720 | 29.79 | 30.00 |
| ECE | 0.025634 | 17.44 | 20.00 |
| Spread | 84.18 | 16.84 | 20.00 |

**Headroom Analysis:**
- K-S: 0.07 points available (essentially maxed)
- MSE: 0.21 points available (essentially maxed)
- ECE: 2.56 points available (moderate opportunity)
- Spread: 3.16 points available (**largest opportunity**)
- **Total headroom: ~6 points. Target: capture 1-3 of them.**

**Prediction Distribution at Plateau:**
```
0.0-0.1:  5.0%  ██
0.1-0.2:  4.5%  ██
0.2-0.3:  3.8%  █
0.3-0.4:  2.4%  █
0.4-0.5: 32.7%  ████████████████   <-- THE PROBLEM: model hedges here
0.5-0.6: 18.5%  █████████
0.6-0.7: 16.9%  ████████
0.7-0.8:  9.7%  ████
0.8-0.9:  5.0%  ██
0.9-1.0:  1.6%
```

The model is hedging on the 54% ambiguous events by predicting ~0.45-0.55. To improve spread without hurting MSE, we need the model to be **confidently right** — pushing predictions away from 0.50 only when it has genuine signal to do so.

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

### Click Timing (indices 0-2)
- `[0] time_to_first_click_sec`: Seconds from send to first click (-1 if no click)
- `[1] avg_inter_click_sec`: Average seconds between consecutive clicks (-1 if <2 clicks)
- `[2] click_span_sec`: Seconds between first and last click (-1 if <2 clicks)

### Click Volume (indices 3-5)
- `[3] raw_total_clicks`: Total click events
- `[4] nhi_clicks`: Non-human-interaction clicks (Sailthru's own bot flag)
- `[5] unique_urls_clicked`: Distinct URLs clicked

### Open Timing (indices 6-8)
- `[6] time_to_first_open_sec`: Seconds from send to first open (-1 if no open)
- `[7] open_span_sec`: Seconds between first and last open (-1 if <2 opens)
- `[8] first_nhi_open_sec`: Seconds to first NHI-flagged open (-1 if none)

### Open Volume (indices 9-10)
- `[9] raw_total_opens`: Total open events
- `[10] nhi_opens`: Non-human-interaction opens

### User History (indices 11-12)
- `[11] user_historical_open_rate`: This user's lifetime open rate (0-1)
- `[12] user_lifetime_verified_opens`: Total verified opens across all sends

### Derived Ratios (indices 13-18)
- `[13] clicks_per_second`
- `[14] url_diversity_ratio`
- `[15] nhi_open_ratio`
- `[16] nhi_click_ratio`
- `[17] has_any_clicks`
- `[18] has_any_opens`

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

## NEXT WAVE: Experiments 46-60 (Ranked by Expected Impact)

These are the highest-priority experiments to run next. Execute them IN ORDER. Each builds on the previous if successful.

### Tier 1: Feature Engineering (Highest Expected Impact)

**Experiment 46: Feature Ablation Study**
- **Hypothesis:** Some of the 43 engineered features are noise. Removing them will let the model focus on real signal, improving calibration and potentially spread.
- **Method:** Run the current model, then systematically remove features one at a time. Track which removals improve or don't hurt the composite. Start with the quantile bins and threshold flags.
- **What to change:** In `engineer_features()`, comment out one feature group at a time (e.g., all quantile bins, all threshold flags, all log transforms). Run after each removal.
- **Expected outcome:** Identify 5-10 features that can be removed. Small composite improvement from reduced noise.
- **Success criteria:** Composite >= 94.00 with fewer features.

**Experiment 47: Ratio of NHI-to-Total as Continuous Feature**
- **Hypothesis:** The model currently has `nhi_click_ratio` and `nhi_open_ratio` as raw features plus threshold flags (>0.95, >0.75, <0.10). But it doesn't have the *interaction* between NHI ratios and volume. A subscriber with nhi_click_ratio=0.95 and 1 click is very different from one with nhi_click_ratio=0.95 and 20 clicks.
- **Method:** Add `nhi_click_ratio * raw_total_clicks` and `nhi_open_ratio * raw_total_opens` as new features.
- **What to change:** Add two lines to `engineer_features()`.
- **Expected outcome:** Better separation in the ambiguous zone where NHI ratio alone is ambiguous.

**Experiment 48: Missing-Value Indicator Features**
- **Hypothesis:** The -1 sentinel value for missing data (no clicks, no opens) is being treated as a numeric value by the tree. Explicit binary indicators for "has no click data" and "has no open data" may help the model handle these cases differently.
- **Method:** Add `(X[:, 0] == -1)` and `(X[:, 6] == -1)` as binary features.
- **What to change:** Add two lines to `engineer_features()`.
- **Expected outcome:** Cleaner splits in the tree for missing-data cases.

**Experiment 49: Open-to-Click Time Gap**
- **Hypothesis:** The time between first open and first click is a strong human signal (humans open, read, then click). Bots often click before or simultaneously with the open. This feature is NOT in the current set.
- **Method:** Add `time_to_first_click_sec - time_to_first_open_sec` (only when both > 0) as a new feature.
- **What to change:** Add one computed feature to `engineer_features()`.
- **Expected outcome:** New signal axis for the ambiguous zone.

**Experiment 50: Velocity Consistency Score**
- **Hypothesis:** Bots have extremely consistent inter-click timing (std dev near 0). Humans have variable timing. We don't currently capture this — we only have the average.
- **Method:** We can't compute std dev from the current features, but we CAN approximate it: if `click_span_sec > 0` and `raw_total_clicks > 2`, then `click_span_sec / raw_total_clicks` vs `avg_inter_click_sec` — if they're very close, timing is regular (bot-like).
- **What to change:** Add `abs(click_span_sec / max(raw_total_clicks - 1, 1) - avg_inter_click_sec)` as a feature.
- **Expected outcome:** Captures timing regularity without needing raw event data.

### Tier 2: Label Strategy (Medium Expected Impact)

**Experiment 51: PU-Learning Framework**
- **Hypothesis:** The 54% ambiguous events are NOT "0.50 probability human." They are UNLABELED. Treating them as 0.50 forces the model to predict 0.50 for genuinely uncertain cases AND for cases where the model could be more confident. PU-learning (Positive-Unlabeled) treats these as unlabeled and learns from the labeled extremes.
- **Method:** Instead of soft labels for the ambiguous zone, use a PU-learning loss that only penalizes the model on labeled examples and treats unlabeled examples as having unknown labels.
- **What to change:** Replace the LightGBM regression objective with a custom PU-learning objective function. The simplest version: set sample_weight=0 for ambiguous samples during training, then let the model extrapolate.
- **Expected outcome:** Model will predict more extreme values for the ambiguous zone based on feature similarity to labeled examples. Spread should improve significantly.
- **Risk:** MSE will increase because we're no longer targeting 0.50 for ambiguous samples. Need to check if spread gain > MSE loss.

**Experiment 52: Confidence-Weighted Labels**
- **Hypothesis:** Not all 0.50 labels are equally uncertain. Some ambiguous events have features that are CLOSE to triggering a rule (e.g., time_to_first_click = 62 seconds, just barely missing the 60s bot threshold). These should have labels closer to the rule they almost triggered.
- **Method:** For ambiguous samples, compute distance to nearest rule threshold and adjust label proportionally. Events that almost triggered a bot rule get labels < 0.50; events that almost triggered a human rule get labels > 0.50.
- **What to change:** New label adjustment function in `train.py` that runs AFTER `spread_ambiguous_labels()`.
- **Expected outcome:** More informative labels for the ambiguous zone, especially for near-threshold cases.

**Experiment 53: Two-Phase Training**
- **Hypothesis:** Train the model in two phases: (1) train on ONLY the labeled extremes (hard_label != NULL) to learn the definitive patterns, then (2) use that model's predictions as pseudo-labels for the ambiguous zone and retrain on everything.
- **Method:** First train on ~730K labeled samples. Use predictions on the 2.36M ambiguous samples as new soft labels. Retrain on all 4.37M with these improved labels.
- **What to change:** Add a two-phase training loop in `train()`.
- **Expected outcome:** Better pseudo-labels than the rule-based 0.50, because the model can use all 19 features to estimate probability.
- **Risk:** This is similar to self-training (exp17-18) which didn't work. The key difference: start from hard labels only, not soft labels.

### Tier 3: Architecture (Lower Expected Impact, Higher Risk)

**Experiment 54: Focal Loss for Ambiguous Samples**
- **Hypothesis:** Standard MSE loss treats all prediction errors equally. Focal loss down-weights easy examples (the definitive bots/humans the model already gets right) and up-weights hard examples (the ambiguous zone). This should force the model to spend more capacity on the uncertain middle.
- **Method:** Implement focal loss as a custom LightGBM objective.
- **What to change:** Add custom objective function to `train()`.
- **Expected outcome:** Better calibration in the ambiguous zone.

**Experiment 55: Quantile Regression Ensemble**
- **Hypothesis:** Train 3 models at quantiles 0.25, 0.50, 0.75. The median model gives the point estimate; the spread between 0.25 and 0.75 quantiles gives a confidence interval. Use the confidence interval to adjust predictions: when the model is confident (narrow interval), push predictions further from 0.50.
- **Method:** Train 3 LightGBM models with `objective=quantile` at different `alpha` values. Combine predictions.
- **What to change:** Multi-model training in `train()`.
- **Expected outcome:** More calibrated uncertainty estimates that improve spread.

**Experiment 56: Small Neural Network (MLP)**
- **Hypothesis:** Tree-based models make axis-aligned splits. A small MLP can learn smooth, non-linear decision boundaries that may capture interactions trees miss.
- **Method:** 3-layer MLP (64-32-16) with dropout, trained on the same features. Use PyTorch or sklearn MLPRegressor.
- **What to change:** Replace LightGBM with MLP in `train()`.
- **Expected outcome:** Different prediction surface that may complement trees.
- **Risk:** Neural nets are harder to tune and may overfit on 4.37M rows without careful regularization.

### Tier 4: Ensemble & Post-Processing

**Experiment 57: Temperature Scaling**
- **Hypothesis:** The model's raw predictions may be well-ordered but poorly calibrated. Temperature scaling (dividing logits by a learned temperature parameter) can improve calibration without changing the ranking.
- **Method:** After training, learn a single temperature parameter T on the validation set that minimizes ECE. Apply `pred = sigmoid(logit(pred) / T)` to all predictions.
- **What to change:** Add post-processing step after `model.predict()`.
- **Expected outcome:** Improved ECE with no change to K-S.

**Experiment 58: Platt Scaling (Careful)**
- **Hypothesis:** Similar to temperature scaling but with two parameters (a, b) in `sigmoid(a * logit(pred) + b)`. More flexible than temperature scaling.
- **Method:** Fit a, b on validation set to minimize NLL. Apply to test set.
- **What to change:** Add post-processing step.
- **Expected outcome:** Better calibration. Risk: isotonic calibration destroyed spread (exp tried), but Platt scaling is much gentler.

## How to Run

```bash
cd ~/Documents/bot-b-gone-ml
python3 train.py 2>&1 | tee run.log
grep "COMPOSITE SCORE\|Validation Composite\|Test Composite" run.log
```

## The Experiment Loop

LOOP FOREVER:

1. Read the current composite score and experiment history in `results.tsv`
2. Read this file (`program.md`) for the next experiment to try
3. Form ONE hypothesis about what will improve the score
4. Make ONE change to `train.py` to test it
5. Git commit with a descriptive message: `git add train.py && git commit -m "expN: ONE change - description"`
6. Run: `python3 train.py 2>&1 | tee run.log`
7. Read results: `grep "Composite\|composite" run.log`
8. Analyze: WHY did it move? What metric changed? Does the histogram look better?
9. If composite improved → keep commit, compound next change on top
10. If composite is equal or worse → `git checkout HEAD~1 -- train.py` to revert
11. Log to results.tsv (automatic via prepare.py)
12. NEVER STOP — run until the human interrupts you

## Important Constraints

- **Memory:** The dataset is 339MB / 4.37M rows. Ensure your changes don't cause OOM. If you need to subsample for testing, use the first 1M rows, but always validate on the full dataset.
- **Speed:** Each experiment should complete in < 5 minutes. If it takes longer, something is wrong.
- **Reproducibility:** Always use `seed=42` for random operations. The data split is deterministic.
- **Git discipline:** Every experiment gets its own commit. Reverted experiments get `git checkout HEAD~1 -- train.py`.
