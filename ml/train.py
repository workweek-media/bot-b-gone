"""
Bot-B-Gone ML — train.py (THE AGENT'S PLAYGROUND)
===================================================
EXPERIMENT 2: Feature-based label spreading + original params

The core problem: 54.8% of training labels are exactly 0.50 (ambiguous).
The model learns to predict 0.50 for everything in the middle.

Solution: Use behavioral features to SPREAD the 0.50 labels before training.
If a 0.50-labeled event has strong human behavioral signals (high open rate,
slow timing, re-opens), nudge its label toward 0.65. If it has bot-like
signals (fast timing, NHI-heavy), nudge toward 0.35.

This creates a continuous label distribution that the model can learn from.
"""

import sys
import time
import numpy as np
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    load_data,
    split_data,
    evaluate,
    print_evaluation,
    log_result,
)


# ============================================================
# FEATURE ENGINEERING (same as baseline + 4 new interaction features)
# ============================================================

def engineer_features(X):
    """
    Raw feature indices:
      0: time_to_first_click_sec    6: time_to_first_open_sec
      1: avg_inter_click_sec        7: open_span_sec
      2: click_span_sec             8: first_nhi_open_sec
      3: raw_total_clicks           9: raw_total_opens
      4: nhi_clicks                10: nhi_opens
      5: unique_urls_clicked       11: user_historical_open_rate
     12: user_lifetime_verified_opens
     13: clicks_per_second         14: url_diversity_ratio
     15: nhi_open_ratio            16: nhi_click_ratio
     17: has_any_clicks            18: has_any_opens
    """
    def safe_log(col_idx):
        vals = X[:, col_idx].copy()
        vals[vals < 0] = 0
        return np.log1p(vals).reshape(-1, 1)
    
    features = [
        X,
        safe_log(0),   # log_time_to_first_click
        safe_log(1),   # log_avg_inter_click
        safe_log(2),   # log_click_span
        safe_log(6),   # log_time_to_first_open
        safe_log(7),   # log_open_span
        safe_log(8),   # log_first_nhi_open
        ((X[:, 0] >= 0) & (X[:, 0] < 60) & (X[:, 5] >= 5)).astype(np.float32).reshape(-1, 1),  # fast_and_many
        ((X[:, 4] > 3) & (X[:, 0] >= 0) & (X[:, 0] < 30)).astype(np.float32).reshape(-1, 1),   # nhi_heavy_and_fast
        (X[:, 9] > 1).astype(np.float32).reshape(-1, 1),   # reopens
        (X[:, 9] > 3).astype(np.float32).reshape(-1, 1),   # many_reopens
        ((X[:, 6] >= 0) & (X[:, 6] < 60)).astype(np.float32).reshape(-1, 1),   # open_in_bot_zone
        (X[:, 6] > 900).astype(np.float32).reshape(-1, 1),  # open_in_human_zone
        (X[:, 11] > 0.75).astype(np.float32).reshape(-1, 1),  # strong_history
        (X[:, 11] < 0.10).astype(np.float32).reshape(-1, 1),  # weak_history
        (X[:, 16] > 0.95).astype(np.float32).reshape(-1, 1),  # nhi_dominant_clicks
        (X[:, 15] > 0.95).astype(np.float32).reshape(-1, 1),  # nhi_dominant_opens
    ]
    
    return np.hstack(features)


# ============================================================
# LABEL SPREADING: Use features to spread the 0.50 cluster
# ============================================================

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.15):
    """
    For events labeled 0.50 (ambiguous), compute a behavioral score
    from raw features and nudge the label up or down.
    
    This creates a continuous distribution from the binary cluster.
    """
    new_labels = soft_labels.copy()
    ambiguous_mask = np.abs(soft_labels - 0.50) < 0.01
    
    if ambiguous_mask.sum() == 0:
        return new_labels
    
    # Compute a "humanness" score from 3 key behavioral signals:
    # 1. User historical open rate (0-1, higher = more human)
    # 2. Time-to-first-open (log-scaled, slower = more human)
    # 3. Re-opens (>1 open = more human)
    
    X_amb = X_raw[ambiguous_mask]
    
    # Signal 1: Historical open rate (already 0-1)
    hist_rate = np.clip(X_amb[:, 11], 0, 1)
    
    # Signal 2: Time-to-first-open (normalize to 0-1, log scale)
    # Bot zone: 0-60s → 0.0, Human zone: >3600s → 1.0
    ttfo = X_amb[:, 6].copy()
    ttfo[ttfo < 0] = 300  # missing → neutral
    ttfo_score = np.clip(np.log1p(ttfo) / np.log1p(86400), 0, 1)
    
    # Signal 3: Re-opens (binary: >1 open = human signal)
    reopen_score = (X_amb[:, 9] > 1).astype(np.float32)
    
    # Combine: weighted average
    humanness = 0.5 * hist_rate + 0.3 * ttfo_score + 0.2 * reopen_score
    
    # Map humanness (0-1) to label adjustment (-spread_amount to +spread_amount)
    # humanness=0.5 → no change, humanness=1.0 → +spread_amount, humanness=0.0 → -spread_amount
    adjustment = (humanness - 0.5) * 2 * spread_amount
    
    new_labels[ambiguous_mask] = np.clip(0.50 + adjustment, 0.0, 1.0)
    
    return new_labels


# ============================================================
# MODEL CONFIGURATION (back to baseline, proven good)
# ============================================================

MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# TRAINING
# ============================================================
def train():
    print("Bot-B-Gone ML — Experiment 2: Feature-Based Label Spreading")
    print("=" * 60)
    
    # Load and split data
    X_raw, soft_labels, hard_labels, feature_cols = load_data()
    splits = split_data(X_raw, soft_labels, hard_labels)
    X_train_raw, X_val_raw, X_test_raw = splits[0], splits[1], splits[2]
    sl_train, sl_val, sl_test = splits[3], splits[4], splits[5]
    hl_train, hl_val, hl_test = splits[6], splits[7], splits[8]
    
    # Spread the ambiguous labels in training set
    print("Spreading ambiguous labels using behavioral features...")
    sl_train_spread = spread_ambiguous_labels(X_train_raw, sl_train, spread_amount=0.15)
    
    # Show the effect
    amb_mask = np.abs(sl_train - 0.50) < 0.01
    print(f"  Ambiguous events: {amb_mask.sum():,}")
    print(f"  Before: all at 0.50")
    print(f"  After:  mean={sl_train_spread[amb_mask].mean():.4f}, "
          f"std={sl_train_spread[amb_mask].std():.4f}, "
          f"min={sl_train_spread[amb_mask].min():.4f}, "
          f"max={sl_train_spread[amb_mask].max():.4f}")
    
    # Feature engineering
    print("\nEngineering features...")
    X_train = engineer_features(X_train_raw)
    X_val = engineer_features(X_val_raw)
    X_test = engineer_features(X_test_raw)
    
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {len(sl_train):,}")
    
    # Train on SPREAD labels
    print(f"\nTraining XGBoost Regressor...")
    start_time = time.time()
    
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(
        X_train, sl_train_spread,
        eval_set=[(X_val, sl_val)],  # Evaluate on ORIGINAL labels
        verbose=False,
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")
    
    # Predict
    val_preds = np.clip(model.predict(X_val), 0.0, 1.0)
    test_preds = np.clip(model.predict(X_test), 0.0, 1.0)
    
    # Evaluate against ORIGINAL soft labels (not spread ones)
    val_metrics = evaluate(sl_val, hl_val, val_preds, dataset_name="validation")
    test_metrics = evaluate(sl_test, hl_test, test_preds, dataset_name="test")
    
    print_evaluation(val_metrics)
    print_evaluation(test_metrics)
    
    # Feature importance
    print("\nTop 15 Feature Importances:")
    eng_feature_names = feature_cols + [
        "log_time_to_first_click", "log_avg_inter_click", "log_click_span",
        "log_time_to_first_open", "log_open_span", "log_first_nhi_open",
        "fast_and_many", "nhi_heavy_and_fast",
        "reopens", "many_reopens",
        "open_in_bot_zone", "open_in_human_zone",
        "strong_history", "weak_history",
        "nhi_dominant_clicks", "nhi_dominant_opens",
    ]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i, idx in enumerate(sorted_idx[:15]):
        name = eng_feature_names[idx] if idx < len(eng_feature_names) else f"feature_{idx}"
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    # Log results
    log_result(
        val_metrics, test_metrics,
        experiment_name="exp2_label_spreading",
        notes=f"spread_amount=0.15, baseline_params, features={X_train.shape[1]}, "
              f"train_time={train_time:.1f}s"
    )
    
    # Save model
    model_path = Path(__file__).parent / "model.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    model, val_metrics, test_metrics = train()
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Validation Composite: {val_metrics['composite_score']:.2f} / 100")
    print(f"  Test Composite:       {test_metrics['composite_score']:.2f} / 100")
    print(f"  Validation K-S:       {val_metrics['ks_score']}")
    print(f"  Validation MSE:       {val_metrics['mse']}")
    print(f"  Validation ECE:       {val_metrics['ece']}")
    print(f"  Validation Spread:    {val_metrics['spread']}")
    print(f"{'='*60}")
