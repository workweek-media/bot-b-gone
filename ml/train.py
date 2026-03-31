"""
Bot-B-Gone ML — train.py (THE AGENT'S PLAYGROUND)
===================================================
This is the ONLY file the autoresearch agent is allowed to modify.

It trains an XGBoost classifier on the Gold Standard dataset and
evaluates it using the immutable prepare.py harness.

The agent's goal: MAXIMIZE the K-S score on the validation set
while keeping FPR < 1%.

WHAT THE AGENT CAN CHANGE:
  - Feature engineering (add interaction terms, transforms, etc.)
  - Hyperparameters (max_depth, learning_rate, n_estimators, etc.)
  - Class weights (scale_pos_weight)
  - Threshold tuning
  - Feature selection

WHAT THE AGENT CANNOT CHANGE:
  - prepare.py (the evaluation harness)
  - The Gold Standard dataset
  - The train/val/test split
"""

import sys
import time
import numpy as np
import xgboost as xgb
from pathlib import Path

# Import the immutable evaluation harness
sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    load_data,
    split_data,
    evaluate,
    print_evaluation,
    log_result,
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
# The agent can add new derived features here.
# The raw features from BigQuery are (19 features, no leaky columns):
#   0: time_to_first_click_sec
#   1: avg_inter_click_sec
#   2: click_span_sec
#   3: raw_total_clicks
#   4: nhi_clicks
#   5: unique_urls_clicked
#   6: time_to_first_open_sec
#   7: open_span_sec
#   8: first_nhi_open_sec
#   9: raw_total_opens
#  10: nhi_opens
#  11: user_historical_open_rate
#  12: user_lifetime_verified_opens
#  13: clicks_per_second
#  14: url_diversity_ratio
#  15: nhi_open_ratio
#  16: nhi_click_ratio
#  17: has_any_clicks
#  18: has_any_opens

def engineer_features(X):
    """
    Add derived features. The agent can modify this function.
    Input X has 20 raw features. Returns X with additional columns appended.
    """
    # Log transforms for heavy-tailed timing features
    # Adding 1 to avoid log(0), using -1 sentinel for missing
    time_to_first_click = X[:, 0].copy()
    time_to_first_click[time_to_first_click < 0] = 0
    log_time_to_first_click = np.log1p(time_to_first_click).reshape(-1, 1)
    
    avg_inter_click = X[:, 1].copy()
    avg_inter_click[avg_inter_click < 0] = 0
    log_avg_inter_click = np.log1p(avg_inter_click).reshape(-1, 1)
    
    click_span = X[:, 2].copy()
    click_span[click_span < 0] = 0
    log_click_span = np.log1p(click_span).reshape(-1, 1)
    
    time_to_first_open = X[:, 6].copy()
    time_to_first_open[time_to_first_open < 0] = 0
    log_time_to_first_open = np.log1p(time_to_first_open).reshape(-1, 1)
    
    first_nhi_open = X[:, 8].copy()
    first_nhi_open[first_nhi_open < 0] = 0
    log_first_nhi_open = np.log1p(first_nhi_open).reshape(-1, 1)
    
    # Interaction: fast click + many URLs = scanner
    fast_and_many = ((X[:, 0] >= 0) & (X[:, 0] < 60) & (X[:, 5] >= 5)).astype(np.float32).reshape(-1, 1)
    
    # Interaction: high NHI clicks + fast timing = definitive bot
    nhi_heavy_and_fast = ((X[:, 4] > 3) & (X[:, 0] >= 0) & (X[:, 0] < 30)).astype(np.float32).reshape(-1, 1)
    
    # Ratio: nhi clicks / total clicks (high = bot)
    total_clicks = X[:, 3].copy()
    total_clicks[total_clicks == 0] = 1
    nhi_click_frac = (X[:, 4] / total_clicks).reshape(-1, 1)
    
    # Ratio: nhi opens / total opens (high = bot)
    total_opens = X[:, 9].copy()
    total_opens[total_opens == 0] = 1
    nhi_open_frac = (X[:, 10] / total_opens).reshape(-1, 1)
    
    # Stack all features
    X_engineered = np.hstack([
        X,
        log_time_to_first_click,
        log_avg_inter_click,
        log_click_span,
        log_time_to_first_open,
        log_first_nhi_open,
        fast_and_many,
        nhi_heavy_and_fast,
        nhi_click_frac,
        nhi_open_frac,
    ])
    
    return X_engineered


# ============================================================
# MODEL CONFIGURATION
# ============================================================
# The agent can tune these hyperparameters.

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,  # Agent can adjust for class imbalance
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

THRESHOLD = 0.5  # The agent can tune this


# ============================================================
# TRAINING
# ============================================================
def train():
    """Train the model and evaluate on validation + test sets."""
    print("Bot-B-Gone ML — Training Pipeline")
    print("=" * 60)
    
    # Load and split data
    X_raw, y, feature_cols = load_data()
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(X_raw, y)
    
    # Feature engineering
    print("Engineering features...")
    X_train = engineer_features(X_train_raw)
    X_val = engineer_features(X_val_raw)
    X_test = engineer_features(X_test_raw)
    
    print(f"  Raw features: {X_train_raw.shape[1]}")
    print(f"  Engineered features: {X_train.shape[1]}")
    print(f"  Training samples: {len(y_train):,}")
    
    # Train XGBoost
    print(f"\nTraining XGBoost (n_estimators={MODEL_PARAMS['n_estimators']})...")
    start_time = time.time()
    
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s")
    
    # Predict probabilities (probability of being HUMAN)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    val_metrics = evaluate(y_val, val_probs, threshold=THRESHOLD, dataset_name="validation")
    test_metrics = evaluate(y_test, test_probs, threshold=THRESHOLD, dataset_name="test")
    
    print_evaluation(val_metrics)
    print_evaluation(test_metrics)
    
    # Feature importance
    print("Top 10 Feature Importances:")
    raw_feature_names = feature_cols + [
        "log_time_to_first_click", "log_avg_inter_click", "log_click_span",
        "log_time_to_first_open", "log_first_nhi_open",
        "fast_and_many", "nhi_heavy_and_fast",
        "nhi_click_frac", "nhi_open_frac",
    ]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i, idx in enumerate(sorted_idx[:10]):
        name = raw_feature_names[idx] if idx < len(raw_feature_names) else f"feature_{idx}"
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    # Log results
    log_result(
        val_metrics, test_metrics,
        experiment_name="baseline_xgboost",
        notes=f"depth={MODEL_PARAMS['max_depth']}, lr={MODEL_PARAMS['learning_rate']}, "
              f"n_est={MODEL_PARAMS['n_estimators']}, features={X_train.shape[1]}, "
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
    print(f"  Validation K-S:  {val_metrics['ks_score']}")
    print(f"  Test K-S:        {test_metrics['ks_score']}")
    print(f"  Validation FPR:  {val_metrics['fpr']}%")
    print(f"  Test FPR:        {test_metrics['fpr']}%")
    print(f"{'='*60}")
    
    if val_metrics['ks_score'] >= 90:
        print("  STATUS: EXCELLENT — Model has strong discriminatory power")
    elif val_metrics['ks_score'] >= 70:
        print("  STATUS: GOOD — Model separates bots from humans well")
    elif val_metrics['ks_score'] >= 50:
        print("  STATUS: MODERATE — Room for improvement")
    else:
        print("  STATUS: POOR — Model needs significant work")
