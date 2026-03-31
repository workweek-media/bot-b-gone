"""
Bot-B-Gone ML — prepare.py (IMMUTABLE EVALUATION HARNESS)
============================================================
DO NOT MODIFY THIS FILE. This is the fixed evaluation framework.

METRIC DESIGN:
  The model is trained on SOFT LABELS (0.0-1.0) from our rule-based system.
  The primary metric is a COMPOSITE SCORE that rewards:
    1. Low MSE on soft labels (can it learn the rule-based spectrum?)
    2. High K-S on the hard-labeled extremes (can it still separate bots from humans?)
    3. Calibration quality (are the probabilities meaningful?)
    4. Gap spread (does it produce a distribution, not a binary collapse?)

  composite_score = (ks_component * 0.3) + (mse_component * 0.3)
                  + (calibration_component * 0.2) + (spread_component * 0.2)

  Higher is better. Max = 100.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SOFT_LABEL_FILE = DATA_DIR / "soft_labeled.csv"
FEATURE_COLUMNS_FILE = DATA_DIR / "feature_columns.txt"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load soft-labeled dataset. Returns X, soft_labels, hard_labels, feature_names."""
    df = pd.read_csv(SOFT_LABEL_FILE)
    
    with open(FEATURE_COLUMNS_FILE) as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=-1.0, posinf=-1.0, neginf=-1.0)
    
    soft_labels = df['soft_label'].values.astype(np.float32)
    hard_labels = df['hard_label'].values  # contains NaN for uncertain
    
    return X, soft_labels, hard_labels, feature_cols


def split_data(X, soft_labels, hard_labels):
    """
    Fixed stratified split: 60% train, 20% val, 20% test.
    Stratified on binned soft_label to ensure all tiers are represented.
    """
    np.random.seed(RANDOM_SEED)
    n = len(X)
    
    # Bin soft labels for stratification
    bins = np.digitize(soft_labels, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    train_idx, val_idx, test_idx = [], [], []
    
    for b in np.unique(bins):
        mask = bins == b
        indices = np.where(mask)[0]
        np.random.shuffle(indices)
        
        n_bin = len(indices)
        n_train = int(n_bin * 0.6)
        n_val = int(n_bin * 0.2)
        
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])
    
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    return (
        X[train_idx], X[val_idx], X[test_idx],
        soft_labels[train_idx], soft_labels[val_idx], soft_labels[test_idx],
        hard_labels[train_idx], hard_labels[val_idx], hard_labels[test_idx],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ks(hard_labels, predictions):
    """K-S statistic on hard-labeled extremes only. Returns 0-100."""
    mask = ~np.isnan(hard_labels)
    if mask.sum() < 10:
        return 0.0
    
    hl = hard_labels[mask]
    preds = predictions[mask]
    
    bot_preds = preds[hl == 0]
    human_preds = preds[hl == 1]
    
    if len(bot_preds) == 0 or len(human_preds) == 0:
        return 0.0
    
    # K-S: max difference between CDFs
    all_thresholds = np.sort(np.unique(np.concatenate([bot_preds, human_preds])))
    max_diff = 0.0
    for t in all_thresholds:
        bot_cdf = (bot_preds <= t).mean()
        human_cdf = (human_preds <= t).mean()
        max_diff = max(max_diff, abs(human_cdf - bot_cdf))
    
    return round(max_diff * 100, 2)


def compute_mse(soft_labels, predictions):
    """Mean squared error on soft labels. Lower is better."""
    return float(np.mean((soft_labels - predictions) ** 2))


def compute_calibration_error(soft_labels, predictions, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predictions, compares mean prediction to mean soft label in each bin.
    Lower is better. Returns 0-1.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predictions)
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # include 1.0 in last bin
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
        
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        
        avg_pred = predictions[mask].mean()
        avg_label = soft_labels[mask].mean()
        ece += (n_bin / total) * abs(avg_pred - avg_label)
    
    return float(ece)


def compute_spread(predictions):
    """
    Measures how well the model spreads predictions across the probability range.
    Uses entropy of the prediction histogram. Higher entropy = better spread.
    Returns 0-100 where 100 = perfectly uniform distribution.
    """
    hist, _ = np.histogram(predictions, bins=20, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # remove empty bins
    
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(20)  # uniform distribution over 20 bins
    
    return round((entropy / max_entropy) * 100, 2)


def evaluate(soft_labels, hard_labels, predictions, dataset_name="validation"):
    """
    Compute the composite evaluation score.
    
    Returns dict with all metrics and the composite score.
    """
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Component 1: K-S on hard labels (0-100, higher is better)
    ks = compute_ks(hard_labels, predictions)
    ks_component = ks  # already 0-100
    
    # Component 2: MSE on soft labels (0-1, lower is better -> invert)
    mse = compute_mse(soft_labels, predictions)
    mse_component = max(0, (1 - mse * 4)) * 100  # MSE of 0.25 -> score 0, MSE of 0 -> score 100
    
    # Component 3: Calibration (0-1, lower is better -> invert)
    ece = compute_calibration_error(soft_labels, predictions)
    calibration_component = max(0, (1 - ece * 5)) * 100  # ECE of 0.2 -> score 0
    
    # Component 4: Spread (0-100, higher is better)
    spread = compute_spread(predictions)
    spread_component = spread
    
    # Composite score
    composite = (
        ks_component * 0.3 +
        mse_component * 0.3 +
        calibration_component * 0.2 +
        spread_component * 0.2
    )
    
    # Prediction distribution stats
    hist, _ = np.histogram(predictions, bins=10, range=(0, 1))
    
    return {
        'dataset': dataset_name,
        'composite_score': round(composite, 2),
        'ks_score': ks,
        'ks_component': round(ks_component * 0.3, 2),
        'mse': round(mse, 6),
        'mse_component': round(mse_component * 0.3, 2),
        'ece': round(ece, 6),
        'calibration_component': round(calibration_component * 0.2, 2),
        'spread': spread,
        'spread_component': round(spread_component * 0.2, 2),
        'pred_mean': round(float(predictions.mean()), 4),
        'pred_std': round(float(predictions.std()), 4),
        'pred_histogram': hist.tolist(),
    }


def print_evaluation(metrics):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {metrics['dataset'].upper()} RESULTS")
    print(f"{'='*60}")
    print(f"  COMPOSITE SCORE:    {metrics['composite_score']:.2f} / 100")
    print(f"  |-- K-S (x0.3):     {metrics['ks_score']:.2f} -> {metrics['ks_component']:.2f}")
    print(f"  |-- MSE (x0.3):     {metrics['mse']:.6f} -> {metrics['mse_component']:.2f}")
    print(f"  |-- ECE (x0.2):     {metrics['ece']:.6f} -> {metrics['calibration_component']:.2f}")
    print(f"  |-- Spread (x0.2):  {metrics['spread']:.2f} -> {metrics['spread_component']:.2f}")
    print(f"")
    print(f"  Prediction Stats:")
    print(f"    Mean: {metrics['pred_mean']:.4f}  Std: {metrics['pred_std']:.4f}")
    print(f"    Histogram (10 bins, 0-1):")
    bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
            '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    total = sum(metrics['pred_histogram'])
    for label, count in zip(bins, metrics['pred_histogram']):
        pct = count / total * 100 if total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {label}: {count:>7,} ({pct:>5.1f}%) {bar}")
    print(f"{'='*60}")


def log_result(val_metrics, test_metrics, experiment_name="", notes=""):
    """Append result to results.tsv."""
    header = "timestamp\texperiment\tval_composite\tval_ks\tval_mse\tval_ece\tval_spread\ttest_composite\ttest_ks\tstatus\tnotes"
    
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'w') as f:
            f.write(header + "\n")
    
    row = "\t".join([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        experiment_name,
        str(val_metrics['composite_score']),
        str(val_metrics['ks_score']),
        str(val_metrics['mse']),
        str(val_metrics['ece']),
        str(val_metrics['spread']),
        str(test_metrics['composite_score']),
        str(test_metrics['ks_score']),
        "keep",
        notes,
    ])
    
    with open(RESULTS_FILE, 'a') as f:
        f.write(row + "\n")
    
    print(f"\nLogged to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# MAIN (for standalone testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Bot-B-Gone ML — Evaluation Harness (Soft Labels)")
    print("=" * 60)
    
    X, soft_labels, hard_labels, feature_cols = load_data()
    print(f"Loaded {len(soft_labels):,} samples with {len(feature_cols)} features")
    print(f"  Soft label range: {soft_labels.min():.2f} - {soft_labels.max():.2f}")
    print(f"  Hard labels: {(~np.isnan(hard_labels)).sum():,} labeled, {np.isnan(hard_labels).sum():,} uncertain")
    
    splits = split_data(X, soft_labels, hard_labels)
    X_train, X_val, X_test = splits[0], splits[1], splits[2]
    sl_train, sl_val, sl_test = splits[3], splits[4], splits[5]
    
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(sl_train):,}")
    print(f"  Validation: {len(sl_val):,}")
    print(f"  Test:       {len(sl_test):,}")
    
    # Sanity check: constant prediction of 0.5 should give baseline
    print("\nSanity check with constant 0.5 predictions:")
    constant_preds = np.full(len(sl_val), 0.5)
    constant_metrics = evaluate(sl_val, hard_labels[len(sl_train):len(sl_train)+len(sl_val)], constant_preds, dataset_name="constant_0.5")
    print_evaluation(constant_metrics)
    
    print("Harness ready. Run train.py to train the model.")
