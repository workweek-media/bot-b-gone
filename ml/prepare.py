"""
Bot-B-Gone ML — prepare.py (IMMUTABLE EVALUATION HARNESS)
==========================================================
This file is READ-ONLY. The autoresearch agent must NOT modify it.

It handles:
  1. Loading the Gold Standard dataset
  2. Splitting into Train / Validation / Test (60/20/20, stratified)
  3. Evaluating model predictions with the K-S statistic
  4. Computing FPR, Recall, Precision, and AUC-ROC
  5. Writing results to results.tsv

GROUND TRUTH LABELS:
  - label=1 (HUMAN): Sailthru real_opens > 0
  - label=0 (BOT):   Machinegun definitive bots (clicked every link in <5s)

The K-S statistic measures the maximum separation between the cumulative
distribution of bot scores and human scores. A perfect model scores 100.
"""

import os
import sys
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path(__file__).parent / "data" / "gold_standard.csv"
FEATURES_PATH = Path(__file__).parent / "data" / "feature_columns.txt"
RESULTS_PATH = Path(__file__).parent / "results.tsv"
RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # 20% of the remaining 80% = 16% of total


# ============================================================
# DATA LOADING & SPLITTING
# ============================================================
def load_data():
    """Load the Gold Standard dataset and return features + labels."""
    df = pd.read_csv(DATA_PATH)
    feature_cols = FEATURES_PATH.read_text().strip().split("\n")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    
    # Replace any NaN/inf with -1 (sentinel for missing)
    X = np.nan_to_num(X, nan=-1.0, posinf=-1.0, neginf=-1.0)
    
    return X, y, feature_cols


def split_data(X, y):
    """
    Stratified split into Train / Validation / Test.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: 80% train+val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: 75% train, 25% val (of the 80%) = 60/20 overall
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=RANDOM_SEED, stratify=y_trainval
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# K-S STATISTIC
# ============================================================
def compute_ks_statistic(y_true, y_prob):
    """
    Compute the Kolmogorov-Smirnov statistic for model evaluation.
    
    Measures the maximum distance between the CDF of predicted probabilities
    for the BOT class (label=0) and the HUMAN class (label=1).
    
    Returns: KS statistic (0-100 scale), KS p-value
    """
    bot_probs = y_prob[y_true == 0]
    human_probs = y_prob[y_true == 1]
    
    ks_stat, ks_pvalue = stats.ks_2samp(bot_probs, human_probs)
    
    return round(ks_stat * 100, 2), ks_pvalue


# ============================================================
# FULL EVALUATION
# ============================================================
def evaluate(y_true, y_prob, threshold=0.5, dataset_name="validation"):
    """
    Full evaluation of model predictions.
    
    Args:
        y_true: Ground truth labels (0=bot, 1=human)
        y_prob: Predicted probability of being HUMAN (0.0 to 1.0)
        threshold: Classification threshold
        dataset_name: Name for logging
    
    Returns: dict of all metrics
    """
    # K-S statistic (the primary metric)
    ks_score, ks_pvalue = compute_ks_statistic(y_true, y_prob)
    
    # Binary predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Standard classification metrics
    auc_roc = round(roc_auc_score(y_true, y_prob) * 100, 2)
    precision = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
    recall = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = round(fp / max(fp + tn, 1) * 100, 4)  # False Positive Rate
    fnr = round(fn / max(fn + tp, 1) * 100, 4)  # False Negative Rate
    
    # Distribution stats
    bot_probs = y_prob[y_true == 0]
    human_probs = y_prob[y_true == 1]
    
    metrics = {
        "dataset": dataset_name,
        "ks_score": ks_score,
        "ks_pvalue": f"{ks_pvalue:.2e}",
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "bot_prob_mean": round(float(bot_probs.mean()), 4),
        "bot_prob_median": round(float(np.median(bot_probs)), 4),
        "human_prob_mean": round(float(human_probs.mean()), 4),
        "human_prob_median": round(float(np.median(human_probs)), 4),
        "n_samples": len(y_true),
        "n_humans": int((y_true == 1).sum()),
        "n_bots": int((y_true == 0).sum()),
    }
    
    return metrics


def print_evaluation(metrics):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {metrics['dataset'].upper()}")
    print(f"{'='*60}")
    print(f"  K-S Score:        {metrics['ks_score']}  (target: maximize)")
    print(f"  K-S p-value:      {metrics['ks_pvalue']}")
    print(f"  AUC-ROC:          {metrics['auc_roc']}%")
    print(f"  Precision:        {metrics['precision']}%")
    print(f"  Recall:           {metrics['recall']}%")
    print(f"  F1:               {metrics['f1']}%")
    print(f"  FPR:              {metrics['fpr']}%  (bots misclassified as human)")
    print(f"  FNR:              {metrics['fnr']}%  (humans misclassified as bot)")
    print(f"  ---")
    print(f"  Bot scores:       mean={metrics['bot_prob_mean']}, median={metrics['bot_prob_median']}")
    print(f"  Human scores:     mean={metrics['human_prob_mean']}, median={metrics['human_prob_median']}")
    print(f"  ---")
    print(f"  Confusion Matrix: TP={metrics['true_positives']:,} FP={metrics['false_positives']:,}")
    print(f"                    FN={metrics['false_negatives']:,} TN={metrics['true_negatives']:,}")
    print(f"  Samples:          {metrics['n_samples']:,} ({metrics['n_humans']:,} human, {metrics['n_bots']:,} bot)")
    print(f"{'='*60}\n")


# ============================================================
# RESULTS LOGGING
# ============================================================
def log_result(val_metrics, test_metrics, experiment_name="baseline", notes=""):
    """Append results to results.tsv for the autoresearch agent to track."""
    timestamp = datetime.datetime.now().isoformat()
    
    row = {
        "timestamp": timestamp,
        "experiment": experiment_name,
        "val_ks_score": val_metrics["ks_score"],
        "val_auc_roc": val_metrics["auc_roc"],
        "val_fpr": val_metrics["fpr"],
        "val_fnr": val_metrics["fnr"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "test_ks_score": test_metrics["ks_score"],
        "test_auc_roc": test_metrics["auc_roc"],
        "test_fpr": test_metrics["fpr"],
        "test_fnr": test_metrics["fnr"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "notes": notes,
    }
    
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a") as f:
        if write_header:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    
    print(f"Results logged to {RESULTS_PATH}")


# ============================================================
# MAIN (for standalone testing)
# ============================================================
if __name__ == "__main__":
    print("Bot-B-Gone ML — Evaluation Harness")
    print("=" * 60)
    
    X, y, feature_cols = load_data()
    print(f"Loaded {len(y):,} samples with {len(feature_cols)} features")
    print(f"  Humans: {(y==1).sum():,}  |  Bots: {(y==0).sum():,}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(y_train):,} ({(y_train==1).sum():,} human, {(y_train==0).sum():,} bot)")
    print(f"  Validation: {len(y_val):,} ({(y_val==1).sum():,} human, {(y_val==0).sum():,} bot)")
    print(f"  Test:       {len(y_test):,} ({(y_test==1).sum():,} human, {(y_test==0).sum():,} bot)")
    
    # Sanity check: random predictions should give K-S ~ 0
    print("\nSanity check with random predictions:")
    random_probs = np.random.rand(len(y_val))
    random_metrics = evaluate(y_val, random_probs, dataset_name="random_baseline")
    print_evaluation(random_metrics)
    
    print("Harness ready. Run train.py to train the model.")
