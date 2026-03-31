"""
Bot-B-Gone ML — train.py
Exp 32: Two-model blend — soft-label LightGBM + hard-label LightGBM.
The soft-label model handles the full spectrum.
The hard-label model is decisive on the extremes (trained only on gold standard).
Blend: where hard labels exist, trust the hard model more.
"""
import sys, time
import numpy as np
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, split_data, evaluate, print_evaluation, log_result

def engineer_features(X):
    def safe_log(i):
        v = X[:, i].copy(); v[v < 0] = 0; return np.log1p(v).reshape(-1, 1)
    
    def safe_div(a, b):
        result = np.zeros_like(a, dtype=np.float32)
        mask = b > 0
        result[mask] = a[mask] / b[mask]
        return result.reshape(-1, 1)
    
    def quantile_bin(col_idx, n_bins=10):
        v = X[:, col_idx].copy()
        valid = v[v >= 0]
        if len(valid) == 0:
            return np.zeros(len(v), dtype=np.float32).reshape(-1, 1)
        edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
        return np.digitize(v, edges[1:-1]).astype(np.float32).reshape(-1, 1)
    
    base = [X]
    for i in [0, 1, 2, 6, 7, 8]:
        base.append(safe_log(i))
    base.append(((X[:,0]>=0)&(X[:,0]<60)&(X[:,5]>=5)).astype(np.float32).reshape(-1,1))
    base.append(((X[:,4]>3)&(X[:,0]>=0)&(X[:,0]<30)).astype(np.float32).reshape(-1,1))
    base.append((X[:,9]>1).astype(np.float32).reshape(-1,1))
    base.append((X[:,9]>3).astype(np.float32).reshape(-1,1))
    base.append(((X[:,6]>=0)&(X[:,6]<60)).astype(np.float32).reshape(-1,1))
    base.append((X[:,6]>900).astype(np.float32).reshape(-1,1))
    base.append((X[:,11]>0.75).astype(np.float32).reshape(-1,1))
    base.append((X[:,11]<0.10).astype(np.float32).reshape(-1,1))
    base.append((X[:,16]>0.95).astype(np.float32).reshape(-1,1))
    base.append((X[:,15]>0.95).astype(np.float32).reshape(-1,1))
    base.append((X[:, 15] * np.clip(np.log1p(np.maximum(X[:, 6], 0)) / 12, 0, 1)).reshape(-1, 1))
    base.append((X[:, 11] * (1 - X[:, 16])).reshape(-1, 1))
    base.append((np.clip(X[:, 13], 0, 100) * X[:, 5]).reshape(-1, 1))
    base.append(safe_div(X[:, 7], np.maximum(X[:, 2], 1)))
    base.append(safe_div(X[:, 12], np.maximum(X[:, 9], 1)))
    base.append(quantile_bin(0))
    base.append(quantile_bin(6))
    base.append(quantile_bin(11))
    return np.hstack(base)

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.42):
    """Enhanced label spreading with 8 behavioral signals."""
    new_labels = soft_labels.copy()
    amb = np.abs(soft_labels - 0.50) < 0.01
    if amb.sum() == 0: return new_labels
    X_a = X_raw[amb]
    
    hist_rate = np.clip(X_a[:, 11], 0, 1)
    ttfo = X_a[:, 6].copy(); ttfo[ttfo < 0] = 300
    ttfo_score = np.clip(np.log1p(ttfo) / np.log1p(86400), 0, 1)
    reopen_score = np.clip(X_a[:, 9] / 5.0, 0, 1)
    nhi_click = 1.0 - np.clip(X_a[:, 16], 0, 1)
    nhi_open = 1.0 - np.clip(X_a[:, 15], 0, 1)
    clicks = X_a[:, 2].copy(); clicks[clicks < 0] = 0
    click_score = np.where(clicks == 0, 0.5,
                  np.where(clicks <= 3, 0.8,
                  np.where(clicks <= 10, 0.6, 0.1)))
    avg_ic = X_a[:, 1].copy(); avg_ic[avg_ic < 0] = 30
    ic_score = np.clip(np.log1p(avg_ic) / np.log1p(300), 0, 1)
    ttfc = X_a[:, 0].copy(); ttfc[ttfc < 0] = 300
    ttfc_score = np.clip(np.log1p(ttfc) / np.log1p(86400), 0, 1)
    
    humanness = (0.25 * hist_rate + 
                 0.15 * ttfo_score + 
                 0.10 * reopen_score +
                 0.15 * nhi_click +
                 0.10 * nhi_open +
                 0.05 * click_score +
                 0.10 * ic_score +
                 0.10 * ttfc_score)
    
    adjustment = (humanness - 0.5) * 2 * spread_amount
    new_labels[amb] = np.clip(0.50 + adjustment, 0.0, 1.0)
    
    print(f"  Label spread stats: mean={new_labels[amb].mean():.4f} std={new_labels[amb].std():.4f}")
    print(f"  Below 0.3: {(new_labels[amb] < 0.3).sum():,}  Above 0.7: {(new_labels[amb] > 0.7).sum():,}")
    
    return new_labels

def train():
    X_raw, soft_labels, hard_labels, feature_cols = load_data()
    splits = split_data(X_raw, soft_labels, hard_labels)
    X_tr, X_v, X_te = splits[0], splits[1], splits[2]
    sl_tr, sl_v, sl_te = splits[3], splits[4], splits[5]
    hl_tr, hl_v, hl_te = splits[6], splits[7], splits[8]
    sl_tr_s = spread_ambiguous_labels(X_tr, sl_tr, spread_amount=0.42)
    X_train = engineer_features(X_tr)
    X_val = engineer_features(X_v)
    X_test = engineer_features(X_te)
    
    print(f"Features: {X_train.shape[1]}")
    
    t0 = time.time()
    
    # ============================================================
    # MODEL 1: Soft-label regression (full spectrum)
    # ============================================================
    train_data = lgb.Dataset(X_train, label=sl_tr_s)
    val_data = lgb.Dataset(X_val, label=sl_v, reference=train_data)
    
    params_soft = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    
    model_soft = lgb.train(
        params_soft, train_data,
        num_boost_round=800,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(0)],
    )
    
    # ============================================================
    # MODEL 2: Hard-label classifier (gold standard only)
    # ============================================================
    hard_mask_tr = ~np.isnan(hl_tr)
    X_hard_tr = X_train[hard_mask_tr]
    hl_hard_tr = hl_tr[hard_mask_tr]
    
    hard_mask_v = ~np.isnan(hl_v)
    X_hard_v = X_val[hard_mask_v]
    hl_hard_v = hl_v[hard_mask_v]
    
    print(f"  Hard-label train: {len(X_hard_tr):,} ({hl_hard_tr.sum():,.0f} human, {len(hl_hard_tr)-hl_hard_tr.sum():,.0f} bot)")
    
    train_hard = lgb.Dataset(X_hard_tr, label=hl_hard_tr)
    val_hard = lgb.Dataset(X_hard_v, label=hl_hard_v, reference=train_hard)
    
    params_hard = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 1.0,
        "lambda_l2": 3.0,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    
    model_hard = lgb.train(
        params_hard, train_hard,
        num_boost_round=500,
        valid_sets=[val_hard],
        callbacks=[lgb.log_evaluation(0)],
    )
    
    train_time = time.time() - t0
    
    # ============================================================
    # BLEND: Soft model for gray zone, hard model anchors the extremes
    # ============================================================
    def blend_predictions(X_feat, hard_labels_subset):
        soft_preds = np.clip(model_soft.predict(X_feat), 0, 1)
        hard_preds = np.clip(model_hard.predict(X_feat), 0, 1)
        
        # Where we have hard labels, trust the hard model more (70/30)
        # Where we don't, trust the soft model more (80/20)
        has_hard = ~np.isnan(hard_labels_subset)
        blended = np.where(
            has_hard,
            0.30 * soft_preds + 0.70 * hard_preds,  # gold standard: trust hard model
            0.80 * soft_preds + 0.20 * hard_preds    # gray zone: trust soft model
        )
        return np.clip(blended, 0, 1)
    
    val_preds = blend_predictions(X_val, hl_v)
    test_preds = blend_predictions(X_test, hl_te)
    
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    log_result(val_m, test_m, experiment_name="exp32_two_model_blend",
               notes=f"Soft+hard model blend, spread=0.42, features={X_train.shape[1]}, train_time={train_time:.1f}s")
    model_soft.save_model(str(Path(__file__).parent / "model.txt"))
    return val_m, test_m

if __name__ == "__main__":
    val_m, test_m = train()
    print(f"\n  Validation Composite: {val_m['composite_score']:.2f} / 100")
    print(f"  Test Composite:       {test_m['composite_score']:.2f} / 100")
    print(f"  Validation K-S:       {val_m['ks_score']}")
    print(f"  Validation MSE:       {val_m['mse']}")
    print(f"  Validation ECE:       {val_m['ece']}")
    print(f"  Validation Spread:    {val_m['spread']}")
