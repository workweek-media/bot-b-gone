"""
Bot-B-Gone ML — train.py
Exp 25: Richer label spreading with 8 signals + spread_amount=0.20.
Current spreading uses only 3 signals (hist_rate, ttfo, reopen).
Add: nhi_click_ratio, nhi_open_ratio, click_count, unique_urls, avg_inter_click.
These are the features the model finds most important — use them to create
better pseudo-labels for the ambiguous zone.
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

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.35):
    """Enhanced label spreading with 8 behavioral signals."""
    new_labels = soft_labels.copy()
    amb = np.abs(soft_labels - 0.50) < 0.01
    if amb.sum() == 0: return new_labels
    X_a = X_raw[amb]
    
    # Signal 1: Historical open rate (col 11) — higher = more human
    hist_rate = np.clip(X_a[:, 11], 0, 1)
    
    # Signal 2: Time to first open (col 6) — later = more human
    ttfo = X_a[:, 6].copy(); ttfo[ttfo < 0] = 300
    ttfo_score = np.clip(np.log1p(ttfo) / np.log1p(86400), 0, 1)
    
    # Signal 3: Reopen count (col 9) — more reopens = more human
    reopen_score = np.clip(X_a[:, 9] / 5.0, 0, 1)
    
    # Signal 4: NHI click ratio (col 16) — lower = more human
    nhi_click = 1.0 - np.clip(X_a[:, 16], 0, 1)
    
    # Signal 5: NHI open ratio (col 15) — lower = more human
    nhi_open = 1.0 - np.clip(X_a[:, 15], 0, 1)
    
    # Signal 6: Click count (col 2) — moderate clicks = human, extreme = bot
    clicks = X_a[:, 2].copy(); clicks[clicks < 0] = 0
    click_score = np.where(clicks == 0, 0.5,
                  np.where(clicks <= 3, 0.8,
                  np.where(clicks <= 10, 0.6, 0.1)))
    
    # Signal 7: Avg inter-click seconds (col 1) — slower = more human
    avg_ic = X_a[:, 1].copy(); avg_ic[avg_ic < 0] = 30
    ic_score = np.clip(np.log1p(avg_ic) / np.log1p(300), 0, 1)
    
    # Signal 8: Time to first click (col 0) — later = more human
    ttfc = X_a[:, 0].copy(); ttfc[ttfc < 0] = 300
    ttfc_score = np.clip(np.log1p(ttfc) / np.log1p(86400), 0, 1)
    
    # Weighted combination
    humanness = (0.25 * hist_rate + 
                 0.15 * ttfo_score + 
                 0.10 * reopen_score +
                 0.15 * nhi_click +
                 0.10 * nhi_open +
                 0.05 * click_score +
                 0.10 * ic_score +
                 0.10 * ttfc_score)
    
    # Asymmetric spreading: push bot-like samples harder (cleaner signal)
    bot_spread = spread_amount * 1.3  # more aggressive for bot side
    human_spread = spread_amount * 0.8  # gentler for human side
    adjustment = np.where(
        humanness < 0.5,
        (humanness - 0.5) * 2 * bot_spread,
        (humanness - 0.5) * 2 * human_spread
    )
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
    train_data = lgb.Dataset(X_train, label=sl_tr_s)
    val_data = lgb.Dataset(X_val, label=sl_v, reference=train_data)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    
    model = lgb.train(
        params, train_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(0)],
    )
    train_time = time.time() - t0
    
    val_preds = np.clip(model.predict(X_val), 0, 1)
    test_preds = np.clip(model.predict(X_test), 0, 1)
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    log_result(val_m, test_m, experiment_name="exp25_rich_spread",
               notes=f"8-signal spreading, spread=0.20, features={X_train.shape[1]}, train_time={train_time:.1f}s")
    model.save_model(str(Path(__file__).parent / "model.txt"))
    return val_m, test_m

if __name__ == "__main__":
    val_m, test_m = train()
    print(f"\n  Validation Composite: {val_m['composite_score']:.2f} / 100")
    print(f"  Test Composite:       {test_m['composite_score']:.2f} / 100")
    print(f"  Validation K-S:       {val_m['ks_score']}")
    print(f"  Validation MSE:       {val_m['mse']}")
    print(f"  Validation ECE:       {val_m['ece']}")
    print(f"  Validation Spread:    {val_m['spread']}")
