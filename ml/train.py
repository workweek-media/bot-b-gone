"""
Bot-B-Gone ML — train.py
Exp 6: LightGBM instead of XGBoost
Hypothesis: LightGBM uses leaf-wise growth (vs XGBoost's level-wise), which
can produce more nuanced splits. May naturally spread predictions better.
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
    return np.hstack([
        X,
        safe_log(0), safe_log(1), safe_log(2), safe_log(6), safe_log(7), safe_log(8),
        ((X[:,0]>=0)&(X[:,0]<60)&(X[:,5]>=5)).astype(np.float32).reshape(-1,1),
        ((X[:,4]>3)&(X[:,0]>=0)&(X[:,0]<30)).astype(np.float32).reshape(-1,1),
        (X[:,9]>1).astype(np.float32).reshape(-1,1),
        (X[:,9]>3).astype(np.float32).reshape(-1,1),
        ((X[:,6]>=0)&(X[:,6]<60)).astype(np.float32).reshape(-1,1),
        (X[:,6]>900).astype(np.float32).reshape(-1,1),
        (X[:,11]>0.75).astype(np.float32).reshape(-1,1),
        (X[:,11]<0.10).astype(np.float32).reshape(-1,1),
        (X[:,16]>0.95).astype(np.float32).reshape(-1,1),
        (X[:,15]>0.95).astype(np.float32).reshape(-1,1),
    ])

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.15):
    new_labels = soft_labels.copy()
    amb = np.abs(soft_labels - 0.50) < 0.01
    if amb.sum() == 0: return new_labels
    X_a = X_raw[amb]
    hist_rate = np.clip(X_a[:, 11], 0, 1)
    ttfo = X_a[:, 6].copy(); ttfo[ttfo < 0] = 300
    ttfo_score = np.clip(np.log1p(ttfo) / np.log1p(86400), 0, 1)
    reopen_score = (X_a[:, 9] > 1).astype(np.float32)
    humanness = 0.5*hist_rate + 0.3*ttfo_score + 0.2*reopen_score
    adjustment = (humanness - 0.5) * 2 * spread_amount
    new_labels[amb] = np.clip(0.50 + adjustment, 0.0, 1.0)
    return new_labels

def train():
    X_raw, soft_labels, hard_labels, feature_cols = load_data()
    splits = split_data(X_raw, soft_labels, hard_labels)
    X_tr, X_v, X_te = splits[0], splits[1], splits[2]
    sl_tr, sl_v, sl_te = splits[3], splits[4], splits[5]
    hl_tr, hl_v, hl_te = splits[6], splits[7], splits[8]
    sl_tr_s = spread_ambiguous_labels(X_tr, sl_tr, spread_amount=0.15)
    X_train = engineer_features(X_tr)
    X_val = engineer_features(X_v)
    X_test = engineer_features(X_te)
    
    t0 = time.time()
    train_data = lgb.Dataset(X_train, label=sl_tr_s)
    val_data = lgb.Dataset(X_val, label=sl_v, reference=train_data)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
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
    
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(0)],
    )
    train_time = time.time() - t0
    
    val_preds = np.clip(model.predict(X_val), 0, 1)
    test_preds = np.clip(model.predict(X_test), 0, 1)
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    log_result(val_m, test_m, experiment_name="exp6_lightgbm",
               notes=f"LightGBM, leaves=63, lr=0.05, n=500, train_time={train_time:.1f}s")
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
