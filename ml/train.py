"""
Bot-B-Gone ML — train.py
Exp 19: XGBoost + LightGBM ensemble.
Average predictions from both models. Different gradient boosting implementations
may disagree on the gray zone, producing better-spread probabilities.
Also: increase spread_amount to 0.18 (sweet spot between 0.15 and 0.20).
"""
import sys, time
import numpy as np
import xgboost as xgb
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

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.18):
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
    sl_tr_s = spread_ambiguous_labels(X_tr, sl_tr, spread_amount=0.18)
    X_train = engineer_features(X_tr)
    X_val = engineer_features(X_v)
    X_test = engineer_features(X_te)
    
    print(f"Features: {X_train.shape[1]}")
    
    t0 = time.time()
    
    # Model 1: LightGBM
    lgb_train = lgb.Dataset(X_train, label=sl_tr_s)
    lgb_val = lgb.Dataset(X_val, label=sl_v, reference=lgb_train)
    lgb_params = {
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
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=800,
                          valid_sets=[lgb_val], callbacks=[lgb.log_evaluation(0)])
    
    # Model 2: XGBoost with different hyperparams
    xgb_dtrain = xgb.DMatrix(X_train, label=sl_tr_s)
    xgb_dval = xgb.DMatrix(X_val, label=sl_v)
    xgb_dtest = xgb.DMatrix(X_test)
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_weight": 10,
        "seed": 123,
        "nthread": -1,
        "verbosity": 0,
    }
    xgb_model = xgb.train(xgb_params, xgb_dtrain, num_boost_round=600,
                          evals=[(xgb_dval, "val")], verbose_eval=False)
    
    train_time = time.time() - t0
    
    # Ensemble: weighted average (0.6 LightGBM + 0.4 XGBoost)
    lgb_val_preds = np.clip(lgb_model.predict(X_val), 0, 1)
    xgb_val_preds = np.clip(xgb_model.predict(xgb_dval), 0, 1)
    val_preds = 0.6 * lgb_val_preds + 0.4 * xgb_val_preds
    
    lgb_test_preds = np.clip(lgb_model.predict(X_test), 0, 1)
    xgb_test_preds = np.clip(xgb_model.predict(xgb_dtest), 0, 1)
    test_preds = 0.6 * lgb_test_preds + 0.4 * xgb_test_preds
    
    # Check disagreement
    disagreement = np.abs(lgb_val_preds - xgb_val_preds)
    print(f"Model disagreement: mean={disagreement.mean():.4f}, max={disagreement.max():.4f}")
    print(f"  LGB mean={lgb_val_preds.mean():.4f} std={lgb_val_preds.std():.4f}")
    print(f"  XGB mean={xgb_val_preds.mean():.4f} std={xgb_val_preds.std():.4f}")
    print(f"  Ensemble mean={val_preds.mean():.4f} std={val_preds.std():.4f}")
    
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    log_result(val_m, test_m, experiment_name="exp19_ensemble",
               notes=f"LGB+XGB ensemble 60/40, spread=0.18, features={X_train.shape[1]}, train_time={train_time:.1f}s")
    lgb_model.save_model(str(Path(__file__).parent / "model.txt"))
    return val_m, test_m

if __name__ == "__main__":
    val_m, test_m = train()
    print(f"\n  Validation Composite: {val_m['composite_score']:.2f} / 100")
    print(f"  Test Composite:       {test_m['composite_score']:.2f} / 100")
    print(f"  Validation K-S:       {val_m['ks_score']}")
    print(f"  Validation MSE:       {val_m['mse']}")
    print(f"  Validation ECE:       {val_m['ece']}")
    print(f"  Validation Spread:    {val_m['spread']}")
