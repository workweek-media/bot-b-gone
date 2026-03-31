"""
Bot-B-Gone ML — train.py
Exp 18: Iterative self-training (3 rounds) with 80/20 blend.
Exp 17 was 92.73 with 60/40 blend and 1 round. Try:
  - 80% model prediction / 20% original label (more aggressive)
  - 3 rounds of self-training (model refines its own beliefs)
  - Each round uses the previous round's model to update pseudo-labels
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
    
    print(f"Features: {X_train.shape[1]}")
    
    t0 = time.time()
    amb = np.abs(sl_tr - 0.50) < 0.01
    
    params = {
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
    
    current_labels = sl_tr_s.copy()
    blend_ratios = [0.5, 0.7, 0.8]  # increasingly trust the model
    
    for round_i, blend in enumerate(blend_ratios):
        print(f"\n--- Self-training round {round_i+1}, blend={blend} ---")
        
        train_data = lgb.Dataset(X_train, label=current_labels)
        val_data = lgb.Dataset(X_val, label=sl_v, reference=train_data)
        
        n_rounds = 400 if round_i < 2 else 800
        model = lgb.train(params, train_data, num_boost_round=n_rounds,
                          valid_sets=[val_data], callbacks=[lgb.log_evaluation(0)])
        
        # Update pseudo-labels for ambiguous samples
        preds = np.clip(model.predict(X_train), 0, 1)
        current_labels[amb] = blend * preds[amb] + (1 - blend) * sl_tr_s[amb]
        current_labels = np.clip(current_labels, 0, 1)
        
        print(f"  Amb label std: {current_labels[amb].std():.4f}")
        print(f"  Amb label mean: {current_labels[amb].mean():.4f}")
    
    train_time = time.time() - t0
    
    val_preds = np.clip(model.predict(X_val), 0, 1)
    test_preds = np.clip(model.predict(X_test), 0, 1)
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    log_result(val_m, test_m, experiment_name="exp18_iterative_self_train",
               notes=f"3-round self-training, blends=[0.5,0.7,0.8], features={X_train.shape[1]}, train_time={train_time:.1f}s")
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
