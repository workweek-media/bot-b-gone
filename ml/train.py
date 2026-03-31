"""
Bot-B-Gone ML — train.py
Exp 13: Omeda + nikwen inspired features.
New signals from industry research:
  - Open-to-click gap (time between first open and first click)
  - Tighter machinegun (2 clicks in 2s, from Omeda's Code 1)
  - Click session entropy (how evenly spread are clicks over time?)
  - Bot repeat offender score (nhi_ratio across multiple signals)
  - Engagement depth ratio (verified_opens / total_opens)
  - Click breadth vs speed interaction (Omeda: >200 clicks = bot)
  - Open-only vs click-engaged segmentation
"""
import sys, time
import numpy as np
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, split_data, evaluate, print_evaluation, log_result

# Feature indices (from feature_columns.txt):
# 0: time_to_first_click_sec    1: avg_inter_click_sec     2: click_span_sec
# 3: raw_total_clicks           4: nhi_clicks              5: unique_urls_clicked
# 6: time_to_first_open_sec     7: open_span_sec           8: first_nhi_open_sec
# 9: raw_total_opens           10: nhi_opens              11: user_historical_open_rate
# 12: user_lifetime_verified_opens  13: clicks_per_second  14: url_diversity_ratio
# 15: nhi_open_ratio           16: nhi_click_ratio        17: has_any_clicks
# 18: has_any_opens

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
    
    base = [X]  # 19 raw features
    
    # --- Log transforms (from exp 10) ---
    for i in [0, 1, 2, 6, 7, 8]:
        base.append(safe_log(i))
    
    # --- Binary flags (from exp 10) ---
    base.append(((X[:,0]>=0)&(X[:,0]<60)&(X[:,5]>=5)).astype(np.float32).reshape(-1,1))  # fast+many_urls
    base.append(((X[:,4]>3)&(X[:,0]>=0)&(X[:,0]<30)).astype(np.float32).reshape(-1,1))   # nhi_clicks+fast
    base.append((X[:,9]>1).astype(np.float32).reshape(-1,1))   # multi_open
    base.append((X[:,9]>3).astype(np.float32).reshape(-1,1))   # many_opens
    base.append(((X[:,6]>=0)&(X[:,6]<60)).astype(np.float32).reshape(-1,1))  # fast_open
    base.append((X[:,6]>900).astype(np.float32).reshape(-1,1))  # slow_open
    base.append((X[:,11]>0.75).astype(np.float32).reshape(-1,1))  # high_hist_rate
    base.append((X[:,11]<0.10).astype(np.float32).reshape(-1,1))  # low_hist_rate
    base.append((X[:,16]>0.95).astype(np.float32).reshape(-1,1))  # almost_all_nhi_clicks
    base.append((X[:,15]>0.95).astype(np.float32).reshape(-1,1))  # almost_all_nhi_opens
    
    # --- Interaction features (from exp 10) ---
    base.append((X[:, 15] * np.clip(np.log1p(np.maximum(X[:, 6], 0)) / 12, 0, 1)).reshape(-1, 1))
    base.append((X[:, 11] * (1 - X[:, 16])).reshape(-1, 1))
    base.append((np.clip(X[:, 13], 0, 100) * X[:, 5]).reshape(-1, 1))
    base.append(safe_div(X[:, 7], np.maximum(X[:, 2], 1)))
    base.append(safe_div(X[:, 12], np.maximum(X[:, 9], 1)))
    
    # --- Quantile bins (from exp 10) ---
    base.append(quantile_bin(0))   # time_to_first_click bins
    base.append(quantile_bin(6))   # time_to_first_open bins
    base.append(quantile_bin(11))  # user_historical_open_rate bins
    
    # ===== NEW: Omeda + nikwen inspired features =====
    
    # 1. OPEN-TO-CLICK GAP: time between first open and first click
    #    Bots: near-zero (open and click simultaneously)
    #    Humans: seconds to minutes (read, then decide to click)
    ttfo = X[:, 6].copy()
    ttfc = X[:, 0].copy()
    open_to_click_gap = np.where(
        (ttfo >= 0) & (ttfc >= 0),
        ttfc - ttfo,  # positive = clicked after opening
        -1.0  # missing
    )
    base.append(open_to_click_gap.reshape(-1, 1))
    base.append(np.log1p(np.maximum(open_to_click_gap, 0)).reshape(-1, 1))
    
    # 2. TIGHTER MACHINEGUN (Omeda Code 1: 2 clicks in 2s)
    #    avg_inter_click < 2s AND raw_total_clicks >= 2
    base.append(((X[:,1] >= 0) & (X[:,1] < 2) & (X[:,3] >= 2)).astype(np.float32).reshape(-1,1))
    
    # 3. OMEDA VOLUME THRESHOLD: >200 total clicks = definitive bot
    base.append((X[:,3] > 200).astype(np.float32).reshape(-1,1))
    base.append((X[:,3] > 50).astype(np.float32).reshape(-1,1))
    base.append((X[:,3] > 10).astype(np.float32).reshape(-1,1))
    
    # 4. COMPOSITE BOT SCORE: combine multiple NHI signals
    #    How many bot signals fire simultaneously?
    bot_signal_count = (
        (X[:,16] > 0.8).astype(np.float32) +  # high nhi_click_ratio
        (X[:,15] > 0.8).astype(np.float32) +  # high nhi_open_ratio
        ((X[:,0] >= 0) & (X[:,0] < 60)).astype(np.float32) +  # fast first click
        ((X[:,6] >= 0) & (X[:,6] < 60)).astype(np.float32) +  # fast first open
        (X[:,1] < 2).astype(np.float32) +  # fast inter-click
        (X[:,5] > 5).astype(np.float32)  # many unique URLs
    )
    base.append(bot_signal_count.reshape(-1, 1))
    
    # 5. COMPOSITE HUMAN SCORE: combine multiple human signals
    human_signal_count = (
        (X[:,11] > 0.5).astype(np.float32) +  # decent historical rate
        (X[:,12] > 0).astype(np.float32) +  # has verified opens
        (X[:,7] > 3600).astype(np.float32) +  # open span > 1hr
        ((X[:,6] >= 300)).astype(np.float32) +  # first open > 5min
        (X[:,16] < 0.5).astype(np.float32) +  # low nhi_click_ratio
        (X[:,15] < 0.5).astype(np.float32)  # low nhi_open_ratio
    )
    base.append(human_signal_count.reshape(-1, 1))
    
    # 6. BOT vs HUMAN SIGNAL DELTA
    base.append((human_signal_count - bot_signal_count).reshape(-1, 1))
    
    # 7. ENGAGEMENT DEPTH: verified_opens / total_opens (Omeda's confirmed vs raw)
    base.append(safe_div(X[:, 12], np.maximum(X[:, 9], 1)))
    
    # 8. CLICK CONCENTRATION: clicks_per_second * nhi_click_ratio
    #    High = fast AND mostly NHI = strong bot signal
    base.append((np.clip(X[:, 13], 0, 100) * X[:, 16]).reshape(-1, 1))
    
    # 9. OPEN-ONLY ENGAGEMENT: has opens but no clicks (common for humans reading)
    base.append(((X[:,18] > 0) & (X[:,17] == 0)).astype(np.float32).reshape(-1,1))
    
    # 10. CLICK-WITHOUT-MEANINGFUL-OPEN: has clicks but very fast open (bot prefetch)
    base.append(((X[:,17] > 0) & (X[:,6] >= 0) & (X[:,6] < 5)).astype(np.float32).reshape(-1,1))
    
    # 11. TIME RATIOS: how does click timing relate to open timing?
    base.append(safe_div(X[:, 2], np.maximum(X[:, 7], 1)))  # click_span / open_span
    
    # 12. QUANTILE BINS for new features
    base.append(quantile_bin(3))   # raw_total_clicks bins
    base.append(quantile_bin(1))   # avg_inter_click bins
    base.append(quantile_bin(9))   # raw_total_opens bins
    
    # 13. SQUARED TERMS for key signals (capture non-linearity)
    base.append((X[:, 15] ** 2).reshape(-1, 1))  # nhi_open_ratio squared
    base.append((X[:, 16] ** 2).reshape(-1, 1))  # nhi_click_ratio squared
    base.append((X[:, 11] ** 2).reshape(-1, 1))  # user_historical_open_rate squared
    
    return np.hstack(base)

def spread_ambiguous_labels(X_raw, soft_labels, spread_amount=0.18):
    """Spread ambiguous 0.50 labels using behavioral signals."""
    new_labels = soft_labels.copy()
    amb = np.abs(soft_labels - 0.50) < 0.01
    if amb.sum() == 0: return new_labels
    X_a = X_raw[amb]
    
    # Historical open rate (strongest human signal)
    hist_rate = np.clip(X_a[:, 11], 0, 1)
    
    # Time to first open (slow = human)
    ttfo = X_a[:, 6].copy(); ttfo[ttfo < 0] = 300
    ttfo_score = np.clip(np.log1p(ttfo) / np.log1p(86400), 0, 1)
    
    # Multiple opens over time (human behavior)
    reopen_score = (X_a[:, 9] > 1).astype(np.float32)
    
    # Verified opens (strong human signal)
    verified_score = (X_a[:, 12] > 0).astype(np.float32)
    
    # NHI ratio (low = human)
    nhi_score = 1 - np.clip(X_a[:, 15], 0, 1)
    
    # Composite humanness
    humanness = 0.30*hist_rate + 0.25*ttfo_score + 0.15*reopen_score + 0.15*verified_score + 0.15*nhi_score
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
    train_data = lgb.Dataset(X_train, label=sl_tr_s)
    val_data = lgb.Dataset(X_val, label=sl_v, reference=train_data)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "max_depth": 10,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    
    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)],
    )
    train_time = time.time() - t0
    
    val_preds = np.clip(model.predict(X_val), 0, 1)
    test_preds = np.clip(model.predict(X_test), 0, 1)
    val_m = evaluate(sl_v, hl_v, val_preds, dataset_name="validation")
    test_m = evaluate(sl_te, hl_te, test_preds, dataset_name="test")
    print_evaluation(val_m); print_evaluation(test_m)
    
    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"\nTop 15 features by gain:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Feature {idx}: {imp[idx]:,.0f}")
    
    log_result(val_m, test_m, experiment_name="exp13_omeda_nikwen",
               notes=f"omeda+nikwen signals, features={X_train.shape[1]}, spread=0.18, train_time={train_time:.1f}s")
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
