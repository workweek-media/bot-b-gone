"""
Bot-B-Gone ML — score_gap.py
==============================
Apply the trained model to the UNCERTAIN GAP (events that are neither
Sailthru-real nor machinegun-definitive) and analyze the probability
distribution.

This answers the critical question: Does the model produce a meaningful
probability spectrum, or does it collapse to 0/1?
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from google.cloud import bigquery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/skills/workweek-bigquery/references/service_account.json"

sys.path.insert(0, str(Path(__file__).parent))
from train import engineer_features

# Load the trained model
model_path = Path(__file__).parent / "model.json"
model = xgb.XGBClassifier()
model.load_model(str(model_path))
print(f"Loaded model from {model_path}")

# Query the uncertain gap from BigQuery (sample for speed)
client = bigquery.Client()

query = """
SELECT
    -- Same features as gold_standard extraction
    COALESCE(click_time_after_send_sec, -1) AS time_to_first_click_sec,
    COALESCE(avg_inter_click_sec, -1) AS avg_inter_click_sec,
    COALESCE(click_span_sec, -1) AS click_span_sec,
    COALESCE(sailthru_raw_total_clicks, 0) AS raw_total_clicks,
    COALESCE(sailthru_nhi_clicks, 0) AS nhi_clicks,
    COALESCE(unique_urls_clicked, 0) AS unique_urls_clicked,
    COALESCE(open_time_after_send_sec, -1) AS time_to_first_open_sec,
    COALESCE(open_span_sec, -1) AS open_span_sec,
    COALESCE(first_nhi_open_sec, -1) AS first_nhi_open_sec,
    COALESCE(sailthru_raw_total_opens, 0) AS raw_total_opens,
    COALESCE(sailthru_nhi_opens, 0) AS nhi_opens,
    COALESCE(user_core_open_rate, 0) AS user_historical_open_rate,
    COALESCE(user_core_verified_opens, 0) AS user_lifetime_verified_opens,
    CASE 
      WHEN COALESCE(sailthru_raw_total_clicks, 0) > 1 AND COALESCE(click_span_sec, 0) > 0
      THEN CAST(sailthru_raw_total_clicks AS FLOAT64) / CAST(click_span_sec AS FLOAT64)
      ELSE 0
    END AS clicks_per_second,
    CASE
      WHEN COALESCE(unique_urls_clicked, 0) > 0 AND COALESCE(sailthru_raw_total_clicks, 0) > 0
      THEN CAST(unique_urls_clicked AS FLOAT64) / CAST(sailthru_raw_total_clicks AS FLOAT64)
      ELSE 0
    END AS url_diversity_ratio,
    CASE
      WHEN COALESCE(sailthru_raw_total_opens, 0) > 0
      THEN CAST(COALESCE(sailthru_nhi_opens, 0) AS FLOAT64) / CAST(sailthru_raw_total_opens AS FLOAT64)
      ELSE 0
    END AS nhi_open_ratio,
    CASE
      WHEN COALESCE(sailthru_raw_total_clicks, 0) > 0
      THEN CAST(COALESCE(sailthru_nhi_clicks, 0) AS FLOAT64) / CAST(sailthru_raw_total_clicks AS FLOAT64)
      ELSE 0
    END AS nhi_click_ratio,
    CASE WHEN COALESCE(sailthru_raw_total_clicks, 0) > 0 THEN 1 ELSE 0 END AS has_any_clicks,
    CASE WHEN COALESCE(sailthru_raw_total_opens, 0) > 0 THEN 1 ELSE 0 END AS has_any_opens
FROM `ww-analytics-dev.unified.fact_user_send_engagement`
WHERE send_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
  AND (sailthru_raw_total_opens > 0 OR sailthru_raw_total_clicks > 0)
  AND (sailthru_real_opens IS NULL OR sailthru_real_opens = 0)
  AND (click_rule_bot_machinegun_definitive IS NULL OR click_rule_bot_machinegun_definitive = FALSE)
ORDER BY RAND()
LIMIT 500000
"""

print("Querying BigQuery for uncertain gap sample (500K)...")
df = client.query(query).to_dataframe()
print(f"Loaded {len(df):,} uncertain gap events")

# Prepare features
feature_cols = [c for c in df.columns]
X_raw = df[feature_cols].values.astype(np.float32)
X_raw = np.nan_to_num(X_raw, nan=-1.0, posinf=-1.0, neginf=-1.0)

# Engineer features (same as training)
X = engineer_features(X_raw)

# Score
probs = model.predict_proba(X)[:, 1]  # P(human)

# Analyze distribution
print(f"\n{'='*60}")
print(f"  UNCERTAIN GAP PROBABILITY DISTRIBUTION")
print(f"{'='*60}")
print(f"  Mean P(human):   {probs.mean():.4f}")
print(f"  Median P(human): {np.median(probs):.4f}")
print(f"  Std:             {probs.std():.4f}")
print(f"  Min:             {probs.min():.4f}")
print(f"  Max:             {probs.max():.4f}")

# Histogram buckets
buckets = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
bucket_labels = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                 '50-60%', '60-70%', '70-80%', '80-90%', '90-95%', '95-100%']

print(f"\n  Probability Distribution:")
for i in range(len(buckets)-1):
    count = ((probs >= buckets[i]) & (probs < buckets[i+1])).sum()
    pct = count / len(probs) * 100
    bar = '#' * int(pct)
    print(f"  {bucket_labels[i]:>8s}: {count:>7,} ({pct:>5.1f}%) {bar}")

# Classification at different thresholds
print(f"\n  Classification at Different Thresholds:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    n_human = (probs >= threshold).sum()
    n_bot = (probs < threshold).sum()
    print(f"  Threshold {threshold}: {n_human:>7,} human ({n_human/len(probs)*100:.1f}%) | {n_bot:>7,} bot ({n_bot/len(probs)*100:.1f}%)")

print(f"\n{'='*60}")
