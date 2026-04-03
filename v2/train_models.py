#!/usr/bin/env python3
"""
Train per-open and per-click Random Forest classifiers using your subscriber
GMM labels as weak supervision.

Usage:
  python3 train_models.py --events events.csv --profiles profiles/ --output models/

Prerequisites:
  - Run build_profiles.py first to create subscriber_classifications.csv
"""

import os
import gc
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
warnings.filterwarnings('ignore')

BOT_DEVICES = ['Internet Explorer', 'Other', 'Windows Tablet']


def train_open_model(df, bot_set, human_set, n_trees=200):
    """Train per-open classifier."""
    print("\nTraining open model...")
    opens = df[df['open_timestamp'].notna()].copy()
    if len(opens) < 1000:
        print("  Not enough open events to train. Need 1000+.")
        return None

    opens['send_timestamp'] = pd.to_datetime(opens['send_timestamp'], errors='coerce')
    opens['open_timestamp'] = pd.to_datetime(opens['open_timestamp'], errors='coerce')
    opens['click_timestamp'] = pd.to_datetime(opens['click_timestamp'], errors='coerce')

    sas = (opens['open_timestamp'] - opens['send_timestamp']).dt.total_seconds().clip(0, 2592000)
    dev = opens['device'].fillna('Unknown') if 'device' in opens.columns else pd.Series('Unknown', index=opens.index)

    X = pd.DataFrame({
        'seconds_after_send': sas,
        'device_is_apple_mail': (dev == 'Apple Mail').astype(int),
        'device_is_gmail': (dev == 'Gmail').astype(int),
        'device_is_bot': dev.isin(BOT_DEVICES).astype(int),
        'device_is_iphone': (dev == 'iPhone').astype(int),
        'device_is_android': (dev == 'Android').astype(int),
        'had_click': opens['click_timestamp'].notna().astype(int),
        'hour_of_open': opens['open_timestamp'].dt.hour,
        'is_weekday': opens['open_timestamp'].dt.dayofweek.lt(5).astype(int),
        'click_within_5s_of_open': (
            (opens['click_timestamp'] - opens['open_timestamp']).dt.total_seconds().between(0, 5)
        ).astype(int),
    }).fillna(0)

    # Labels from GMM (weak supervision)
    sid = opens['subscriber_id'] if 'subscriber_id' in opens.columns else opens.iloc[:, 0]
    y = sid.map(lambda s: 1 if s in bot_set else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_trees, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Open model AUC: {auc:.3f} (expect ~0.96)")
    print(f"  Trained on {len(X_train):,} events")
    return model


def train_click_model(df, bot_set, human_set, n_trees=200):
    """Train per-click classifier."""
    print("\nTraining click model...")
    clicks = df[df['click_timestamp'].notna()].copy()
    if len(clicks) < 1000:
        print("  Not enough click events to train. Need 1000+.")
        return None

    clicks['send_timestamp'] = pd.to_datetime(clicks['send_timestamp'], errors='coerce')
    clicks['open_timestamp'] = pd.to_datetime(clicks['open_timestamp'], errors='coerce')
    clicks['click_timestamp'] = pd.to_datetime(clicks['click_timestamp'], errors='coerce')

    sas = (clicks['click_timestamp'] - clicks['send_timestamp']).dt.total_seconds().clip(0, 2592000)

    sid_col = 'subscriber_id' if 'subscriber_id' in clicks.columns else clicks.columns[0]
    click_counts = clicks.groupby(sid_col).size()
    clicks['total_clicks_in_blast'] = clicks[sid_col].map(click_counts).fillna(1)
    clicks = clicks.sort_values([sid_col, 'click_timestamp'])
    clicks['inter_click_interval'] = clicks.groupby(sid_col)['click_timestamp'].diff().dt.total_seconds().fillna(0)

    dev = clicks['device'].fillna('Unknown') if 'device' in clicks.columns else pd.Series('Unknown', index=clicks.index)
    otc = (clicks['click_timestamp'] - clicks['open_timestamp']).dt.total_seconds()

    X = pd.DataFrame({
        'seconds_after_send': sas,
        'inter_click_interval': clicks['inter_click_interval'],
        'total_clicks_in_blast': clicks['total_clicks_in_blast'],
        'device_is_bot': dev.isin(BOT_DEVICES).astype(int),
        'device_is_apple': dev.isin(['Apple Mail', 'iPhone', 'iPad']).astype(int),
        'device_is_mobile': dev.isin(['iPhone', 'Android', 'iPad']).astype(int),
        'had_open': clicks['open_timestamp'].notna().astype(int),
        'is_weekday': clicks['click_timestamp'].dt.dayofweek.lt(5).astype(int),
        'hour_of_click': clicks['click_timestamp'].dt.hour,
        'seconds_open_to_click': otc.fillna(-1).clip(-1, 2592000),
        'click_within_5min': (sas <= 300).astype(int),
        'click_within_10s': (sas <= 10).astype(int),
        'multi_click': (clicks['total_clicks_in_blast'] > 1).astype(int),
        'phantom_click': (~clicks['open_timestamp'].notna()).astype(int),
    }).fillna(0)

    y = clicks[sid_col].map(lambda s: 1 if s in bot_set else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_trees, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Click model AUC: {auc:.3f} (expect ~0.99)")
    print(f"  Trained on {len(X_train):,} events")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train bot detection ML models')
    parser.add_argument('--events', required=True, help='Path to events CSV')
    parser.add_argument('--profiles', default='profiles', help='Profiles directory (from build_profiles.py)')
    parser.add_argument('--output', default='models', help='Output directory for models')
    parser.add_argument('--subscriber-col', default='subscriber_id')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load subscriber classifications
    class_path = os.path.join(args.profiles, 'subscriber_classifications.csv')
    print(f"Loading subscriber classifications from {class_path}...")
    subs = pd.read_csv(class_path)
    bot_set = set(subs[subs['classification'] == 'Bot'].index)
    human_set = set(subs[subs['classification'] == 'Human'].index)
    print(f"  {len(bot_set):,} bots, {len(human_set):,} humans")

    # Load events
    print(f"Loading events from {args.events}...")
    df = pd.read_csv(args.events, engine='python', on_bad_lines='skip')
    print(f"  {len(df):,} rows")

    # Train open model
    open_model = train_open_model(df, bot_set, human_set)
    if open_model:
        path = os.path.join(args.output, 'open_model.pkl')
        pickle.dump(open_model, open(path, 'wb'))
        print(f"  Saved to {path}")

    # Train click model
    click_model = train_click_model(df, bot_set, human_set)
    if click_model:
        path = os.path.join(args.output, 'click_model.pkl')
        pickle.dump(click_model, open(path, 'wb'))
        print(f"  Saved to {path}")

    print(f"\nDone. Next step: python3 classify.py --input-dir blasts/ --profiles {args.profiles} --models {args.output}")


if __name__ == '__main__':
    main()
