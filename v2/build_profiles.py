#!/usr/bin/env python3
"""
Build subscriber behavioral profiles from your ESP event data.

This creates three outputs:
  1. subscriber_classifications.csv — GMM clustering (Bot / Human / Scanner-Affected)
  2. click_profiles.json — per-subscriber click rate
  3. social_signal.json — per-subscriber social footer click fraction

Usage:
  python3 build_profiles.py --input events.csv --output profiles/

Input CSV must have columns:
  subscriber_id, blast_id, send_timestamp, open_timestamp, click_timestamp, device

Optional but recommended:
  click_urls (space-separated URLs clicked)
"""

import os
import gc
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore')

SOCIAL_DOMAINS = ['linkedin.com', 'twitter.com', 'x.com', 'instagram.com',
                  'youtube.com', 'facebook.com', 'tiktok.com']


def build_gmm_profiles(df, subscriber_col='subscriber_id'):
    """Build 8 behavioral signals per subscriber and run GMM clustering."""
    print("Building subscriber profiles...")

    sends = df.groupby(subscriber_col).size().rename('n_sends')
    opens = df[df['open_timestamp'].notna()].groupby(subscriber_col).size().rename('n_opens')
    clicks = df[df['click_timestamp'].notna()].groupby(subscriber_col).size().rename('n_clicks')

    profiles = pd.DataFrame(sends).join(opens, how='left').join(clicks, how='left').fillna(0)
    profiles['open_rate'] = (profiles['n_opens'] / profiles['n_sends']).clip(0, 1)
    profiles['click_rate'] = (profiles['n_clicks'] / profiles['n_sends']).clip(0, 1)

    # Timing features
    df['_sas'] = (pd.to_datetime(df['open_timestamp'], errors='coerce') -
                  pd.to_datetime(df['send_timestamp'], errors='coerce')).dt.total_seconds()
    avg_open_time = df[df['_sas'].notna()].groupby(subscriber_col)['_sas'].mean().rename('avg_seconds_to_open')
    profiles = profiles.join(avg_open_time, how='left').fillna(0)

    df['_click_sas'] = (pd.to_datetime(df['click_timestamp'], errors='coerce') -
                        pd.to_datetime(df['send_timestamp'], errors='coerce')).dt.total_seconds()
    avg_click_time = df[df['_click_sas'].notna()].groupby(subscriber_col)['_click_sas'].mean().rename('avg_seconds_to_click')
    profiles = profiles.join(avg_click_time, how='left').fillna(0)

    # Device diversity
    if 'device' in df.columns:
        dev_div = df.groupby(subscriber_col)['device'].nunique().rename('device_diversity')
        profiles = profiles.join(dev_div, how='left').fillna(1)
    else:
        profiles['device_diversity'] = 1

    # MPP open fraction (Apple Mail)
    if 'device' in df.columns:
        apple_opens = df[df['open_timestamp'].notna() & df['device'].str.contains('Apple', case=False, na=False)]
        apple_frac = (apple_opens.groupby(subscriber_col).size() /
                      df[df['open_timestamp'].notna()].groupby(subscriber_col).size()).rename('mpp_open_pct')
        profiles = profiles.join(apple_frac, how='left').fillna(0)
    else:
        profiles['mpp_open_pct'] = 0

    # Placeholder for machinegun/phantom (need per-click detail)
    profiles['machinegun_rate'] = 0
    profiles['phantom_click_rate'] = 0

    # GMM clustering
    feature_cols = ['open_rate', 'click_rate', 'machinegun_rate', 'phantom_click_rate',
                    'mpp_open_pct', 'avg_seconds_to_open', 'avg_seconds_to_click', 'device_diversity']
    X = profiles[feature_cols].fillna(0).values
    X[:, 5] = np.log1p(X[:, 5])
    X[:, 6] = np.log1p(X[:, 6])

    print("Training GMM (3 components)...")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=10)
    clusters = gmm.fit_predict(X)

    # Label clusters by behavior
    cluster_stats = pd.DataFrame(X, columns=feature_cols, index=profiles.index)
    cluster_stats['cluster'] = clusters
    means = cluster_stats.groupby('cluster')[['open_rate', 'click_rate', 'avg_seconds_to_open']].mean()

    # Fastest average open time with high open rate = bot
    bot_cluster = means['avg_seconds_to_open'].idxmin()
    human_cluster = means['avg_seconds_to_open'].idxmax()
    scanner_clusters = [c for c in range(3) if c not in (bot_cluster, human_cluster)]

    labels = []
    for c in clusters:
        if c == bot_cluster:
            labels.append('Bot')
        elif c == human_cluster:
            labels.append('Human')
        else:
            labels.append('Scanner-Affected')

    profiles['classification'] = labels

    bot_n = sum(1 for l in labels if l == 'Bot')
    human_n = sum(1 for l in labels if l == 'Human')
    scanner_n = sum(1 for l in labels if l == 'Scanner-Affected')
    print(f"  GMM results: {bot_n:,} bots, {human_n:,} humans, {scanner_n:,} scanner-affected")

    return profiles


def build_click_profiles(df, subscriber_col='subscriber_id'):
    """Compute per-subscriber historical click rate."""
    print("Building click profiles...")
    sends = df.groupby(subscriber_col)['blast_id'].nunique()
    clicks = df[df['click_timestamp'].notna()].groupby(subscriber_col)['blast_id'].nunique()
    rates = (clicks / sends).fillna(0)

    profiles = {}
    for sub_id, rate in rates.items():
        n = sends.get(sub_id, 0)
        if n >= 3:
            profiles[str(sub_id)] = {
                'click_rate': round(float(rate), 4),
                'n_blasts': int(n)
            }
    print(f"  {len(profiles):,} subscribers with 3+ blasts")
    return profiles


def build_social_signal(df, subscriber_col='subscriber_id', url_col='click_urls'):
    """Compute per-subscriber social footer click fraction."""
    if url_col not in df.columns:
        print("  No click URL column found — skipping social signal")
        return {}

    print("Building social signal...")
    clicked = df[df['click_timestamp'].notna() & df[url_col].notna()]

    sub_total = {}
    sub_social = {}

    for _, row in clicked.iterrows():
        sid = str(row[subscriber_col])
        urls = str(row[url_col]).split(' ') if pd.notna(row[url_col]) else []
        sub_total[sid] = sub_total.get(sid, 0) + 1
        if any(any(d in u.lower() for d in SOCIAL_DOMAINS) for u in urls if u.startswith('http')):
            sub_social[sid] = sub_social.get(sid, 0) + 1

    signal = {}
    for sid, total in sub_total.items():
        social = sub_social.get(sid, 0)
        signal[sid] = {
            'total_clicks': total,
            'social_clicks': social,
            'social_fraction': round(social / total, 4)
        }

    social_bots = sum(1 for v in signal.values() if v['social_fraction'] >= 0.80)
    print(f"  {len(signal):,} subscribers with clicks, {social_bots:,} social bots (>80%)")
    return signal


def main():
    parser = argparse.ArgumentParser(description='Build subscriber profiles for bot detection')
    parser.add_argument('--input', required=True, help='Path to events CSV')
    parser.add_argument('--output', default='profiles', help='Output directory')
    parser.add_argument('--subscriber-col', default='subscriber_id', help='Subscriber ID column name')
    parser.add_argument('--url-col', default='click_urls', help='Click URLs column name')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, engine='python', on_bad_lines='skip')
    print(f"  {len(df):,} rows")

    # 1. GMM subscriber classification
    profiles = build_gmm_profiles(df, args.subscriber_col)
    out_path = os.path.join(args.output, 'subscriber_classifications.csv')
    profiles.to_csv(out_path)
    print(f"  Saved to {out_path}")

    # 2. Click profiles
    click_prof = build_click_profiles(df, args.subscriber_col)
    out_path = os.path.join(args.output, 'click_profiles.json')
    json.dump(click_prof, open(out_path, 'w'))
    print(f"  Saved to {out_path}")

    # 3. Social signal
    social = build_social_signal(df, args.subscriber_col, args.url_col)
    out_path = os.path.join(args.output, 'social_signal.json')
    json.dump(social, open(out_path, 'w'))
    print(f"  Saved to {out_path}")

    print("\nDone. Next step: python3 train_models.py --events your_events.csv --profiles profiles/")


if __name__ == '__main__':
    main()
