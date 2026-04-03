#!/usr/bin/env python3
"""
Bot-B-Gone V2 Quickstart — See it work in 60 seconds.

Generates synthetic email blast data with realistic bot patterns,
runs the full pipeline, and shows before/after metrics.

Usage:
  python3 quickstart.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(SCRIPT_DIR, 'sample_data')
os.makedirs(SAMPLE_DIR, exist_ok=True)

np.random.seed(42)


def generate_synthetic_blast(blast_id, n_sends=10000, bot_pct=0.55):
    """Generate a realistic synthetic email blast with bot patterns."""
    send_time = datetime(2025, 1, 15, 10, 0, 0)
    rows = []

    n_bots = int(n_sends * bot_pct)
    n_humans = n_sends - n_bots

    # Human subscribers
    for i in range(n_humans):
        sid = f"human_{i:06d}"
        row = {
            'subscriber_id': sid,
            'blast_id': blast_id,
            'send_timestamp': send_time,
            'open_timestamp': None,
            'click_timestamp': None,
            'device': None,
            'click_urls': None,
        }

        # 30% of humans open (true open rate ~30% × 45% = ~13.5% of total)
        if np.random.random() < 0.30:
            # Humans open 1-48 hours after send, peak at 2-6 hours
            delay_hrs = np.random.lognormal(mean=1.5, sigma=0.8)
            delay_hrs = np.clip(delay_hrs, 0.1, 72)
            row['open_timestamp'] = send_time + timedelta(hours=delay_hrs)

            # Device distribution: 55% Apple Mail, 20% Gmail, 15% iPhone, 10% other
            dev_roll = np.random.random()
            if dev_roll < 0.55:
                row['device'] = 'Apple Mail'
            elif dev_roll < 0.75:
                row['device'] = 'Gmail'
            elif dev_roll < 0.90:
                row['device'] = 'iPhone'
            else:
                row['device'] = np.random.choice(['Outlook', 'Android', 'Yahoo Mail'])

            # 15% of openers click (1-2 content links, minutes to hours after open)
            if np.random.random() < 0.15:
                click_delay = np.random.lognormal(mean=4, sigma=1.5)  # seconds
                click_delay = np.clip(click_delay, 30, 7200)
                row['click_timestamp'] = row['open_timestamp'] + timedelta(seconds=click_delay)
                row['click_urls'] = 'https://example.com/article-about-topic'

                # 5% of clickers click 2 links (super fans)
                if np.random.random() < 0.05:
                    row['click_urls'] += ' https://example.com/related-article'

        rows.append(row)

    # Bot subscribers
    for i in range(n_bots):
        sid = f"bot_{i:06d}"
        row = {
            'subscriber_id': sid,
            'blast_id': blast_id,
            'send_timestamp': send_time,
            'open_timestamp': None,
            'click_timestamp': None,
            'device': None,
            'click_urls': None,
        }

        bot_type = np.random.choice(['apple_mpp', 'scanner', 'phantom'], p=[0.6, 0.25, 0.15])

        if bot_type == 'apple_mpp':
            # Apple MPP: opens immediately, no click
            row['open_timestamp'] = send_time + timedelta(seconds=np.random.uniform(0.5, 3))
            row['device'] = 'Apple Mail'

        elif bot_type == 'scanner':
            # Corporate scanner: opens + clicks every link within seconds
            row['open_timestamp'] = send_time + timedelta(seconds=np.random.uniform(1, 30))
            row['click_timestamp'] = row['open_timestamp'] + timedelta(seconds=np.random.uniform(0.5, 5))
            row['device'] = np.random.choice(['Internet Explorer', 'Other', 'Windows Tablet', 'Apple Mail'])
            # Scanner clicks every link including social
            row['click_urls'] = (
                'https://example.com/article-1 '
                'https://example.com/article-2 '
                'https://linkedin.com/company/example '
                'https://twitter.com/example '
                'https://instagram.com/example '
                'https://example.com/unsubscribe'
            )

        elif bot_type == 'phantom':
            # Phantom: clicks without opening (link scanner)
            row['click_timestamp'] = send_time + timedelta(seconds=np.random.uniform(5, 120))
            row['click_urls'] = 'https://linkedin.com/company/example https://twitter.com/example'

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Bot-B-Gone V2 — Quickstart Demo")
    print("=" * 60)

    # Generate 3 synthetic blasts
    print("\n1. Generating synthetic blast data...")
    blasts = []
    for i, blast_id in enumerate(['demo_blast_001', 'demo_blast_002', 'demo_blast_003']):
        df = generate_synthetic_blast(blast_id, n_sends=5000, bot_pct=0.55)
        path = os.path.join(SAMPLE_DIR, f'{blast_id}.csv')
        df.to_csv(path, index=False)
        blasts.append(df)
        print(f"   {blast_id}: {len(df):,} subscribers")

    all_data = pd.concat(blasts, ignore_index=True)

    # Show raw (ESP-reported) metrics
    print("\n2. Raw metrics (what your ESP shows you):")
    raw_opens = all_data['open_timestamp'].notna().sum()
    raw_clicks = all_data['click_timestamp'].notna().sum()
    total = len(all_data)
    print(f"   Open rate:  {raw_opens/total*100:.1f}%  ({raw_opens:,} opens)")
    print(f"   Click rate: {raw_clicks/total*100:.1f}%  ({raw_clicks:,} clicks)")
    print(f"   CTOR:       {raw_clicks/max(raw_opens,1)*100:.1f}%")

    # Run the V2 classifier
    print("\n3. Running Bot-B-Gone V2 classifier...")

    # Quick inline classification (no trained models — rules only)
    for col in ['send_timestamp', 'open_timestamp', 'click_timestamp']:
        all_data[col] = pd.to_datetime(all_data[col], errors='coerce')

    # Classify opens
    opens = all_data[all_data['open_timestamp'].notna()].copy()
    sas = (opens['open_timestamp'] - opens['send_timestamp']).dt.total_seconds()
    dev = opens['device'].fillna('Unknown')
    had_click = opens['click_timestamp'].notna()
    is_apple = dev == 'Apple Mail'
    is_peak = opens['open_timestamp'].dt.hour.isin([7,8,9,10,11,12,17,18,19,20])

    # MPP strip: Apple opens without click and outside peak = bot
    human_opens = ~is_apple | had_click | is_peak
    # Scanner: <10s after send, no click = bot
    human_opens = human_opens & ~((sas < 10) & ~had_click)

    # Classify clicks
    clicks = all_data[all_data['click_timestamp'].notna()].copy()
    click_sas = (clicks['click_timestamp'] - clicks['send_timestamp']).dt.total_seconds()
    click_phantom = clicks['open_timestamp'].isna()
    click_urls = clicks['click_urls'].fillna('')

    has_social = click_urls.str.contains('linkedin|twitter|instagram|facebook|youtube', case=False, na=False)
    social_only = has_social & ~click_urls.str.contains('example.com/article', na=False)
    n_urls = click_urls.str.count('http')

    human_clicks = (
        ~(click_sas < 5) &                          # not instant
        ~(click_phantom & (click_sas < 300)) &       # not early phantom
        ~social_only &                                # not social-only
        ~((n_urls >= 3) & has_social & (click_sas < 60))  # not scanner pattern
    )

    clean_opens = int(human_opens.sum())
    clean_clicks = int(human_clicks.sum())

    print(f"\n4. Bot-B-Gone results:")
    print(f"   {'Metric':<20} {'Raw (ESP)':>12} {'Cleaned':>12} {'Bot %':>10}")
    print(f"   {'-'*56}")
    print(f"   {'Open rate':<20} {raw_opens/total*100:>11.1f}% {clean_opens/total*100:>11.1f}% {(1-clean_opens/max(raw_opens,1))*100:>9.0f}%")
    print(f"   {'Click rate':<20} {raw_clicks/total*100:>11.1f}% {clean_clicks/total*100:>11.2f}% {(1-clean_clicks/max(raw_clicks,1))*100:>9.0f}%")
    print(f"   {'CTOR':<20} {raw_clicks/max(raw_opens,1)*100:>11.1f}% {clean_clicks/max(clean_opens,1)*100:>11.1f}%")

    print(f"\n5. Validation against benchmarks:")
    clean_or = clean_opens / total * 100
    ctor = clean_clicks / max(clean_opens, 1) * 100
    print(f"   Clean open rate: {clean_or:.1f}%  {'PASS (18-30%)' if 18 <= clean_or <= 30 else 'CHECK — outside 18-30% range'}")
    print(f"   CTOR:            {ctor:.1f}%  {'PASS (4-7%)' if 4 <= ctor <= 7 else 'CHECK — outside 4-7% range'}")
    print(f"   Bot open %:      {(1-clean_opens/max(raw_opens,1))*100:.0f}%  {'PASS (40-60%)' if 40 <= (1-clean_opens/max(raw_opens,1))*100 <= 60 else 'CHECK'}")

    print(f"\n{'='*60}")
    print(f"Next steps:")
    print(f"  1. Export your real ESP data (see /docs/ESP_GUIDE.md)")
    print(f"  2. python3 build_profiles.py --input your_events.csv --output profiles/")
    print(f"  3. python3 train_models.py --events your_events.csv --profiles profiles/")
    print(f"  4. python3 classify.py --input-dir blasts/ --profiles profiles/ --models models/")
    print(f"  5. Compare results against BENCHMARKS.md")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
