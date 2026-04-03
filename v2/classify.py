#!/usr/bin/env python3
"""
Bot-B-Gone V2 Classifier — Production runner.

Applies the four-layer cascade to classify opens and clicks as human or bot.

Usage:
  python3 classify.py --blast events.csv --profiles profiles/ --models models/
  python3 classify.py --input-dir blasts/ --profiles profiles/ --models models/ --output results.csv

Prerequisites:
  1. Run build_profiles.py to create subscriber profiles
  2. Run train_models.py to train ML models
  3. (Optional) Place click_grid.json in reference_data/ for empirical prior
"""

import os
import gc
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Load reference data ──

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default thresholds (override by editing reference_data/thresholds.json)
THRESHOLDS = {
    'mpp_devices': ['Apple Mail'],
    'human_peak_hours': [7, 8, 9, 10, 11, 12, 17, 18, 19, 20],
    'scanner_instant_sec': 10,
    'open_ml_bot_threshold': 0.5,
    'instant_click_sec': 5.0,
    'machinegun_interval_sec': 2.0,
    'dead_zone_sec': 300,
    'dead_zone_interval_sec': 10,
    'late_click_sec': 3600,
    'social_bot_fraction': 0.80,
    'selective_click_rate': 0.10,
    'uncertain_combined_threshold': 0.40,
    'bot_devices': ['Internet Explorer', 'Other', 'Windows Tablet'],
    'social_domains': ['linkedin.com', 'twitter.com', 'x.com', 'instagram.com',
                       'youtube.com', 'facebook.com', 'tiktok.com'],
}

# Try loading custom thresholds
_th_path = os.path.join(SCRIPT_DIR, 'reference_data', 'thresholds.json')
if os.path.exists(_th_path):
    with open(_th_path) as f:
        custom = json.load(f)
    if 'open_cascade' in custom:
        for k, v in custom['open_cascade'].items():
            THRESHOLDS[k] = v
    if 'click_cascade' in custom:
        for k, v in custom['click_cascade'].items():
            THRESHOLDS[k] = v
    if 'bot_devices' in custom:
        THRESHOLDS['bot_devices'] = custom['bot_devices']
    if 'social_domains' in custom:
        THRESHOLDS['social_domains'] = custom['social_domains']

# Load click grid
_grid_path = os.path.join(SCRIPT_DIR, 'reference_data', 'click_grid.json')
CLICK_GRID = {}
if os.path.exists(_grid_path):
    with open(_grid_path) as f:
        CLICK_GRID = json.load(f)


def load_model_data(profiles_dir, models_dir, community_file=None):
    """Load all required data for classification."""
    data = {}

    # Subscriber classifications
    class_path = os.path.join(profiles_dir, 'subscriber_classifications.csv')
    if os.path.exists(class_path):
        subs = pd.read_csv(class_path, index_col=0)
        data['bot_set'] = set(subs[subs['classification'] == 'Bot'].index)
        data['scanner_set'] = set(subs[subs['classification'] == 'Scanner-Affected'].index)
        data['human_set'] = set(subs[subs['classification'] == 'Human'].index)
    else:
        data['bot_set'] = data['scanner_set'] = data['human_set'] = set()

    # Community verified humans
    data['community_set'] = set()
    if community_file and os.path.exists(community_file):
        comm = pd.read_csv(community_file)
        id_col = [c for c in comm.columns if 'id' in c.lower()][0] if comm.columns.size > 0 else comm.columns[0]
        data['community_set'] = set(comm[id_col].dropna())

    # Click profiles
    cp_path = os.path.join(profiles_dir, 'click_profiles.json')
    data['click_profiles'] = json.load(open(cp_path)) if os.path.exists(cp_path) else {}

    # Social signal
    ss_path = os.path.join(profiles_dir, 'social_signal.json')
    data['social_signal'] = json.load(open(ss_path)) if os.path.exists(ss_path) else {}

    # ML models
    om_path = os.path.join(models_dir, 'open_model.pkl')
    data['open_model'] = pickle.load(open(om_path, 'rb')) if os.path.exists(om_path) else None

    cm_path = os.path.join(models_dir, 'click_model.pkl')
    data['click_model'] = pickle.load(open(cm_path, 'rb')) if os.path.exists(cm_path) else None

    return data


# ── Open classification ──

def classify_opens(opens_df, data, subscriber_col='subscriber_id'):
    """V2 four-layer open cascade. Returns boolean array (True = human)."""
    n = len(opens_df)
    pids = opens_df[subscriber_col].values

    is_community = pd.Series(pids).isin(data['community_set']).values
    is_human = pd.Series(pids).isin(data['human_set'] | data['scanner_set']).values
    is_bot = pd.Series(pids).isin(data['bot_set']).values

    sas = (opens_df['open_timestamp'] - opens_df['send_timestamp']).dt.total_seconds().clip(0, 2592000).values
    dev = opens_df['device'].fillna('Unknown').values if 'device' in opens_df.columns else np.full(n, 'Unknown')
    had_click = opens_df['click_timestamp'].notna().astype(int).values
    hour = opens_df['open_timestamp'].dt.hour.values

    # Layer 1: MPP stripping
    is_apple = np.array([d in THRESHOLDS['mpp_devices'] for d in dev])
    is_peak = np.isin(hour, THRESHOLDS['human_peak_hours'])
    mpp_confirmed = is_apple & ((had_click == 1) | (is_human & is_peak))
    mpp_bot = is_apple & ~mpp_confirmed

    # Layer 2: Scanner detection
    scanner_bot = (sas < THRESHOLDS['scanner_instant_sec']) & is_bot & (had_click == 0) & ~mpp_bot

    # Layer 3: ML model
    if data['open_model'] is not None:
        X = _build_open_features(opens_df, sas, dev, had_click, hour)
        prob = data['open_model'].predict_proba(X)[:, 1]
        model_human = prob < THRESHOLDS['open_ml_bot_threshold']
    else:
        model_human = np.ones(n, dtype=bool)

    decided = mpp_bot | mpp_confirmed | scanner_bot
    ml_human = ~decided & (is_human | (is_bot & model_human) | (~is_human & ~is_bot & model_human))

    # Layer 4: Community override
    return is_community | mpp_confirmed | ml_human


def _build_open_features(opens_df, sas, dev, had_click, hour):
    n = len(opens_df)
    bd = THRESHOLDS['bot_devices']
    X = pd.DataFrame({
        'seconds_after_send': sas,
        'device_is_apple_mail': np.array([d == 'Apple Mail' for d in dev]).astype(int),
        'device_is_gmail': np.array([d == 'Gmail' for d in dev]).astype(int),
        'device_is_bot': np.array([d in bd for d in dev]).astype(int),
        'device_is_iphone': np.array([d == 'iPhone' for d in dev]).astype(int),
        'device_is_android': np.array([d == 'Android' for d in dev]).astype(int),
        'had_click': had_click,
        'hour_of_open': hour,
        'is_weekday': opens_df['open_timestamp'].dt.dayofweek.lt(5).astype(int).values,
        'click_within_5s_of_open': (
            (opens_df['click_timestamp'] - opens_df['open_timestamp']).dt.total_seconds().between(0, 5)
        ).fillna(False).astype(int).values,
    })
    return X.fillna(0)


# ── Click classification ──

def classify_clicks(clicks_df, data, subscriber_col='subscriber_id'):
    """V2 click cascade. Returns boolean array (True = human)."""
    df = clicks_df.copy()
    T = THRESHOLDS
    sd = T['social_domains']

    sas = (df['click_timestamp'] - df['send_timestamp']).dt.total_seconds().clip(0, 2592000)
    df['_sas'] = sas
    is_phantom = df['open_timestamp'].isna()

    pids = df[subscriber_col]
    is_bot = pids.isin(data['bot_set']).values
    is_community = pids.isin(data['community_set']).values
    social_fracs = np.array([data['social_signal'].get(str(p), {}).get('social_fraction', 0.0) for p in pids])
    is_social_bot = social_fracs >= T['social_bot_fraction']
    click_rates = np.array([data['click_profiles'].get(str(p), {}).get('click_rate', 0.5) for p in pids])
    is_selective = click_rates < T['selective_click_rate']

    # URL features
    actual_clicks = np.ones(len(df))
    has_social = np.zeros(len(df), dtype=bool)
    social_only = np.zeros(len(df), dtype=bool)
    url_types = np.ones(len(df))
    click_span = np.zeros(len(df))

    if 'click_urls' in df.columns:
        for i, row in df.iterrows():
            urls = [u for u in str(row.get('click_urls', '')).split(' ') if u.startswith('http')]
            types = set()
            for u in urls:
                ul = u.lower()
                if any(d in ul for d in sd): types.add('social')
                elif 'unsubscribe' in ul or 'manage' in ul: types.add('unsub')
                else: types.add('content')
            idx = df.index.get_loc(i)
            actual_clicks[idx] = max(len(urls), 1)
            has_social[idx] = 'social' in types
            social_only[idx] = types == {'social'} if types else False
            url_types[idx] = len(types)

    sas_arr = sas.values

    # Step 1: Known human
    known_human = (
        is_community |
        ((actual_clicks == 1) & ~has_social & ~is_phantom.values & ~is_social_bot & (sas_arr > 60)) |
        ((sas_arr > 15) & ~is_phantom.values & ~is_social_bot & ~is_bot & ~social_only) |
        (is_bot & is_selective & (actual_clicks == 1) & ~has_social & ~is_phantom.values & (sas_arr > 60)) |
        ((sas_arr > 3600) & ~is_phantom.values & ~is_social_bot) |
        ((actual_clicks >= 2) & (click_span > 1800) & ~social_only)
    )

    # Step 2: Known bot
    known_bot = (
        ((actual_clicks >= 3) & has_social & (click_span < 60) & (click_span > 0)) |
        ((url_types >= 4) & (click_span < 300) & (click_span > 0)) |
        (social_only & (actual_clicks >= 1)) |
        is_social_bot |
        (is_phantom.values & (sas_arr < 300)) |
        (sas_arr < T['instant_click_sec'])
    ) & ~known_human

    # Step 3: Uncertain — ML + grid
    uncertain = ~known_human & ~known_bot
    if data['click_model'] is not None:
        X = _build_click_features(df, sas)
        ml_prob = data['click_model'].predict_proba(X)[:, 1]
    else:
        ml_prob = np.full(len(df), 0.5)

    grid_probs = np.array([_grid_prob(s, n) for s, n in zip(sas_arr, actual_clicks)])
    combined = np.where(is_bot, 0.3 * ml_prob + 0.7 * grid_probs, 0.5 * ml_prob + 0.5 * grid_probs)
    uncertain_human = uncertain & (combined < T['uncertain_combined_threshold'])

    return known_human | uncertain_human


def _build_click_features(df, sas):
    bd = THRESHOLDS['bot_devices']
    dev = df['device'].fillna('Unknown') if 'device' in df.columns else pd.Series('Unknown', index=df.index)
    otc = (df['click_timestamp'] - df['open_timestamp']).dt.total_seconds()
    return pd.DataFrame({
        'seconds_after_send': sas,
        'inter_click_interval': 0,
        'total_clicks_in_blast': 1,
        'device_is_bot': dev.isin(bd).astype(int),
        'device_is_apple': dev.isin(['Apple Mail', 'iPhone', 'iPad']).astype(int),
        'device_is_mobile': dev.isin(['iPhone', 'Android', 'iPad']).astype(int),
        'had_open': df['open_timestamp'].notna().astype(int),
        'is_weekday': df['click_timestamp'].dt.dayofweek.lt(5).astype(int),
        'hour_of_click': df['click_timestamp'].dt.hour,
        'seconds_open_to_click': otc.fillna(-1).clip(-1, 2592000),
        'click_within_5min': (sas <= 300).astype(int),
        'click_within_10s': (sas <= 10).astype(int),
        'multi_click': 0,
        'phantom_click': df['open_timestamp'].isna().astype(int),
    }).fillna(0)


def _grid_prob(sas_val, n_clicks):
    cb = 1 if n_clicks == 1 else 2 if n_clicks == 2 else 3 if n_clicks == 3 else 5 if n_clicks <= 5 else 10 if n_clicks <= 10 else 99
    for t in [5, 10, 15, 30, 45, 60, 120, 300, 600, 1800, 3600, 7200, 14400, 43200, 86400, 999999]:
        if sas_val < t:
            return CLICK_GRID.get(f'{cb}_{t}', {}).get('bot_prob', 0.5)
    return 0.5


# ── Process blast ──

def process_blast(df, data, subscriber_col='subscriber_id'):
    """Process a single blast DataFrame."""
    for col in ['send_timestamp', 'open_timestamp', 'click_timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    n_sends = len(df)
    has_open = df['open_timestamp'].notna()
    has_click = df['click_timestamp'].notna()

    unique_opens = 0
    if has_open.sum() > 0:
        unique_opens = int(classify_opens(df[has_open].copy(), data, subscriber_col).sum())

    human_clicks = 0
    if has_click.sum() > 0:
        human_clicks = int(classify_clicks(df[has_click].copy(), data, subscriber_col).sum())

    gc.collect()
    return {
        'sends': n_sends,
        'raw_opens': int(has_open.sum()),
        'unique_human_opens': unique_opens,
        'raw_clicks': int(has_click.sum()),
        'human_clicks': human_clicks,
        'clean_open_rate': round(unique_opens / max(n_sends, 1) * 100, 2),
        'clean_click_rate': round(human_clicks / max(n_sends, 1) * 100, 2),
        'bot_open_pct': round((1 - unique_opens / max(has_open.sum(), 1)) * 100, 1),
        'bot_click_pct': round((1 - human_clicks / max(has_click.sum(), 1)) * 100, 1),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Bot-B-Gone V2 Classifier')
    parser.add_argument('--blast', help='Single blast CSV')
    parser.add_argument('--input-dir', help='Directory of blast CSVs')
    parser.add_argument('--profiles', default='profiles', help='Profiles directory')
    parser.add_argument('--models', default='models', help='Models directory')
    parser.add_argument('--community', help='Community verified humans CSV (optional)')
    parser.add_argument('--output', default='bbg_results.csv')
    parser.add_argument('--subscriber-col', default='subscriber_id')
    args = parser.parse_args()

    print("Loading model data...")
    data = load_model_data(args.profiles, args.models, args.community)
    print(f"  Bots: {len(data['bot_set']):,}, Humans: {len(data['human_set']):,}, Community: {len(data['community_set']):,}")

    files = []
    if args.blast:
        files = [args.blast]
    elif args.input_dir:
        files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.csv')])

    print(f"Processing {len(files)} blasts...")
    results = []
    for i, fpath in enumerate(files):
        try:
            df = pd.read_csv(fpath, engine='python', on_bad_lines='skip')
            if len(df) == 0:
                continue
            result = process_blast(df, data, args.subscriber_col)
            result['blast_id'] = os.path.basename(fpath).replace('.csv', '')
            results.append(result)
        except Exception as e:
            print(f"  ERROR {fpath}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)
    print(f"\nDone. {len(results)} blasts → {args.output}")

    if len(results) > 0:
        ts = out['sends'].sum()
        uo = out['unique_human_opens'].sum()
        hc = out['human_clicks'].sum()
        print(f"  Open rate: {uo / ts * 100:.1f}% (bot: {(1 - uo / out['raw_opens'].sum()) * 100:.0f}%)")
        print(f"  Click rate: {hc / ts * 100:.2f}% (bot: {(1 - hc / out['raw_clicks'].sum()) * 100:.0f}%)")


if __name__ == '__main__':
    main()
