#!/usr/bin/env python3
"""
Bot-B-Gone V2 Diagnostic — Validate your model is working correctly.

Run after classify.py to check your results against industry benchmarks
and flag potential issues.

Usage:
  python3 diagnose.py --results bbg_results.csv
"""

import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def diagnose(results_path):
    df = pd.read_csv(results_path)
    n = len(df)
    ts = df['sends'].sum()
    uo = df['unique_human_opens'].sum()
    ro = df['raw_opens'].sum()
    hc = df['human_clicks'].sum()
    rc = df['raw_clicks'].sum()

    open_rate = uo / ts * 100
    click_rate = hc / ts * 100
    bot_open_pct = (1 - uo / max(ro, 1)) * 100
    bot_click_pct = (1 - hc / max(rc, 1)) * 100
    ctor = hc / max(uo, 1) * 100

    print("=" * 60)
    print("Bot-B-Gone V2 — Diagnostic Report")
    print("=" * 60)
    print(f"\nBlasts analyzed: {n}")
    print(f"Total sends: {ts:,}")

    print(f"\n{'Metric':<25} {'Value':>10} {'Benchmark':>15} {'Status':>10}")
    print("-" * 65)

    checks = []

    # Open rate
    status = 'PASS' if 18 <= open_rate <= 30 else 'WARN' if 15 <= open_rate <= 35 else 'FAIL'
    checks.append(status)
    print(f"{'Clean open rate':<25} {open_rate:>9.1f}% {'18-30%':>15} {status:>10}")

    # Bot open %
    status = 'PASS' if 40 <= bot_open_pct <= 60 else 'WARN' if 30 <= bot_open_pct <= 70 else 'FAIL'
    checks.append(status)
    print(f"{'Bot open %':<25} {bot_open_pct:>9.0f}% {'40-60%':>15} {status:>10}")

    # CTOR
    status = 'PASS' if 4 <= ctor <= 7 else 'WARN' if 2 <= ctor <= 10 else 'FAIL'
    checks.append(status)
    print(f"{'CTOR':<25} {ctor:>9.1f}% {'4-7%':>15} {status:>10}")

    # Bot click %
    status = 'PASS' if 50 <= bot_click_pct <= 80 else 'WARN' if 40 <= bot_click_pct <= 90 else 'FAIL'
    checks.append(status)
    print(f"{'Bot click %':<25} {bot_click_pct:>9.0f}% {'50-80%':>15} {status:>10}")

    # Click rate
    status = 'PASS' if 0.5 <= click_rate <= 2.5 else 'WARN'
    checks.append(status)
    print(f"{'Clean click rate':<25} {click_rate:>9.2f}% {'0.5-2.5%':>15} {status:>10}")

    # Per-blast variance check
    or_std = df['clean_open_rate'].std()
    status = 'PASS' if or_std < 15 else 'WARN'
    checks.append(status)
    print(f"{'Open rate std dev':<25} {or_std:>9.1f}pp {'<15pp':>15} {status:>10}")

    # Summary
    fails = checks.count('FAIL')
    warns = checks.count('WARN')
    print(f"\n{'='*60}")
    if fails == 0 and warns == 0:
        print("ALL CHECKS PASSED. Your bot detection model is calibrated correctly.")
    elif fails == 0:
        print(f"{warns} WARNING(S). Review the flagged metrics but model is likely working.")
    else:
        print(f"{fails} FAILURE(S), {warns} WARNING(S). See recommendations below.")

    # Recommendations
    if open_rate > 30:
        print(f"\n  HIGH OPEN RATE ({open_rate:.0f}%): You're not filtering enough bot opens.")
        print(f"  → Check that MPP stripping is enabled (Apple Mail opens should be stripped)")
        print(f"  → Lower the peak hours window or open ML threshold")

    if open_rate < 18:
        print(f"\n  LOW OPEN RATE ({open_rate:.0f}%): You may be over-filtering real opens.")
        print(f"  → Check that community override is enabled for verified humans")
        print(f"  → Widen the peak hours window for MPP confirmation")

    if bot_open_pct < 40:
        print(f"\n  LOW BOT OPEN % ({bot_open_pct:.0f}%): Apple MPP opens are probably leaking through.")
        print(f"  → Verify that Apple Mail is in your MPP device list")

    if ctor < 4:
        print(f"\n  LOW CTOR ({ctor:.1f}%): Opens are still inflated relative to clicks.")
        print(f"  → Be more aggressive on open filtering")

    if bot_click_pct < 50:
        print(f"\n  LOW BOT CLICK % ({bot_click_pct:.0f}%): You may not be catching scanner clicks.")
        print(f"  → Check that social footer links are being detected")
        print(f"  → Verify phantom click rule is active")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='bbg_results.csv')
    args = parser.parse_args()
    diagnose(args.results)
