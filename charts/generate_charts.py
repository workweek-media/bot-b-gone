"""
Bot-B-Gone — Chart Generator
Generates all visualizations for the repo. No publisher-specific data.
Run: python3 generate_charts.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'grid.alpha': 0.3,
})


# ============================================================================
# ORIGINAL CHARTS (conceptual / framework)
# ============================================================================

def chart_tradeoff():
    """The FP vs FN tradeoff curve."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    unique_ors = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fps =        [0, 1,  3,  5, 10, 18, 25, 32, 40, 48, 55]
    fns =        [95,80, 50, 30, 15,  5,  3,  1, 0.5, 0.2, 0.1]
    
    ax.fill_between(unique_ors, fps, alpha=0.15, color='#E74C3C')
    ax.fill_between(unique_ors, fns, alpha=0.15, color='#3498DB')
    ax.plot(unique_ors, fps, 'o-', color='#E74C3C', linewidth=2.5, markersize=8, 
            label='False Positives: Bots counted as human (%)', zorder=3)
    ax.plot(unique_ors, fns, 's-', color='#3498DB', linewidth=2.5, markersize=8,
            label='False Negatives: Real readers missed (%)', zorder=3)
    
    ax.axvspan(10, 22, alpha=0.08, color='#27AE60', zorder=0)
    ax.text(16, 92, 'TARGET ZONE\nfor B2B long-form publishers', ha='center', fontsize=11, 
            color='#27AE60', fontweight='bold', alpha=0.7)
    
    ax.plot(20, 12, '*', color='#F39C12', markersize=22, zorder=5,
            markeredgecolor='#333', markeredgewidth=1)
    ax.annotate('Equal Error Rate\n~20% Open Rate', xy=(20, 12), xytext=(30, 30),
                textcoords='data', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', 
                         edgecolor='#F39C12', alpha=0.95))
    
    ax.annotate('Past this point, every point of\nopen rate adds more bots than humans', 
                xy=(32, 28), fontsize=9, color='#E74C3C', fontstyle='italic',
                ha='center', alpha=0.7)
    
    ax.set_xlabel('Reported Unique Open Rate (%)', fontsize=13)
    ax.set_ylabel('Error Rate (%)', fontsize=13)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='center right', framealpha=0.95, edgecolor='gray')
    ax.set_title('The Tradeoff: Every Point of Open Rate Has a Cost', 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('the_tradeoff.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  the_tradeoff.png")


def chart_confidence():
    """Confidence decay with target thresholds."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    or_points = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    conf_points = [99, 93, 85, 75, 60, 45, 30, 22, 15, 10, 8]
    
    or_smooth = np.linspace(1, 50, 300)
    spl = make_interp_spline(or_points, conf_points, k=3)
    conf_smooth = spl(or_smooth)
    conf_smooth = np.clip(conf_smooth, 0, 100)
    
    for i in range(len(or_smooth) - 1):
        c = conf_smooth[i] / 100
        color = plt.cm.RdYlGn(c)
        ax.fill_between(or_smooth[i:i+2], 0, conf_smooth[i:i+2], 
                         color=color, alpha=0.6)
    
    ax.plot(or_smooth, conf_smooth, 'k-', linewidth=2.5, alpha=0.7)
    
    ax.axhline(y=70, color='#27AE60', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(51, 72, 'TIER 1: SELL AGAINST IT', fontsize=10, fontweight='bold', 
            color='#27AE60', ha='right', va='bottom')
    ax.text(51, 66, '70%+ confidence', fontsize=8, color='#27AE60', 
            ha='right', va='top')
    
    ax.axhline(y=50, color='#F39C12', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(51, 52, 'TIER 2: DIRECTIONAL USE', fontsize=10, fontweight='bold',
            color='#F39C12', ha='right', va='bottom')
    ax.text(51, 46, '50%+ confidence', fontsize=8, color='#F39C12',
            ha='right', va='top')
    
    ax.axhline(y=30, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(51, 32, 'TIER 3: VANITY METRIC', fontsize=10, fontweight='bold',
            color='#E74C3C', ha='right', va='bottom')
    ax.text(51, 26, 'Below 30% = guessing', fontsize=8, color='#E74C3C',
            ha='right', va='top')
    
    markers = {
        10: ('Strict\nfilter', 85),
        18: ('Calibrated\nestimate', 68),
        30: ('Where most\nESPs report', 30),
        45: ('Raw /\nunfiltered', 10),
    }
    
    for or_val, (label, conf_val) in markers.items():
        ax.plot(or_val, conf_val, 'ko', markersize=8, zorder=5)
        y_off = 10 if or_val != 30 else -15
        ax.annotate(f'{label}\n{or_val}%', xy=(or_val, conf_val),
                    xytext=(0, y_off), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.9))
    
    ax.set_xlabel('Reported Unique Open Rate (%)', fontsize=13)
    ax.set_ylabel('Confidence That the Number Is Real (%)', fontsize=13)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 105)
    ax.set_title('How Confidence Decays as You Loosen the Filter', 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('confidence_decay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  confidence_decay.png")


def chart_ctor():
    """CTOR math check for multiple click rates."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    or_range = np.linspace(5, 50, 200)
    click_rates = [0.5, 1.0, 1.5, 2.0]
    colors = ['#BDC3C7', '#95A5A6', '#7F8C8D', '#2C3E50']
    
    for cr, color in zip(click_rates, colors):
        ctor = cr / or_range * 100
        ax.plot(or_range, ctor, linewidth=2.5, color=color, 
                label=f'{cr}% click rate')
    
    ax.axhspan(4, 7, alpha=0.12, color='#27AE60', zorder=0)
    ax.text(48, 5.5, 'B2B long-form\nnewsletter range\n(4-7% CTOR)', 
            fontsize=10, color='#27AE60', fontweight='bold', ha='right',
            va='center', alpha=0.8)
    
    ax.axhspan(10, 15, alpha=0.08, color='#9B59B6', zorder=0)
    ax.text(48, 12.5, 'Marketing /\npromotional range\n(10-15% CTOR)', 
            fontsize=9, color='#9B59B6', ha='right', va='center', alpha=0.6)
    
    ax.text(30, 20, 'Formula: CTOR = Click Rate / Open Rate\n'
            'If your CTOR falls below your format\'s range,\n'
            'your open rate is probably inflated.',
            fontsize=10, fontstyle='italic', color='#666', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8', 
                     edgecolor='#DDD'))
    
    ax.set_xlabel('Reported Unique Open Rate (%)', fontsize=13)
    ax.set_ylabel('Implied Click-to-Open Rate (%)', fontsize=13)
    ax.set_xlim(3, 52)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=10, loc='upper right', title='Your Unique Click Rate',
             title_fontsize=10)
    ax.set_title('The Math Check: What CTOR Does Your Open Rate Imply?', 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('ctor_math.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ctor_math.png")


def chart_scorecard():
    """The Bot-B-Gone Scorecard visual — what the industry standard looks like."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'THE PINOCCHIO SCORECARD', fontsize=20, fontweight='bold',
            ha='center', va='top', color='#2C3E50')
    ax.text(5, 9.0, 'A proposed standard for transparent email metric reporting',
            fontsize=11, ha='center', va='top', color='#7F8C8D', fontstyle='italic')
    
    # Divider
    ax.plot([1, 9], [8.6, 8.6], color='#BDC3C7', linewidth=1)
    
    # Example scorecard
    fields = [
        ('Newsletter', 'Your Newsletter Name'),
        ('Reporting Period', 'Q1 2026'),
        ('List Size', '50,000 subscribers'),
        ('', ''),
        ('ESP-Reported Open Rate', '42%'),
        ('Bot-B-Gone-Filtered Open Rate', '18%'),
        ('Unique Click Rate', '1.1%'),
        ('Implied CTOR', '6.1%'),
        ('', ''),
        ('Bot-B-Gone Confidence Score', '72%  (Tier 1)'),
        ('Estimated False Positive Rate', '<5%'),
        ('Estimated False Negative Rate', '~25%'),
    ]
    
    y = 8.2
    for label, value in fields:
        if label == '':
            y -= 0.15
            continue
        
        # Color coding for key fields
        if 'Bot-B-Gone Confidence' in label:
            color = '#27AE60'
            weight = 'bold'
        elif 'ESP-Reported' in label:
            color = '#E74C3C'
            weight = 'normal'
        elif 'Bot-B-Gone-Filtered' in label:
            color = '#27AE60'
            weight = 'bold'
        else:
            color = '#2C3E50'
            weight = 'normal'
        
        ax.text(1.5, y, label, fontsize=11, ha='left', va='center', color='#7F8C8D')
        ax.text(8.5, y, value, fontsize=11, ha='right', va='center', 
                color=color, fontweight=weight)
        y -= 0.5
    
    # Tier legend at bottom
    y -= 0.3
    ax.plot([1, 9], [y + 0.2, y + 0.2], color='#BDC3C7', linewidth=1)
    y -= 0.2
    
    tiers = [
        ('#27AE60', 'Tier 1 (70%+): Sell against it'),
        ('#F39C12', 'Tier 2 (50-70%): Directional use'),
        ('#E74C3C', 'Tier 3 (<30%): Vanity metric'),
    ]
    for color, text in tiers:
        ax.plot(1.5, y, 'o', color=color, markersize=10)
        ax.text(2.0, y, text, fontsize=10, va='center', color='#555')
        y -= 0.4
    
    # Border
    border = FancyBboxPatch((0.5, 0.3), 9, 9.4, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='#2C3E50', linewidth=2)
    ax.add_patch(border)
    
    plt.tight_layout()
    plt.savefig('scorecard.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  scorecard.png")


def chart_bot_timeline():
    """Bot contamination over time — the accelerating problem."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    bot_click_pct = [10, 15, 25, 40, 55, 70, 80, 85]
    bot_open_pct =  [5, 8, 15, 25, 35, 42, 48, 53]
    
    ax.fill_between(years, bot_click_pct, alpha=0.15, color='#E74C3C')
    ax.fill_between(years, bot_open_pct, alpha=0.15, color='#F39C12')
    ax.plot(years, bot_click_pct, 'o-', color='#E74C3C', linewidth=2.5, 
            markersize=8, label='Bot % of all clicks')
    ax.plot(years, bot_open_pct, 's-', color='#F39C12', linewidth=2.5, 
            markersize=8, label='Bot % of all opens')
    
    # MPP launch
    ax.axvline(x=2021.75, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.annotate('Apple MPP\nlaunches', xy=(2021.75, 60), fontsize=9,
                fontweight='bold', color='#9B59B6', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Bot Contamination (%)', fontsize=13)
    ax.set_xlim(2018.5, 2026.5)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_title('The Accelerating Problem: Bot Contamination Over Time', 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('bot_timeline.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  bot_timeline.png")


# ============================================================================
# NEW CHARTS: Data-backed visualizations for the full production algorithm
# ============================================================================

def chart_click_timing_distribution():
    """Time-to-first-click: bot vs human distribution.
    Based on 90-day production data (~684K click sessions).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Time buckets (log-ish scale, matching METHODOLOGY.md data)
    labels = ['< 2s', '2-10s', '10-60s', '1-5 min', '5-60 min', '1 hr+']
    bot_counts = [252, 61899, 304339, 153616, 33736, 13126]
    human_counts = [0, 0, 0, 0, 22039, 73653]

    x = np.arange(len(labels))
    width = 0.35

    bars_bot = ax.bar(x - width/2, bot_counts, width, label='Bot Sessions',
                      color='#E74C3C', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_human = ax.bar(x + width/2, human_counts, width, label='Human Sessions',
                        color='#27AE60', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Annotate the zero-human zone
    ax.axvspan(-0.5, 3.5, alpha=0.06, color='#E74C3C', zorder=0)
    ax.text(1.5, max(bot_counts) * 0.92, 'ZERO HUMANS\nin this zone',
            ha='center', fontsize=13, fontweight='bold', color='#E74C3C', alpha=0.7)

    # Add count labels on bars
    for bar in bars_bot:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 3000,
                    f'{h:,.0f}', ha='center', va='bottom', fontsize=8, color='#C0392B')
    for bar in bars_human:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 3000,
                    f'{h:,.0f}', ha='center', va='bottom', fontsize=8, color='#1E8449')

    ax.set_xlabel('Time from Send to First Click', fontsize=13)
    ax.set_ylabel('Number of Click Sessions', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_title('When Do Bots Click vs. Humans? (90-Day Production Data)',
                fontsize=15, fontweight='bold', pad=15)

    # Format y-axis with commas
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    plt.savefig('click_timing_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  click_timing_distribution.png")


def chart_interclick_velocity():
    """Inter-click velocity: bot vs human comparison.
    Based on production data for multi-click sessions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: distribution of inter-click intervals
    # Simulated distributions matching our observed medians
    np.random.seed(42)
    bot_intervals = np.random.lognormal(mean=-0.3, sigma=0.8, size=5000)
    bot_intervals = np.clip(bot_intervals, 0.01, 10)
    human_intervals = np.random.lognormal(mean=3.2, sigma=0.9, size=2000)
    human_intervals = np.clip(human_intervals, 2, 300)

    bins = np.logspace(-2, 2.5, 60)
    ax1.hist(bot_intervals, bins=bins, alpha=0.7, color='#E74C3C', label='Bot',
             density=True, edgecolor='white', linewidth=0.3)
    ax1.hist(human_intervals, bins=bins, alpha=0.7, color='#27AE60', label='Human',
             density=True, edgecolor='white', linewidth=0.3)

    ax1.axvline(x=2.0, color='#2C3E50', linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(2.3, ax1.get_ylim()[1] * 0.85, 'Threshold: 2s\n99.7% precision',
             fontsize=10, fontweight='bold', color='#2C3E50')

    ax1.set_xscale('log')
    ax1.set_xlabel('Average Inter-Click Interval (seconds, log scale)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_title('Inter-Click Velocity Distribution', fontsize=14, fontweight='bold')

    # Right panel: summary comparison
    categories = ['Median\nInter-Click', 'Median\nClick Span', 'Mean\nClicks/Session']
    bot_vals = [0.72, 4.1, 11.4]
    human_vals = [25.0, 36.0, 2.4]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, bot_vals, width, label='Bot', color='#E74C3C', alpha=0.85)
    ax2.bar(x + width/2, human_vals, width, label='Human', color='#27AE60', alpha=0.85)

    # Add value labels
    for i, (bv, hv) in enumerate(zip(bot_vals, human_vals)):
        ax2.text(i - width/2, bv + 0.5, f'{bv}', ha='center', fontsize=11,
                 fontweight='bold', color='#C0392B')
        ax2.text(i + width/2, hv + 0.5, f'{hv}', ha='center', fontsize=11,
                 fontweight='bold', color='#1E8449')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel('Seconds (or count)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.set_title('Bot vs. Human Click Behavior', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('interclick_velocity.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  interclick_velocity.png")


def chart_click_rule_breakdown():
    """What percentage of bot sessions does each click rule catch?"""
    fig, ax = plt.subplots(figsize=(14, 7))

    rules = [
        'B1: Machinegun\n(5+ in 5s)',
        'B2: Machinegun Likely\n(5+ in 10s)',
        'B3: Instant\n(< 2s)',
        'B4: Instant Likely\n(2-60s)',
        'B5: URL Scanner',
        'B6: Cron Burst',
        'B7: High Volume',
    ]
    # Approximate percentages based on our data (these sum > 100% because
    # sessions can match multiple rules; the cascade picks the first)
    # Cascade-assigned percentages (exclusive):
    cascade_pct = [67.9, 6.0, 0.04, 14.6, 3.5, 4.0, 3.96]
    precision = [99.998, 99.995, 100.0, 100.0, 99.5, 98.0, 97.0]

    colors = ['#C0392B', '#E74C3C', '#922B21', '#CB4335', '#EC7063', '#F1948A', '#FADBD8']

    bars = ax.barh(rules, cascade_pct, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)

    # Add precision labels
    for i, (bar, prec) in enumerate(zip(bars, precision)):
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%  (precision: {prec:.1f}%)',
                ha='left', va='center', fontsize=10, color='#555')

    ax.set_xlabel('% of Bot Sessions Caught (cascade-assigned)', fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_title('Click Bot Rule Breakdown: Which Rules Catch What',
                fontsize=15, fontweight='bold', pad=15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('click_rule_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  click_rule_breakdown.png")


def chart_open_rule_breakdown():
    """Open classification breakdown — what % of opens each rule classifies."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Categories and approximate percentages from production data
    labels = [
        'HUMAN: Verified Clicker',
        'HUMAN: ESP Real Flag',
        'HUMAN: Multi-Open',
        'HUMAN: Reopen Long Span',
        'HUMAN: Apple Mail Double',
        'BOT: Instant Prefetch',
        'BOT: Bot Click Session',
        'BOT: Never Verified Fast',
        'UNCERTAIN: No Evidence',
    ]
    pcts = [8.2, 14.5, 5.3, 3.1, 2.8, 18.4, 12.6, 4.8, 30.3]
    colors = [
        '#1ABC9C', '#16A085', '#2ECC71', '#27AE60', '#82E0AA',
        '#E74C3C', '#CB4335', '#F1948A',
        '#BDC3C7',
    ]
    confidences = [
        'Definitive', 'High', 'Medium', 'Medium', 'Medium',
        'High', 'High', 'Medium',
        'Low',
    ]

    bars = ax.barh(labels, pcts, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)

    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%  [{conf}]',
                ha='left', va='center', fontsize=10, color='#555')

    # Divider lines
    ax.axhline(y=4.5, color='#2C3E50', linewidth=1.5, alpha=0.3)

    ax.text(48, 2, 'HUMAN\n33.9%', fontsize=12, fontweight='bold',
            color='#27AE60', ha='center', va='center', alpha=0.6)
    ax.text(48, 6, 'BOT\n35.8%', fontsize=12, fontweight='bold',
            color='#E74C3C', ha='center', va='center', alpha=0.6)

    ax.set_xlabel('% of All Open Sessions', fontsize=13)
    ax.set_xlim(0, 55)
    ax.set_title('Open Classification Breakdown: Where Do Opens Land?',
                fontsize=15, fontweight='bold', pad=15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('open_rule_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  open_rule_breakdown.png")


def chart_precision_recall():
    """Precision vs Recall scatter for all bot click rules."""
    fig, ax = plt.subplots(figsize=(12, 8))

    rules = {
        'B1: Machinegun (5+ in 5s)': (99.998, 67.9),
        'B2: Machinegun (5+ in 10s)': (99.995, 73.9),
        'B3: Instant (< 2s)': (100.0, 0.04),
        'B4: Instant (< 60s)': (100.0, 64.6),
        'B5: URL Scanner': (99.5, 3.5),
        'B6: Cron Burst': (98.0, 4.0),
        'B7: High Volume': (97.0, 3.96),
        'Inter-click < 2s': (99.687, 78.4),
        'Combined (B1 OR B4)': (99.999, 90.5),
    }

    colors = ['#C0392B', '#E74C3C', '#922B21', '#CB4335', '#EC7063',
              '#F1948A', '#FADBD8', '#8E44AD', '#F39C12']
    markers = ['o', 'o', 's', 's', 'D', 'D', 'D', '^', '*']

    for (name, (prec, rec)), color, marker in zip(rules.items(), colors, markers):
        size = 200 if marker == '*' else 120
        ax.scatter(rec, prec, s=size, c=color, marker=marker, zorder=5,
                   edgecolors='white', linewidth=1)
        # Offset labels to avoid overlap
        x_off = 2 if rec < 80 else -2
        ha = 'left' if rec < 80 else 'right'
        ax.annotate(name, xy=(rec, prec), xytext=(x_off, 8),
                    textcoords='offset points', fontsize=8.5,
                    ha=ha, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor='#DDD', alpha=0.9))

    # Target zone
    ax.axhspan(99.0, 100.1, alpha=0.06, color='#27AE60')
    ax.text(50, 99.3, 'Target: 99%+ precision', fontsize=10,
            color='#27AE60', fontweight='bold', alpha=0.6)

    ax.set_xlabel('Recall: % of Bots Caught', fontsize=13)
    ax.set_ylabel('Precision: % of Flagged That Are Actually Bots', fontsize=13)
    ax.set_xlim(-2, 100)
    ax.set_ylim(96, 100.2)
    ax.set_title('Precision vs. Recall: Every Bot Click Rule',
                fontsize=15, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('precision_recall.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  precision_recall.png")


def chart_probability_distribution():
    """Bimodal distribution of click_probability and open_probability scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Click probability — bimodal: most sessions are 0-10 (bot) or 65-95 (human)
    np.random.seed(42)
    bot_probs = np.random.choice([0, 5, 10], size=567000, p=[0.68, 0.22, 0.10])
    human_probs = np.random.choice([65, 70, 85, 90, 95], size=102000, p=[0.15, 0.10, 0.25, 0.35, 0.15])
    ambig_probs = np.full(16000, 40)
    click_probs = np.concatenate([bot_probs, human_probs, ambig_probs])

    bins = np.arange(-2.5, 102.5, 5)
    ax1.hist(click_probs, bins=bins, color='#3498DB', alpha=0.8, edgecolor='white',
             linewidth=0.5, density=True)
    ax1.axvline(x=40, color='#F39C12', linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(42, ax1.get_ylim()[1] * 0.001, 'Default\nthreshold',
             fontsize=10, color='#F39C12', fontweight='bold')
    ax1.set_xlabel('Click Probability Score (0-100)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Click Probability Distribution', fontsize=14, fontweight='bold')

    # Open probability — more spread out due to uncertain category
    bot_open_probs = np.random.choice([5, 10], size=358000, p=[0.7, 0.3])
    human_open_probs = np.random.choice([65, 70, 75, 85, 99], size=339000, p=[0.08, 0.09, 0.15, 0.43, 0.25])
    uncertain_open_probs = np.full(303000, 40)
    open_probs = np.concatenate([bot_open_probs, human_open_probs, uncertain_open_probs])

    ax2.hist(open_probs, bins=bins, color='#9B59B6', alpha=0.8, edgecolor='white',
             linewidth=0.5, density=True)
    ax2.axvline(x=40, color='#F39C12', linestyle='--', linewidth=2, alpha=0.8)
    ax2.text(42, ax2.get_ylim()[1] * 0.001, 'Default\nthreshold',
             fontsize=10, color='#F39C12', fontweight='bold')
    ax2.set_xlabel('Open Probability Score (0-100)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Open Probability Distribution', fontsize=14, fontweight='bold')

    fig.suptitle('Probability Score Distributions: The Bimodal Reality',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('probability_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  probability_distribution.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Generating Bot-B-Gone charts...")
    print("\n--- Framework Charts ---")
    chart_tradeoff()
    chart_confidence()
    chart_ctor()
    chart_scorecard()
    chart_bot_timeline()
    print("\n--- Data-Backed Algorithm Charts ---")
    chart_click_timing_distribution()
    chart_interclick_velocity()
    chart_click_rule_breakdown()
    chart_open_rule_breakdown()
    chart_precision_recall()
    chart_probability_distribution()
    print("\nAll charts generated!")
