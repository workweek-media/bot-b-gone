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


if __name__ == '__main__':
    print("Generating Bot-B-Gone charts...")
    chart_tradeoff()
    chart_confidence()
    chart_ctor()
    chart_scorecard()
    chart_bot_timeline()
    print("All charts generated!")
