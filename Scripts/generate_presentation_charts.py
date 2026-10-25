#!/usr/bin/env python3
"""Generates clean, non-technical, presentation-ready PNG charts
for the Spotify Lyrics project presentation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modern visual styling
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette
COLOR_GREEN = '#1DB954'    # Spotify Green
COLOR_DARK = '#0F172A'     # Slate 900
COLOR_MUTED = '#64748B'    # Slate 500
COLOR_LIGHT = '#F8FAFC'    # Slate 50
COLOR_ACCENT = '#6366F1'   # Indigo
COLOR_RED = '#EF4444'      # Coral Red


def make_slide1_the_myth():
    """Chart 1: The Lyric Myth (Hits vs Non-Hits look identical)."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    metrics = [
        'Word Count\n(Scaled / 10)',
        'Chorus Line\nRepetition (%)',
        'Title Mentions\nin Lyrics',
        'Positive Emotion\nScore (x100)'
    ]
    # Data from Mann-Whitney U test on 587 tracks
    hits = [37.6, 31.8, 7.7, 14.4]
    non_hits = [37.3, 33.7, 7.8, 14.9]

    x = np.arange(len(metrics))
    width = 0.32

    rects1 = ax.bar(x - width/2, hits, width, label='Top 50 Hits', color=COLOR_GREEN, edgecolor='none', zorder=3)
    rects2 = ax.bar(x + width/2, non_hits, width, label='Ranks 51–200', color='#94A3B8', edgecolor='none', zorder=3)

    ax.set_title("The Myth: 'Hit songs have distinct, catchy lyric formulas'", fontsize=14, fontweight='bold', color=COLOR_DARK, pad=18, loc='left')
    ax.text(0.0, 1.02, "Reality: Across 580+ tracks, hit and non-hit lyrics are statistically indistinguishable (all p > 0.10).",
            transform=ax.transAxes, fontsize=10.5, color=COLOR_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color=COLOR_DARK, fontweight='medium')
    ax.set_ylim(0, 50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0', zorder=0)
    ax.set_yticks([])

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=COLOR_DARK)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9.5, color=COLOR_MUTED)

    ax.legend(frameon=False, loc='upper right', fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / "slide1_the_lyric_myth.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out}")


def make_slide2_fandom_wall():
    """Chart 2: The Fandom Wall (Superstars vs Emerging Artists)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    categories = ['Superstars\n(10+ Charting Songs)', 'Emerging / Single-Track Artists\n(1–2 Charting Songs)']
    hit_rates = [61.5, 27.8]
    colors = [COLOR_GREEN, '#94A3B8']

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, hit_rates, height=0.45, color=colors, zorder=3)

    ax.set_title("The 'Fandom Wall': Who Actually Hits the Top 50?", fontsize=14, fontweight='bold', color=COLOR_DARK, pad=18, loc='left')
    ax.text(0.0, 1.02, "Superstars enter Top 50 on brand momentum (61.5%), while emerging artists face a 3x harder barrier (27.8%).",
            transform=ax.transAxes, fontsize=10.5, color=COLOR_MUTED)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11, color=COLOR_DARK, fontweight='medium')
    ax.set_xlim(0, 85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0', zorder=0)
    ax.set_xticks([])

    for bar, rate in zip(bars, hit_rates):
        w = bar.get_width()
        ax.annotate(f'{rate:.1f}% Top 50 Rate', xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(8, 0), textcoords="offset points", ha='left', va='center',
                    fontsize=11, fontweight='bold', color=COLOR_DARK)

    plt.tight_layout()
    out = OUTPUT_DIR / "slide2_fandom_wall.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out}")


def make_slide3_lyric_themes():
    """Chart 3: 2024 Lyric Theme Playbook (What actually works)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    themes = [
        "Raw Emotional Vulnerability\n(Olivia Rodrigo, Gracie Abrams)",
        "Aggressive Trap / Street Anthem\n(Drake, Future)",
        "K-Pop & High-Energy Hooks\n(aespa, BTS, ENHYPEN)",
        "Indie / Acoustic Melancholy\n(Tyler, The Creator, Hozier)",
        "Generic Pop / Dance Fillers\n(WizTheMc, Benson Boone)",
        "Standard Nostalgia / Reflection\n(Taylor Swift, Noah Kahan)",
        "Cliche Club Flirtation\n(turn, sweat, kiss)"
    ]
    rates = [82.6, 62.9, 61.5, 50.0, 47.1, 40.9, 8.3]
    colors = [
        '#10B981',  # Emerald green (winner)
        '#3B82F6',  # Blue
        '#8B5CF6',  # Purple
        '#94A3B8',  # Gray
        '#CBD5E1',  # Light gray
        '#CBD5E1',  # Light gray
        '#F43F5E'   # Rose red (loser)
    ]

    y_pos = np.arange(len(themes))[::-1]
    rates_sorted = rates[::-1]
    themes_sorted = themes[::-1]
    colors_sorted = colors[::-1]

    bars = ax.barh(y_pos, rates_sorted, height=0.55, color=colors_sorted, zorder=3)

    ax.set_title("The 2024 Lyric Playbook: Top 50 Hit Rate by Theme", fontsize=14, fontweight='bold', color=COLOR_DARK, pad=18, loc='left')
    ax.text(0.0, 1.02, "In 2024, emotional vulnerability and sharp genre hooks achieve the highest mainstream breakthrough.",
            transform=ax.transAxes, fontsize=10.5, color=COLOR_MUTED)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(themes_sorted, fontsize=9.5, color=COLOR_DARK, fontweight='medium')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0', zorder=0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color=COLOR_MUTED)

    for bar, rate in zip(bars, rates_sorted):
        w = bar.get_width()
        txt = f' {rate:.1f}%'
        ax.annotate(txt, xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(4, 0), textcoords="offset points", ha='left', va='center',
                    fontsize=9.5, fontweight='bold', color=COLOR_DARK)

    plt.tight_layout()
    out = OUTPUT_DIR / "slide3_lyric_themes.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out}")


if __name__ == "__main__":
    make_slide1_the_myth()
    make_slide2_fandom_wall()
    make_slide3_lyric_themes()
