import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

st.set_page_config(
    page_title="The Lyric Illusion | Spotify Top 200 Audit",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "spotify_data" / "final_dataset_for_ml_lyrics_en.csv"
SUMMARY_FULL_PATH = ROOT / "artifacts" / "m1_bertopic_summary_song_level.json"
SUMMARY_2024_PATH = ROOT / "artifacts" / "m1_bertopic_summary_2024plus_song_level.json"
TOPICS_FULL_PATH = ROOT / "artifacts" / "bertopic_topic_metrics.csv"
TOPICS_2024_PATH = ROOT / "artifacts" / "bertopic_topic_metrics_2024plus.csv"
SONG_TOPICS_PATH = ROOT / "artifacts" / "bertopic_song_topics.csv"

# Clean Topic Display Names (No arbitrary topic numbers)
TOPIC_DISPLAY_ALIASES = {
    -1: "Unclustered / Outliers (Topic -1)",
    0: "Club Grooves & Flirtation",
    1: "Street Rap & Trap Anthems",
    2: "Urban Swagger & Nostalgic R&B",
    3: "High-Energy K-Pop & Performance Hooks",
    4: "Atmospheric Chill & Late-Night Vibes",
    5: "Dance Floor Anthems & Party Pop",
    6: "Acoustic Longing & Soulful Pop",
    7: "Classic Rock & Power Ballads",
    8: "Deep Emotion, Vulnerability & Catharsis",
    9: "Bittersweet Melancholy & Summer Heartbreak",
    10: "Breakups, Distance & Relationship Tension",
    11: "Contemporary R&B, Swag & Collabs",
    12: "Funk, Classic Rhythm & Nostalgic Chants",
    13: "Melodic Romance & Stadium Hooks",
    14: "Folk-Pop Introspection & Acoustic Roots",
    15: "Indie Folk Melancholy & Healing",
    16: "Atmospheric Midnight Romance",
    17: "Dream Pop & Slow-Burn Longing",
    18: "Emotional Diving, Pain & Release",
    19: "Hip-Hop Braggadocio & Hustle",
    20: "Latin Pop, Global Grooves & Summer Heat",
    21: "Raw Acoustic Pain & Folk-Rock Grit",
    22: "Theatrical Soundtracks & Dramatic Ballads",
    23: "Motivational Anthems & Fighter Spirit"
}

def wilson_ci(k, n, confidence=0.95):
    """Calculate 95% Wilson score confidence interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    diff = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - diff), min(1.0, centre + diff)

def _calc_line_repetition(text: str) -> float:
    """Compute genuine line-level repetition rate: 1 - (unique lines / total non-empty lines)."""
    lines = [line.strip().lower() for line in str(text).splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))

@st.cache_data
def load_standard_cohort():
    if not DATA_PATH.exists():
        return None
    raw_df = pd.read_csv(DATA_PATH)
    
    # Exact filtering matching Scripts/m1_bertopic_model.py
    seasonal_title_pattern = (
        r"(?<![a-z0-9])(?:christmas|xmas|sleigh|reindeer|rudolph|mistletoe|"
        r"jingle(?:\s+bells?)?|silent\s+night|winter\s+wonderland|one\s+more\s+sleep|"
        r"last\s+christmas|fairytale\s+of\s+new\s+york|run\s+rudolph|"
        r"santa\s+(?:baby|claus|clause|tell\s+me|is\s+coming\s+to\s+town))(?![a-z0-9])"
    )
    track_type = raw_df["track_type"].fillna("").astype(str).str.strip().str.casefold()
    holiday_type = track_type.isin({"holiday", "event_anthem"})
    holiday_genre = raw_df["genre"].fillna("").astype(str).str.contains(
        r"(?<![a-z0-9])(?:holiday|christmas)(?![a-z0-9])", case=False, regex=True
    )
    seasonal_title = raw_df["track_name"].fillna("").astype(str).str.contains(
        seasonal_title_pattern, case=False, regex=True
    )
    raw_df["_exclude"] = holiday_type | holiday_genre | seasonal_title
    
    agg_rules = {
        "best_rank_this_year": "min",
        "all_time_peak_rank": "min",
        "all_time_weeks_on_chart": "max",
        "total_streams_this_year": "sum",
        "release_year": "first",
        "genre": "first",
        "lyrics": "first",
        "cleaned_lyrics_transformer": "first",
        "word_count": "first",
        "lexical_diversity": "first",
        "lemma_lexical_diversity": "first",
        "_exclude": "any"
    }
    df_dedup = raw_df.groupby(["artist_names", "track_name"], as_index=False).agg(agg_rules)
    df_clean = df_dedup[~df_dedup["_exclude"]].copy().reset_index(drop=True)
    
    # Compute true line-level repetition rate (distinct from lexical_diversity / TTR)
    df_clean["line_repetition_rate"] = df_clean["cleaned_lyrics_transformer"].apply(_calc_line_repetition)
    
    # Compute VADER compound sentiment score
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
    df_clean["vader_compound"] = df_clean["cleaned_lyrics_transformer"].fillna("").apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    
    df_clean["target_hit"] = (df_clean["best_rank_this_year"] <= 50).astype(int)
    df_clean["Hit_Tier"] = df_clean["target_hit"].map({1: "Top 50 Breakthrough", 0: "Ranks 51–200 Charting"})
    
    # Calculate artist track counts
    artist_counts = df_clean["artist_names"].value_counts()
    df_clean["artist_track_count"] = df_clean["artist_names"].map(artist_counts)
    return df_clean

clean_df = load_standard_cohort()

# ==============================================================================
# MAIN PAGE HERO: PROJECT OVERVIEW & EXECUTIVE BRIEFING
# ==============================================================================
st.title("The Lyric Illusion: Deconstructing Chart Position in the Spotify Top 200")
st.caption("An Empirical Data Science Audit on Commercial Music, Confounding Bias, and Feature Value")

with st.container():
    st.markdown("""<div style="background-color: #0F172A; padding: 26px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
<h3 style="color: #38BDF8; margin-top: 0; font-size: 1.3rem;">Project Overview &amp; Executive Briefing</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; font-size: 0.95rem; line-height: 1.65;">
<div>
<p><b>Why We Started This Project (Core Purpose):</b><br>
In the music industry, it is widely believed that a song needs a specific "lyric formula"—shorter verses, a heavily repeated chorus, simple words, or a trendy emotional theme—to become a massive hit. We launched this project to test whether lyrics actually separate songs that break into the <b>Spotify Global Top 50</b> from songs that stay in the lower tiers of the chart (ranks 51–200).</p>
<p><b>Target Audience:</b><br>
Emerging artists, independent songwriters, music producers, and data science readers who want an honest, evidence-based audit of what actually drives chart rankings—and what is merely a myth.</p>
</div>
<div>
<p><b>Dataset Used:</b><br>
One full year of weekly <b>Spotify Global Top 200</b> chart records (2025–2026), carefully filtered to focus on commercial English releases while removing seasonal Christmas carols, one-off global event anthems, and duplicate album editions.</p>
<p><b>Data Science Techniques Used:</b><br>
We evaluated lyrics across three layers: (1) <b>Surface Structure</b> (word length, chorus repetition, and vocabulary richness), (2) <b>Emotional Tone</b> (sentiment analysis), and (3) <b>Deep Meaning</b> (AI language models that group songs by theme), alongside machine learning models that test whether lyrics can predict chart success for unseen artists.</p>
</div>
</div>
<div style="background-color: #1E293B; padding: 22px 24px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #10B981; font-size: 0.95rem; line-height: 1.7;">
<h4 style="margin-top: 0; color: #10B981; font-size: 1.1rem;">Executive Summary: Three Core Insights</h4>
<p><b>1. The "Hit Lyric Formula" Is a Myth Within the Top 200:</b><br>
Once a song is strong enough to enter the Spotify Top 200, its written lyrics do not determine whether it climbs into the Top 50. Top 50 hits and lower-ranking songs have virtually identical lyric lengths, chorus repetition rates, vocabulary complexity, and emotional tone. Even advanced AI language models cannot tell a Top 50 lyric apart from a rank 150 lyric any better than a coin flip.</p>
<p><b>2. Apparent Genre and Theme Advantages Are Market Illusions:</b><br>
At first glance, certain genres or "diary-style confessional" lyric themes look like shortcuts to the Top 50. However, our audit revealed two hidden illusions behind those numbers: (a) <b>Release Timing</b>—older classic songs sit in the lower half of the chart as they slowly fade over time, while brand-new releases sit at the top; and (b) <b>The Superstar Effect</b>—a handful of mega-stars (such as Taylor Swift, Olivia Rodrigo, Ariana Grande, and Drake) flood specific genres and themes with dozens of songs at once. When we control for artist fame and release timing, no lyric theme or genre holds a real advantage.</p>
<p style="margin-bottom: 0;"><b>3. Real Differentiation Happens Outside the Lyric Sheet:</b><br>
Because every song in the Top 200 has already passed a high quality bar to reach the chart (survivorship bias), written text alone no longer separates the biggest hits from the rest. For songs competing at this level, reaching the Top 50 is driven by factors outside the written words—such as vocal delivery, audio production, short-form video virality (TikTok/Reels), playlist placement, and existing fanbase power.</p>
</div>
</div>""", unsafe_allow_html=True)

# Unified clean navigation tabs
tabs = st.tabs([
    "Act 1: The Lyric Surface Myth",
    "Act 2: Market Realities & Model Ablation",
    "Act 3: Lyrical Themes & Benchmark Database",
    "Act 4: Exploratory Patterns & Case Studies",
    "Act 5: Honest Guide for Emerging Artists",
    "Act 6: Limitations & Future Horizons",
    "Act 7: Incidental Data Quality Audit"
])

# ==============================================================================
# TAB 1: THE LYRIC SURFACE MYTH (Interactive Inspector & Robustness Checks)
# ==============================================================================
with tabs[0]:
    st.header("Act 1: The Lyric Surface Myth")
    st.markdown("""
    **Core Question**: Do songs reaching the Top 50 feature higher chorus line repetition, shorter verses, simpler vocabulary, or more extreme emotional polarity than songs peaking in ranks 51–200?  
    **Empirical Result**: Evaluated across 582 charting tracks using non-parametric Mann-Whitney U tests, **none of the four surface linguistic or sentiment metrics show statistically significant separation ($p > 0.20$).**
    """)
    
    st.subheader("Interactive Lyric Attribute Inspector")
    st.markdown("Select a linguistic attribute below to inspect its distribution between **Top 50 Breakthroughs** ($y=1$, $n=228$) and **Ranks 51–200 Charting Tracks** ($y=0$, $n=354$):")
    
    attr_option = st.radio(
        "Choose an attribute to analyze:",
        [
            "Chorus Line Repetition Rate",
            "Word Count (Lyric Length)",
            "Lexical Diversity (Type-Token Ratio)",
            "Sentiment Polarity (VADER Compound)"
        ],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    if attr_option == "Chorus Line Repetition Rate":
        col1.metric("Top 50 Breakthroughs (Median)", "31.8%", help="Median percentage of repeated lyrical lines (1 - unique_lines / total_lines) in Top 50 tracks (n = 228; mean = 33.4%).")
        col2.metric("Ranks 51–200 Charting (Median)", "33.3%", help="Median percentage of repeated lyrical lines in ranks 51–200 tracks (n = 354; mean = 35.1%).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.2154",
            delta="Difference: -1.5 pp (Not Significant)",
            delta_color="off",
            help="Why Mann-Whitney U? Line repetition rates are bounded ratios with skewed tails. Mann-Whitney U compares rank sums rather than assuming a normal bell curve. Note: In the raw CSV, the column named 'repetition_rate' was an alias for Type-Token Ratio; here we compute true line-level chorus repetition (1 - unique_lines / total_lines)."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="line_repetition_rate",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Breakthrough": "#10B981", "Ranks 51–200 Charting": "#64748B"},
                points="outliers",
                labels={"line_repetition_rate": "Chorus Line Repetition Rate (1 - Unique/Total Lines)", "Hit_Tier": "Chart Tier"},
                title="Chorus Line Repetition Rate by Chart Tier (Full Cohort, N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Note on Null Finding: Top 50 tracks actually have a slightly lower median line repetition rate (31.8% vs 33.3%), a difference indistinguishable from random sampling noise ($p = 0.2154$).")

    elif attr_option == "Word Count (Lyric Length)":
        col1.metric("Top 50 Breakthroughs (Median)", "352.5 words", help="Median cleaned word count of Top 50 breakthrough lyrics (n = 228; mean = 376.7 words).")
        col2.metric("Ranks 51–200 Charting (Median)", "346.5 words", help="Median cleaned word count of ranks 51–200 charting lyrics (n = 354; mean = 374.0 words).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.5294",
            delta="Difference: +6.0 words (Not Significant)",
            delta_color="off",
            help="Why Mann-Whitney U? Lyric lengths range from short 80-word dance chants to 900+ word rap verses, creating strong positive skew. Mann-Whitney U tests rank distributions so high-word-count rap outliers do not distort the comparison."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="word_count",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Breakthrough": "#10B981", "Ranks 51–200 Charting": "#64748B"},
                points="outliers",
                labels={"word_count": "Total Cleaned Word Count", "Hit_Tier": "Chart Tier"},
                title="Total Word Count by Chart Tier (Full Cohort, N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Note on Null Finding: Total lyric length differs by only 6 words at the median between tiers ($p = 0.5294$).")

    elif attr_option == "Lexical Diversity (Type-Token Ratio)":
        col1.metric("Top 50 Breakthroughs (Median)", "0.372", help="Unique words divided by total words (Type-Token Ratio) in Top 50 tracks (exact median = 0.3715, mean = 0.3741, n = 228).")
        col2.metric("Ranks 51–200 Charting (Median)", "0.365", help="Unique words divided by total words in ranks 51–200 tracks (exact median = 0.3650, mean = 0.3672, n = 354).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.3255",
            delta="Difference: +0.0065 (Not Significant)",
            delta_color="off",
            help="Why Mann-Whitney U? Type-Token Ratio (unique words / total words) is bounded in [0, 1] with non-normal clustering. Note: Lemma-level TTR after stopword removal also shows no difference (Hit median 0.4385 vs Mid 0.4275, p = 0.2671)."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="lexical_diversity",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Breakthrough": "#10B981", "Ranks 51–200 Charting": "#64748B"},
                points="outliers",
                labels={"lexical_diversity": "Lexical Diversity (Type-Token Ratio)", "Hit_Tier": "Chart Tier"},
                title="Lexical Diversity (TTR) by Chart Tier (Full Cohort, N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Note on Null Finding: Vocabulary richness (TTR) is virtually identical across tiers ($p = 0.3255$; content-word lemma TTR $p = 0.2671$).")

    else:
        col1.metric("Top 50 Breakthroughs (Mean / Med)", "+0.335 / +0.948", help="VADER compound sentiment score in [-1, +1] for Top 50 tracks (mean = +0.3345, median = +0.9481, n = 228).")
        col2.metric("Ranks 51–200 Charting (Mean / Med)", "+0.369 / +0.959", help="VADER compound sentiment score in [-1, +1] for ranks 51–200 tracks (mean = +0.3691, median = +0.9591, n = 354).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.4205",
            delta="Difference: -0.035 mean (Not Significant)",
            delta_color="off",
            help="Why Mann-Whitney U? Cumulative song-level VADER compound scores pile up near -1.0 and +1.0 for full-length lyrics (bimodal skew), violating parametric t-test assumptions. Separate positive (p = 0.5172) and negative (p = 0.8051) token ratios are also null."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="vader_compound",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Breakthrough": "#10B981", "Ranks 51–200 Charting": "#64748B"},
                points="outliers",
                labels={"vader_compound": "VADER Compound Sentiment Score (-1 to +1)", "Hit_Tier": "Chart Tier"},
                title="VADER Sentiment Polarity by Chart Tier (Full Cohort, N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Note on Null Finding: Emotional polarity does not separate Top 50 songs from ranks 51–200 ($p = 0.4205$).")

    st.markdown("---")
    
    # Three-Way Robustness Verification
    st.subheader("Robustness Verification Across Sub-Cohorts & Alternative Target Definitions")
    st.markdown("""
    To ensure our null surface lyric findings are not artifacts of (a) superstar discography dominance, (b) observation-window left-truncation of older catalog releases, or (c) the window-specific target definition, we re-evaluated all four metrics across three sensitivity specifications:
    """)
    
    rob_df = pd.DataFrame({
        "Sensitivity Specification": [
            "1. Full Modeling Cohort (Baseline)",
            "2. Emerging / Low-Presence Artists Only (<= 2 charted tracks)",
            "3. In-Window Releases Only (Released in 2025–2026)",
            "4. All-Time Peak Rank Target (all_time_peak_rank <= 50)"
        ],
        "Sample Size (N)": [
            "N = 582 (228 Top 50 / 354 Mid)",
            "N = 276 (77 Top 50 / 199 Mid)",
            "N = 276 (149 Top 50 / 127 Mid)",
            "N = 582 (320 All-Time Top 50 / 262 Mid)"
        ],
        "Line Repetition (p)": [
            "31.8% vs 33.3% (p = 0.2154)",
            "33.3% vs 35.3% (p = 0.8539)",
            "31.0% vs 28.3% (p = 0.8767)",
            "32.4% vs 33.3% (p = 0.3740)"
        ],
        "Word Count (p)": [
            "352.5 vs 346.5 (p = 0.5294)",
            "371.0 vs 355.0 (p = 0.2412)",
            "353.0 vs 350.0 (p = 0.4907)",
            "352.0 vs 348.5 (p = 0.2590)"
        ],
        "Lexical Diversity TTR (p)": [
            "0.372 vs 0.365 (p = 0.3255)",
            "0.344 vs 0.356 (p = 0.5350)",
            "0.392 vs 0.390 (p = 0.9049)",
            "0.372 vs 0.365 (p = 0.6041)"
        ],
        "VADER Sentiment (p)": [
            "Mean +0.335 vs +0.369 (p = 0.4205)",
            "Mean +0.367 vs +0.377 (p = 0.9752)",
            "Mean +0.326 vs +0.268 (p = 0.4399)",
            "Mean +0.352 vs +0.360 (p = 0.7888)"
        ]
    })
    st.dataframe(rob_df, use_container_width=True)
    
    st.info("""
    **Robustness Takeaway**: Across every specification—removing multi-track superstars ($N=276$), isolating in-window 2025–2026 releases ($N=276$, where release year itself has $p = 0.4836$), or defining the target via all-time historical peak rank ($N=582$)—every surface lyric metric remains statistically null ($p > 0.20$).
    """)

# ==============================================================================
# TAB 2: MARKET REALITIES & MODEL ABLATION
# ==============================================================================
with tabs[1]:
    st.header("Act 2: Market Realities & Model Ablation")
    st.markdown("""
    To test whether dense Transformer representations or thematic clusters add predictive value beyond basic track metadata, we conducted a controlled ablation study under **5-Fold Stratified Group K-Fold Cross-Validation** across 191 artist collaboration clusters.
    """)
    
    st.subheader("Methodological Design & Leak Prevention")
    st.markdown("""
    * **Artist-Cluster Grouped Cross-Validation (191 Connected Components):**  
      Collaborating artists are decomposed into a graph of 191 connected components. Grouping by artist cluster prevents the same artist or collaborating circle from appearing in both training and test splits within any fold, stopping the classifier from memorizing artist idiolects. *(Scope note: Group CV prevents same-artist leakage, though it cannot prevent cross-artist stylistic similarity or shared external songwriting teams.)*
    * **Fold-Internal BERTopic Fitting vs. Direct 384-Dim Embeddings:**  
      Inside the cross-validation loop (`Scripts/m1_bertopic_model.py`), BERTopic (`UMAP` + `HDBSCAN` + `CountVectorizer`) is fit **strictly on each training fold** (`classifier_fits_topic_model_on_training_songs_only = True`, yielding 2 to 22 topics per fold) before transforming the held-out test fold. Because density clustering can lose continuous semantic nuance, we also evaluate **raw 384-dimensional Sentence-BERT (`all-MiniLM-L6-v2`) mean-pooled embeddings** directly in L2-regularized Logistic Regression.
    * **Decomposing the Release Year Artifact (Left-Truncation vs. Catalog Aging):**  
      Our target $y = (\text{best\_rank\_this\_year} \le 50)$ is observed during Sept 2025–Sept 2026. Songs released prior to 2025 are observed on their post-peak decay tail: pre-2024 songs have an all-time Top 50 rate of **45.1%**, which drops to **23.7%** in our window; 2024 releases drop from **75.0% all-time** to **40.0% in-window**; whereas in-window 2025–2026 releases convert at **54.0%**. To prove that `Release Year` alone drives the metadata baseline without changing sample size, we evaluate `Release Year Only` and `Genre Only` directly on the full $N=582$ cohort alongside the $N=316$ (2024+) cohort.
    """)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("1. Standard Full Cohort (N = 582, 191 Clusters)")
        cohort_full = pd.DataFrame({
            "Model Architecture": [
                "Release Year Only (Single Feature)",
                "M1: Metadata Baseline (Release Year + Genre)",
                "M1 + Raw 384-dim SBERT Embeddings",
                "M1 + BERTopic (Fold-Fit Topic Probabilities)",
                "Genre Only (One-Hot Genre)",
                "Raw 384-dim SBERT Embeddings Only",
                "Pure Surface Lyrics (WC + LineRep + TTR + VADER)"
            ],
            "Mean ROC-AUC": [0.6676, 0.6600, 0.6249, 0.6237, 0.6039, 0.5080, 0.4558],
            "Fold Std Dev": ["± 0.0786", "± 0.1008", "± 0.0621", "± 0.0756", "± 0.0638", "± 0.0450", "± 0.0519"],
            "Macro F1": ["0.5925 ± 0.039", "0.5845 ± 0.068", "0.5934 ± 0.048", "0.5861 ± 0.064", "0.5790 ± 0.065", "0.5090 ± 0.023", "0.4560 ± 0.044"]
        })
        fig_full = px.bar(
            cohort_full,
            x="Mean ROC-AUC",
            y="Model Architecture",
            orientation='h',
            color="Mean ROC-AUC",
            color_continuous_scale="Tealgrn",
            text="Mean ROC-AUC"
        )
        fig_full.add_vline(x=0.50, line_dash="dash", line_color="#EF4444")
        fig_full.update_layout(yaxis={'categoryorder': 'total ascending'}, height=380, xaxis_range=[0.30, 0.78])
        st.plotly_chart(fig_full, use_container_width=True)
        st.markdown("""
        * **Release Year Alone (`0.6676 ± 0.0786`) Exceeds M1 (`0.6600 ± 0.1008`)**: Holding sample size fixed at $N=582$, a single feature (`release_year`) accounts for 100% of M1's discrimination.
        * **Direct SBERT (`0.5080`) & Surface Lyrics (`0.4558`)**: Raw 384-dim Transformer embeddings perform at random chance (`0.5080`), and adding either SBERT (`0.6249`) or fold-fit BERTopic (`0.6237`) to M1 provides no predictive gain (mean $\Delta\text{AUC} = -0.0363$; note that a paired $t$-test across 5 correlated CV folds, $p = 0.078$, is underpowered and non-independent, so we conclude there is no reliable evidence of improvement).
        """)

    with colB:
        st.subheader("2. Contemporary Cohort (2024+, N = 316, 92 Clusters)")
        cohort_2024 = pd.DataFrame({
            "Model Architecture": [
                "Genre Only (One-Hot Genre)",
                "Release Year Only (2024 vs 2025 vs 2026)",
                "M1 + Raw 384-dim SBERT Embeddings",
                "M1: Metadata Baseline (Release Year + Genre)",
                "M1 + BERTopic (Fold-Fit Topic Probabilities)",
                "Raw 384-dim SBERT Embeddings Only",
                "Pure Surface Lyrics (WC + LineRep + TTR + VADER)"
            ],
            "Mean ROC-AUC": [0.5667, 0.5424, 0.5289, 0.5239, 0.5088, 0.4878, 0.3411],
            "Fold Std Dev": ["± 0.1346", "± 0.1122", "± 0.1437", "± 0.1087", "± 0.1629", "± 0.1186", "± 0.0561"],
            "Macro F1": ["0.5440 ± 0.116", "0.3430 ± 0.004", "0.5001 ± 0.113", "0.5440 ± 0.116", "0.5234 ± 0.094", "0.4820 ± 0.102", "0.3634 ± 0.036"]
        })
        fig_2024 = px.bar(
            cohort_2024,
            x="Mean ROC-AUC",
            y="Model Architecture",
            orientation='h',
            color="Mean ROC-AUC",
            color_continuous_scale="Purples",
            text="Mean ROC-AUC"
        )
        fig_2024.add_vline(x=0.50, line_dash="dash", line_color="#EF4444")
        fig_2024.update_layout(yaxis={'categoryorder': 'total ascending'}, height=380, xaxis_range=[0.25, 0.78])
        st.plotly_chart(fig_2024, use_container_width=True)
        st.markdown("""
        * **Restricting to 2024–2026 Releases Collapses M1 (`0.5239 ± 0.1087`)**: Note that in our pipeline (`Scripts/m1_bertopic_model.py`), `M1` on the 2024+ cohort includes both `release_year` (2024, 2025, 2026) and `genre_cleaned`.
        * Even within 2024+, residual left-truncation exists between 2024 releases (40.0% window Top 50 rate, $n=40$) and in-window 2025–2026 releases (54.0%, $n=276$), yielding `AUC = 0.5424` for Release Year Only and `0.5667 ± 0.1346` for Genre Only—all within one standard deviation of random guessing (`0.50`).
        """)

    st.markdown("---")
    st.subheader("Statistical Screening Summary: Track-Level vs. Artist-Cluster Level ($N = 191$)")
    st.markdown("""
    **Why Cluster-Level Verification Matters (Direction of Non-Independence Bias):**  
    When prolific artists (e.g., Taylor Swift, Drake, Olivia Rodrigo, Ariana Grande with 10+ tracks) contribute multiple songs, treating tracks as independent observations underestimates standard errors and **inflates Type I error (false positives)**. This bias threatens *statistically significant* track-level associations (`Release Year`, `Superstar Volume`, `Genre`), NOT our null lyrical results. Below, we report both naive track-level tests ($N=582$) and **Artist-Cluster aggregated tests ($N=191$ independent clusters)**, and we consolidate the 31 raw iTunes genres (where 23/31 had expected counts $<5$) into **8 Macro-Genres** where every cell satisfies expected count $\ge 5.48$:
    """)
    
    stat_summary = pd.DataFrame({
        "Feature Tested": [
            "Release Year (Recency / Window Left-Truncation)",
            "Chart Volume Proxy (10+ Tracks in Window)",
            "Consolidated Macro-Genre (8 Groups, all E >= 5.5)",
            "Raw iTunes Genre (31 Categories, 23 with E < 5)",
            "Chorus Line Repetition Rate",
            "Lexical Diversity (Type-Token Ratio)",
            "VADER Sentiment Polarity (Compound)",
            "Total Cleaned Word Count",
            "BERTopic Clusters (24 Topics, 6 with E < 5)"
        ],
        "Naive Track-Level Test (N = 582)": [
            "Mann-Whitney p = 3.47e-11",
            "Chi2 = 51.32, df = 1, p = 7.86e-13",
            "Chi2 = 30.76, df = 7, p = 6.89e-05",
            "Chi2 = 58.30, df = 30, p = 0.0015",
            "Mann-Whitney p = 0.2154",
            "Mann-Whitney p = 0.3255",
            "Mann-Whitney p = 0.4205",
            "Mann-Whitney p = 0.5294",
            "Chi2 = 33.48, df = 23, p = 0.0731 (perm p = 0.0702)"
        ],
        "Artist-Cluster Level Test (N = 191 Clusters)": [
            "Mann-Whitney p = 5.16e-04 (Spearman p = 5.09e-04)",
            "Mann-Whitney p = 2.01e-04 (13 vs 178 clusters)",
            "Chi2 = 11.29, p = 0.1265 | Kruskal-Wallis p = 0.2592",
            "Not valid (45/62 cells have expected count < 5)",
            "Kruskal / Mann-Whitney p > 0.30 (Null holds)",
            "Kruskal / Mann-Whitney p > 0.30 (Null holds)",
            "Kruskal / Mann-Whitney p > 0.40 (Null holds)",
            "Kruskal / Mann-Whitney p > 0.25 (Null holds)",
            "Kruskal-Wallis H = 17.42, p = 0.7399 (N = 131 clusters)"
        ],
        "Methodological Verdict & Interpretation": [
            "Robust across clusters, but heavily driven by left-truncation of pre-2025 releases in our 2025–2026 observation window.",
            "Endogenous observational proxy: defined on the same chart window; reflects superstar catalog flood (62.1% vs 29.8% track hit rate).",
            "LOSES SIGNIFICANCE at artist-cluster level (p = 0.1265)! Track-level significance was inflated by multi-track Pop/K-Pop superstars.",
            "Violates Chi-Square expected frequency assumptions (23/31 genres have E < 5); must be consolidated into Macro-Genres.",
            "Robust Null: No difference in line-level repetition across tiers.",
            "Robust Null: No difference in vocabulary richness (lemma TTR p = 0.2671).",
            "Robust Null: No difference in compound, positive, or negative sentiment.",
            "Robust Null: No difference in lyric length across tiers.",
            "Not Significant: Marginal track-level p = 0.0731 collapses to p = 0.7399 when accounting for artist clusters."
        ]
    })
    st.dataframe(stat_summary, use_container_width=True)

# ==============================================================================
# TAB 3: LYRICAL THEMES & BENCHMARK DATABASE
# ==============================================================================
with tabs[2]:
    st.header("Act 3: Lyrical Themes & Benchmark Database")
    st.markdown("""
    While lyrical themes do not predict chart rank in supervised classification, fitting **BERTopic** on the full 582-track corpus segments songs into descriptive thematic clusters: **459 tracks (78.9%)** map into **24 non-outlier topics**, while **123 tracks (21.1%)** are assigned by HDBSCAN to **`Topic -1` (unclustered outliers**, which converted to the Top 50 at **34.1%**, 42/123 tracks).
    """)
    
    if TOPICS_FULL_PATH.exists():
        topics_df = pd.read_csv(TOPICS_FULL_PATH)
        topics_df["Thematic Name"] = topics_df["topic_id"].map(TOPIC_DISPLAY_ALIASES).fillna("Other / Uncategorized")
        valid_topics = topics_df[topics_df["topic_id"] >= 0].copy()
        
        # Calculate Wilson CIs
        ci_minus, ci_plus = [], []
        for idx, r in valid_topics.iterrows():
            low, high = wilson_ci(r["top_50_count"], r["song_count"])
            rate = r["top_50_rate"]
            ci_minus.append((rate - low) * 100)
            ci_plus.append((high - rate) * 100)
            
        valid_topics["top_50_pct"] = valid_topics["top_50_rate"] * 100
        valid_topics["ci_minus"] = ci_minus
        valid_topics["ci_plus"] = ci_plus
        valid_topics = valid_topics.sort_values(by="top_50_pct", ascending=True)
        valid_topics["label_with_n"] = valid_topics.apply(lambda r: f"{r['Thematic Name']} (n={r['song_count']})", axis=1)
        valid_topics["pct_label"] = valid_topics["top_50_pct"].apply(lambda x: f" {x:.1f}%")
        
        st.subheader("Thematic Breakthrough Rates with 95% Wilson Confidence Intervals (24 Clustered Topics, n = 459)")
        st.markdown("Horizontal bar chart displaying observed Top 50 rates (%), cluster sample sizes ($n$), and unadjusted 95% Wilson score confidence intervals (excluding the $n=123$ unclustered `Topic -1` tracks):")
        
        fig_topics = go.Figure()
        fig_topics.add_trace(go.Bar(
            x=valid_topics["top_50_pct"],
            y=valid_topics["label_with_n"],
            orientation="h",
            marker=dict(color=valid_topics["top_50_pct"], colorscale="Viridis"),
            text=valid_topics["pct_label"],
            textposition="outside",
            textfont=dict(size=12, color="#F8FAFC"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=valid_topics["ci_plus"],
                arrayminus=valid_topics["ci_minus"],
                color="#94A3B8",
                thickness=1.5,
                width=3
            )
        ))
        
        # Add baseline reference line at overall Top 50 rate (39.2%)
        fig_topics.add_vline(
            x=39.2, line_dash="dash", line_color="#EF4444",
            annotation_text="Full Cohort Baseline: 39.2% (Clustered Mean: 40.5%)", annotation_position="top right"
        )
        
        fig_topics.update_layout(
            height=950,
            margin=dict(l=330, r=60, t=50, b=50),
            yaxis=dict(dtick=1, tickfont=dict(size=13, color="#F8FAFC")),
            xaxis=dict(title="Top 50 Breakthrough Rate (%) with Unadjusted 95% Wilson CI", tickfont=dict(size=12, color="#F8FAFC"), range=[0, 88])
        )
        st.plotly_chart(fig_topics, use_container_width=True)
        
        st.warning("""
        **Statistical Reality Check on Thematic Differences**:  
        * **Global Contingency & Permutation Tests Are Null ($p > 0.05$)**: Across the 24 clustered topics ($n = 459, df = 23$), Pearson $\chi^2 = 33.48$ yields asymptotic $p = 0.0731$ and Monte Carlo permutation $p = 0.0702$ (10,000 permutations). Including the $n = 123$ unclustered `Topic -1` tracks ($N = 582, df = 24$) yields $\chi^2 = 35.52$, asymptotic $p = 0.0611$, and permutation $p = 0.0581$. When aggregated to the artist-cluster level ($N = 131$ clusters), Kruskal-Wallis $H = 17.42, p = 0.7399$.
        * **Sparse Cells & Multiple Testing**: Exactly **6 of the 24 topics** (those with $n \le 12$) have expected Top 50 cell counts $< 5$. While 2 of the 24 unadjusted 95% Wilson intervals—*High-Energy K-Pop & Performance Hooks* (`64.3%`, CI `[45.8%, 79.3%]`) and *Club Grooves & Flirtation* (`22.6%`, CI `[13.5%, 35.5%]`)—barely exclude the 39.2% baseline, testing 24 intervals simultaneously at $\alpha = 0.05$ expects $24 \times 0.05 = 1.2$ false positives by chance alone (none survive Bonferroni correction at $\alpha = 0.0021$). Moreover, the K-Pop cluster reflects a genre/fandom release effect rather than a pure lyric theme.
        """)

    st.markdown("---")
    
    # Clarified Purpose of the Database
    st.subheader("Explore Songs in the Dataset: Benchmark Repository (All 582 Tracks)")
    st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 10px; border-left: 4px solid #38BDF8; margin-bottom: 20px;">
<h4 style="margin-top: 0; color: #38BDF8;">Purpose &amp; Guidance for Using the Database</h4>
<p>This is an <b>Empirical Benchmark Repository</b> covering all 582 modeling tracks (459 clustered + 123 Topic -1 outliers):</p>
<ol style="margin-bottom: 0;">
<li><b>Inspect Peer Benchmarks:</b> Search for specific artists (e.g., <i>Olivia Rodrigo, Drake, Noah Kahan, aespa</i>) to see which unsupervised cluster their lyrics mapped to.</li>
<li><b>Verify Target Neutrality:</b> Filter by any theme (e.g., <i>Deep Emotion, Vulnerability &amp; Catharsis</i>, <i>n</i> = 18, 61.1% Top 50 rate in the full cohort) to observe that both high-peaking and lower-charting songs share similar vocabularies.</li>
<li><b>Understand Artist Clustering in Rap Themes:</b> While the consolidated <i>Hip-Hop/Rap</i> macro-genre had a 37.1% Top 50 rate (26/70 tracks), specific Rap lyric clusters (<i>Street Rap &amp; Trap Anthems</i>: 44.4%; <i>Hip-Hop Braggadocio &amp; Hustle</i>: 54.5%, <i>n</i> = 11) show higher rates because prolific superstars like Drake (<i>n</i> = 9 Top 50 tracks) concentrate heavily inside those specific clusters.</li>
</ol>
</div>""", unsafe_allow_html=True)
    
    if SONG_TOPICS_PATH.exists():
        songs_df = pd.read_csv(SONG_TOPICS_PATH)
        songs_df["Thematic Category"] = songs_df["topic_id"].map(TOPIC_DISPLAY_ALIASES).fillna("Unclustered / Outliers (Topic -1)")
        
        col_search, col_topic_filt = st.columns([2, 1])
        with col_search:
            artist_filter = st.text_input("Filter by artist or song title:", placeholder="e.g. Drake, Olivia Rodrigo, Djo, Noah Kahan")
        with col_topic_filt:
            topic_options = ["All Categories (N = 582)"] + [v for k, v in TOPIC_DISPLAY_ALIASES.items()]
            topic_filter = st.selectbox("Filter by thematic category:", topic_options)
            
        filtered_df = songs_df.copy()
        if artist_filter.strip():
            mask_art = filtered_df["artist_names"].str.contains(artist_filter, case=False, na=False)
            mask_trk = filtered_df["track_name"].str.contains(artist_filter, case=False, na=False)
            filtered_df = filtered_df[mask_art | mask_trk]
        if topic_filter != "All Categories (N = 582)":
            filtered_df = filtered_df[filtered_df["Thematic Category"] == topic_filter]
            
        st.dataframe(
            filtered_df[["artist_names", "track_name", "release_year", "best_rank_this_year", "Thematic Category", "topic_keywords"]].rename(
                columns={
                    "artist_names": "Artist",
                    "track_name": "Song Title",
                    "release_year": "Release Year",
                    "best_rank_this_year": "Window Peak Rank",
                    "topic_keywords": "Top Cluster Keywords"
                }
            ),
            use_container_width=True
        )

# ==============================================================================
# TAB 4: EXPLORATORY PATTERNS & CASE STUDIES
# ==============================================================================
with tabs[3]:
    st.header("Act 4: Exploratory Patterns & Track Case Studies")
    
    st.markdown("""<div style="background-color: #1E293B; padding: 18px 22px; border-radius: 8px; border-left: 4px solid #F59E0B; margin-bottom: 22px;">
<b>Methodological Clarification &amp; Scope:</b><br>
Our primary statistical tests confirm that lyrics do not reliably predict Top 50 placement within the Top 200 (full-cohort BERTopic &chi;<sup>2</sup> <i>p</i> = 0.0731, artist-cluster Kruskal-Wallis <i>p</i> = 0.7399; 2024+ BERTopic &chi;<sup>2</sup> <i>p</i> = 0.2472; 2024+ classifier AUC = 0.5088). Furthermore, while genre correlates at the raw track level due to catalog age and superstar clustering, it loses statistical significance at the artist-cluster level (<i>p</i> = 0.1265) and drops to AUC = 0.5667 in the 2024+ cohort. The notes below examine <b>why apparent patterns in sub-cohorts are driven by confounding</b> rather than causal songwriting rules.
</div>""", unsafe_allow_html=True)
    
    st.subheader("Deconstructing Apparent Sub-Cohort Patterns")
    
    colR1, colR2, colR3 = st.columns(3)
    with colR1:
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #10B981; height: 100%;">
<h4 style="color: #10B981; margin-top: 0;">1. The 2024+ Topic 5 Artifact (82.6%)</h4>
<p style="font-size: 0.9rem;">When BERTopic was <b>re-fit separately on the 2024+ cohort</b> (<i>N</i> = 316, where overall Top 50 baseline rate is <b>52.2%</b>, 165/316), its newly formed Topic 5 (<code>5_big_luck_thing_feeling</code>, confessional pop) showed 19/23 tracks (82.6%) in the Top 50.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Why This Is Confounded:</b> First, this comes from a separate 13-topic model fit on 2024+ tracks (where overall topic &chi;<sup>2</sup> = 16.04, <i>p</i> = 0.2472 is non-significant), not the full-cohort model (where Topic 8 had <i>n</i> = 18, 61.1%). Second, inspection of the 23 tracks reveals that <b>7 are Olivia Rodrigo deluxe tracks and 3 are Ariana Grande tracks</b>—meaning two mega-stars account for 10 of the 19 hits!</p>
</div>""", unsafe_allow_html=True)
    with colR2:
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #38BDF8; height: 100%;">
<h4 style="color: #38BDF8; margin-top: 0;">2. Why Track-Level Genre Collapses</h4>
<p style="font-size: 0.9rem;">In the full 582-track corpus, consolidated Macro-Genres show track-level differences (Pop 48.7% and K-Pop 48.3% vs Rock/Alt 30.5% and R&amp;B/Soul 20.7%, &chi;<sup>2</sup> <i>p</i> = 6.89e-05).</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Why This Is Confounded:</b> Two mechanisms drive this: (1) Rock/Alt tracks in the Top 200 are overwhelmingly older catalog classics observed on their decay tail, and (2) prolific Pop/K-Pop superstars contribute dozens of tracks. When aggregated to the 191 independent artist clusters, genre differences become statistically non-significant (<i>p</i> = 0.1265).</p>
</div>""", unsafe_allow_html=True)
    with colR3:
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #F59E0B; height: 100%;">
<h4 style="color: #F59E0B; margin-top: 0;">3. Repetition vs. Vocabulary Richness</h4>
<p style="font-size: 0.9rem;">Commercial songwriting folklore frequently claims that maximizing chorus line repetition or simplifying vocabulary guarantees a higher chart peak.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Empirical Reality:</b> Across all 582 tracks, genuine line-level repetition (median 31.8% vs 33.3%, <i>p</i> = 0.2154) and lexical diversity TTR (median 0.372 vs 0.365, <i>p</i> = 0.3255) show zero statistical relationship with reaching the Top 50.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Illustrative Track Profiles: Dataset Records vs. External Industry Context")
    st.markdown("""
    Below are five illustrative charting tracks across our full cohort ($N=582$). To maintain strict separation between **variables measured in our dataset** and **external qualitative context**, we report the exact CSV columns alongside strictly textual lyric descriptions and external industry background:
    """)
    
    playbook_cases = [
        {
            "Artist & Track": "Djo — End of Beginning",
            "Window Peak Rank": "#1",
            "All-Time Peak Rank": "#1 (135 wks)",
            "Dataset Release Year & Genre": "2024 (Alternative)",
            "Lyrical Theme (Text Only)": "Second-person nostalgic reflection on moving away from Chicago and past identity ('back in Chicago I feel it').",
            "External Industry & Temporal Context (Not in Dataset)": "Originally released on the 2022 album DECIDE; iTunes API recorded 2024 due to its 2024 viral single re-indexing. Artist Joe Keery had substantial pre-existing celebrity recognition from Netflix's Stranger Things."
        },
        {
            "Artist & Track": "Malcolm Todd — Earrings",
            "Window Peak Rank": "#1",
            "All-Time Peak Rank": "#1 (25 wks)",
            "Dataset Release Year & Genre": "2024 (Alternative)",
            "Lyrical Theme (Text Only)": "Conversational first-person lines centered on everyday romantic objects and relationship aftermath.",
            "External Industry & Temporal Context (Not in Dataset)": "Released in 2024; gained rapid short-form social video traction and signed with Columbia Records."
        },
        {
            "Artist & Track": "Benson Boone — Beautiful Things",
            "Window Peak Rank": "#28",
            "All-Time Peak Rank": "#1 (139 wks)",
            "Dataset Release Year & Genre": "2024 (Pop)",
            "Lyrical Theme (Text Only)": "Direct first-person gratitude mingled with anxiety over losing loved ones ('please stay, I want you, I need you').",
            "External Industry & Temporal Context (Not in Dataset)": "Illustrates window left-truncation: peaked at #1 all-time in early 2024, but was observed at #28 on its multi-year decay tail during our 2025–2026 window. Backed by Night Street / Warner Records."
        },
        {
            "Artist & Track": "Teddy Swims — Lose Control",
            "Window Peak Rank": "#34",
            "All-Time Peak Rank": "#5 (149 wks)",
            "Dataset Release Year & Genre": "2023 (Pop)",
            "Lyrical Theme (Text Only)": "First-person narrative of emotional dependency and loss of self-restraint in a turbulent relationship.",
            "External Industry & Temporal Context (Not in Dataset)": "Pre-2024 catalog release (2023) that peaked at #5 all-time and remained at #34 during our 2025–2026 observation window. Supported by Warner Records radio promotion."
        },
        {
            "Artist & Track": "Olivia Rodrigo — begged",
            "Window Peak Rank": "#14",
            "All-Time Peak Rank": "#14 (4 wks)",
            "Dataset Release Year & Genre": "2026 (Pop)",
            "Lyrical Theme (Text Only)": "Confessional diary-style confrontation over romantic betrayal (assigned to 2024+ Topic 5).",
            "External Industry & Temporal Context (Not in Dataset)": "In-window 2026 release backed by Geffen / Interscope; illustrates how superstar album/deluxe drops immediately populate the Top 50 and dominate small BERTopic clusters."
        }
    ]
    st.table(pd.DataFrame(playbook_cases))
    st.caption("Selection Bias Disclaimer: These five tracks are qualitative illustrations from the charting cohort. Inspecting successful tracks without comparing them against non-charting tracks with similar lyrics is selection on the dependent variable and cannot establish causal efficacy.")

# ==============================================================================
# TAB 5: HONEST GUIDE FOR EMERGING ARTISTS & SONGWRITERS (THE ANTI-PLAYBOOK)
# ==============================================================================
with tabs[4]:
    st.header("Act 5: Does This Project Actually Help Emerging Artists & Songwriters?")
    
    st.markdown("""<div style="background-color: #0F172A; padding: 24px; border-radius: 10px; border-left: 5px solid #EF4444; margin-bottom: 24px;">
<h3 style="color: #F87171; margin-top: 0; font-size: 1.2rem;">The Cold, Honest Answer: 0% Value as a "Hit Songwriting Formula"</h3>
<p style="font-size: 0.96rem; line-height: 1.7; margin-bottom: 0;">
If an emerging songwriter or independent artist opens this project hoping to learn <b>"how to write lyrics that guarantee a Top 50 hit,"</b> this study offers <b>zero prescriptive songwriting formulas</b>. Because our empirical data proved that lyric length, chorus repetition, vocabulary simplicity, emotional tone, and lyrical topics have no statistical power to separate Top 50 hits from lower-charting songs, giving you a "magic checklist of words or themes to write" would be fabricating advice that contradicts the evidence.
</p>
</div>""", unsafe_allow_html=True)

    st.subheader("Where This Audit DOES Help: The 'Anti-Playbook' (What NOT to Waste Time On)")
    st.markdown("""
    While data science cannot write a hit song for you, **proving what does NOT work is just as valuable for decision-making**. For an emerging artist with limited time, energy, and budget, our findings provide three practical lessons on **traps to avoid**:
    """)

    colG1, colG2, colG3 = st.columns(3)
    with colG1:
        st.markdown("""<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border-top: 4px solid #10B981; height: 100%;">
<h4 style="color: #10B981; margin-top: 0;">1. Ignore Fake "Lyric Formulas" (Keep Your Creative Freedom)</h4>
<p style="font-size: 0.92rem; line-height: 1.65;">
Commercial songwriting blogs and self-proclaimed industry gurus often tell new artists: <i>"Keep your lyrics under 250 words," "Repeat your chorus line at least 40% of the time,"</i> or <i>"Use simpler vocabulary so listeners don't skip."</i>
</p>
<p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6; margin-bottom: 0;">
<b>Practical Takeaway:</b> None of those claims hold up in real Spotify chart data. Top 50 hits and lower-tier tracks have the exact same median repetition and word complexity. Do not compromise your artistic voice or rewrite a lyric you love just to satisfy an imaginary algorithm rule.
</p>
</div>""", unsafe_allow_html=True)

    with colG2:
        st.markdown("""<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border-top: 4px solid #38BDF8; height: 100%;">
<h4 style="color: #38BDF8; margin-top: 0;">2. Don't Blindly Copy Superstar Lyric Trends</h4>
<p style="font-size: 0.92rem; line-height: 1.65;">
When emerging artists see a specific lyric style—such as raw, diary-like heartbreak confessions—dominating the Top 50, it is tempting to think, <i>"If I write about that exact same topic, my song will chart higher too."</i>
</p>
<p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6; margin-bottom: 0;">
<b>Practical Takeaway:</b> Our audit shows this is a classic <b>Superstar Illusion</b>. Those lyric clusters only show high Top 50 rates because mega-stars like Olivia Rodrigo and Ariana Grande dropped entire albums in that style, and their massive fanbases pushed every track into the Top 50. Copying a superstar's diary topic will not give a new artist a superstar's fanbase.
</p>
</div>""", unsafe_allow_html=True)

    with colG3:
        st.markdown("""<div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border-top: 4px solid #F59E0B; height: 100%;">
<h4 style="color: #F59E0B; margin-top: 0;">3. Invest Your Energy Outside the Lyric Sheet</h4>
<p style="font-size: 0.92rem; line-height: 1.65;">
Every machine learning model we trained on lyric text alone performed no better than a coin toss. Once a song's writing meets basic professional standards, endlessly tweaking individual words will not move it into the Top 50.
</p>
<p style="font-size: 0.88rem; color: #94A3B8; line-height: 1.6; margin-bottom: 0;">
<b>Practical Takeaway:</b> If your goal is commercial growth, stop obsessing over word-count or rhyme tweaks and redirect your energy toward the factors that actually move streams: <b>vocal delivery, audio production polish, a catchy 15-second sonic hook for TikTok/Reels, and release promotion</b>.
</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Actionable Summary: Where Emerging Artists Should Focus vs. Stop Worrying")
    
    guide_df = pd.DataFrame({
        "Common Songwriting Myth (What People Think)": [
            "\"I need to repeat my chorus more often so it sticks in listeners' heads and charts higher.\"",
            "\"I should dumb down my vocabulary or shorten my verses to fit modern streaming attention spans.\"",
            "\"Diary-style breakup lyrics have the highest Top 50 rate right now—I should switch to that topic.\"",
            "\"If my song isn't blowing up, the problem must be the written lyrics.\""
        ],
        "What Our Data Science Audit Proved": [
            "Top 50 songs actually have a slightly lower median line repetition rate (31.8%) than ranks 51–200 (33.3%), with no statistical difference.",
            "Vocabulary richness and word counts are virtually identical between Top 50 hits and lower-ranking songs.",
            "High Top 50 rates in specific themes are driven by a few mega-stars (e.g., Olivia Rodrigo, Ariana Grande, Drake) flooding those clusters.",
            "Even advanced AI language models reading the full lyrics cannot predict chart rank any better than a 50/50 coin flip."
        ],
        "Realistic Advice for Emerging Artists": [
            "Write the chorus repetition that fits your melody naturally; more repetition will not boost your chart peak.",
            "Write with the vocabulary and story length your song needs—listeners do not penalize richer or longer lyrics.",
            "Write themes that are authentic to your own identity rather than chasing whatever topic superstars released this month.",
            "Once your lyrics feel honest and complete, focus 80% of your remaining effort on vocal performance, sound production, and short-form video hooks."
        ]
    })
    st.table(guide_df)

# ==============================================================================
# TAB 6: LIMITATIONS & FUTURE HORIZONS
# ==============================================================================
with tabs[5]:
    st.header("Act 6: Methodological Limitations & Future Horizons")
    st.markdown("""
    A rigorous data science audit requires transparent disclosure of target-label artifacts, sample range restriction, omitted variables, and statistical dependencies.
    """)
    
    colL1, colL2 = st.columns(2)
    with colL1:
        st.subheader("1. Target Label Left-Truncation (Observation Window Artifact)")
        st.markdown("""
        * **The Mechanism**: Our binary outcome $y = (\text{best\_rank\_this\_year} \le 50)$ captures a song's highest rank **within the 53-week observation window (Sept 2025–Sept 2026)**, not its launch-year peak.
        * **Empirical Consequence**: Songs released before Sept 2025 are observed only on their downward post-peak decay tail. Pre-2024 songs in our cohort achieved a **45.1% all-time Top 50 rate** (median all-time peak `#59.0`), but only **23.7%** in our 2025–2026 window (median `#99.0`, mean decay gap `+30.7` ranks). Similarly, 2024 releases dropped from **75.0% all-time** (median `#6.5`) to **40.0% in-window** (median `#85.5`, mean gap `+58.8` ranks).
        * **Why It Matters**: The strong association between `Release Year` and $y$ ($p = 3.47 \times 10^{-11}$, `AUC = 0.6676`) reflects **window left-truncation** as much as real-world catalog aging. While our sensitivity checks in Act 1 confirm that lyrics remain null even among in-window 2025–2026 releases ($N=276, p > 0.43$) and under an `all_time_peak_rank <= 50` target ($p > 0.25$), future studies should align observation windows strictly to each track's first $K$ weeks post-release.
        """)
        
        st.subheader("2. Range Restriction & Survivorship Bias")
        st.markdown("""
        * **The Bottleneck**: Every song in our dataset already reached the Spotify Global Top 200—an elite commercial tier representing a tiny fraction of all released music.
        * **Unmeasured Hygiene Hypothesis**: We conceptualize songwriting craft as a *hygiene factor* (a baseline threshold assumed to be largely satisfied among charting tracks), though we emphasize that "lyric quality" is an unmeasured theoretical construct here rather than a quantified metric. Comparing Top 50 vs. ranks 51–200 cannot test whether lyrics differentiate charting songs from **released but low-stream (non-charting)** tracks.
        """)

    with colL2:
        st.subheader("3. Omitted Commercial & Audio Variables")
        st.markdown("""
        * **Unobserved Marketing & Distribution Drivers**: Chart rank within the Top 200 is heavily shaped by variables absent from lyric text:
          - Editorial playlist placement (*Today's Top Hits*, *New Music Friday*) and autoplay queueing
          - Short-form video velocity (TikTok / Reels usage) and external celebrity capital
          - Major-label promotional budgets and radio airplay campaigns
        * **Unmeasured Audio Production**: Vocal delivery, timbre, hook melody, tempo (BPM), and mixing dynamics are completely invisible to text-only NLP models.
        """)
        
        st.subheader("4. Multi-Track Artist Non-Independence & Modeling Scope")
        st.markdown("""
        * **Type I Error Inflation in Track-Level Tests**: Treating multiple songs by the same superstar as independent observations underestimates standard errors and **inflates false positives (Type I error)** in track-level $\chi^2$ and Mann-Whitney tests. As shown in Act 2, when we aggregate from $N=582$ tracks to $N=191$ independent artist clusters, Macro-Genre loses statistical significance ($p = 6.89 \times 10^{-5} \rightarrow p = 0.1265$).
        * **Descriptive vs. Cross-Validated BERTopic**: While our 5-fold classifier fit BERTopic strictly inside each training fold, the descriptive topic charts in Act 3 and Act 4 were fit on the pooled 582-track and 316-track cohorts for exploratory visualization.
        * **Exact Attrition Accounting ($885 \rightarrow 582$)**: 885 raw unique songs $\rightarrow$ 709 English tracks $\rightarrow$ 669 deduplicated songs $\rightarrow$ 596 after initial `track_type` holiday/event filter $\rightarrow$ **582 final modeling tracks** after removing 14 residual holiday songs (11 with Holiday/Christmas genre metadata missed by initial script ordering + 3 untagged carols surfaced during BERTopic inspection).
        """)

    st.markdown("---")
    st.subheader("Roadmap for Future Research")
    st.markdown("""
    1. **Release-Aligned Survival & Cohort Design**: Track every song for a fixed window (e.g., Weeks 1 to 12 post-release) to eliminate left-truncation and separate debut velocity from multi-year longevity.
    2. **Incorporate Non-Charting Negative Controls**: Sample released but low-stream (non-charting) tracks matched by release week and distributor tier to test whether lyrics help songs enter the Top 200.
    3. **Multimodal Audio-Text Fusion**: Combine lyric embeddings with acoustic representations (e.g., spectrogram embeddings, BPM, harmonic novelty) to isolate vocal and production effects from text.
    4. **Multilingual Corpus Expansion**: Evaluate the 176 excluded non-English tracks (Spanish reggaeton, Korean-majority K-Pop, Brazilian phonk) using multilingual encoders (`XLM-RoBERTa`).
    """)

# ==============================================================================
# TAB 7: INCIDENTAL DATA QUALITY AUDIT
# ==============================================================================
with tabs[6]:
    st.header("Act 7: Incidental Data Quality Audit")
    st.markdown("""
    While Sentence-BERT and BERTopic did not improve downstream chart rank prediction, inspecting their unsupervised clusters during exploratory data analysis provided a **practical workflow convenience**: surfacing residual seasonal tracks that had bypassed our initial metadata filter.
    """)
    
    st.subheader("How Unsupervised Clustering Surfaced Residual Holiday Tracks")
    st.markdown("""
    During initial dataset construction (`Scripts/build_dataset.py`), seasonal tracks were flagged using a title regex and a genre check (`track_type == 'holiday'`), removing 72 holiday tracks and 1 event anthem ($669 \rightarrow 596$ tracks in `EDA_spotify_lyrics.ipynb`).  
    However, 14 seasonal songs remained in the 596-track EDA corpus due to two pipeline gaps:
    1. **Script Precedence & Genre Tag Omissions (11 tracks)**: In `build_dataset.py`, remix detection (`'remix'` in title) executed *before* holiday detection (causing *Little Saint Nick - 1991 Remix* to be labeled `remix`), and several tracks with `Holiday` or `Christmas` in their iTunes genre field had `track_type == 'standard'`.
    2. **Title-Only Regex Blind Spot (3 tracks)**: Songs like The Pogues' *Fairytale of New York* (`genre = Alternative`), Chuck Berry's *Run Rudolph Run* (`genre = Rock`), and Leona Lewis's *One More Sleep* (`genre = Pop`) had neither `"Christmas"` in their titles nor `"Holiday"` in their genre metadata.
    
    When we ran BERTopic on the 596-track corpus during exploratory auditing, **8 of these residual holiday tracks clustered together into a single semantic topic** (alongside *Sleigh Ride* and 5 additional `Holiday`/`Christmas`-genre tracks caught when expanding the exclusion rule in `Scripts/m1_bertopic_model.py`, removing all 14 tracks to yield the final **582-track cohort**):
    """)
    
    holiday_audit_df = pd.DataFrame({
        "Artist": [
            "Britney Spears", "Chuck Berry", "Burl Ives", "Dean Martin",
            "The Beach Boys", "The Pogues, Kirsty MacColl", "Leona Lewis", "Meghan Trainor", "Ella Fitzgerald"
        ],
        "Track Name": [
            "My Only Wish (This Year)", "Run Rudolph Run", "Rudolph The Red-Nosed Reindeer",
            "Rudolph The Red-Nosed Reindeer", "Little Saint Nick - 1991 Remix",
            "Fairytale of New York (feat. Kirsty MacColl)", "One More Sleep", "I'll Be Home", "Sleigh Ride"
        ],
        "Raw track_type / Genre in CSV": [
            "standard / Christmas: Pop", "standard / Rock", "standard / Christmas", "standard / Christmas: Classic",
            "remix / Holiday", "standard / Alternative", "standard / Pop", "standard / Holiday", "standard / Holiday"
        ],
        "How Surfaced in Our Workflow": [
            "BERTopic Cluster + Genre Tag", "BERTopic Cluster (Missed by Title/Genre)", "BERTopic Cluster + Genre Tag", "BERTopic Cluster + Genre Tag",
            "BERTopic Cluster + Genre Tag (Masked by Remix)", "BERTopic Cluster (Missed by Title/Genre)", "BERTopic Cluster (Missed by Title/Genre)", "BERTopic Cluster + Genre Tag", "Expanded Title/Genre Regex Audit"
        ],
        "Seasonal Keywords Present in Lyric Text": [
            "christmas, santa, sleigh, mistletoe", "christmas, santa", "christmas, santa, reindeer, sleigh",
            "christmas, santa, reindeer, sleigh", "christmas, santa, reindeer, saint nick",
            "christmas", "christmas, mistletoe", "christmas, santa", "sleigh"
        ],
        "Window Peak Rank": [47, 27, 82, 152, 57, 45, 166, 134, 118]
    })
    st.table(holiday_audit_df)
    st.caption("Methodological Reality Check: Notice the fifth column—every single one of these tracks contains explicit seasonal words ('christmas', 'santa', 'reindeer', 'sleigh', 'mistletoe') inside its lyric body. Had our initial rule scanned the lyric text with a simple keyword regex (rather than checking only track titles and genre tags), all 9 tracks would have been caught without needing a Transformer model.")

    st.markdown("---")
    st.subheader("Realistic Scope for Unsupervised NLP in Catalog Maintenance")
    st.markdown("""
    To avoid overstating this incidental finding, we emphasize that **Transformer clustering is not necessary for simple keyword filtering** (where a regex over lyric text is faster, deterministic, and cheaper). Rather, unsupervised embedding clusters can serve as an **exploratory diagnostic aid** when analysts do not know in advance which anomalous categories exist in a messy corpus:
    """)

    colU1, colU2 = st.columns(2)
    with colU1:
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #38BDF8; height: 100%;">
<h4 style="color: #38BDF8; margin-top: 0;">1. Unanticipated Ingestion Auditing</h4>
<p style="font-size: 0.9rem;">Rule-based regex filters work well when the target words are known a priori (e.g., holiday keywords), but fail when unexpected sub-corpora (e.g., comedy skits, children's nursery rhymes, or soundtrack dialogue) enter a chart dataset under generic <code>Pop</code> tags.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Practical Role:</b> Unsupervised clustering groups semantically similar outliers together during EDA so human analysts can spot unanticipated pipeline leaks and write deterministic rules to fix them.</p>
</div>""", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #10B981; height: 100%;">
<h4 style="color: #10B981; margin-top: 0;">2. Cold-Start Descriptive Tagging</h4>
<p style="font-size: 0.9rem;">Newly uploaded tracks lack user co-listening graphs. While audio models capture tempo and instrumentation, they do not index lyrical subject matter.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Practical Role:</b> Lightweight text embeddings provide a descriptive thematic index for catalog search (unrelated to predicting commercial chart rank).</p>
</div>""", unsafe_allow_html=True)

    with colU2:
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #F59E0B; height: 100%;">
<h4 style="color: #F59E0B; margin-top: 0;">3. Sync Licensing &amp; Thematic Retrieval</h4>
<p style="font-size: 0.9rem;">Music supervisors searching for television or film sync placements often query by narrative scenario (e.g., moving away from a hometown) rather than rigid iTunes genre labels.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Practical Role:</b> Semantic similarity search over lyric embeddings allows qualitative retrieval across catalog archives without claiming predictive hit power.</p>
</div>""", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        st.markdown("""<div style="background-color: #1E293B; padding: 18px; border-radius: 8px; border-top: 4px solid #EC4899; height: 100%;">
<h4 style="color: #EC4899; margin-top: 0;">4. Human-in-the-Loop Data Hygiene</h4>
<p style="font-size: 0.9rem;">As demonstrated in our own 669 &rarr; 596 &rarr; 582 attrition pipeline, automated rules and unsupervised clustering are complementary rather than competing tools.</p>
<p style="font-size: 0.85rem; color: #94A3B8;"><b>Practical Role:</b> Clustering surfaces candidate anomalies during exploratory audits, which engineers then codify into fast, reproducible regex/metadata masks.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("""
    **Methodological Takeaway**: Neither surface lyric metrics, VADER sentiment, raw 384-dim Sentence-BERT embeddings, nor BERTopic clusters improve Top 50 chart rank prediction within the Spotify Top 200. However, inspecting unsupervised clusters during EDA served a useful diagnostic role by exposing gaps in our initial title-only holiday filter—illustrating how negative predictive findings and transparent error analysis strengthen data science pipelines.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""<div style="text-align: center; font-size: 0.85rem; color: #64748B;">
<b>Spotify-TOP200-Lyrics Project Deliverable</b> | 
Evaluated with 5-Fold Artist-Cluster Grouped Cross-Validation (191 Connected Collaboration Components) | 
Fold-Internal BERTopic Fitting &amp; Direct 384-Dim SBERT Verification
</div>""", unsafe_allow_html=True)
