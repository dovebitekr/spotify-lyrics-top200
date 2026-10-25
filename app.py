import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Spotify Top 200 Lyrics Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants & Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "spotify_data" / "final_dataset_for_ml_lyrics_en.csv"
TOPICS_FULL_PATH = PROJECT_ROOT / "artifacts" / "bertopic_topic_metrics.csv"
SONG_TOPICS_PATH = PROJECT_ROOT / "artifacts" / "bertopic_song_topics.csv"

# Descriptive Names for the 24 BERTopic Clusters (Matched to bertopic_topic_metrics.csv)
TOPIC_DISPLAY_ALIASES = {
    -1: "Unclustered / Outliers (Topic -1)",
    0: "Club Grooves & Flirtation",
    1: "Street Rap & Trap Anthems",
    2: "Urban Pop, Style & Late-Night Confidence",
    3: "High-Energy K-Pop & Performance Hooks",
    4: "Dark Alt-Pop, Angst & Obsession",
    5: "Rhythmic Dance-Pop & Party Anthems",
    6: "Heartbreak, Letting Go & Breakup Ballads",
    7: "Classic Pop Storytelling & Nostalgia",
    8: "Confessional Pop & Emotional Vulnerability",
    9: "Melancholic Reflection, Summer & Solitude",
    10: "Post-Breakup Distance & Holding On",
    11: "Contemporary R&B, Swag & Collabs",
    12: "Funk, Classic Rhythm & Nostalgic Chants",
    13: "Melodic Romance & Stadium Hooks",
    14: "Folk-Pop Introspection & Acoustic Roots",
    15: "Indie Folk Melancholy & Soulful R&B",
    16: "Atmospheric Midnight Romance",
    17: "Dream Pop & Slow-Burn Longing",
    18: "Emotional Diving, Pain & Release",
    19: "Hip-Hop Braggadocio & Hustle",
    20: "Latin Pop, Global Grooves & Summer Heat",
    21: "Raw Acoustic Pain & Folk-Rock Grit",
    22: "Theatrical Soundtracks & Dramatic Ballads",
    23: "Motivational Anthems & Fighter Spirit"
}

def _calc_line_repetition(text: str) -> float:
    """Compute line-level repetition rate: 1 - (unique lines / total non-empty lines)."""
    lines = [line.strip().lower() for line in str(text).splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))

@st.cache_data
def load_standard_cohort():
    if not DATA_PATH.exists():
        return None
    raw_df = pd.read_csv(DATA_PATH)
    
    # Filter out holiday carols and event anthems matching Scripts/m1_bertopic_model.py
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
    
    # Compute line-level repetition rate
    df_clean["line_repetition_rate"] = df_clean["cleaned_lyrics_transformer"].apply(_calc_line_repetition)
    
    df_clean["target_hit"] = (df_clean["best_rank_this_year"] <= 50).astype(int)
    df_clean["Hit_Tier"] = df_clean["target_hit"].map({1: "Top 50 Hits", 0: "Ranks 51–200"})
    
    artist_counts = df_clean["artist_names"].value_counts()
    df_clean["artist_track_count"] = df_clean["artist_names"].map(artist_counts)
    return df_clean

clean_df = load_standard_cohort()

# ==============================================================================
# MAIN PAGE HERO: PROJECT OVERVIEW & EXECUTIVE BRIEFING
# ==============================================================================
st.title("Do Lyrics Make a Top 50 Hit? Analyzing the Spotify Global Top 200")
st.caption("An Exploratory Data Analysis, Topic Modeling, and Machine Learning Project on Commercial Song Lyrics")

with st.container():
    st.markdown(r"""<div style="background-color: #0F172A; padding: 26px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
<h3 style="color: #38BDF8; margin-top: 0; font-size: 1.3rem;">Project Overview &amp; Executive Briefing</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; font-size: 0.95rem; line-height: 1.65;">
<div>
<p><b>Why We Started This Project:</b><br>
Many songwriting blogs claim there is a secret "lyric formula" for making a hit song—such as keeping verses short, repeating the chorus as much as possible, using simple words, or writing about a specific emotional topic. Our goal was to test whether written lyrics actually separate songs that reach the <b>Spotify Global Top 50</b> from songs that stay in ranks 51–200.</p>
<p><b>Who This Project Is For:</b><br>
Emerging artists, songwriters, and data science students who want to see what real chart data says about lyrics versus market factors like release timing and artist fame.</p>
</div>
<div>
<p><b>Dataset Used:</b><br>
One year of weekly <b>Spotify Global Top 200</b> chart data (Sept 2025 – Sept 2026). After filtering for English songs and removing duplicate album versions and seasonal Christmas tracks, our final modeling dataset contains <b>582 unique songs</b> (228 Top 50 hits and 354 songs peaking in ranks 51–200).</p>
<p><b>Methods Used:</b><br>
We analyzed the lyrics in three steps: (1) <b>Exploratory Data Analysis (EDA)</b> on word count, line repetition, and vocabulary diversity; (2) <b>Topic Modeling (BERTopic)</b> to group songs by lyrical theme; and (3) <b>Logistic Regression with 5-Fold Grouped Cross-Validation</b> to test whether lyrics help predict Top 50 placement.</p>
</div>
</div>
<div style="background-color: #1E293B; padding: 22px 24px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #10B981; font-size: 0.95rem; line-height: 1.7;">
<h4 style="margin-top: 0; color: #10B981; font-size: 1.1rem;">Key Takeaways: Three Simple Findings</h4>
<p><b>1. Basic Lyric Features Look the Same in Top 50 Hits and Ranks 51–200:</b><br>
Whether a song peaks in the Top 50 or stays between ranks 51 and 200, its lyric length, chorus repetition rate, and vocabulary diversity are almost identical. Statistical tests (Mann-Whitney U test) show no meaningful difference on any of these features.</p>
<p><b>2. Adding Lyric Topics or Text Embeddings Does Not Improve Machine Learning Predictions:</b><br>
A simple baseline Logistic Regression model using only <b>Release Year and Genre</b> achieves a moderate cross-validated ROC-AUC of <b>0.66</b>. Adding BERTopic lyric themes does not improve the score, and models trained only on lyrics perform around <b>0.47–0.51</b> (the same as a 50/50 coin flip).</p>
<p style="margin-bottom: 0;"><b>3. Apparent "Winning Themes" Are Driven by Superstars and Release Timing:</b><br>
Some genres and lyric topics appear to have high Top 50 rates at first glance. However, looking closer at the data shows why: brand-new songs naturally peak higher during the current chart year than older catalog songs, and a few mega-stars (like Taylor Swift, Drake, Olivia Rodrigo, and Ariana Grande) release many songs at once that all land in the Top 50.</p>
</div>
</div>""", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "Part 1: Basic Lyric EDA",
    "Part 2: Machine Learning Models",
    "Part 3: Lyric Topics (BERTopic)",
    "Part 4: Song Examples & Artist Guide",
    "Part 5: Limitations & Data Cleaning"
])

# ==============================================================================
# TAB 1: BASIC LYRIC EDA
# ==============================================================================
with tabs[0]:
    st.header("Part 1: Do Top 50 Hits Have Different Lyric Statistics?")
    st.markdown(r"""
    **Question**: Do Top 50 songs (`n = 228`) have more repetition, shorter lyrics, or simpler vocabulary than songs peaking at ranks 51–200 (`n = 354`)?  
    **Method**: Because song statistics have skewed distributions (for example, rap songs have much higher word counts than pop songs), we compare medians and means and run the non-parametric **Mann-Whitney U test**.  
    **Result**: **All three lyric features have $p > 0.20$**, meaning there is no statistically significant difference between Top 50 hits and lower-charting tracks.
    """)
    
    st.subheader("Interactive Feature Distribution Explorer")
    attr_option = st.radio(
        "Select a lyric feature to compare:",
        [
            "Chorus Line Repetition Rate",
            "Word Count (Song Length)",
            "Vocabulary Diversity (Type-Token Ratio)"
        ],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    if attr_option == "Chorus Line Repetition Rate":
        col1.metric("Top 50 Hits (Median / Mean)", "31.8% / 32.2%", help="Percentage of repeated lines (1 - unique_lines / total_lines) in Top 50 tracks (n = 228).")
        col2.metric("Ranks 51–200 (Median / Mean)", "33.3% / 33.9%", help="Percentage of repeated lines in ranks 51–200 tracks (n = 354).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.2154",
            delta="No Significant Difference",
            delta_color="off",
            help="Since p > 0.05, there is no statistically significant difference in line repetition between the two groups."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="line_repetition_rate",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Hits": "#10B981", "Ranks 51–200": "#64748B"},
                points="outliers",
                labels={"line_repetition_rate": "Line Repetition Rate (0 to 1)", "Hit_Tier": "Chart Group"},
                title="Chorus Line Repetition Rate: Top 50 vs. Ranks 51–200 (N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Takeaway: Top 50 songs actually have a slightly lower median line repetition rate (31.8% vs. 33.3%), and the difference is not statistically significant ($p = 0.2154$).")

    elif attr_option == "Word Count (Song Length)":
        col1.metric("Top 50 Hits (Median / Mean)", "352.5 / 376.7 words", help="Total word count in Top 50 tracks (n = 228).")
        col2.metric("Ranks 51–200 (Median / Mean)", "346.5 / 374.0 words", help="Total word count in ranks 51–200 tracks (n = 354).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.5294",
            delta="No Significant Difference",
            delta_color="off",
            help="Since p > 0.05, song word count does not differ significantly between Top 50 and ranks 51–200."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="word_count",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Hits": "#10B981", "Ranks 51–200": "#64748B"},
                points="outliers",
                labels={"word_count": "Total Word Count", "Hit_Tier": "Chart Group"},
                title="Total Word Count: Top 50 vs. Ranks 51–200 (N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Takeaway: Both groups have a median length of about 350 words ($p = 0.5294$).")

    else:
        col1.metric("Top 50 Hits (Median / Mean)", "0.372 / 0.374", help="Unique words divided by total words (Type-Token Ratio) in Top 50 tracks (n = 228).")
        col2.metric("Ranks 51–200 (Median / Mean)", "0.365 / 0.367", help="Unique words divided by total words in ranks 51–200 tracks (n = 354).")
        col3.metric(
            "Mann-Whitney U Test",
            "p = 0.3255",
            delta="No Significant Difference",
            delta_color="off",
            help="Since p > 0.05, vocabulary diversity does not differ significantly between the two groups."
        )
        
        if clean_df is not None:
            fig = px.box(
                clean_df,
                x="Hit_Tier",
                y="lexical_diversity",
                color="Hit_Tier",
                color_discrete_map={"Top 50 Hits": "#10B981", "Ranks 51–200": "#64748B"},
                points="outliers",
                labels={"lexical_diversity": "Vocabulary Diversity (Unique / Total Words)", "Hit_Tier": "Chart Group"},
                title="Vocabulary Diversity (TTR): Top 50 vs. Ranks 51–200 (N = 582)"
            )
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Takeaway: Top 50 hits and lower-ranking songs use roughly the same proportion of unique words ($p = 0.3255$).")

    st.markdown("---")
    st.subheader("Summary Table of Basic Lyric Features (N = 582)")
    eda_summary_df = pd.DataFrame({
        "Lyric Feature": [
            "Chorus Line Repetition Rate (1 - unique_lines / total_lines)",
            "Word Count (Total Cleaned Words)",
            "Vocabulary Diversity (Unique Words / Total Words)"
        ],
        "Top 50 Hits (n = 228)": [
            "Median: 31.8% (Mean: 32.2%)",
            "Median: 352.5 (Mean: 376.7)",
            "Median: 0.372 (Mean: 0.374)"
        ],
        "Ranks 51–200 (n = 354)": [
            "Median: 33.3% (Mean: 33.9%)",
            "Median: 346.5 (Mean: 374.0)",
            "Median: 0.365 (Mean: 0.367)"
        ],
        "Mann-Whitney U p-value": [
            "p = 0.2154",
            "p = 0.5294",
            "p = 0.3255"
        ],
        "Spearman Correlation with Peak Rank": [
            "r = +0.044 (p = 0.2842)",
            "r = -0.030 (p = 0.4735)",
            "r = -0.023 (p = 0.5807)"
        ],
        "Conclusion": [
            "Not Significant",
            "Not Significant",
            "Not Significant"
        ]
    })
    st.table(eda_summary_df)

# ==============================================================================
# TAB 2: MACHINE LEARNING MODELS
# ==============================================================================
with tabs[1]:
    st.header("Part 2: Can Machine Learning Predict Top 50 Hits from Lyrics?")
    st.markdown(r"""
    To test whether lyrics help predict Top 50 hits ($y = 1$ if `best_rank_this_year <= 50`, else $0$), we trained **Logistic Regression** models and evaluated them using **5-Fold Stratified Group K-Fold Cross-Validation**.
    """)
    
    st.subheader("How We Set Up the Machine Learning Pipeline")
    st.markdown(r"""
    1. **Grouping by Artist (`StratifiedGroupKFold`)**:  
       Some artists have multiple songs in the Top 200. If an artist like Taylor Swift or Drake appeared in both the training set and the test set, the model could simply memorize that artist's style. Grouping by artist ensures that **all songs by the same artist (or collaborating artists) stay in the same fold**, so the test set always evaluates unseen artists.
    2. **Preventing Data Leakage & Feature Scaling**:  
       In every fold, numeric features like `release_year` are standardized using `StandardScaler` fit only on the training set, and the BERTopic model is trained only on the training fold before predicting topic probabilities on the test fold.
    3. **Evaluation Metric (`ROC-AUC`)**:  
       A score of **`0.50`** means the model is no better than flipping a coin, while **`1.00`** is a perfect classifier.
    """)
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("1. Full Dataset Models (All 582 Songs)")
        cohort_full = pd.DataFrame({
            "Model": [
                "M1: Metadata Baseline (Release Year + Genre)",
                "M2: Metadata + BERTopic Lyric Themes",
                "M3: Text Embeddings Only (Sentence-BERT)",
                "M4: Basic Lyric Stats Only (WC, Repetition, TTR)"
            ],
            "Mean ROC-AUC": [0.6611, 0.6463, 0.5080, 0.4696],
            "Fold Std Dev": ["± 0.1000", "± 0.0566", "± 0.0450", "± 0.0394"],
            "Macro F1": ["0.5845", "0.5861", "0.5090", "0.4686"]
        })
        fig_full = px.bar(
            cohort_full,
            x="Mean ROC-AUC",
            y="Model",
            orientation="h",
            color="Mean ROC-AUC",
            color_continuous_scale="Tealgrn",
            text="Mean ROC-AUC"
        )
        fig_full.add_vline(x=0.50, line_dash="dash", line_color="#EF4444", annotation_text="0.50 (Coin Flip)")
        fig_full.update_layout(yaxis={"categoryorder": "total ascending"}, height=360, xaxis_range=[0.30, 0.78])
        st.plotly_chart(fig_full, use_container_width=True)
        st.markdown(r"""
        * **Metadata Baseline (`ROC-AUC = 0.6611`)**: Using just `Release Year` and `Genre` gives moderate predictive performance.
        * **Adding Lyrics Does Not Help (`ROC-AUC = 0.6463`)**: Adding BERTopic lyric probabilities slightly lowers the average score from `0.6611` to `0.6463`.
        * **Lyrics Alone Perform Like a Coin Toss (`0.47–0.51`)**: Both Sentence-BERT text embeddings (`0.5080`) and basic lyric stats (`0.4696`) perform around `0.50`.
        """)

    with colB:
        st.subheader("2. Recent Songs Only (2024–2026 Releases, N = 316)")
        cohort_2024 = pd.DataFrame({
            "Model": [
                "M2: Metadata + BERTopic Lyric Themes",
                "M1: Metadata Baseline (Release Year + Genre)",
                "M3: Text Embeddings Only (Sentence-BERT)",
                "M4: Basic Lyric Stats Only (WC, Repetition, TTR)"
            ],
            "Mean ROC-AUC": [0.5313, 0.5239, 0.4878, 0.3525],
            "Fold Std Dev": ["± 0.1190", "± 0.1087", "± 0.1186", "± 0.0372"],
            "Macro F1": ["0.5234", "0.5440", "0.4820", "0.3957"]
        })
        fig_2024 = px.bar(
            cohort_2024,
            x="Mean ROC-AUC",
            y="Model",
            orientation="h",
            color="Mean ROC-AUC",
            color_continuous_scale="Purples",
            text="Mean ROC-AUC"
        )
        fig_2024.add_vline(x=0.50, line_dash="dash", line_color="#EF4444", annotation_text="0.50 (Coin Flip)")
        fig_2024.update_layout(yaxis={"categoryorder": "total ascending"}, height=360, xaxis_range=[0.25, 0.78])
        st.plotly_chart(fig_2024, use_container_width=True)
        st.markdown(r"""
        * **Why Does Metadata Drop to `0.5239` on Recent Songs?**  
          In the full dataset (left), `Release Year` helped the model separate brand-new hits from older 1980s–2022 catalog songs that sit in ranks 51–200. When we filter the dataset to only recent 2024–2026 songs (`N = 316`), `Release Year` loses most of its contrast, and every model drops close to `0.50`.
        """)

    st.markdown("---")
    st.subheader("Why Did Metadata Work in the Full Dataset? (Metadata vs. Lyrics Comparison)")
    st.markdown(r"""
    To understand why non-lyric metadata (`Release Year`, `Artist Chart Volume`, and `Genre`) looked strong in our initial exploration while lyrics did not, we compare their statistical tests side by side:
    """)
    
    meta_vs_lyrics_df = pd.DataFrame({
        "Feature": [
            "Release Year (Newer vs. Older Songs)",
            "Artist Chart Presence (10+ Charting Songs vs. 1–2 Songs)",
            "Music Genre (Consolidated into 8 Major Genres)",
            "Lyric Themes (24 BERTopic Clusters)",
            "Basic Lyric Features (Word Count, Repetition, Vocabulary Diversity)"
        ],
        "Statistical Test (N = 582)": [
            "Mann-Whitney U test: p < 0.0001",
            "Chi-Square test: p < 0.0001",
            "Chi-Square test: p < 0.0001",
            "Chi-Square test: p = 0.0731",
            "Mann-Whitney U tests: all p > 0.20"
        ],
        "What We Observe in the Data": [
            "2025–2026 releases reach the Top 50 at 54.0%, compared to 23.7% for pre-2024 songs.",
            "Superstars with 10+ songs hit the Top 50 at 62.1%, vs. 29.8% for artists with only 1–2 songs.",
            "Pop (48.7%) and K-Pop (48.3%) have higher Top 50 rates than Rock/Alternative (30.5%).",
            "Top 50 rates vary across topics, but overall differences are not statistically significant (p > 0.05).",
            "Top 50 and ranks 51–200 have almost identical averages and medians."
        ],
        "Simple Interpretation": [
            "Older songs peaked years ago and now sit in the lower half of the chart, while new releases debut high.",
            "Famous artists have large fanbases that stream their new albums straight into the Top 50.",
            "Pop and K-Pop have more brand-new superstar releases in the current chart year than Rock.",
            "Lyric topics alone do not reliably separate Top 50 hits from lower-charting songs.",
            "Writing shorter or more repetitive lyrics does not boost chart rank."
        ]
    })
    st.table(meta_vs_lyrics_df)

# ==============================================================================
# TAB 3: LYRIC TOPICS (BERTopic) & SONG EXPLORER
# ==============================================================================
with tabs[2]:
    st.header("Part 3: Exploring Lyric Themes with BERTopic")
    st.markdown(r"""
    We used **BERTopic** (which combines `Sentence-BERT` text embeddings, `UMAP` dimensionality reduction, and `HDBSCAN` clustering) to group our 582 songs by lyrical theme:
    * **459 songs (78.9%)** were grouped into **24 specific lyric topics**.
    * **123 songs (21.1%)** did not fit neatly into a single cluster and were labeled by HDBSCAN as **`Topic -1` (Outliers / Mixed Themes)**, which had a **34.1%** Top 50 rate.
    """)
    
    if TOPICS_FULL_PATH.exists():
        topics_df = pd.read_csv(TOPICS_FULL_PATH)
        topics_df["Thematic Name"] = topics_df["topic_id"].map(TOPIC_DISPLAY_ALIASES).fillna("Other / Uncategorized")
        valid_topics = topics_df[topics_df["topic_id"] >= 0].copy()
        valid_topics["top_50_pct"] = valid_topics["top_50_rate"] * 100
        valid_topics = valid_topics.sort_values(by="top_50_pct", ascending=True)
        valid_topics["label_with_n"] = valid_topics.apply(lambda r: f"{r['Thematic Name']} (n={r['song_count']})", axis=1)
        valid_topics["pct_label"] = valid_topics["top_50_pct"].apply(lambda x: f" {x:.1f}%")
        
        st.subheader("Top 50 Hit Rate by Lyric Theme (24 BERTopic Clusters, n = 459 Songs)")
        
        fig_topics = go.Figure()
        fig_topics.add_trace(go.Bar(
            x=valid_topics["top_50_pct"],
            y=valid_topics["label_with_n"],
            orientation="h",
            marker=dict(color=valid_topics["top_50_pct"], colorscale="Viridis"),
            text=valid_topics["pct_label"],
            textposition="outside",
            textfont=dict(size=12, color="#F8FAFC")
        ))
        
        fig_topics.add_vline(
            x=39.2, line_dash="dash", line_color="#EF4444",
            annotation_text="Overall Dataset Average: 39.2%", annotation_position="top right"
        )
        
        fig_topics.update_layout(
            height=880,
            margin=dict(l=320, r=60, t=40, b=40),
            yaxis=dict(dtick=1, tickfont=dict(size=13, color="#F8FAFC")),
            xaxis=dict(title="Top 50 Hit Rate (%)", tickfont=dict(size=12, color="#F8FAFC"), range=[0, 80])
        )
        st.plotly_chart(fig_topics, use_container_width=True)
        
        st.info(r"""
        **Why You Should Be Careful Interpreting This Chart**:  
        1. **No Overall Statistical Significance ($p = 0.0731$)**: A Chi-Square test across all 24 topics gives $p = 0.0731$ (above the standard $0.05$ cutoff), meaning topic differences could easily happen by chance.
        2. **Small Sample Sizes & Superstar Concentration**: Many topics have only 10 to 20 songs. When a topic shows a high Top 50 rate—like *Confessional Pop & Emotional Vulnerability* (`61.1%`, $n=18$) or *Hip-Hop Braggadocio* (`54.5%`, $n=11$)—it is largely because mega-stars like **Olivia Rodrigo, Ariana Grande, or Drake** have multiple songs inside that specific cluster.
        """)

    st.markdown("---")
    st.subheader("Search & Explore All 582 Songs in the Dataset")
    st.markdown("Use the filters below to look up any artist, song title, or BERTopic theme in our 582-song dataset:")
    
    if SONG_TOPICS_PATH.exists():
        songs_df = pd.read_csv(SONG_TOPICS_PATH)
        songs_df["Thematic Category"] = songs_df["topic_id"].map(TOPIC_DISPLAY_ALIASES).fillna("Unclustered / Outliers (Topic -1)")
        
        col_search, col_topic_filt = st.columns([2, 1])
        with col_search:
            artist_filter = st.text_input("Search by artist or song title:", placeholder="e.g. Drake, Olivia Rodrigo, Djo, Taylor Swift")
        with col_topic_filt:
            topic_options = ["All Categories (N = 582)"] + [v for k, v in TOPIC_DISPLAY_ALIASES.items()]
            topic_filter = st.selectbox("Filter by lyric topic:", topic_options)
            
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
                    "best_rank_this_year": "Peak Rank (This Year)",
                    "topic_keywords": "Top Topic Words"
                }
            ),
            use_container_width=True
        )

# ==============================================================================
# TAB 4: SONG EXAMPLES & PRACTICAL GUIDE FOR NEW ARTISTS
# ==============================================================================
with tabs[3]:
    st.header("Part 4: Real Song Examples & Practical Guide for Emerging Artists")
    
    st.subheader("1. Five Example Songs from Our Dataset")
    st.markdown(r"""
    Looking at individual songs helps illustrate why chart rank depends heavily on **when a song was released** and **who released it**, rather than just the words on the page:
    """)
    
    playbook_cases = [
        {
            "Artist & Song": "Djo — End of Beginning",
            "Peak Rank (2025–2026)": "#1",
            "All-Time Peak Rank": "#1 (135 wks)",
            "Dataset Year & Genre": "2024 (Alternative)",
            "What the Lyrics Are About": "Nostalgic reflection on moving away from Chicago and growing into a new version of yourself.",
            "Real-World Context": "Originally released in 2022, it went viral on TikTok in 2024 (which updated its iTunes year to 2024). Performed by actor Joe Keery (Stranger Things)."
        },
        {
            "Artist & Song": "Malcolm Todd — Earrings",
            "Peak Rank (2025–2026)": "#1",
            "All-Time Peak Rank": "#1 (25 wks)",
            "Dataset Year & Genre": "2024 (Alternative)",
            "What the Lyrics Are About": "Simple, conversational lines about everyday memories after a breakup.",
            "Real-World Context": "Blew up through short-form social media clips and major-label distribution (Columbia Records)."
        },
        {
            "Artist & Song": "Benson Boone — Beautiful Things",
            "Peak Rank (2025–2026)": "#28",
            "All-Time Peak Rank": "#1 (139 wks)",
            "Dataset Year & Genre": "2024 (Pop)",
            "What the Lyrics Are About": "Grateful reflection on love mixed with the fear of losing it.",
            "Real-World Context": "Shows why release year matters in our dataset: it hit #1 all-time in 2024, and by 2025–2026 it was still charting at #28."
        },
        {
            "Artist & Song": "Teddy Swims — Lose Control",
            "Peak Rank (2025–2026)": "#34",
            "All-Time Peak Rank": "#5 (149 wks)",
            "Dataset Year & Genre": "2023 (Pop)",
            "What the Lyrics Are About": "Emotional ballad about losing self-control in an intense relationship.",
            "Real-World Context": "A 2023 hit that peaked at #5 all-time and stayed in the Top 50 (#34) more than two years later thanks to vocal power and radio airplay."
        },
        {
            "Artist & Song": "Olivia Rodrigo — begged",
            "Peak Rank (2025–2026)": "#14",
            "All-Time Peak Rank": "#14 (4 wks)",
            "Dataset Year & Genre": "2026 (Pop)",
            "What the Lyrics Are About": "Diary-style breakup confrontation.",
            "Real-World Context": "Shows the Superstar Effect: when a major pop star drops new tracks, their existing fanbase immediately pushes them into the Top 50."
        }
    ]
    st.table(pd.DataFrame(playbook_cases))

    st.markdown("---")
    st.subheader("2. Honest Takeaway for Emerging Artists & Songwriters")
    
    st.markdown(r"""<div style="background-color: #0F172A; padding: 22px; border-radius: 10px; border-left: 5px solid #10B981; margin-bottom: 20px;">
<h4 style="color: #10B981; margin-top: 0;">Why "No Lyric Formula" Is Actually Good News for Songwriters</h4>
<p style="font-size: 0.95rem; line-height: 1.65; margin-bottom: 0;">
Our data shows that <b>there is no magic word count, repetition percentage, or lyric topic that guarantees a Top 50 hit</b>. While that means data science cannot hand you a "cheat code" for writing hit lyrics, it also means <b>you do not need to restrict your creativity</b> to follow fake internet songwriting rules.
</p>
</div>""", unsafe_allow_html=True)

    guide_df = pd.DataFrame({
        "Common Songwriting Myth": [
            "\"You have to repeat your chorus lines 40%+ of the time to get a hit.\"",
            "\"Listeners today only want short lyrics and super simple vocabulary.\"",
            "\"If I copy the breakup topics of big pop stars, my song will chart higher.\"",
            "\"My song didn't chart high because the written lyrics weren't catchy enough.\""
        ],
        "What Our Data Actually Shows": [
            "Top 50 songs have a 31.8% median line repetition rate, compared to 33.3% for ranks 51–200 (no significant difference).",
            "Word count (~350 words) and vocabulary diversity (TTR ~0.37) are almost identical in both groups.",
            "Those breakup clusters only have high Top 50 rates because superstars like Olivia Rodrigo and Ariana Grande released many songs in them.",
            "Machine learning models using only lyrics score around 0.50 ROC-AUC (no better than a coin flip)."
        ],
        "Practical Advice for New Artists": [
            "Repeat your chorus as much as fits your melody naturally—don't force extra repetition.",
            "Write the story and vocabulary that fit your artistic style.",
            "Write themes that feel authentic to you rather than chasing a superstar's recent album topic.",
            "Spend your extra time on melody, vocal delivery, audio production, and short-form video promotion (TikTok/Reels)."
        ]
    })
    st.table(guide_df)

# ==============================================================================
# TAB 5: LIMITATIONS & DATA CLEANING NOTES
# ==============================================================================
with tabs[4]:
    st.header("Part 5: Project Limitations & Data Cleaning Notes")
    st.markdown(r"""
    Every data science project has dataset boundaries and limitations. Here are the main limitations of our study and how we cleaned the data:
    """)
    
    colL1, colL2 = st.columns(2)
    with colL1:
        st.subheader("1. Our Chart Data Covers a Fixed 1-Year Window")
        st.markdown(r"""
        * **What Happened**: Our target variable (`best_rank_this_year <= 50`) measures a song's highest rank between **September 2025 and September 2026**.
        * **Why It Matters**: Songs released in 2023 or early 2024 already had their biggest chart peak before our 1-year window started, so we only observe them as they slowly move down the chart. For example, pre-2024 songs in our dataset had a **45.1% all-time Top 50 rate**, but only **23.7%** during our 2025–2026 window. That is the main reason `Release Year` is the strongest predictor in our baseline model.
        """)
        
        st.subheader("2. Survivorship Bias (Only Top 200 Songs)")
        st.markdown(r"""
        * **What Happened**: All 582 songs in our dataset already succeeded in reaching the **Spotify Global Top 200**.
        * **Why It Matters**: Because we do not have unstreamed or non-charting songs in our dataset, our study only compares **Top 50 hits vs. Ranks 51–200**. We cannot test whether lyrics help an unknown song get into the Top 200 in the first place.
        """)

    with colL2:
        st.subheader("3. Missing Audio & Marketing Features")
        st.markdown(r"""
        * **What Happened**: Spotify deprecated its public Audio Features API (which previously provided tempo, danceability, and energy), and our dataset does not include marketing or social media metrics.
        * **Why It Matters**: In the real music market, **melody, beat, vocal performance, TikTok trends, playlist placement, and label promotion** play a huge role in pushing a song into the Top 50.
        """)
        
        st.subheader("4. Dataset Filtering Summary (885 → 582 Songs)")
        st.markdown(r"""
        * **885 unique songs** collected from 53 weekly Spotify Global Top 200 charts
        * **→ 709 English songs** (`language == 'en'`)
        * **→ 669 deduplicated songs** after merging duplicate album/single versions
        * **→ 596 songs** after removing initial `holiday` and `event_anthem` tags in EDA
        * **→ 582 final songs** after removing **14 leftover Christmas songs** discovered during BERTopic clustering
        """)

    st.markdown("---")
    st.subheader("Data Cleaning Note: How BERTopic Helped Us Catch 14 Leftover Christmas Songs")
    st.markdown(r"""
    Even though BERTopic did not improve Top 50 prediction, running topic clustering during exploratory analysis was very helpful for **data cleaning**.  
    In our first cleaning script, we filtered out Christmas songs by checking if `track_type == 'holiday'`, which left **596 songs** in our initial EDA notebook. However, when we ran BERTopic on those 596 songs, we noticed that one cluster grouped together **leftover Christmas songs** that had slipped through (for example, *Little Saint Nick - 1991 Remix* was tagged as `remix` instead of `holiday`, and *Fairytale of New York* was tagged as `Alternative` without `"Christmas"` in the title).  
    Inspecting that cluster allowed us to update our filtering rule and remove all 14 remaining holiday songs, giving us our clean **582-song dataset**:
    """)
    
    holiday_audit_df = pd.DataFrame({
        "Artist": [
            "Britney Spears", "Chuck Berry", "Burl Ives", "Dean Martin",
            "The Beach Boys", "The Pogues, Kirsty MacColl", "Leona Lewis", "Meghan Trainor", "Ella Fitzgerald"
        ],
        "Song Title": [
            "My Only Wish (This Year)", "Run Rudolph Run", "Rudolph The Red-Nosed Reindeer",
            "Rudolph The Red-Nosed Reindeer", "Little Saint Nick - 1991 Remix",
            "Fairytale of New York (feat. Kirsty MacColl)", "One More Sleep", "I'll Be Home", "Sleigh Ride"
        ],
        "Original Tag / Genre in CSV": [
            "standard / Christmas: Pop", "standard / Rock", "standard / Christmas", "standard / Christmas: Classic",
            "remix / Holiday", "standard / Alternative", "standard / Pop", "standard / Holiday", "standard / Holiday"
        ],
        "Why Our Initial Filter Missed It": [
            "track_type was 'standard'", "Genre was 'Rock', no 'Christmas' in title", "track_type was 'standard'", "track_type was 'standard'",
            "Tagged as 'remix' before holiday check", "Genre was 'Alternative', no 'Christmas' in title", "Genre was 'Pop', no 'Christmas' in title", "track_type was 'standard'", "track_type was 'standard'"
        ],
        "Christmas Words Found in the Lyrics": [
            "christmas, santa, sleigh, mistletoe", "christmas, santa", "christmas, santa, reindeer, sleigh",
            "christmas, santa, reindeer, sleigh", "christmas, santa, reindeer, saint nick",
            "christmas", "christmas, mistletoe", "christmas, santa", "sleigh"
        ]
    })
    st.table(holiday_audit_df)

# --- Footer ---
st.markdown("---")
st.markdown(r"""<div style="text-align: center; font-size: 0.85rem; color: #64748B;">
<b>Spotify Top 200 Lyrics Project</b> | 
582 English Charting Songs | 
EDA, BERTopic Clustering &amp; 5-Fold Artist-Grouped Logistic Regression
</div>""", unsafe_allow_html=True)
