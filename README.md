# The Lyric Myth: Can Lyrics Predict Spotify Hits?

Does singing about heartbreak, partying, or flexing make a song more likely to become a Top 50 hit? Or does lyrical content matter far less than who is singing it?

This project investigates whether lyric-level features (thematic clusters, vocabulary richness, repetition, and sentiment) can predict whether a song reaches the Top 50 of the Spotify Global Weekly Top 200.

---

## Key Findings

- **Lyrics alone cannot predict hits:** Machine learning models trained strictly on lyric features (TF-IDF, sentence embeddings, sentiment, line repetition) achieved an average cross-validated ROC-AUC of **0.504**—essentially equivalent to a random guess (0.50).
- **The Superstar Leakage Effect:** Standard cross-validation gives a false sense of accuracy because models simply memorise prolific superstars (like Taylor Swift or Drake) across train and test sets. When evaluated using **5-Fold Artist-Grouped Cross-Validation**, the predictive power drops back to baseline.
- **Lyrical formulas are overrated:** Using BERTopic, we identified 24 distinct lyrical themes across 582 charting songs. However, over **21% of songs (123 tracks)** were categorized as unclustered outliers—and these formula-free tracks achieved a **34.1% Top 50 hit rate**, outperforming several repetitive pop tropes.

---

## Interactive Web Dashboard

An interactive dashboard built with **Streamlit** lets you explore the data and models:

- **Model Comparison:** Visualizing ROC-AUC scores across 4 machine learning baselines.
- **The 24 Lyrical Themes:** Browse each topic cluster with core keywords, hit rates, and real-world track examples.
- **Song Explorer:** A searchable table of all 582 songs with real-time filtering by artist, title, and theme.

### Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/dovebitekr/spotify-lyrics-top200.git
cd spotify-lyrics-top200

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## How it Was Built

1. **Data Pipeline:**
   - Collected weekly chart data from the Spotify Global Top 200 (2024–2025).
   - Fetched full lyrics via the Genius API and computed lexical metrics (Type-Token Ratio, repetition rates, sentiment scores).

2. **Topic Modeling:**
   - Used **BERTopic** with sentence embeddings (`all-MiniLM-L6-v2`), UMAP for dimensionality reduction, and HDBSCAN for density clustering.
   - Extracted representative keywords for each cluster using class-based TF-IDF (c-TF-IDF).

3. **Machine Learning & Validation:**
   - Tested L2-regularized Logistic Regression, Random Forest, and XGBoost to predict Top 50 status.
   - Implemented artist-grouped splitting to prevent data leakage and isolate the true effect of lyrics from artist popularity.

---

## Repository Structure

```text
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # App dependencies
├── EDA_spotify_lyrics.ipynb    # Exploratory data analysis notebook
├── Scripts/
│   ├── build_dataset.py        # Chart and lyric dataset compilation
│   ├── lyrics_cleaning_en.py   # Text preprocessing and feature extraction
│   └── m1_bertopic_model.py    # BERTopic modeling script
├── artifacts/
│   ├── bertopic_song_topics.csv    # Clustered songs with assigned themes
│   ├── bertopic_topic_metrics.csv  # Cluster metadata and keywords
│   └── topic_cluster_wordclouds.png
└── spotify_data/
    └── final_dataset_for_ml_lyrics_en.csv # Modeling dataset
```
