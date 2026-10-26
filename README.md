# Spotify Top 200 Lyric Analysis: Guidance for Aspiring Songwriters

When aspiring songwriters sit down to write their next track, they often wonder: **"What should I write about?"** Should you follow proven pop themes like nightlife and heartbreak, or is there room for unconventional, niche storytelling?

This project explores one year of the Spotify Global Weekly Top 200 charts combined with Genius lyrics to provide data-driven insights for songwriters, music producers, and music analysts.

---

## Project Motivation & Target Audience

* **Target Audience:** Aspiring songwriters, independent artists, and music producers aiming for their first entry into major streaming charts.
* **Core Question:** Do specific lyrical themes, vocabulary diversity, or repetition patterns give a song a measurable advantage in breaking from the Top 200 into the **Top 50**?
* **Original Vision & Scope:** We initially set out to explore whether seasonal variations affect which lyrical themes dominate the charts (e.g., whether melancholic ballads perform better in autumn). Because one year of chart data is not sufficient to reliably isolate seasonal trends, we refined our focus to examine whether **lyrical embeddings and topic clusters correlate with higher chart performance (Top 50 status)**.

---

## Key Takeaways for Songwriters

Rather than treating lyrics as a magic formula for virality, our findings offer practical, scientifically grounded guidance:

1. **No Thematic "Silver Bullet" (Beware of Surface Illusions):**
   Surface-level averages can easily deceive. For example, a small cluster like *Moody Alt-Pop* (themes of darkness, dreams, and misery) initially seemed to boast a high average chart rank. However, this was simply driven by a few superstar hits in a small sample. When rigorously evaluated with machine learning, adding lyrical topic embeddings provided **zero predictive improvement (ROC-AUC ~0.504)** over baseline models. There is no specific lyrical subject that mechanically guarantees chart success.

2. **Freedom from Formulaic Pop Tropes:**
   While party anthems and breakup tropes take up large portions of the charts, over **21% of all charting songs (123 tracks)** formed unclustered outliers. These formula-free, unique songs achieved a **34.1% Top 50 hit rate**. Aspiring writers do not need to force themselves into generic pop clichés—unconventional storytelling performs just as well as standard tropes.

3. **Lyrics Do Not Exist in a Vacuum:**
   Because machine learning models cannot predict Top 50 placement from lyrics alone, success is fundamentally driven by unmeasured musical factors (production quality, melodic hooks, vocal delivery, tempo) and artist reach. For songwriters, lyrical authenticity matters more than chasing "viral keywords."

---

## The Interactive Web Dashboard

To make these 582 songs and 24 lyrical themes genuinely usable for creators and researchers, we built an interactive dashboard with **Streamlit**:

* **Model Comparison & Baselines:** Inspect the ROC-AUC evaluation across models and see why artist-grouped cross-validation is essential to avoid data leakage.
* **The 24 Lyrical Themes:** Browse each cluster's core keywords (c-TF-IDF tags), average peak ranks, and Top 50 hit rates.
* **Interactive Song Explorer:** Filter and search all 582 charting tracks by artist, song title, and lyrical theme to study real-world songwriting examples.

### Running Locally

```bash
# 1. Clone this repository
git clone https://github.com/dovebitekr/spotify-lyrics-top200.git
cd spotify-lyrics-top200

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

---

## Data Pipeline & Methodology

### 1. Data Collection
* **Spotify Charts:** 52 weekly Global Top 200 chart CSVs (September 2025 – September 2026) collected via charts.spotify.com, capturing chart rank, track URI, peak position, and weeks on chart.
* **Genius API:** Retrieved full song lyrics via `lyricsgenius`.
* **iTunes Search API:** Supplemented metadata with release year and primary genre.

### 2. Preprocessing & Quality Control
* **Exclusions:** Filtered out instrumental tracks, karaoke versions, and holiday/Christmas songs (using regex matching and genre metadata).
* **Fan-Translation Filtering:** Identified and resolved foreign fan-translations returned by the Genius API by stripping "feat." tags and blacklisting translation keywords.
* **Language Detection:** Used `lingua` (with `langdetect` as a fallback) to reliably identify English lyrics while handling colloquial slang and informal text.
* **Three Text Representations:** 
  * *Cleaned:* Structural headers removed (e.g. `[Chorus]`, `[Bridge]`).
  * *Expanded:* Contractions expanded (`don't` $\rightarrow$ `do not`) and custom informal slang normalized.
  * *Lemmatized:* Lowercased, stripped of custom stopwords, and lemmatized via `spaCy`.

### 3. Clustering & Machine Learning
* **Topic Modeling (BERTopic):** Generated dense sentence embeddings using `all-MiniLM-L6-v2`, reduced dimensionality with UMAP, and clustered lyrics with HDBSCAN. Extracted cluster keywords via class-based TF-IDF (c-TF-IDF).
* **Predictive Modeling:** Built baseline (M1) and topic-augmented (M2) logistic regression classifiers to predict Top 50 placement. Evaluated models using **5-Fold Artist-Grouped Cross-Validation** to prevent artist memorization and text embedding overfitting.

---

## Limitations & Future Work

* **Selection Bias:** By sampling exclusively from the Spotify Top 200, the dataset only contains songs that have already achieved global commercial success. A future extension would include a non-charting control group.
* **Legacy Tracks vs. New Releases:** Pre-2024 legacy hits had a 45.1% Top 50 rate compared to 23.7% for newer releases, reflecting persistent streaming longevity rather than purely modern lyrical trends.
* **Unmeasured Factors:** Popularity is heavily influenced by musical components (tempo, vocal delivery, production) and promotional reach, which are beyond pure lyric text.

---

## Repository Structure

```text
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Python dependencies
├── EDA_spotify_lyrics.ipynb    # Exploratory Data Analysis notebook
├── Scripts/
│   ├── build_dataset.py        # Chart and lyric dataset compilation
│   ├── lyrics_cleaning_en.py   # Text preprocessing and NLP metric calculation
│   └── m1_bertopic_model.py    # BERTopic training and artifact generation
├── artifacts/
│   ├── bertopic_song_topics.csv    # Clustered song dataset with assigned topics
│   ├── bertopic_topic_metrics.csv  # Topic metadata, hit rates, and keywords
│   └── topic_cluster_wordclouds.png
└── spotify_data/
    └── final_dataset_for_ml_lyrics_en.csv # Cleaned modeling dataset
```
