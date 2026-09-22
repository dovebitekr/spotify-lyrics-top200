import os
import re
import glob
import time
import urllib.parse
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
import lyricsgenius

# Detect language: prefers lingua, falls back to langdetect
try:
    from lingua import LanguageDetectorBuilder
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    def detect_language(text: str) -> str:
        """Detect language code using lingua library (returns 'missing' if empty)."""
        if pd.isna(text) or not str(text).strip():
            return "missing"
        detected = detector.detect_language_of(str(text))
        return detected.iso_code_639_1.name.lower() if detected else "unknown"
except ImportError:
    from langdetect import detect
    def detect_language(text: str) -> str:
        """Fallback language detector using langdetect."""
        if pd.isna(text) or not str(text).strip():
            return "missing"
        try:
            return detect(str(text))
        except Exception:
            return "unknown"


def clean_title(title: str) -> str:
    """Clean track title by stripping features, remasters, and soundtrack tags."""
    title = re.sub(r'\s*\(.*?(feat|with|from|remaster|version|soundtrack|deluxe|anniversary|live|prod).*?\)', '', str(title), flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*.*?(remaster|version|live|soundtrack|from|deluxe|anniversary|radio|edit|mono|stereo|album).*', '', str(title), flags=re.IGNORECASE)
    return title.strip()


def clean_artist(artist: str) -> str:
    """Extract the primary artist from a comma-separated artist string."""
    return str(artist).split(',')[0].strip()


def aggregate_weekly_charts(csv_pattern: str) -> pd.DataFrame:
    """Read all 53 weekly chart CSVs and aggregate unique songs with summary metrics."""
    files = glob.glob(csv_pattern)
    if not files:
        raise FileNotFoundError(f"No chart CSV files found matching: {csv_pattern}")

    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    grouped = df_all.groupby('uri').agg({
        'artist_names': 'first',
        'track_name': 'first',
        'rank': 'min',
        'peak_rank': 'min',
        'weeks_on_chart': 'max',
        'streams': 'sum'
    }).reset_index()

    grouped.rename(columns={
        'rank': 'best_rank_this_year',
        'peak_rank': 'all_time_peak_rank',
        'weeks_on_chart': 'all_time_weeks_on_chart',
        'streams': 'total_streams_this_year'
    }, inplace=True)

    return grouped


def fetch_metadata(track: str, artist: str, genius_client=None):
    """Fetch release year and genre. Returns None if not found (preserves missing values)."""
    c_title = clean_title(track)
    c_artist = clean_artist(artist)
    year = None
    genre = None

    # 1. Try iTunes Search API
    try:
        query = urllib.parse.quote(f"{c_title} {c_artist}")
        url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        if res.get("resultCount", 0) > 0:
            item = res["results"][0]
            r_date = item.get("releaseDate")
            year = int(r_date[:4]) if r_date else None
            genre = item.get("primaryGenreName")
    except Exception:
        pass

    # 2. Fallback to Genius API for release year
    if not year and genius_client:
        try:
            song = genius_client.search_song(c_title, c_artist)
            if song:
                components = song.to_dict().get("release_date_components")
                if components and components.get("year"):
                    year = int(components["year"])
        except Exception:
            pass

    return year, genre


def compute_nlp_metrics(lyrics: str):
    """Calculate word count, repetition rate, and detected language for lyrics."""
    if pd.isna(lyrics) or not str(lyrics).strip():
        return 0, 0.0, "missing"

    words = str(lyrics).lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    repetition_rate = round(unique_words / word_count, 3) if word_count > 0 else 0.0
    lang = detect_language(lyrics)
    return word_count, repetition_rate, lang


def determine_track_type(row) -> str:
    """Classify track into standard, holiday, remix, instrumental, or event_anthem."""
    title = str(row['track_name']).lower()
    genre = str(row.get('genre', '')).lower()
    wc = row.get('word_count', 0)

    if any(k in title for k in ['remix', 'rmx', 'sped up', 'speed up', 'slowed']):
        return 'remix'

    if wc <= 15:
        return 'instrumental'

    if any(k in title for k in ['waka waka', 'fifa', 'world cup']):
        return 'event_anthem'

    # Exception: V's 'Winter Ahead' is a standard pop ballad
    if 'winter ahead' in title:
        return 'standard'

    holiday_keywords = [
        'christmas', 'xmas', 'santa', 'jingle', 'holiday', 'carol', 
        'mistletoe', 'noel', 'navidad', 'snowman', 'let it snow', 
        'winter wonderland', 'frosty'
    ]
    if any(k in title for k in holiday_keywords) or any(k in genre for k in ['holiday', 'christmas']):
        return 'holiday'

    return 'standard'


def main():
    """Run full end-to-end data collection and enrichment pipeline."""
    load_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    chart_pattern = str(project_root / "spotify_data" / "regional-global-weekly-*.csv")
    output_csv = project_root / "spotify_data" / "final_dataset_for_ml_enriched.csv"

    # Step 1: Aggregate Spotify charts
    print("Step 1: Aggregating weekly chart records...")
    df = aggregate_weekly_charts(chart_pattern)
    print(f"Loaded {len(df)} unique songs.")

    # Step 2: Initialize Genius API client
    genius = lyricsgenius.Genius(
        os.getenv("GENIUS_ACCESS_TOKEN"),
        remove_section_headers=True,
        skip_non_songs=True,
        timeout=15,
        retries=3
    )
    genius.verbose = False

    # Step 3: Fetch lyrics with cleaned titles
    print("Step 2: Fetching lyrics from Genius API...")
    lyrics_list = []
    for idx, row in df.iterrows():
        c_title = clean_title(row['track_name'])
        c_artist = clean_artist(row['artist_names'])
        try:
            song = genius.search_song(c_title, c_artist)
            lyrics_list.append(song.lyrics if song and song.lyrics else None)
        except Exception:
            lyrics_list.append(None)
        time.sleep(0.2)
    df['lyrics'] = lyrics_list

    # Step 4: Extract metadata and compute NLP features
    print("Step 3: Extracting metadata and computing NLP features...")
    years, genres, word_counts, rep_rates, languages = [], [], [], [], []
    for idx, row in df.iterrows():
        year, genre = fetch_metadata(row['track_name'], row['artist_names'], genius)
        w_count, rep_rate, lang = compute_nlp_metrics(row['lyrics'])
        years.append(year)
        genres.append(genre)
        word_counts.append(w_count)
        rep_rates.append(rep_rate)
        languages.append(lang)
        time.sleep(0.1)

    df['release_year'] = years
    df['genre'] = genres
    df['word_count'] = word_counts
    df['repetition_rate'] = rep_rates
    df['language'] = languages

    # Step 5: Classify track types
    print("Step 4: Tagging track types...")
    df['track_type'] = df.apply(determine_track_type, axis=1)

    # Step 6: Save enriched dataset
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Success! Enriched dataset saved to: {output_csv}")


if __name__ == "__main__":
    main()
