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

try:
    from lingua import LanguageDetectorBuilder
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    def detect_language(text: str) -> str:
        if pd.isna(text) or not str(text).strip() or str(text).strip().lower() == 'nan':
            return "missing"
        clean = re.sub(r'(?i)\b(la|fa|ha|da|ba|sha|ooh|woah|yeah|uh)[\s\-,.!?]+', ' ', str(text))
        detected = detector.detect_language_of(clean)
        return detected.iso_code_639_1.name.lower() if detected else "unknown"
except ImportError:
    from langdetect import detect
    def detect_language(text: str) -> str:
        """Fallback language detector using langdetect."""
        if pd.isna(text) or not str(text).strip() or str(text).strip().lower() == 'nan':
            return "missing"
        clean = re.sub(r'(?i)\b(la|fa|ha|da|ba|sha|ooh|woah|yeah|uh)[\s\-,.!?]+', ' ', str(text))
        try:
            return detect(clean)
        except Exception:
            return "unknown"


def clean_title(title: str) -> str:
    title = re.sub(r'\s*\(.*?(feat|with|from|remaster|version|soundtrack|deluxe|anniversary|live|prod|explicit).*?\)', '', str(title), flags=re.IGNORECASE)
    title = re.sub(r'\s*-\s*.*?(remaster|version|live|soundtrack|from|deluxe|anniversary|radio|edit|mono|stereo|album|bonus|spotify singles).*', '', str(title), flags=re.IGNORECASE)
    return title.strip().rstrip('.')


def clean_artist(artist: str) -> str:
    return str(artist).split(',')[0].strip()


def aggregate_weekly_charts(csv_pattern: str) -> pd.DataFrame:
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
    c_title = clean_title(track)
    c_artist = clean_artist(artist)
    year = None
    genre = None

    # Try iTunes Search API (retry up to 3 times on rate limit)
    query = urllib.parse.quote(f"{c_title} {c_artist}")
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            data = res.json()
            if data.get("resultCount", 0) > 0:
                item = data["results"][0]
                r_date = item.get("releaseDate")
                year = int(r_date[:4]) if r_date else None
                genre = item.get("primaryGenreName")
            break
        except Exception:
            time.sleep(2)

    # Fallback to Genius API for release year
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


def fetch_lyrics(title: str, artist: str, genius_client) -> str:
    c_title = clean_title(title)
    c_artist = clean_artist(artist)
    try:
        res = genius_client.search_songs(f"{c_title} {c_artist}")
        if res and res.get('hits'):
            for hit in res['hits'][:5]:
                s = hit['result']
                p_name = s['primary_artist']['name'].lower()
                if not any(k in p_name for k in ['translation', 'çeviri', 'tradu', 'traduction', 'übersetzung']):
                    song = genius_client.search_song(s['title'], s['primary_artist']['name'])
                    if song and song.lyrics:
                        return song.lyrics
        song = genius_client.search_song(c_title, c_artist)
        return song.lyrics if song and song.lyrics else None
    except Exception:
        return None


def compute_nlp_metrics(lyrics: str):
    """Calculate word count, lexical diversity (Type-Token Ratio), and detected language."""
    if pd.isna(lyrics) or not str(lyrics).strip() or str(lyrics).strip().lower() == 'nan':
        return 0, 0.0, "missing"

    words = str(lyrics).lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    lexical_diversity = round(unique_words / word_count, 3) if word_count > 0 else 0.0
    lang = detect_language(lyrics)
    return word_count, lexical_diversity, lang


def determine_track_type(row) -> str:
    title = str(row['track_name']).lower()
    genre = str(row.get('genre', '')).lower()
    wc = row.get('word_count', 0)

    if any(k in title for k in ['remix', 'rmx', 'sped up', 'speed up', 'slowed']):
        return 'remix'

    if any(k in title for k in ['instrumental', 'orchestral', 'karaoke']) or 'no. 29' in title or 'montagem' in title:
        return 'instrumental'

    if any(k in title for k in ['waka waka', 'fifa', 'world cup']):
        return 'event_anthem'

    if 'winter ahead' in title:
        return 'standard'

    # Use whole word boundary \b so words like 'Santana' won't match 'santa'
    holiday_regex = r'\b(christmas|xmas|santa|jingle|holiday|carol|mistletoe|noel|navidad|snowman|frosty)\b|let it snow|winter wonderland'
    if re.search(holiday_regex, title) or any(k in genre for k in ['holiday', 'christmas']):
        return 'holiday'

    return 'standard'


def main():
    load_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    chart_pattern = str(project_root / "spotify_data" / "regional-global-weekly-*.csv")
    output_csv = project_root / "spotify_data" / "final_dataset_for_ml_enriched.csv"

    # Step 1: Aggregate Spotify charts
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

    # Fetch lyrics with cleaned titles and translation filtering
    lyrics_list = []
    for idx, row in df.iterrows():
        lyrics = fetch_lyrics(row['track_name'], row['artist_names'], genius)
        lyrics_list.append(lyrics)
        time.sleep(0.15)
    df['lyrics'] = lyrics_list

    # Extract metadata and compute NLP features
    years, genres, word_counts, rep_rates, languages = [], [], [], [], []
    for idx, row in df.iterrows():
        year, genre = fetch_metadata(row['track_name'], row['artist_names'], genius)
        w_count, rep_rate, lang = compute_nlp_metrics(row['lyrics'])
        years.append(year)
        genres.append(genre)
        word_counts.append(w_count)
        rep_rates.append(rep_rate)
        languages.append(lang)
        time.sleep(0.5)

    df['release_year'] = pd.Series(years, dtype='Int64')
    df['genre'] = genres
    df['word_count'] = word_counts
    df['lexical_diversity'] = rep_rates
    df['language'] = languages

    # Classify track types
    df['track_type'] = df.apply(determine_track_type, axis=1)

    #  Save enriched dataset
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Success! Enriched dataset saved to: {output_csv}")


if __name__ == "__main__":
    main()
