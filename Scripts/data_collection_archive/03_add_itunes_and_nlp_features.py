from pathlib import Path
import pandas as pd
import requests
import urllib.parse
import time
from langdetect import detect
from lingua import Language, LanguageDetectorBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "spotify_data"
INPUT_CSV = DATA_DIR / "final_dataset_for_ml.csv"
OUTPUT_CSV = DATA_DIR / "final_dataset_for_ml_enriched.csv"


# Language detector for lingua
detector = LanguageDetectorBuilder.from_all_spoken_languages().build()

def get_itunes_metadata(track, artist):
    try:
        term = urllib.parse.quote(f"{track} {artist.split(',')[0]}")
        url = f'https://itunes.apple.com/search?term={term}&entity=song&limit=1'
        res = requests.get(url, timeout=5).json()
        if res['resultCount'] > 0:
            item = res['results'][0]
            return item.get('releaseDate', '')[:4], item.get('primaryGenreName')
    except Exception:
        pass
    return None, None

def detect_lyrics_language(lyrics):
        # If lyrics are missing, language is also missing
        # Can't use NaN: 'nan' -> tagalog in language detectors
        if pd.isna(lyrics) or not str(lyrics).strip():
            return 'missing'
        
        # If language is detected, return it
        # otherwise unknown 
        detected = detector.detect_language_of(str(lyrics))
        if detected:
            return detected.iso_code_639_1.name.lower()
        return 'unknown'


def main():
    """
    [Phase 3] Fetch Genre/Year via iTunes API and generate NLP features (Language, Word Count, Repetition).
    """

    if not INPUT_CSV.exists():
            raise FileNotFoundError(f"Input .csv not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    years, genres, languages, word_counts, rep_rates = [], [], [], [], []

    df['language'] = df['lyrics'].apply(detect_lyrics_language)

    for idx, row in df.iterrows():
        year, genre = get_itunes_metadata(row['track_name'], row['artist_names'])
        years.append(year)
        genres.append(genre)

        # Here is the langdetect version
        ## try:
        ##     lang = detect(str(row['lyrics']))
        ## except Exception:
        ##     lang = 'unknown'
        ## languages.append(lang)
        
        words = str(row['lyrics']).lower().split()
        wc = len(words)
        word_counts.append(wc)
        rep_rates.append(round(len(set(words))/wc, 3) if wc > 0 else 0)
        time.sleep(0.3)
        
    df['release_year'], df['genre'] = years, genres
    df['word_count'], df['repetition_rate'] = word_counts, rep_rates
 
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()
