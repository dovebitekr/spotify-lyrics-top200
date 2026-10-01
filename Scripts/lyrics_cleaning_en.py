import re
from pathlib import Path
from typing import Optional, Set, Tuple, List
import pandas as pd
from nltk.corpus import stopwords

# Data paths relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "spotify_data"
INPUT_CSV = PROJECT_ROOT / "spotify_data" / "final_dataset_for_ml_enriched.csv"
OUTPUT_CSV = PROJECT_ROOT / "spotify_data" / "data_cleaned_lyrics_en.csv"

# Do you have more ideas?
# Or is there something in the set we want to keep in word tokens? 
CUSTOM_STOPWORDS: Set[str] = {
    'oh', 'yeah', 'ooh', 'la', 'da', 'ah', 'uh', 'woah', 'hey', 'ayy', 'na', 'ha', 'ya', 'yo', 
    'chorus', 'verse', 'intro', 'outro', 'bridge', 'hook', 'pre-chorus','embed', 'contributor', 'contributors'
}

# NLTK stopwords and custom stop words
NLTK_STOPWORDS: Set[str] = set(stopwords.words('english'))
STOPWORDS: Set[str] = NLTK_STOPWORDS | CUSTOM_STOPWORDS


def clean_english_lyrics(df, min_word_len):
    df_out = df[df["language"] == "en"].copy()

    df_out["cleaned_lyrics"] = (
        df_out["lyrics"]
        # square brackets and new lines
        .str.replace(r"(?s)\[.*?\]", " ", regex=True)
        # Genius structure
        .str.replace(r"\(.*?(chorus|verse|outro|bridge).*?\)", " ", regex=True, flags=re.IGNORECASE)
        .str.replace(r"\d*\s*contributors?", " ", regex=True, flags=re.IGNORECASE)
        .str.replace(r"\bembed\b", " ", regex=True, flags=re.IGNORECASE)
        # lowercase and special characters
        .str.lower()
        .str.replace(r"[^\w\s]+", " ", regex=True)
        # empty space and new lines
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Stop words
    df_out["cleaned_lyrics"] = df_out["cleaned_lyrics"].apply(lambda lyrics : " ".join(
        [w for w in str(lyrics).split() if w not in STOPWORDS and len(w) >= min_word_len]))

    # TODO: 
    # lemmatisation
    # other lyrical issues like spoken versions of words ain't -> am not etc
    
    print(df_out[["lyrics", "cleaned_lyrics"]])
    return df_out

if __name__ == "__main__":

    # using a test dummy not to destroy first one
    df = pd.read_csv(INPUT_CSV)

    # Cleaning english only lyrics into words
    # min length of word is >= 2
    min_word_len = 2
    df_out = clean_english_lyrics(df, min_word_len)
    # print(df_out.columns)

    # write to csv
    df_out.to_csv(OUTPUT_CSV, index=False)

