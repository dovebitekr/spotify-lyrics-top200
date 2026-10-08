import re
from pathlib import Path
from typing import Set
import pandas as pd
from nltk.corpus import stopwords
import spacy
import contractions

# Data paths relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "spotify_data"
INPUT_CSV = PROJECT_ROOT / "spotify_data" / "final_dataset_for_ml_enriched.csv"
OUTPUT_CSV = PROJECT_ROOT / "spotify_data" / "final_dataset_for_ml_lyrics_en.csv"

# Do you have more ideas?
# Or is there something in the set we want to keep in word tokens? Like "bridge"?
CUSTOM_STOPWORDS: Set[str] = {
    'oh', 'yeah', 'ooh', 'la', 'da', 'ah', 'uh', 'woah', 'hey', 'ayy', 'na', 'ha', 'ya', 'yo', 
    'chorus', 'verse', 'intro', 'outro', 'bridge', 'hook', 'pre-chorus','embed', 'contributor', 'contributors'
}

# NLTK stopwords and custom stop words
NLTK_STOPWORDS: Set[str] = set(stopwords.words('english'))
STOPWORDS: Set[str] = NLTK_STOPWORDS | CUSTOM_STOPWORDS

# slang words not in contractions library
SLANG_MAP = {
    r"\bdunno\b": "do not know",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\btryna\b": "trying to",
    r"\bfinna\b": "going to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\boutta\b": "out of",
    r"\bgimme\b": "give me",
    r"\blemme\b": "let me",
    r"\bcoulda\b": "could have",
    r"\bshoulda\b": "should have",
    r"\bwoulda\b": "would have",
    r"\bain'?t\b": "am not",
    r"\blil\b": "little",
    r"(?<!\w)'cause\b|\bcause\b|\bcoz\b": "because",
    r"(?<!\w)'til\b|\btill\b": "until",
    r"(?<!\w)'bout\b": "about",
    r"(?<!\w)'em\b": "them",
    r"\bima\b|\bi'mma\b": "i am going to",
    r"\b([a-zA-Z]{2,})in'(?!\w)": r"\1ing"
}

# Spacy lemmatizer
# "parser" and "ner" are heavy analytic tools not needed for lemmatizing
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def expand_slang_and_contractions(text):
    for pattern, repl in SLANG_MAP.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = contractions.fix(text)
    return text

def normalize_unicode(text: str) -> str:
    """Normalize common Unicode whitespace, zero-width characters, and quotation marks."""
    text = str(text).replace('\xa0', ' ').replace('\u200b', '').replace('\ufeff', '')
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return text.replace('…', '...')


def remove_genius_structure(series):
    output_series = (
        series
        .astype(str)
        .apply(normalize_unicode)
        # square brackets and section headers
        .str.replace(r"(?s)\[.*?\]", " ", regex=True)
        # Genius structure tags and basic brackets
        .str.replace(r"\(.*?(chorus|verse|outro|bridge).*?\)", " ", regex=True, flags=re.IGNORECASE)
        .str.replace(r"\d*\s*contributors?", " ", regex=True, flags=re.IGNORECASE)
        .str.replace(r"\bembed\b", " ", regex=True, flags=re.IGNORECASE)
        .str.replace(r"[ \t]+", " ", regex=True)
        .str.replace(r"\n\s*\n+", "\n\n", regex=True)
        .str.strip()
    )
    return output_series

def clean_english_lyrics(df, min_word_len=2):
    df_out = df[df["language"] == "en"].copy()

    # Backup raw metrics before updating
    df_out["raw_word_count"] = df_out["word_count"]
    df_out["raw_lexical_diversity"] = df_out.get("lexical_diversity", df_out.get("repetition_rate"))

    # 1. Raw data for transformer: Genius structure cleaning before lemmatization
    df_out["cleaned_lyrics_transformer"] = remove_genius_structure(df_out["lyrics"])

    # 2. Expanded lyrics: slang & contractions expansion
    expanded_lyrics = df_out["cleaned_lyrics_transformer"].apply(expand_slang_and_contractions)

    # 3. Lemmatization from expanded_lyrics:
    docs = nlp.pipe(expanded_lyrics, batch_size=50)

    cleaned_lemmas = []
    for doc in docs:
        tokens = [
            token.lemma_.lower() for token in doc 
            if token.is_alpha
            and token.is_ascii
            and token.lemma_.lower() not in STOPWORDS 
            and len(token.lemma_) >= min_word_len
        ]
        cleaned_lemmas.append(" ".join(tokens))
    df_out["lemmas"] = cleaned_lemmas

    # For expanded lyrics, special characters are taken out and words in lower case
    df_out["expanded_lyrics"] = (
            expanded_lyrics
            .str.lower()
            .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # 4. Update word_count and lexical_diversity based on cleaned transformer lyrics
    clean_words = df_out["cleaned_lyrics_transformer"].str.lower().str.split()
    df_out["word_count"] = clean_words.apply(len)
    df_out["lexical_diversity"] = clean_words.apply(
        lambda w: round(len(set(w)) / len(w), 3) if len(w) > 0 else 0.0
    )

    # 5. Additional lemma-level vocabulary metrics
    lemma_words = df_out["lemmas"].str.split()
    df_out["lemma_word_count"] = lemma_words.apply(len)
    df_out["lemma_lexical_diversity"] = lemma_words.apply(
        lambda w: round(len(set(w)) / len(w), 3) if len(w) > 0 else 0.0
    )

    return df_out

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    df_out = clean_english_lyrics(df, min_word_len=2)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully processed {len(df_out)} English tracks and saved to: {OUTPUT_CSV.name}")

