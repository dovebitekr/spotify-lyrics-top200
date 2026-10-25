import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "lyrics.db"
CSV_PATH = PROJECT_ROOT / "spotify_data" / "final_dataset_for_ml_enriched.csv"


def build_db():
    """Create empty relational tables (Songs and Lyrics) in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS Lyrics")
        conn.execute("DROP TABLE IF EXISTS Songs")

        conn.execute("""
            CREATE TABLE Songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist TEXT,
                title TEXT,
                best_rank INTEGER,
                all_time_peak_rank INTEGER,
                all_time_weeks INTEGER,
                total_streams INTEGER,
                release_year INTEGER,
                genre TEXT,
                language TEXT,
                word_count INTEGER,
                lexical_diversity REAL,
                track_type TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE Lyrics (
                song_id INTEGER PRIMARY KEY,
                lyrics TEXT,
                FOREIGN KEY(song_id) REFERENCES Songs(id)
            )
        """)


def load_data_to_db():
    """Read the enriched CSV dataset and populate the database tables."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO Songs (
                    artist, title, best_rank, all_time_peak_rank, 
                    all_time_weeks, total_streams, release_year, 
                    genre, language, word_count, lexical_diversity, track_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['artist_names'],
                row['track_name'],
                row.get('best_rank_this_year'),
                row.get('all_time_peak_rank'),
                row.get('all_time_weeks_on_chart'),
                row.get('total_streams_this_year'),
                int(row['release_year']) if pd.notna(row.get('release_year')) else None,
                row.get('genre') if pd.notna(row.get('genre')) else None,
                row.get('language'),
                row.get('word_count'),
                row.get('lexical_diversity', row.get('repetition_rate')),
                row.get('track_type', 'standard')
            ))

            song_id = cursor.lastrowid
            cursor.execute("INSERT INTO Lyrics (song_id, lyrics) VALUES (?, ?)", (song_id, row['lyrics']))

    print(f"Successfully loaded {len(df)} tracks into {DB_PATH.name}.")


if __name__ == "__main__":
    build_db()
    load_data_to_db()
