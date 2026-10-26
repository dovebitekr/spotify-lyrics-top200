import sqlite3 as sq
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "lyrics.db"


def build_db():
    """Create relational Songs and Lyrics tables with the project schema."""
    with sq.connect(DB_PATH) as conn:
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


def add_song(artist, title, best_rank=None, all_time_peak_rank=None, all_time_weeks=None, 
             total_streams=None, release_year=None, genre=None, language=None, 
             word_count=None, lexical_diversity=None, track_type="standard"):
    """Insert a single song record and return its generated ID."""
    with sq.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Songs (
                artist, title, best_rank, all_time_peak_rank, all_time_weeks, 
                total_streams, release_year, genre, language, word_count, lexical_diversity, track_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            artist, title, best_rank, all_time_peak_rank, all_time_weeks,
            total_streams, release_year, genre, language, word_count, lexical_diversity, track_type
        ))
        return cursor.lastrowid


def add_lyrics(song_id, lyrics):
    """Insert lyrics for a specific song ID."""
    with sq.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO Lyrics (song_id, lyrics) VALUES (?, ?)", (song_id, lyrics))
