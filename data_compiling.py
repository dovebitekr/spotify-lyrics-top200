import pandas as pd
import sqlite3
import os

def build_db():
    print("Building database structure...")
    with sqlite3.connect("lyrics.db") as conn:
        conn.execute("DROP TABLE IF EXISTS Songs")
        conn.execute("DROP TABLE IF EXISTS Lyrics")
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
                repetition_rate REAL
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
    csv_path = 'spotify_data/final_dataset_for_ml_enriched.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print("Loading enriched dataset into SQLite Database...")
    df = pd.read_csv(csv_path)
    
    with sqlite3.connect("lyrics.db") as conn:
        cursor = conn.cursor()
        
        for idx, row in df.iterrows():
            cursor.execute("""
                INSERT INTO Songs (artist, title, best_rank, all_time_peak_rank, all_time_weeks, total_streams, release_year, genre, language, word_count, repetition_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['artist_names'],
                row['track_name'],
                row.get('best_rank_this_year'),
                row.get('all_time_peak_rank'),
                row.get('all_time_weeks_on_chart'),
                row.get('total_streams_this_year'),
                row.get('release_year'),
                row.get('genre'),
                row.get('language'),
                row.get('word_count'),
                row.get('repetition_rate')
            ))
            
            song_id = cursor.lastrowid
            
            cursor.execute("""
                INSERT INTO Lyrics (song_id, lyrics)
                VALUES (?, ?)
            """, (song_id, row['lyrics']))
            
    print("Successfully populated lyrics.db with all tracks and lyrics!")

if __name__ == "__main__":
    build_db()
    load_data_to_db()
