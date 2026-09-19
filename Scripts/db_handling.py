import sqlite3 as sq
import api_scripts as api

def build_db():
    with sq.connect("lyrics.db") as conn:
        conn.execute("""create table Songs (
            artist TEXT, title TEXT UNIQUE, peak_rank INTEGER, release_year INTEGER, genre TEXT, total_weeks INTEGER, peak_date""")

        conn.execute("""create table Lyrics (
            song_id PRIMARY KEY, lyrics TEXT""")

def add_song(artist, title, peak, year, genre, weeks, peak_date):
    with sq.connect("lyrics.db") as conn:
        conn.execute("""INSERT INTO Songs (
            artist, title, peak_rank, release_year, genre, total_weeks, peak_date TEXT) 
            VALUES(?,?,?,?,?,?)""",(artist, title, peak, year, genre, weeks, peak_date))

def add_lyrics_by_title(title, artist, id):
    lyrics = api.get_lyrics(title, artist)
    with sq.connect("lyrics.db") as conn:
        conn.execute("""INSERT INTO Lyrics (
            song_id, lyrics) VALUES(?,?)""",(id, lyrics))
