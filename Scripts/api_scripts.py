import os
import csv
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()
USER = os.getenv("ID")
KEY = os.getenv("SECRET")
TOKEN = os.getenv("TOKEN")

def get_lyrics(artist, title):
    genius = lyricsgenius.Genius(TOKEN)
    song = genius.search_song(
        f"{title}",
        f"{artist}"
    )
    if song:
        return song.lyrics
    return None
