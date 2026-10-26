import os
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()
TOKEN = os.getenv("GENIUS_ACCESS_TOKEN") or os.getenv("TOKEN")


def get_lyrics(title: str, artist: str):
    """Fetch lyrics from Genius API for a given title and artist."""
    if not TOKEN:
        raise ValueError("GENIUS_ACCESS_TOKEN not found in environment.")

    genius = lyricsgenius.Genius(TOKEN, timeout=10, retries=2)
    genius.verbose = False
    song = genius.search_song(title, artist)
    return song.lyrics if song else None
