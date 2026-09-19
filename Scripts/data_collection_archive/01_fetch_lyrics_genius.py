import glob
import time
import pandas as pd
from dotenv import load_dotenv
import os
import lyricsgenius

def main():
    """
    [Phase 1] Extract unique tracks from 52 weeks of Spotify CSVs and fetch lyrics via Genius API.
    """
    load_dotenv()
    csv_files = glob.glob('spotify_data/*.csv')
    df_list = [pd.read_csv(f) for f in csv_files]
    all_df = pd.concat(df_list, ignore_index=True)
    
    unique_tracks = all_df.drop_duplicates(subset=['uri']).copy()
    unique_tracks = unique_tracks.dropna(subset=['track_name', 'artist_names'])
    
    genius = lyricsgenius.Genius(os.getenv("GENIUS_ACCESS_TOKEN"), timeout=15)
    genius.verbose = False
    
    lyrics_list = []
    for idx, row in unique_tracks.iterrows():
        track_name, artist_name = row['track_name'], row['artist_names'].split(',')[0]
        try:
            song = genius.search_song(track_name, artist_name)
            lyrics_list.append(song.lyrics if song and song.lyrics else None)
        except:
            lyrics_list.append(None)
        time.sleep(0.3)
        
    unique_tracks['lyrics'] = lyrics_list
    unique_tracks.to_csv('spotify_data/global_top200_lyrics.csv', index=False)

if __name__ == '__main__':
    main()
