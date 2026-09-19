import pandas as pd
import requests
import urllib.parse
import time
from langdetect import detect

def get_itunes_metadata(track, artist):
    try:
        term = urllib.parse.quote(f"{track} {artist.split(',')[0]}")
        url = f'https://itunes.apple.com/search?term={term}&entity=song&limit=1'
        res = requests.get(url, timeout=5).json()
        if res['resultCount'] > 0:
            item = res['results'][0]
            return item.get('releaseDate', '')[:4], item.get('primaryGenreName')
    except:
        pass
    return None, None

def main():
    """
    [Phase 3] Fetch Genre/Year via iTunes API and generate NLP features (Language, Word Count, Repetition).
    """
    df = pd.read_csv('spotify_data/final_dataset_for_ml.csv')
    years, genres, languages, word_counts, rep_rates = [], [], [], [], []
    
    for idx, row in df.iterrows():
        year, genre = get_itunes_metadata(row['track_name'], row['artist_names'])
        years.append(year)
        genres.append(genre)
        
        try:
            lang = detect(str(row['lyrics']))
        except:
            lang = 'unknown'
        languages.append(lang)
        
        words = str(row['lyrics']).lower().split()
        wc = len(words)
        word_counts.append(wc)
        rep_rates.append(round(len(set(words))/wc, 3) if wc > 0 else 0)
        time.sleep(0.3)
        
    df['release_year'], df['genre'], df['language'] = years, genres, languages
    df['word_count'], df['repetition_rate'] = word_counts, rep_rates
    
    df.to_csv('spotify_data/final_dataset_for_ml_enriched.csv', index=False)

if __name__ == '__main__':
    main()
