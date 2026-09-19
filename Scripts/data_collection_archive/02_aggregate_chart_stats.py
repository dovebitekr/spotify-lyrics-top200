import pandas as pd
import glob

def main():
    """
    [Phase 2] Parse 52 weeks of chart data to calculate true 1-year historical performance metrics.
    """
    csv_files = glob.glob('spotify_data/regional-global-weekly-*.csv')
    all_charts = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    grouped = all_charts.groupby('uri').agg({
        'rank': 'min',              # Best rank in this 1-year period
        'peak_rank': 'min',         # Overall historical peak rank
        'weeks_on_chart': 'max',    # Total weeks on chart up to the latest date
        'streams': 'sum'            # Total streams accumulated in this 1-year period
    }).reset_index()
    
    grouped.rename(columns={
        'rank': 'best_rank_this_year',
        'peak_rank': 'all_time_peak_rank',
        'weeks_on_chart': 'all_time_weeks_on_chart',
        'streams': 'total_streams_this_year'
    }, inplace=True)
    
    lyrics_df = pd.read_csv('spotify_data/global_top200_lyrics.csv')
    final_df = pd.merge(lyrics_df, grouped, on='uri', how='left')
    final_df.to_csv('spotify_data/final_dataset_for_ml.csv', index=False)

if __name__ == "__main__":
    main()
