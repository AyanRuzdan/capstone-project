import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("./data", exist_ok=True)

def load_lda_data(lda_file='./data/lda_results.csv'):
    """
    Load LDA topic assignments from CSV file.
    
    Parameters:
    - lda_file: Path to LDA results CSV.
    
    Returns:
    - lda_df: DataFrame with LDA data.
    """
    lda_df = pd.read_csv(lda_file)
    return lda_df

def analyze_temporal_trends(lda_df, output_path='./data/topic_trends_over_time.png'):
    """
    Analyze and plot temporal trends of LDA topic assignments.
    
    Parameters:
    - lda_df: DataFrame with LDA topic assignments (date, topic).
    - output_path: Path to save the trend plot.
    
    Returns:
    - temporal_data: DataFrame with topic counts by month.
    """
    # Convert date to datetime and extract month
    lda_df['date'] = pd.to_datetime(lda_df['date'])
    lda_df['month'] = lda_df['date'].dt.to_period('M')
    
    # Group by month and topic, count articles
    temporal_data = lda_df.groupby(['month', 'topic']).size().unstack(fill_value=0)
    
    # Plot line graph
    plt.figure(figsize=(10, 6))
    for topic in temporal_data.columns:
        plt.plot(temporal_data.index.astype(str), temporal_data[topic], marker='o', label=f'Topic {topic}')
    
    plt.title('Temporal Trend of LDA Topic Assignments Over Time')
    plt.xlabel('Month')
    plt.ylabel('Number of Articles')
    plt.legend(title='Topics')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    
    return temporal_data

def main():
    # Load LDA data
    lda_df = load_lda_data()
    
    # Analyze and plot temporal trends
    temporal_data = analyze_temporal_trends(lda_df)
    
    # Print results
    print("LDA Temporal Trend Data:")
    print(temporal_data)
    print(f"Temporal trend plot saved to ./data/topic_trends_over_time.png")

if __name__ == '__main__':
    main()