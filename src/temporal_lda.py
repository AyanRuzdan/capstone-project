import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("./data", exist_ok=True)

def load_lda_data(lda_file='./data/lda_results.csv'):
    lda_df = pd.read_csv(lda_file)
    return lda_df

def analyze_temporal_trends(lda_df, output_path='./data/topic_trends_over_time.png'):
    lda_df['date'] = pd.to_datetime(lda_df['date'])
    lda_df['month'] = lda_df['date'].dt.to_period('M')
    temporal_data = lda_df.groupby(['month', 'topic']).size().unstack(fill_value=0)
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
    
    plt.savefig(output_path)
    plt.close()
    
    return temporal_data

def main():
    lda_df = load_lda_data()
    temporal_data = analyze_temporal_trends(lda_df)
    print("LDA Temporal Trend Data:")
    print(temporal_data)
    print(f"Temporal trend plot saved to ./data/topic_trends_over_time.png")

if __name__ == '__main__':
    main()