import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import os

os.makedirs("./data", exist_ok=True)


def load_data(kmeans_file='./data/cluster_assignments.csv', lda_file='./data/lda_results.csv'):
    kmeans_df = pd.read_csv(kmeans_file)
    lda_df = pd.read_csv(lda_file)
    return kmeans_df, lda_df


def compute_lda_coherence(lda_df, texts):
    topic_words = [row['top_words'].split(
        ', ') for _, row in lda_df.drop_duplicates('topic').iterrows()]

    dictionary = Dictionary(texts)
    coherence_model = CoherenceModel(
        topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def generate_wordclouds(lda_df, output_dir='./data'):
    for topic in lda_df['topic'].unique():
        words = lda_df[lda_df['topic'] ==
                       topic]['top_words'].iloc[0].split(', ')
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate(' '.join(words))
        wordcloud.to_file(f'{output_dir}/wordcloud_topic_{topic}.png')


def plot_metrics(kmeans_silhouette, kmeans_db, lda_coherence, output_path='./data/metrics_comparison.png'):
    metrics = {
        'K-means Silhouette': kmeans_silhouette,
        'K-means Davies-Bouldin': kmeans_db,
        'LDA Coherence (C_v)': lda_coherence
    }

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values(),
                   color=['blue', 'orange', 'green'])
    plt.title('Clustering Quality Metrics Comparison')
    plt.ylabel('Score')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    kmeans_df, lda_df = load_data()

    kmeans_silhouette = kmeans_df['silhouette_score'].iloc[0]
    kmeans_db = kmeans_df['davies_bouldin_index'].iloc[0]
    print(f"K-means Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"K-means Davies-Bouldin Index: {kmeans_db:.4f}")

    texts = [row['title'].split() for _, row in lda_df.iterrows()]
    lda_coherence = compute_lda_coherence(lda_df, texts)
    print(f"LDA Topic Coherence (C_v): {lda_coherence:.4f}")

    generate_wordclouds(lda_df)
    print("Word clouds generated for LDA topics in ./data/")

    plot_metrics(kmeans_silhouette, kmeans_db, lda_coherence)
    print("Metrics comparison plot saved to ./data/metrics_comparison.png")


if __name__ == '__main__':
    main()
