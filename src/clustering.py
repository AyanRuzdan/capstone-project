import json
import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from nltk.corpus import stopwords
import pandas as pd

os.makedirs("./data", exist_ok=True)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

custom_stopwords = set([
    "said", "also", "new", "one", "two", "first", "last", "many", "could", "would", "get",
    "news", "report", "study", "researchers", "research", "according", "year", "years"
])


def load_jsonl(file_path):
    articles = []
    with open(file_path, 'r') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles


def extract_text(articles):
    return [article['text'] for article in articles]


def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L12-v2", device='cpu')
    return model.encode(texts)


def perform_clustering(embeddings, num_clusters=12):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


def plot_elbow(embeddings, max_k=10, save_path='./data/elbow_plot.png'):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_keywords(texts, cluster_labels, n_keywords=5):
    cluster_keywords = {}
    for cluster in set(cluster_labels):
        cluster_texts = [text for text, label in zip(
            texts, cluster_labels) if label == cluster]
        all_words = ' '.join(cluster_texts).split()
        filtered_words = [
            word.lower() for word in all_words
            if word.lower() not in stop_words
            and word.lower() not in custom_stopwords
            and word.isalpha()
        ]
        common_words = Counter(filtered_words).most_common(n_keywords)
        cluster_keywords[cluster] = [word for word, _ in common_words]
    return cluster_keywords


def plot_clusters_with_labels(embeddings, kmeans, cluster_keywords, save_path="./data/cluster_plot.png"):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=kmeans.labels_,
        cmap='tab10',
        alpha=0.7
    )

    centroids_2d = pca.transform(kmeans.cluster_centers_)
    for i, coords in enumerate(centroids_2d):
        keywords_list = cluster_keywords.get(i, [])
        half = len(keywords_list) // 2 or 1
        keywords = ", ".join(
            keywords_list[:half]) + "\n" + ", ".join(keywords_list[half:])
        plt.text(
            coords[0], coords[1], keywords,
            fontsize=10, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8,
                      edgecolor='gray', boxstyle='round')
        )

    plt.title("2D Cluster Plot of News Articles with Topic Labels")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def save_cluster_data_for_analysis(articles, kmeans, cluster_keywords, silhouette_avg, db_index, save_path="./data/cluster_assignments.csv"):
    data = []
    for i, article in enumerate(articles):
        cluster = kmeans.labels_[i]
        keywords = ', '.join(cluster_keywords[cluster])
        date = article['date']
        data.append({
            'article_index': i + 1,
            'date': date,
            'cluster': cluster,
            'keywords': keywords,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_index': db_index
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Cluster data saved to {save_path}")


def main():
    articles = load_jsonl('./data/scraped_articles.jsonl')
    texts = extract_text(articles)
    embeddings = create_embeddings(texts)
    plot_elbow(embeddings, max_k=12, save_path='./data/elbow_plot.png')
    kmeans = perform_clustering(embeddings, num_clusters=5)
    silhouette_avg = silhouette_score(
        embeddings, kmeans.labels_, metric='euclidean')
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    db_index = davies_bouldin_score(embeddings, kmeans.labels_)
    print(f"Davies-Bouldin Index: {db_index:.2f}")
    cluster_keywords = extract_keywords(texts, kmeans.labels_, n_keywords=5)
    plot_clusters_with_labels(
        embeddings, kmeans, cluster_keywords, save_path='./data/cluster_plot.png')
    save_cluster_data_for_analysis(
        articles, kmeans, cluster_keywords, silhouette_avg, db_index, save_path='./data/cluster_assignments.csv')
    for i, article in enumerate(articles):
        cluster = kmeans.labels_[i]
        keywords = ', '.join(cluster_keywords[cluster])
        print(f"Article {i+1:04}: Cluster {cluster} Keywords: {keywords}")


if __name__ == '__main__':
    main()
