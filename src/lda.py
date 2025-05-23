import json
import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

os.makedirs("./data", exist_ok=True)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

custom_stopwords = set([
    "said", "also", "new", "one", "two", "first", "last", "many", "could", "would", "get",
    "news", "report", "study", "researchers", "research", "according", "year", "years"
])
all_stopwords = list(stop_words.union(custom_stopwords))


def load_jsonl(file_path):
    articles = []
    with open(file_path, 'r') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles


def extract_text(articles):
    return [article['text'] for article in articles]


def perform_lda_topic_modeling(texts, num_topics=5, n_top_words=10, stop_words=None):
    if stop_words is None:
        stop_words = 'english'

    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_keywords[topic_idx] = top_words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

    return lda, vectorizer, topic_keywords


def save_lda_results(articles, lda_model, vectorizer, topic_keywords, save_path="./data/lda_results.csv"):
    texts = [article['text'] for article in articles]
    doc_term_matrix = vectorizer.transform(texts)

    document_topics = lda_model.transform(doc_term_matrix)
    assigned_topics = np.argmax(document_topics, axis=1)

    data = []
    for i, article in enumerate(articles):
        topic = assigned_topics[i]
        top_words = topic_keywords[topic]
        data.append({
            'article_index': i + 1,
            'title': article['title'],
            'date': article['date'],
            'topic': topic,
            'top_words': ', '.join(top_words)
        })

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"LDA results saved to {save_path}")


def main():
    articles = load_jsonl('./data/scraped_articles.jsonl')
    texts = extract_text(articles)

    lda_model, vectorizer, topic_keywords = perform_lda_topic_modeling(
        texts, num_topics=5, n_top_words=10, stop_words=all_stopwords
    )

    save_lda_results(
        articles, lda_model, vectorizer, topic_keywords, save_path="./data/lda_results.csv"
    )


if __name__ == '__main__':
    main()
