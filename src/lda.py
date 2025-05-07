import json
import os
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Custom stopwords to remove generic news terms
custom_stopwords = set([
    "said", "also", "new", "one", "two", "first", "last", "many", "could", "would", "get",
    "news", "report", "study", "researchers", "research", "according", "year", "years"
])
all_stopwords = list(stop_words.union(custom_stopwords))

# Load JSONL data
def load_jsonl(file_path):
    articles = []
    with open(file_path, 'r') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles

# Extract the 'text' field from each article
def extract_text(articles):
    return [article['text'] for article in articles]

# Perform LDA topic modeling
def perform_lda_topic_modeling(texts, num_topics=5, n_top_words=10, stop_words=None):
    if stop_words is None:
        stop_words = 'english'
    
    # Create and fit CountVectorizer
    vectorizer = CountVectorizer(stop_words=stop_words, max_df=0.95, min_df=2)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Initialize and fit LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Extract top keywords for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_keywords[topic_idx] = top_words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    return lda, vectorizer, topic_keywords

# Save LDA results to CSV
def save_lda_results(articles, lda_model, vectorizer, topic_keywords, save_path="./data/lda_results.csv"):
    # Transform texts to document-term matrix
    texts = [article['text'] for article in articles]
    doc_term_matrix = vectorizer.transform(texts)
    
    # Get topic distributions and assign topics
    document_topics = lda_model.transform(doc_term_matrix)
    assigned_topics = np.argmax(document_topics, axis=1)
    
    # Prepare data for CSV
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
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"LDA results saved to {save_path}")

# Main workflow
def main():
    # Load and process data
    articles = load_jsonl('./data/scraped_articles.jsonl')
    texts = extract_text(articles)
    
    # Perform LDA topic modeling
    lda_model, vectorizer, topic_keywords = perform_lda_topic_modeling(
        texts, num_topics=5, n_top_words=10, stop_words=all_stopwords
    )
    
    # Save results
    save_lda_results(
        articles, lda_model, vectorizer, topic_keywords, save_path="./data/lda_results.csv"
    )

# Run the script
if __name__ == '__main__':
    main()