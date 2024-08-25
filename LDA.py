import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd

# Data Collection
newsgroups = fetch_20newsgroups(subset='all', categories=['rec.autos', 'rec.sport.baseball', 'comp.graphics', 'sci.med'], shuffle=True, random_state=1)
documents = newsgroups.data

# Text Preprocessing
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(documents)

# Topic Modeling with LDA
lda_model = LatentDirichletAllocation(n_components=4, random_state=0)
lda_topics = lda_model.fit_transform(X)

# Display topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

topics = display_topics(lda_model, vectorizer.get_feature_names_out(), 10)

# Plot LDA Topics
def plot_lda_topics(lda_model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(lda_model.components_):
        plt.figure(figsize=(10, 5))
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        top_word_values = [topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        sns.barplot(x=top_word_values, y=top_words)
        plt.title(f"Topic {topic_idx + 1}")
        plt.xlabel('Word Weight')
        plt.ylabel('Words')
        plt.show()

plot_lda_topics(lda_model, vectorizer.get_feature_names_out(), 10)

# Document Similarity
doc_topic_dist = normalize(lda_topics, norm='l1')  # Normalize topic distributions
similarity_matrix = cosine_similarity(doc_topic_dist)

# Plot Document Similarity Matrix
def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', square=True, cbar=True, annot=False)
    plt.title('Document Similarity Matrix')
    plt.xlabel('Document')
    plt.ylabel('Document')
    plt.show()

plot_similarity_matrix(similarity_matrix)

# Print topic keywords for reference
print("Discovered Topics:")
for i, topic in enumerate(topics):
    print(f"Topic {i + 1}: {topic}")
