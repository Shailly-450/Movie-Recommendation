import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

# Load datasets
movies = pd.read_csv("dataset/movies.csv")  # MovieLens dataset
ratings = pd.read_csv("dataset/ratings.csv")

# Merge datasets for better recommendations
movies['genres'] = movies['genres'].fillna('')
movies['title'] = movies['title'].fillna('')
movie_data = movies.copy()

# TF-IDF Vectorizer for Content-Based Recommendations
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movie_data["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def get_recommendations(movie_name, top_n=5):
    """Recommend similar movies based on genres (Content-Based Filtering)."""
    if movie_name not in movie_data['title'].values:
        return ["Movie not found. Try another!"]
    
    idx = movie_data.index[movie_data['title'] == movie_name][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return movie_data['title'].iloc[movie_indices].tolist()

def analyze_sentiment(review_text):
    """Analyze sentiment of user review (Positive/Neutral/Negative)."""
    score = sia.polarity_scores(review_text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
