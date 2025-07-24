import numpy as np
import pandas as pd
import re
import nltk
import string
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Text feature extraction functions
def count_mentions(text):
    """Count @mentions in text."""
    mention_pattern = re.compile(r'@\w+')
    return len(mention_pattern.findall(text))

def count_hashtags(text):
    """Count #hashtags in text."""
    hashtag_pattern = re.compile(r'#\w+')
    return len(hashtag_pattern.findall(text))

def count_punctuation(text):
    """Count punctuation marks in text."""
    return sum(1 for char in text if char in string.punctuation)

def count_capital_letters(text):
    """Count capital letters in text."""
    return sum(1 for c in text if c.isupper())

def text_length(text):
    """Calculate text length."""
    return len(text)

def word_count(text):
    """Calculate word count."""
    return len(text.split())

def average_word_length(text):
    """Calculate average word length."""
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)

def sentiment_analysis(text):
    """Perform sentiment analysis on text."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def has_scientific_keywords(text):
    """Check if text contains scientific keywords."""
    scientific_keywords = [
        'study', 'research', 'science', 'scientist', 'evidence', 'experiment',
        'data', 'analysis', 'hypothesis', 'theory', 'clinical', 'medical',
        'findings', 'journal', 'publication', 'published', 'effects'
    ]
    return any(keyword in text.lower() for keyword in scientific_keywords)

def has_question(text):
    """Check if text contains a question."""
    return '?' in text

def has_numbers(text):
    """Check if text contains numbers."""
    return bool(re.search(r'\d', text))

# Feature matrix creation functions
def create_bow_features(texts, max_features=1000):
    """Create Bag of Words features."""
    vectorizer = CountVectorizer(max_features=max_features)
    bow_features = vectorizer.fit_transform(texts)
    return bow_features, vectorizer

def create_tfidf_features(texts, max_features=1000):
    """Create TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_features = vectorizer.fit_transform(texts)
    return tfidf_features, vectorizer

# Feature extraction pipeline
def extract_basic_features(df, text_column='text'):
    """Extract basic text features from a DataFrame."""
    feature_df = df.copy()

    feature_df['text_length'] = feature_df[text_column].apply(text_length)
    feature_df['word_count'] = feature_df[text_column].apply(word_count)
    feature_df['avg_word_length'] = feature_df[text_column].apply(average_word_length)
    feature_df['mention_count'] = feature_df[text_column].apply(count_mentions)
    feature_df['hashtag_count'] = feature_df[text_column].apply(count_hashtags)
    feature_df['punctuation_count'] = feature_df[text_column].apply(count_punctuation)
    feature_df['capital_letter_count'] = feature_df[text_column].apply(count_capital_letters)
    feature_df['has_question'] = feature_df[text_column].apply(has_question)
    feature_df['has_numbers'] = feature_df[text_column].apply(has_numbers)
    feature_df['has_scientific_keywords'] = feature_df[text_column].apply(has_scientific_keywords)

    sentiments = feature_df[text_column].apply(sentiment_analysis)
    feature_df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
    feature_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
    feature_df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
    feature_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])

    return feature_df

def feature_engineering_pipeline(df, text_column='text', max_features=1000):
    """Complete feature engineering pipeline with basic and TF-IDF features."""
    feature_df = extract_basic_features(df, text_column)
    tfidf_features, tfidf_vectorizer = create_tfidf_features(feature_df[text_column], max_features)

    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])],
        index=feature_df.index
    )

    final_df = pd.concat([feature_df, tfidf_df], axis=1)
    return final_df, tfidf_vectorizer
