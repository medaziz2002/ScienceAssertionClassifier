# -*- coding: utf-8 -*-
"""
Text Preprocessing Script

Preprocessing pipeline for cleaning and normalizing text data, particularly tweets.
"""

# ==============================
# 0 - Imports
# ==============================
import re
import string
import pandas as pd
import nltk
import emoji
import spacy
from nltk.corpus import stopwords

# ==============================
# 1 - Installations and Downloads (only if necessary)
# ==============================
# Install and download resources only if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ==============================
# 2 - Text Cleaning Functions
# ==============================

def replace_emojis(text):
    """Replace emojis with their textual descriptions."""
    return emoji.demojize(text, delimiters=(" ", " "))

def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_mentions(text):
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text):
    """Remove #hashtags from text."""
    return re.sub(r'#\w+', '', text)

def remove_punctuation(text):
    """Remove punctuation from text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_numbers(text):
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)

def remove_stopwords(text):
    """Remove English stopwords from text."""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def lemmatize_text(text):
    """Lemmatize text using spaCy."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# ==============================
# 3 - Full Preprocessing Pipeline
# ==============================

def preprocess_text(text):
    """Apply all preprocessing steps to a single text."""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = replace_emojis(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_dataset(df, text_column='text'):
    """Apply text preprocessing to an entire DataFrame."""
    df_processed = df.copy()
    df_processed['text_processed'] = df_processed[text_column].apply(preprocess_text)
    return df_processed

# ==============================
# 4 - Main Testing Section
# ==============================

if __name__ == "__main__":
    # Sample tweet
    tweet = ("Can any Gynecologist with Cancer Experience explain the dangers of "
             "Transvaginal Douching with Fluoride or other toxins such as Dioxin? "
             "#PDX @doctor http://example.com")

    print("Original Tweet:\n", tweet)
    print("\n--- After Preprocessing ---")
    print(preprocess_text(tweet))

    # Test on a sample DataFrame
    try:
        df = pd.read_csv('/content/scitweets_export.tsv', sep='\t')
        print("\nOriginal DataFrame:")
        print(df.head())

        df_processed = preprocess_dataset(df)
        print("\nProcessed DataFrame:")
        print(df_processed[['text', 'text_processed']].head())
    except FileNotFoundError:
        print("\nNo file found at /content/scitweets_export.tsv.")

