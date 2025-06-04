#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

#print("Path to dataset files:", path)

import pandas as pd
import os
import sys
import string
import joblib
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    return sentiment

def detect_exaggeration(text):
    exaggeration_words = ["always", "never", "everyone", "nobody", "all", "none", "incredbile", "unbelievable", "shocking"]
    count = sum(word in text.lower() for word in exaggeration_words)
    return count > 0

def check_sources(text):
    unreliable_phrases = ["according to", "sources say", "experts claim", "some people say", "anonymous"]
    count = sum(phrase in text.lower() for phrase in unreliable_phrases)
    return count > 0

def load_dataset(dataset_path):
    true_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/True.csv')
    fake_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/Fake.csv')

    true_df["label"] = 1  # Real
    fake_df["label"] = 0  # Fake

    data = pd.concat([true_df, fake_df], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)  # amestecare
    return data

def get_embedding_model(embedding_type):
    if embedding_type == 'tfidf':
        return TfidfVectorizer(stop_words="english", max_df=0.7)
    elif embedding_type == 'spacy':
        nlp = spacy.load('en_core_web_md')
        return nlp
    elif embedding_type == 'sbert':
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        raise ValueError('Unknown embedding type')

def embed_texts(texts, embedding_type, model, fit=False):
    if embedding_type == 'tfidf':
        if fit:
            return model.fit_transform(texts)
        else:
            return model.transform(texts)
    elif embedding_type == 'spacy':
        return np.array([model(text).vector for text in texts])
    elif embedding_type == 'sbert':
        return np.array(model.encode(texts))
    else:
        raise ValueError('Unknown embedding type')

def train_model_with_embedding(data, embedding_type='tfidf'):
    X = data["text"].apply(preprocess_text)
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    embedding_model = get_embedding_model(embedding_type)
    X_train_emb = embed_texts(X_train.tolist(), embedding_type, embedding_model, fit=True)
    X_test_emb = embed_texts(X_test.tolist(), embedding_type, embedding_model, fit=False)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)
    print(f"Results for {embedding_type} embedding:")
    print(classification_report(y_test, y_pred))
    return embedding_model, clf

def train_model(data):
    X = data["text"].apply(preprocess_text)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
        ("clf", RandomForestClassifier(n_estimators=100)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

def predict_from_file(model, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = preprocess_text(text)
        prediction = model.predict([text])[0]
        prob = model.predict_proba([text])[0][prediction]
        probabilities = model.predict_proba([text])[0]
        print(f"Probabilities: REAL={probabilities[1] * 100:.2f}%, FAKE={probabilities[0] * 100:.2f}%")
        threshold = 0.70
        if prob >= threshold:
            if prediction == 1:
                print(f"The news is likely TRUE ({prob*100:.2f}%)")
            else:
                print(f"The news is likely FAKE ({prob*100:.2f}%)")
        else:
            print(f"Probability is too low ({prob*100:.2f}%).")
    except FileNotFoundError:
        print("File not found")

def predict_from_file_with_nlp(model, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = preprocess_text(text)

        prediction = model.predict([text])[0]
        prob = model.predict_proba([text])[0][prediction]
        probabilities = model.predict_proba([text])[0]
        sentiment = analyze_sentiment(text)
        has_exaggeration = detect_exaggeration(text)
        lacks_sources = check_sources(text)

        print(f"Probabilities: REAL={probabilities[1] * 100:.2f}%, FAKE={probabilities[0] * 100:.2f}%")
        print(f"Sentiment Polarity: {sentiment:.2f}")
        print(f"Exaggeration Detected: {'Yes' if has_exaggeration else 'No'}")
        print(f"Lacks Reliable Sources: {'Yes' if lacks_sources else 'No'}")

        threshold = 0.70
        if prob >= threshold:
            if prediction == 1:
                print(f"The news is likely REAL ({prob*100:.2f}%)")
            else:
                print(f"The news is likely FAKE ({prob*100:.2f}%)")
        else:
            print(f"The prediction is uncertain. Probability is too low ({prob*100:.2f}%).")
    except FileNotFoundError:
        print("The .txt file was not found.")

def compare_embedding_speeds(sample_size=100):
    """
    Compare embedding generation speed for TF-IDF, spaCy, and SentenceTransformer.
    """
    # Load dataset
    true_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/True.csv')
    fake_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/Fake.csv')
    df = pd.concat([true_df, fake_df]).reset_index(drop=True)
    texts = df['text'].astype(str).apply(preprocess_text).tolist()[:sample_size]

    # TF-IDF
    tfidf = TfidfVectorizer()
    start = time.time()
    tfidf_embeddings = tfidf.fit_transform(texts)
    tfidf_time = time.time() - start
    print(f"TF-IDF embedding time for {sample_size} samples: {tfidf_time:.4f} seconds")

    # spaCy
    nlp = spacy.load('en_core_web_sm')
    start = time.time()
    spacy_embeddings = np.array([nlp(text).vector for text in texts])
    spacy_time = time.time() - start
    print(f"spaCy embedding time for {sample_size} samples: {spacy_time:.4f} seconds")

    # SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    start = time.time()
    st_embeddings = model.encode(texts, show_progress_bar=False)
    st_time = time.time() - start
    print(f"SentenceTransformer embedding time for {sample_size} samples: {st_time:.4f} seconds")

    print("\nSummary:")
    print(f"TF-IDF: {tfidf_time:.4f}s, spaCy: {spacy_time:.4f}s, SentenceTransformer: {st_time:.4f}s")

# Exemplu de rulare:
compare_embedding_speeds(100)

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Use: python Clasificare.py <path_to_dataset> <file.txt>")
#         sys.exit(1)
#
#     dataset_path = sys.argv[1]
#     txt_file_path = sys.argv[2]
#     print("Loading dataset...")
#     data = load_dataset(dataset_path)
#     print("Training model (TF-IDF)...")
#     model = train_model(data)
#     print("\nClassifying input file...")
#     predict_from_file_with_nlp(model, txt_file_path)
#     print("\nDone.")
