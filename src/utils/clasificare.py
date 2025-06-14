#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

#print("Path to dataset files:", path)

import pandas as pd
import os
import sys
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from textblob import TextBlob

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

def load_dataset():
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use the provided dataset_path or default to the script's directory
    true_file = os.path.join(script_dir, "True.csv")
    fake_file = os.path.join(script_dir, "Fake.csv")

    # Read the CSV files
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)

    true_df["label"] = 1  # Real
    fake_df["label"] = 0  # Fake

    data = pd.concat([true_df, fake_df], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)  # amestecare
    return data

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
        threshold = 0.70  # Prag pentru probabilitate
        if prob >= threshold:
            if prediction == 1:
                print(f"The news is likely TRUE ({prob*100:.2f}%)")
            else:
                print(f"The news is likely FAKE ({prob*100:.2f}%)")
        else:
            print(f"Probability is too low ({prob*100:.2f}%).")
    except FileNotFoundError:
        print("File not found")

def predict_from_file_with_nlp(model, text, threshold):
    try:
        #with open(file_path, "r", encoding="utf-8") as f:
        #    text = f.read()
        text = preprocess_text(text)

        # Model prediction
        prediction = model.predict([text])[0]
        prob = model.predict_proba([text])[0][prediction]
        probabilities = model.predict_proba([text])[0]

        # NLP analyses
        sentiment = analyze_sentiment(text)
        has_exaggeration = detect_exaggeration(text)
        lacks_sources = check_sources(text)
        
        output = []

        output.append(f"Probabilities: REAL={probabilities[1] * 100:.2f}%, FAKE={probabilities[0] * 100:.2f}%")
        output.append(f"Sentiment Polarity: {sentiment:.2f}")
        output.append(f"Exaggeration Detected: {'Yes' if has_exaggeration else 'No'}")
        output.append(f"Lacks Reliable Sources: {'Yes' if lacks_sources else 'No'}")

        # Probability threshold
        if prob >= threshold:
            if prediction == 1:
                output.append(f"The news is likely REAL ({prob*100:.2f}%)")
            else:
                output.append(f"The news is likely FAKE ({prob*100:.2f}%)")
        else:
            output.append(f"The prediction is uncertain. Probability is too low ({prob*100:.2f}%).")
        
        return output
    except FileNotFoundError:
        print("The .txt file was not found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use: python Clasificare.py <path_to_dataset> <file.txt>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    print("Loading model...")
    data = load_dataset(dataset_path)
    model = train_model(data)

    print("\nAnalysing input...")
    predict_from_file_with_nlp(model, txt_file_path, 0.7)