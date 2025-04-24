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

def load_dataset(dataset_path):
    true_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/True.csv')
    fake_df = pd.read_csv('/Users/mihneacucu/Documents/MDS/Fake.csv')

    true_df["label"] = 1  # Real
    fake_df["label"] = 0  # Fake

    data = pd.concat([true_df, fake_df], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)  # amestecare
    return data

def train_model(data):
    X = data["text"]
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
        prediction = model.predict([text])[0]
        prob = model.predict_proba([text])[0][prediction]
        threshold = 0.75  # Prag pentru probabilitate
        if prob >= threshold:
            if prediction == 1:
                print(f"Stirea este probabil REALA ({prob*100:.2f}%)")
            else:
                print(f"Stirea este probabil FALSA ({prob*100:.2f}%)")
        else:
            print(f"Predictia nu este sigura. Probabilitatea este prea mica ({prob*100:.2f}%).")
    except FileNotFoundError:
        print("Fisierul .txt nu a fost gasit.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Foloseste: python Clasificare.py <cale_catre_dataset> <fisier_txt>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    txt_file_path = sys.argv[2]

    print("Incarcare si antrenare model...")
    data = load_dataset(dataset_path)
    model = train_model(data)

    print("\nAnalizam fisierul de input...")
    predict_from_file(model, txt_file_path)