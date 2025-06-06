import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import string
from pathlib import Path


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_dataset():
    '''
    Determinam file_path pentru csv uri daca rulam din afara fisierului
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use the provided dataset_path or default to the script's directory
    true_file = os.path.join(script_dir, "True.csv")
    fake_file = os.path.join(script_dir, "Fake.csv")
    '''

    # Read the CSV files
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")

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


if __name__ == "__main__":
    print("Loading model...")
    data = load_dataset()
    model = train_model(data)
    
    model_dir = Path(__file__).resolve().parent.parent / "pagina_principala" / "ai_models"

    # Create the directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Define the full file path
    model_path = model_dir / "model_clasificare_0.joblib"

    # Save the model
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")