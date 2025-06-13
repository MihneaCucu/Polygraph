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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# =====================
# Admin Advanced Settings
# =====================
# These can be adjusted by an admin for fine-tuning
CREDIBILITY_THRESHOLD = 0.70  # Default threshold for classification confidence
ALGORITHM_SENSITIVITY = 1.0   # Sensitivity multiplier (1.0 = normal, >1 = stricter, <1 = more permissive)
# =====================

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

def train_model(data, model_type='random_forest'):
    X = data["text"].apply(preprocess_text)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100)
    elif model_type == 'logreg':
        clf = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        clf = LinearSVC()
    elif model_type == 'nb':
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
        ("clf", clf),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\nResults for {model_type}:")
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
        # Use admin-configurable threshold and sensitivity
        threshold = CREDIBILITY_THRESHOLD * ALGORITHM_SENSITIVITY
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
        # Check if model supports predict_proba
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            probabilities = model.predict_proba([text])[0]
            prob = probabilities[prediction]
            print(f"Probabilities: REAL={probabilities[1] * 100:.2f}%, FAKE={probabilities[0] * 100:.2f}%")
        else:
            prob = None
            print("Probabilities: Not available for this model.")

        sentiment = analyze_sentiment(text)
        has_exaggeration = detect_exaggeration(text)
        lacks_sources = check_sources(text)

        print(f"Sentiment Polarity: {sentiment:.2f}")
        print(f"Exaggeration Detected: {'Yes' if has_exaggeration else 'No'}")
        print(f"Lacks Reliable Sources: {'Yes' if lacks_sources else 'No'}")
        threshold = CREDIBILITY_THRESHOLD * ALGORITHM_SENSITIVITY
        if prob is not None and prob >= threshold:
            if prediction == 1:
                print(f"The news is likely REAL ({prob*100:.2f}%)")
            else:
                print(f"The news is likely FAKE ({prob*100:.2f}%)")
        else:
            if prediction == 1:
                print("The news is likely REAL.")
            else:
                print("The news is likely FAKE.")
            if prob is not None:
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
    print(f"spacy embedding time for {sample_size} samples: {spacy_time:.4f} seconds")

    # SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    start = time.time()
    st_embeddings = model.encode(texts, show_progress_bar=False)
    st_time = time.time() - start
    print(f"SentenceTransformer embedding time for {sample_size} samples: {st_time:.4f} seconds")

    print("\nSummary:")
    print(f"TF-IDF: {tfidf_time:.4f}s, spaCy: {spacy_time:.4f}s, SentenceTransformer: {st_time:.4f}s")

# compare_embedding_speeds(100)

BERT_MODEL_NAME = 'bert-base-uncased'
BERT_MODEL_PATH = 'model_bert'

class BertNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def train_bert_model(data):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    X = data["text"].apply(preprocess_text).tolist()
    y = data["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = BertNewsDataset(X_train, y_train, tokenizer)
    test_dataset = BertNewsDataset(X_test, y_test, tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        disable_tqdm=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    # Evaluate
    eval_result = trainer.evaluate()
    print(f"\nBERT eval loss: {eval_result['eval_loss']:.4f}")
    model.save_pretrained(BERT_MODEL_PATH)
    tokenizer.save_pretrained(BERT_MODEL_PATH)
    return model, tokenizer

def predict_with_bert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For training: python Clasificare.py train <path_to_dataset>")
        print("  For classification: python Clasificare.py predict <text_file>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "train":
        if len(sys.argv) < 3:
            print("Please specify the path to the dataset: python Clasificare.py train <path_to_dataset> [model_type]")
            sys.exit(1)
        dataset_path = sys.argv[2]
        model_type = sys.argv[3] if len(sys.argv) > 3 else 'random_forest'
        print(f"Loading dataset and training model ({model_type})...")
        data = load_dataset(dataset_path)
        if model_type == 'bert':
            print("Training BERT model...")
            model, tokenizer = train_bert_model(data)
            print(f"BERT model trained and saved in {BERT_MODEL_PATH}.")
            return
        model = train_model(data, model_type)
        model_filename = f"model_{model_type}.joblib"
        joblib.dump(model, model_filename)
        print(f"Model trained and saved as {model_filename}.")
    elif mode == "predict":
        if len(sys.argv) < 4:
            print("Please specify the .txt file to classify and the model: python Clasificare.py predict <text_file> <model_type>")
            sys.exit(1)
        txt_file_path = sys.argv[2]
        model_type = sys.argv[3]
        model_filename = f"model_{model_type}.joblib"
        if model_type == 'bert':
            if not os.path.exists(BERT_MODEL_PATH):
                print(f"The BERT model does not exist. Please train it first: python Clasificare.py train <path_to_dataset> bert")
                sys.exit(1)
            from transformers import BertTokenizer, BertForSequenceClassification
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
            with open(txt_file_path, "r", encoding="utf-8") as f:
                text = f.read()
            text = preprocess_text(text)
            pred, probs = predict_with_bert(text, model, tokenizer)
            print(f"Probabilities: REAL={probs[1]*100:.2f}%, FAKE={probs[0]*100:.2f}%")
            print(f"The news is likely {'REAL' if pred==1 else 'FAKE'} ({probs[pred]*100:.2f}%)")
            return
        if not os.path.exists(model_filename):
            print(f"The model {model_filename} does not exist. Please train the model first: python Clasificare.py train <path_to_dataset> {model_type}")
            sys.exit(1)
        print(f"Loading existing model {model_filename}...")
        model = joblib.load(model_filename)
        print("\nClassifying input file...")
        predict_from_file_with_nlp(model, txt_file_path)
        print("\nDone.")
    else:
        print("Unknown mode. Use 'train' or 'predict'.")
        sys.exit(1)

if __name__ == "__main__":
    main()


'''
Exemple de comenzi corecte:
Pentru a utiliza modelul Random Forest :
python3 Clasificare.py predict trueNews2.txt random_forest

Pentru modelul SVM:
python3 Clasificare.py predict trueNews2.txt svm

Pentru modelul Logistic Regression:
python3 Clasificare.py predict trueNews2.txt logreg

Pentru modelul Naive Bayes:
python3 Clasificare.py predict trueNews2.txt nb

Pentru modelul BERT (dacă a fost antrenat anterior):
python3 Clasificare.py predict trueNews2.txt bert
'''