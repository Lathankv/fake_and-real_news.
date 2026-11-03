"""
Trains the fake news model. Run this once before app.py:
    python train.py

Needs data/Fake.csv and data/True.csv from the Kaggle dataset
"Fake and Real News Dataset" (search that name on kaggle.com).
"""
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean(text: str) -> str:
    """Normalize article text. Must match the version in app.py."""
    text = str(text)
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", text)  # drop the real-news source tag
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def main():
    fake = pd.read_csv("data/Fake.csv")
    fake["label"] = 0
    real = pd.read_csv("data/True.csv")
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df = df.drop_duplicates(subset="text").sample(frac=1, random_state=42)

    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean)
    df = df[df["content"].str.split().str.len() > 20]

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=5, max_df=0.7, max_features=200_000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=["Fake", "Real"]))

    joblib.dump(model, "model/model.joblib")
    print("Saved model/model.joblib")


if __name__ == "__main__":
    main()
