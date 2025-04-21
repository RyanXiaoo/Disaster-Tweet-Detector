# disaster_tweet_classifier.py

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def clean_text(text):
    """Cleans tweet text for modeling."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # EDA - show basic info
    print(train_df["target"].value_counts(normalize=True))
    train_df["target"].value_counts().plot(kind="bar", title="Disaster vs Non-Disaster")
    plt.savefig("target_distribution.png")
    plt.close()

    # Clean text
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y = train_df["target"]

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    print("Validation Performance:")
    print(classification_report(y_val, val_preds))

    # Predict on test data
    test_preds = model.predict(X_test)

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": test_preds
    })
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv created!")


if __name__ == "__main__":
    main()
