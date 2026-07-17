"""
Baseline model: TF-IDF features + Logistic Regression.

This gives a classical-ML reference point so the BERT fine-tuning result
can be judged against something simpler, rather than in isolation.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .evaluate import compute_metrics


def train_baseline(train_df, test_df):
    """
    Train a TF-IDF + Logistic Regression baseline and evaluate it on the
    test set. Returns the fitted vectorizer, model, and a metrics dict.
    """
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy()

    X_test = test_df["text"].tolist()
    y_test = test_df["label"].to_numpy()

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    acc, prec, rec, f1 = compute_metrics(y_test, preds)

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    return tfidf, model, metrics, y_test, preds
