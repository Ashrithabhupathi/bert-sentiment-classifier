"""
End-to-end pipeline: load data -> train baseline -> fine-tune BERT ->
compare -> save results.

Run with: python -m src.main
"""

import random

import numpy as np
import pandas as pd
import torch

from .baseline import train_baseline
from .data import load_imdb_splits
from .train_bert import fine_tune_bert
from .visualize import plot_confusion_matrix, plot_learning_curves, plot_model_comparison


def set_seed(seed: int = 40) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds, val_ds, test_ds = load_imdb_splits()
    train_df = pd.DataFrame(train_ds)
    test_df = pd.DataFrame(test_ds)

    print("\n=== Training baseline: TF-IDF + Logistic Regression ===")
    _, _, baseline_metrics, y_test_base, preds_base = train_baseline(train_df, test_df)
    print(baseline_metrics)
    plot_confusion_matrix(
        y_test_base,
        preds_base,
        "Baseline Confusion Matrix (TF-IDF + Logistic Regression)",
        "results/baseline_confusion_matrix.png",
    )

    print("\n=== Fine-tuning BERT ===")
    _, history_df, bert_metrics, y_true_bert, preds_bert = fine_tune_bert(
        train_ds, val_ds, test_ds, device
    )
    print(bert_metrics)
    plot_learning_curves(history_df)
    plot_confusion_matrix(
        y_true_bert,
        preds_bert,
        "BERT Confusion Matrix (Fine-Tuned)",
        "results/bert_confusion_matrix.png",
    )

    print("\n=== Final comparison ===")
    results = pd.DataFrame([baseline_metrics, bert_metrics])
    print(results)
    results.to_csv("results/comparison_table.csv", index=False)
    plot_model_comparison(baseline_metrics, bert_metrics)


if __name__ == "__main__":
    main()
