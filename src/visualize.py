"""
Plotting helpers: learning curves, confusion matrices, and the final
baseline-vs-BERT comparison chart. All figures are saved to results/.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_learning_curves(history_df, out_path="results/learning_curves.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o")
    plt.title("Learning Curve: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["val_f1"], marker="o")
    plt.title("Learning Curve: Validation F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1-score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.title(title)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(baseline_metrics, bert_metrics, out_path="results/comparison.png"):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    baseline_scores = [baseline_metrics[m] for m in metrics]
    bert_scores = [bert_metrics[m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, baseline_scores, width, label=baseline_metrics["model"])
    plt.bar(x + width / 2, bert_scores, width, label=bert_metrics["model"])

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Performance Comparison: Baseline vs Fine-Tuned BERT")
    plt.ylim(0.85, 0.95)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_path, dpi=150)
    plt.close()
