"""
Shared evaluation utilities used by both the baseline and BERT models,
so metrics are computed identically for a fair comparison.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred):
    """Return (accuracy, precision, recall, f1) for binary sentiment labels."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, prec, rec, f1


@torch.no_grad()
def evaluate_torch_model(model, loader, device):
    """
    Run a PyTorch classification model over a DataLoader in eval mode and
    return (metrics, true_labels, predictions).
    """
    model.eval()
    preds, labels_all = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)

        preds.extend(pred.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    metrics = compute_metrics(labels_all, preds)
    return metrics, np.array(labels_all), np.array(preds)
