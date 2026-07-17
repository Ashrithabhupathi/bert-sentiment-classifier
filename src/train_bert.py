"""
Fine-tunes bert-base-uncased for binary sentiment classification on IMDb.

Trains against the validation split each epoch (never the test set), then
evaluates once on the test set after training completes, to avoid any
test-set leakage into model selection.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .evaluate import compute_metrics, evaluate_torch_model

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5


def tokenize_splits(tokenizer, train_ds, val_ds, test_ds):
    """Pre-tokenize all splits (faster than tokenizing inside the training loop)."""

    def tokenize_function(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN
        )

    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    for split in (tokenized_train, tokenized_val, tokenized_test):
        split.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_train, tokenized_val, tokenized_test


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def fine_tune_bert(train_ds, val_ds, test_ds, device):
    """
    Full fine-tuning pipeline: tokenize, train for EPOCHS epochs while
    tracking validation metrics, then do a single final evaluation on test.

    Returns the trained model, a per-epoch history DataFrame, and the
    final test-set metrics dict.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train, tokenized_val, tokenized_test = tokenize_splits(
        tokenizer, train_ds, val_ds, test_ds
    )

    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        (val_acc, val_prec, val_rec, val_f1), _, _ = evaluate_torch_model(
            model, val_loader, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

    history_df = pd.DataFrame(history)

    (test_acc, test_prec, test_rec, test_f1), y_true, y_pred = evaluate_torch_model(
        model, test_loader, device
    )
    test_metrics = {
        "model": f"Fine-Tuned BERT ({MODEL_NAME})",
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
    }

    return model, history_df, test_metrics, y_true, y_pred
