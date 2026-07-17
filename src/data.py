"""
Data loading and splitting for the IMDb sentiment classification task.

Loads the IMDb dataset via Hugging Face `datasets`, and creates a held-out
validation split from the training data so the test set is never touched
until final evaluation.
"""

from datasets import load_dataset


def load_imdb_splits(seed: int = 40, val_fraction: float = 0.1):
    """
    Load IMDb and return (train, val, test) splits.

    The validation split is carved out of the official training set so
    that model selection never leaks information from the test set.
    """
    dataset = load_dataset("imdb")

    split = dataset["train"].train_test_split(test_size=val_fraction, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = dataset["test"]

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_imdb_splits()
    print("Train size:", len(train_ds))
    print("Val size  :", len(val_ds))
    print("Test size :", len(test_ds))
