# BERT Sentiment Classifier

Fine-tunes `bert-base-uncased` for binary sentiment classification on the
IMDb movie review dataset, benchmarked against a TF-IDF + Logistic
Regression baseline.

## Problem

Manually tagging customer/review sentiment doesn't scale. This project
compares a classical NLP baseline against a fine-tuned transformer to
quantify how much a modern language model actually improves sentiment
classification over bag-of-words methods — and whether that improvement
is worth the extra training cost.

## Approach

1. **Data** — IMDb dataset (25k train / 25k test), with a 10% validation
   split carved out of training data so test-set performance is only
   checked once, after training is finished.
2. **Baseline** — TF-IDF (unigrams + bigrams, 50k features) + Logistic
   Regression.
3. **Fine-tuning** — `bert-base-uncased`, sequence length 128, batch size
   16, 2 epochs, linear LR schedule with warmup, AdamW optimizer.
4. **Evaluation** — accuracy, precision, recall, F1 on the untouched test
   set, plus confusion matrices for both models.

## Results

| Model                          | Accuracy | Precision | Recall | F1     |
|---------------------------------|----------|-----------|--------|--------|
| TF-IDF + Logistic Regression    | 0.8814   | 0.8801    | 0.8832 | 0.8816 |
| Fine-Tuned BERT (bert-base-uncased) | 0.8912   | 0.8804    | 0.9054 | 0.8927 |

BERT improves over the baseline on every metric, most notably recall
(+2.2 points), meaning it catches more true positive sentiment cases.
The overall gap is modest rather than dramatic — IMDb reviews are long
and contain fairly explicit sentiment language, which lets a bag-of-words
model perform well too. The confusion matrices show BERT reduces both
false positives and false negatives slightly, consistent with better
handling of context and negation.

See `results/` for learning curves and confusion matrix plots after
running the pipeline.

## Project structure

```
bert-sentiment-classifier/
├── src/
│   ├── data.py           # dataset loading + train/val/test split
│   ├── baseline.py       # TF-IDF + Logistic Regression baseline
│   ├── train_bert.py     # BERT tokenization + fine-tuning loop
│   ├── evaluate.py       # shared metrics used by both models
│   ├── visualize.py      # learning curves, confusion matrices, comparison chart
│   └── main.py           # end-to-end pipeline entry point
├── notebooks/
│   └── exploration.ipynb # original exploratory notebook (EDA, baseline dev)
├── results/               # generated plots and metrics (created on run)
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python -m src.main
```

Requires a GPU for practical BERT fine-tuning time (trained on a T4 in
the original run); CPU training will work but is significantly slower.

## Notes

- Test set is only touched once, after training — validation is what
  training decisions are based on, to avoid leaking test performance
  into model selection.
- Seed fixed at 40 across Python, NumPy, and PyTorch for reproducibility.
