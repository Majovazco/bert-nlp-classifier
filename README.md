# BERT for Aspect-Based Sentiment Classification

HSE University — Natural Language Processing, Homework №1
*Neural Network Architectures: Transformers*

Fine-tuning encoder transformers for 3-class sentiment classification of hotel reviews, with a baseline, a second encoder, hyperparameter tuning, and a layer-freezing experiment.

**Notebook:** [`HW1_Transformers.ipynb`](HW1_Transformers.ipynb) — runs top to bottom on a free Colab T4 GPU in about 25–30 minutes.

## The problem

Classify TripAdvisor hotel reviews by sentiment for one aspect, on a 1–5 scale mapped to 3 classes:

- `0` — BAD (ratings 1–2)
- `1` — NEUTRAL (rating 3)
- `2` — GOOD (ratings 4–5)

**Dataset:** `jniimi/tripadvisor-review-rating` (Hugging Face, based on Li et al., 2013), 201,295 reviews.
**Aspect used:** `rooms` (`rooms_score`).
**Sample:** 20,000 reviews, `random_state=42`.
**Splits:** train 14,000 / validation 3,000 / test 3,000, stratified, 70 / 15 / 15.

Class distribution in the 20,000-review sample:

| Class | Count | Share |
|---|---:|---:|
| BAD | 1,806 | 9.03% |
| NEUTRAL | 3,519 | 17.60% |
| GOOD | 14,675 | 73.38% |

Tokenization: per-model tokenizer, `MAX_LENGTH=128`, truncation, dynamic padding with `DataCollatorWithPadding`. In train, 7,925 / 14,000 reviews (56.61%) exceed 128 tokens.

## Results

All metrics below are the executed outputs in the notebook (final clean run, Colab T4 GPU, `SEED=42`).

### Baseline vs. second encoder — validation

| Model | Parameters | LR | Batch | Epochs | Val accuracy | Val macro-F1 | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | 109,484,547 | 2e-05 | 32 | 3 | **0.8233** | **0.6813** | 223.44 s |
| distilbert-base-uncased | 66,955,779 | 2e-05 | 32 | 3 | 0.8160 | 0.6578 | 107.74 s |

BERT was chosen for tuning: +0.0172 macro-F1 over DistilBERT, at about 2× the training time.

### Tuned model — final evaluation

Hyperparameter search: Optuna, 4 trials × 2 epochs, on 3,000 train / 1,000 validation examples.
Final model retrained with LR `3.00e-05`, weight decay `0.01`, early stopping on validation.

| Model | LR | Weight decay | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| BERT baseline | 2.00e-05 | 0.01 | 0.8233 | 0.6813 | 0.8330 | 0.6897 |
| **BERT tuned** | **3.00e-05** | **0.01** | **0.8223** | **0.6885** | **0.8357** | **0.7041** |

### Layer freezing — test

Only the last transformer block (`encoder.layer.11`), the pooler, and the classification head were trainable, with LR `2.00e-04`.

| Model | Trainable parameters | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| BERT frozen, last block only | 7,680,771 (7.02% of 109,484,547) | 0.8270 | 0.6855 |

Freezing 92.98% of the parameters kept most of the performance: test accuracy 0.8270 vs. 0.8357 for full fine-tuning of the tuned model.

## Beyond the code

This was not just training a model. The work was in the decisions behind it:

- **A hotel rating problem, not a generic score:** a review is not just “good” or “bad”. The same guest can praise the location and complain about the rooms, so the sentiment is classified for one aspect — the rooms — where the signal is actually useful.
- **Chose one aspect, `rooms`,** instead of the overall rating, because the same review can praise one aspect and criticise another.
- **Dropped reviews with a missing aspect rating instead of imputing one,** to avoid inventing labels the guest never gave.
- **Binned 1–5 into BAD / NEUTRAL / GOOD** on domain meaning, not just numbers: 1–2 are clearly negative, 3 is mixed, 4–5 are positive.
- **Judged the models on macro-F1 as well as accuracy,** because GOOD is 73% of the data and accuracy alone would hide failures on the minority classes.
- **Chose BERT over DistilBERT** even though DistilBERT is about 2× faster, because BERT was stronger on macro-F1 (+0.0172).
- **Tuned, then evaluated on test only once,** with early stopping on validation, so the test result is a genuine check, not a tuning target.
- **Tested the efficiency trade-off:** freezing 93% of the network still reached 0.8270 test accuracy, so full fine-tuning wins on accuracy, but freezing is a real option under a tight compute budget.

## Technologies

Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · Optuna · scikit-learn · pandas · NumPy · matplotlib · seaborn · Google Colab (T4 GPU)

## How to run

1. Open [`HW1_Transformers.ipynb`](HW1_Transformers.ipynb) in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells top to bottom.

Do not upgrade `torch` / `torchvision` in Colab: a mismatched `torchvision` breaks `datasets` with `ImportError: cannot import name 'VideoReader'`.

Installs in the notebook:

```bash
pip install -q "transformers[torch]" datasets optuna
```

See [`requirements.txt`](requirements.txt) for the libraries used.

## Structure

```
bert-nlp-classifier/
├── HW1_Transformers.ipynb   # Full homework: data, training, tuning, freezing, analysis
├── README.md
└── requirements.txt
```
