import csv
import gc
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"
TRAIN_FILE = DATA_DIR / "train_tokens_clean.csv"
TEST_FILE = DATA_DIR / "test_tokens.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "comi_lingua_word_model.joblib"
REPORT_PATH = MODEL_DIR / "comi_lingua_evaluation_report.json"
ERROR_PATH = MODEL_DIR / "comi_lingua_test_errors.csv"

LABELS = np.array(["en", "hi", "other"])
VALID_LABELS = set(LABELS.tolist())
BATCH_SIZE = 2048
EPOCHS = 3
WORD_HASH_FEATURES = 2**18
CHAR_HASH_FEATURES = 2**18

REQUIRED_COLUMNS = {
    "sentence_id", "sentence", "token_index", "token", "language",
    "votes", "annotator_count", "agreement",
}


def normalize_sentence(text):
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.casefold().split())


def get_headers(reader, path):
    if not reader.fieldnames:
        raise ValueError(f"No CSV header found: {path}")
    headers = {
        name.strip().lower(): name
        for name in reader.fieldnames if name
    }
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}. "
            f"Found: {reader.fieldnames}"
        )
    return headers


def scan_dataset(path):
    """First pass: validate key fields, count labels, and collect sentence texts."""
    counts = Counter()
    sentence_id_to_text = {}
    sentence_texts = set()
    rows = 0

    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = get_headers(reader, path)
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            sid = (row.get(headers["sentence_id"]) or "").strip()
            sentence = (row.get(headers["sentence"]) or "").strip()
            token = (row.get(headers["token"]) or "").strip()
            label = (row.get(headers["language"]) or "").strip().lower()

            if not sid or not sentence or not token:
                raise ValueError(
                    f"Empty sentence_id/sentence/token at {path.name}:{line_number}"
                )
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Unexpected label {label!r} at {path.name}:{line_number}"
                )
            try:
                index = int(row[headers["token_index"]])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid token_index at {path.name}:{line_number}"
                )
            if index < 1:
                raise ValueError(
                    f"token_index must be >= 1 at {path.name}:{line_number}"
                )

            old_sentence = sentence_id_to_text.get(sid)
            if old_sentence is not None and old_sentence != sentence:
                raise ValueError(
                    f"Sentence ID {sid!r} maps to more than one sentence in {path.name}"
                )
            sentence_id_to_text[sid] = sentence
            sentence_texts.add(normalize_sentence(sentence))
            counts[label] += 1

    if rows == 0:
        raise ValueError(f"Dataset is empty: {path}")

    return {
        "rows": rows,
        "sentence_count": len(sentence_id_to_text),
        "unique_sentence_texts": len(sentence_texts),
        "sentence_texts": sentence_texts,
        "label_counts": counts,
    }


def infer_script(token):
    has_latin = any(ch.isascii() and ch.isalpha() for ch in token)
    has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in token)
    if has_latin and has_devanagari:
        return "mixed"
    if has_devanagari:
        return "devanagari"
    if has_latin:
        return "latin"
    if any(ch.isdigit() for ch in token):
        return "digit"
    return "symbol"


def token_shape(token):
    if token.isupper() and any(ch.isalpha() for ch in token):
        return "upper"
    if token.istitle():
        return "title"
    if token.islower():
        return "lower"
    return "mixedcase"


def build_word_features(tokens, index):
    token = tokens[index]
    folded = token.casefold()

    def get_word(position):
        if position < 0:
            return "<START>"
        if position >= len(tokens):
            return "<END>"
        return tokens[position].casefold()

    left = [get_word(i) for i in range(max(0, index - 3), index)]
    right = [get_word(i) for i in range(index + 1, min(len(tokens), index + 4))]
    prefix2 = folded[:2] if len(folded) >= 2 else folded
    prefix3 = folded[:3] if len(folded) >= 3 else folded
    suffix2 = folded[-2:] if len(folded) >= 2 else folded
    suffix3 = folded[-3:] if len(folded) >= 3 else folded
    length_bucket = min(len(token), 15)

    return (
        f"TARGET {folded} PREV1 {get_word(index - 1)} NEXT1 {get_word(index + 1)} "
        f"PREV2 {get_word(index - 2)} NEXT2 {get_word(index + 2)} "
        f"LEFT {' '.join(left)} RIGHT {' '.join(right)} "
        f"SCRIPT {infer_script(token)} SHAPE {token_shape(token)} "
        f"PREFIX2 {prefix2} PREFIX3 {prefix3} SUFFIX2 {suffix2} SUFFIX3 {suffix3} "
        f"LENGTH {length_bucket}"
    )


def iter_sentence_examples(path):
    """Stream one sentence group at a time, preserving token-index context."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = get_headers(reader, path)

        def normalized_rows():
            for raw in reader:
                yield {
                    key: (raw.get(headers[key]) or "").strip()
                    for key in REQUIRED_COLUMNS
                }

        # The converter writes rows in source-sentence order; groupby avoids loading
        # hundreds of thousands of rows into RAM at once.
        for sid, grouped in groupby(normalized_rows(), key=lambda row: row["sentence_id"]):
            group_rows = list(grouped)
            if not group_rows:
                continue
            group_rows.sort(key=lambda row: int(row["token_index"]))
            sentence = group_rows[0]["sentence"]
            tokens = [row["token"] for row in group_rows]

            if any(row["sentence"] != sentence for row in group_rows):
                raise ValueError(f"Inconsistent sentence text for ID {sid!r}")

            seen_indices = set()
            for index, row in enumerate(group_rows):
                token_index = int(row["token_index"])
                if token_index in seen_indices:
                    raise ValueError(
                        f"Duplicate token index {token_index} in sentence ID {sid!r}"
                    )
                seen_indices.add(token_index)
                label = row["language"].lower()
                if label not in VALID_LABELS:
                    raise ValueError(f"Unexpected label {label!r} in sentence ID {sid!r}")

                yield {
                    "sentence_id": sid,
                    "sentence": sentence,
                    "token_index": token_index,
                    "token": row["token"],
                    "label": label,
                    "word_text": build_word_features(tokens, index),
                    "char_text": row["token"],
                }


def iter_batches(path, batch_size, shuffle_seed=None):
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    batch = []
    for example in iter_sentence_examples(path):
        batch.append(example)
        if len(batch) >= batch_size:
            if rng is not None:
                rng.shuffle(batch)
            yield batch
            batch = []
    if batch:
        if rng is not None:
            rng.shuffle(batch)
        yield batch


def make_features(batch, word_vectorizer, char_vectorizer):
    word_texts = [item["word_text"] for item in batch]
    char_texts = [item["char_text"] for item in batch]
    word_matrix = word_vectorizer.transform(word_texts)
    char_matrix = char_vectorizer.transform(char_texts)
    return hstack([word_matrix, char_matrix], format="csr")


def main():
    print("Scanning dataset metadata without loading feature matrices into memory...")
    train_info = scan_dataset(TRAIN_FILE)
    test_info = scan_dataset(TEST_FILE)

    overlap = train_info["sentence_texts"] & test_info["sentence_texts"]
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(
            f"Train/test sentence overlap detected ({len(overlap)} texts). "
            f"Example: {sample!r}. Run remove_train_test_overlap.py first."
        )

    if set(train_info["label_counts"]) - VALID_LABELS:
        raise ValueError("Unknown training labels detected.")

    missing_train_classes = VALID_LABELS - set(train_info["label_counts"])
    if missing_train_classes:
        raise ValueError(f"Training file lacks classes: {sorted(missing_train_classes)}")

    print("\n=== COMI-LINGUA MEMORY-EFFICIENT MODEL ===")
    print(f"Training token rows: {train_info['rows']:,}")
    print(f"Training sentence IDs: {train_info['sentence_count']:,}")
    print(f"Training unique sentence texts: {train_info['unique_sentence_texts']:,}")
    print(f"Training label counts: {dict(train_info['label_counts'])}")
    print(f"Test token rows: {test_info['rows']:,}")
    print(f"Test sentence IDs: {test_info['sentence_count']:,}")
    print(f"Test label counts: {dict(test_info['label_counts'])}")
    print("Exact normalized sentence overlap: 0")
    print(f"Batch size: {BATCH_SIZE}; epochs: {EPOCHS}")

    # Balanced sample weights are computed from the training split only.
    total_train = train_info["rows"]
    num_classes = len(VALID_LABELS)
    class_weights = {
        label: total_train / (num_classes * count)
        for label, count in train_info["label_counts"].items()
    }

    word_vectorizer = HashingVectorizer(
        n_features=WORD_HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        dtype=np.float32,
    )
    char_vectorizer = HashingVectorizer(
        n_features=CHAR_HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        dtype=np.float32,
    )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        random_state=42,
        average=True,
        learning_rate="optimal",
    )

    initialized = False
    for epoch in range(EPOCHS):
        print(f"\nTraining epoch {epoch + 1}/{EPOCHS}...")
        processed = 0
        batch_count = 0

        for batch_count, batch in enumerate(
            iter_batches(TRAIN_FILE, BATCH_SIZE, shuffle_seed=2026 + epoch),
            start=1,
        ):
            X_batch = make_features(batch, word_vectorizer, char_vectorizer)
            y_batch = np.asarray([item["label"] for item in batch], dtype=str)
            sample_weights = np.asarray(
                [class_weights[label] for label in y_batch], dtype=np.float64
            )

            if not initialized:
                classifier.partial_fit(
                    X_batch, y_batch, classes=LABELS,
                    sample_weight=sample_weights,
                )
                initialized = True
            else:
                classifier.partial_fit(
                    X_batch, y_batch, sample_weight=sample_weights
                )

            processed += len(batch)
            if batch_count % 25 == 0:
                print(
                    f"  batches: {batch_count}; "
                    f"tokens this epoch: {processed:,}/{train_info['rows']:,}"
                )

            del X_batch, y_batch, sample_weights
            if batch_count % 20 == 0:
                gc.collect()

        print(
            f"Epoch {epoch + 1} completed: "
            f"{processed:,} token rows in {batch_count:,} batches."
        )
        gc.collect()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier": classifier,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "labels": LABELS.tolist(),
        "feature_config": {
            "word_hash_features": WORD_HASH_FEATURES,
            "char_hash_features": CHAR_HASH_FEATURES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
    }
    print("\nSaving trained model...")
    joblib.dump(model_bundle, MODEL_PATH, compress=1)

    # Evaluate only once on the separate held-out test split.
    print("\nEvaluating on the untouched COMI-LINGUA test split...")
    y_true = []
    y_pred = []
    error_count = 0
    evaluated = 0

    with ERROR_PATH.open("w", encoding="utf-8", newline="") as error_file:
        error_columns = [
            "sentence_id", "sentence", "token_index", "token",
            "actual", "predicted",
        ]
        error_writer = csv.DictWriter(error_file, fieldnames=error_columns)
        error_writer.writeheader()

        for batch_number, batch in enumerate(
            iter_batches(TEST_FILE, BATCH_SIZE), start=1
        ):
            X_batch = make_features(batch, word_vectorizer, char_vectorizer)
            batch_y = [item["label"] for item in batch]
            batch_predictions = classifier.predict(X_batch)

            y_true.extend(batch_y)
            y_pred.extend(batch_predictions.tolist())

            for item, actual, predicted in zip(batch, batch_y, batch_predictions):
                if actual != predicted:
                    error_count += 1
                    error_writer.writerow({
                        "sentence_id": item["sentence_id"],
                        "sentence": item["sentence"],
                        "token_index": item["token_index"],
                        "token": item["token"],
                        "actual": actual,
                        "predicted": str(predicted),
                    })

            evaluated += len(batch)
            if batch_number % 20 == 0:
                print(f"  evaluated test tokens: {evaluated:,}/{test_info['rows']:,}")
            del X_batch

    accuracy = float(accuracy_score(y_true, y_pred))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABELS.tolist(),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=LABELS.tolist(),
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS.tolist())

    print("\n=== COMI-LINGUA TEST RESULTS ===")
    print(f"Test tokens evaluated: {len(y_true):,}")
    print(f"Held-out token accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix; label order:", LABELS.tolist())
    print(matrix)
    print(f"Misclassified tokens: {error_count:,}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Evaluation report: {REPORT_PATH}")
    print(f"Test errors: {ERROR_PATH}")

    report_data = {
        "dataset": "COMI-LINGUA",
        "training_rows": train_info["rows"],
        "training_sentence_ids": train_info["sentence_count"],
        "training_label_counts": dict(train_info["label_counts"]),
        "test_rows": test_info["rows"],
        "test_sentence_ids": test_info["sentence_count"],
        "test_label_counts": dict(test_info["label_counts"]),
        "test_sentence_overlap": len(overlap),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "labels_order": LABELS.tolist(),
        "confusion_matrix": matrix.tolist(),
        "misclassified_tokens": error_count,
        "evaluation_note": (
            "Evaluation uses the provided test split after removing exact normalized "
            "sentence overlap from training. Label quality depends on the source annotations. "
            "Do not repeatedly tune to this test set and present it as an untouched final test."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, MemoryError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        if isinstance(error, MemoryError):
            print(
                "The program ran out of memory again. Do not edit individual sections; "
                "send the terminal screenshot for a complete lower-memory replacement.",
                file=sys.stderr,
            )
        sys.exit(1)
