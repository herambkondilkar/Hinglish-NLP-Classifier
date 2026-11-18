import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"
TEST_FILE = DATA_DIR / "test_tokens.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "comi_lingua_dev_tuned_augmented.joblib"
REPORT_PATH = MODEL_DIR / "comi_lingua_augmented_final_test_report.json"
ERROR_PATH = MODEL_DIR / "comi_lingua_augmented_final_test_errors.csv"

LABELS = ["en", "hi", "other"]
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
            f"{path.name} is missing columns: {sorted(missing)}"
        )
    return headers


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
    right = [
        get_word(i)
        for i in range(index + 1, min(len(tokens), index + 4))
    ]
    prefix2 = folded[:2] if len(folded) >= 2 else folded
    prefix3 = folded[:3] if len(folded) >= 3 else folded
    suffix2 = folded[-2:] if len(folded) >= 2 else folded
    suffix3 = folded[-3:] if len(folded) >= 3 else folded
    length_bucket = min(len(token), 15)

    return (
        f"TARGET {folded} PREV1 {get_word(index - 1)} "
        f"NEXT1 {get_word(index + 1)} PREV2 {get_word(index - 2)} "
        f"NEXT2 {get_word(index + 2)} LEFT {' '.join(left)} "
        f"RIGHT {' '.join(right)} SCRIPT {infer_script(token)} "
        f"SHAPE {token_shape(token)} PREFIX2 {prefix2} PREFIX3 {prefix3} "
        f"SUFFIX2 {suffix2} SUFFIX3 {suffix3} LENGTH {length_bucket}"
    )


def iter_sentence_examples(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = get_headers(reader, path)

        def normalized_rows():
            for raw in reader:
                yield {
                    key: (raw.get(headers[key]) or "").strip()
                    for key in REQUIRED_COLUMNS
                }

        for sentence_id, grouped in groupby(
            normalized_rows(), key=lambda row: row["sentence_id"]
        ):
            group_rows = list(grouped)
            if not group_rows:
                continue
            group_rows.sort(key=lambda row: int(row["token_index"]))
            sentence = group_rows[0]["sentence"]
            tokens = [row["token"] for row in group_rows]

            if any(row["sentence"] != sentence for row in group_rows):
                raise ValueError(
                    f"Inconsistent sentence text for ID {sentence_id!r}"
                )

            seen_indices = set()
            for index, row in enumerate(group_rows):
                token_index = int(row["token_index"])
                if token_index in seen_indices:
                    raise ValueError(
                        f"Duplicate token index {token_index} "
                        f"in sentence ID {sentence_id!r}"
                    )
                seen_indices.add(token_index)
                label = row["language"].lower()
                if label not in LABELS:
                    raise ValueError(
                        f"Unexpected label {label!r} in {sentence_id!r}"
                    )
                yield {
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "token_index": token_index,
                    "token": row["token"],
                    "label": label,
                    "word_text": build_word_features(tokens, index),
                    "char_text": row["token"],
                }


def iter_batches(path, batch_size):
    batch = []
    for example in iter_sentence_examples(path):
        batch.append(example)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_features(batch, word_vectorizer, char_vectorizer):
    from scipy.sparse import hstack

    word_matrix = word_vectorizer.transform(
        [item["word_text"] for item in batch]
    )
    char_matrix = char_vectorizer.transform(
        [item["char_text"] for item in batch]
    )
    return hstack([word_matrix, char_matrix], format="csr")


def apply_english_threshold(probabilities, classes, threshold):
    class_index = {label: idx for idx, label in enumerate(classes)}
    if "en" not in class_index:
        raise ValueError("Loaded candidate model has no 'en' class.")

    en_idx = class_index["en"]
    non_en_indices = [idx for label, idx in class_index.items() if label != "en"]
    predictions = []

    for row in probabilities:
        if row[en_idx] >= threshold:
            predictions.append("en")
        else:
            best_non_en = max(non_en_indices, key=lambda idx: row[idx])
            predictions.append(classes[best_non_en])

    return np.asarray(predictions, dtype=str)


def summarize(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "label_order": LABELS,
        "english_precision": float(report["en"]["precision"]),
        "english_recall": float(report["en"]["recall"]),
        "english_f1": float(report["en"]["f1-score"]),
        "false_positive_english": int(
            sum(1 for actual, pred in zip(y_true, y_pred)
                if pred == "en" and actual != "en")
        ),
        "false_negative_english": int(
            sum(1 for actual, pred in zip(y_true, y_pred)
                if actual == "en" and pred != "en")
        ),
    }


def main():
    for path in (TEST_FILE, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading locally augmented development-tuned candidate model...")
    bundle = joblib.load(MODEL_PATH)
    required_bundle_fields = {
        "classifier", "word_vectorizer", "char_vectorizer", "labels",
        "english_threshold",
    }
    if not required_bundle_fields.issubset(bundle):
        raise ValueError(
            "Candidate model bundle is missing required fields: "
            f"{sorted(required_bundle_fields - set(bundle))}"
        )

    threshold = bundle["english_threshold"]
    if threshold is None:
        raise ValueError(
            "No English threshold was selected during development tuning."
        )

    classifier = bundle["classifier"]
    word_vectorizer = bundle["word_vectorizer"]
    char_vectorizer = bundle["char_vectorizer"]
    classes = [str(label) for label in bundle["labels"]]

    # Stream the test set in batches. No training data is loaded or changed.
    y_true = []
    argmax_predictions = []
    tuned_predictions = []
    test_rows_for_errors = []
    label_counts = Counter()
    batch_size = 2048

    print("Evaluating the separate COMI-LINGUA test split once...")
    for batch_number, batch in enumerate(
        iter_batches(TEST_FILE, batch_size), start=1
    ):
        X_batch = make_features(batch, word_vectorizer, char_vectorizer)
        probabilities = classifier.predict_proba(X_batch)
        argmax_idx = np.argmax(probabilities, axis=1)

        argmax_batch = np.asarray(classes, dtype=str)[argmax_idx]
        tuned_batch = apply_english_threshold(
            probabilities, classes, float(threshold)
        )

        y_true.extend(item["label"] for item in batch)
        argmax_predictions.extend(argmax_batch.tolist())
        tuned_predictions.extend(tuned_batch.tolist())
        test_rows_for_errors.extend({
            "sentence_id": item["sentence_id"],
            "sentence": item["sentence"],
            "token_index": item["token_index"],
            "token": item["token"],
        } for item in batch)
        label_counts.update(item["label"] for item in batch)

        if batch_number % 20 == 0:
            print(
                f"  evaluated test tokens: {len(y_true):,}"
            )

    argmax_metrics = summarize(y_true, argmax_predictions)
    tuned_metrics = summarize(y_true, tuned_predictions)

    print("\n=== FROZEN THRESHOLD FINAL TEST RESULTS ===")
    print(f"Test tokens evaluated: {len(y_true):,}")
    print(f"Test label counts: {dict(label_counts)}")
    print(f"Selected English threshold: {float(threshold):.2f}")

    for name, predictions, metrics in [
        ("Default argmax (comparison only)", argmax_predictions, argmax_metrics),
        ("Tuned threshold", tuned_predictions, tuned_metrics),
    ]:
        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"English precision: {metrics['english_precision']:.4f}")
        print(f"English recall: {metrics['english_recall']:.4f}")
        print(f"English F1: {metrics['english_f1']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"False English predictions: {metrics['false_positive_english']}")
        print("Confusion matrix; order ['en', 'hi', 'other']:")
        print(np.array(metrics["confusion_matrix"]))
        print("\nClassification report:")
        print(classification_report(
            y_true,
            predictions,
            labels=LABELS,
            zero_division=0,
        ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with ERROR_PATH.open("w", encoding="utf-8", newline="") as file:
        fields = [
            "sentence_id", "sentence", "token_index", "token", "actual",
            "default_prediction", "threshold_prediction",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item, actual, default, tuned in zip(
            test_rows_for_errors, y_true, argmax_predictions, tuned_predictions
        ):
            if actual != tuned or actual != default:
                writer.writerow({
                    **item,
                    "actual": actual,
                    "default_prediction": default,
                    "threshold_prediction": tuned,
                })

    report = {
        "dataset": "COMI-LINGUA official test split; augmented candidate",
        "test_file": TEST_FILE.name,
        "test_tokens": len(y_true),
        "test_label_counts": dict(label_counts),
        "threshold_selected_on_validation": float(threshold),
        "default_argmax_metrics": argmax_metrics,
        "frozen_threshold_metrics": tuned_metrics,
        "note": (
            "The threshold was selected using the development validation split. "
            "No threshold was selected or adjusted using this test evaluation. "
            "The original dataset labels may contain annotation ambiguity; results "
            "measure agreement with the converted consensus labels."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved final test report: {REPORT_PATH}")
    print(f"Saved test error rows: {ERROR_PATH}")
    print("Evaluation completed for augmented candidate. Do not tune using this test output.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, MemoryError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
