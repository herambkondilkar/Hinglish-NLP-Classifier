import csv
import gc
import json
import random
import sys
import unicodedata
from collections import Counter
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
FIT_FILE = DATA_DIR / "train_tokens_fit_augmented.csv"
DEV_FILE = DATA_DIR / "validation_tokens.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "comi_lingua_dev_tuned_augmented.joblib"
REPORT_PATH = MODEL_DIR / "comi_lingua_dev_tuning_augmented_report.json"
THRESHOLD_PATH = MODEL_DIR / "comi_lingua_english_thresholds_augmented.csv"
ERROR_PATH = MODEL_DIR / "comi_lingua_dev_errors_augmented.csv"

LABELS = np.array(["en", "hi", "other"])
VALID_LABELS = set(LABELS.tolist())
BATCH_SIZE = 2048
EPOCHS = 3
WORD_HASH_FEATURES = 2**18
CHAR_HASH_FEATURES = 2**18
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MIN_EN_RECALL_FOR_RECOMMENDATION = 0.85
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
    headers = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return headers


def scan_dataset(path):
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
                raise ValueError(f"Empty sentence_id/sentence/token at {path.name}:{line_number}")
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected label {label!r} at {path.name}:{line_number}")
            try:
                index = int(row[headers["token_index"]])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid token_index at {path.name}:{line_number}")
            if index < 1:
                raise ValueError(f"token_index must be >= 1 at {path.name}:{line_number}")
            old_sentence = sentence_id_to_text.get(sid)
            if old_sentence is not None and old_sentence != sentence:
                raise ValueError(f"Sentence ID {sid!r} maps to more than one sentence in {path.name}")
            sentence_id_to_text[sid] = sentence
            sentence_texts.add(normalize_sentence(sentence))
            counts[label] += 1
    if rows == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return {
        "rows": rows,
        "sentence_count": len(sentence_id_to_text),
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
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = get_headers(reader, path)

        def normalized_rows():
            for raw in reader:
                yield {key: (raw.get(headers[key]) or "").strip() for key in REQUIRED_COLUMNS}

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
                    raise ValueError(f"Duplicate token index {token_index} in sentence ID {sid!r}")
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
        yield batch


def make_features(batch, word_vectorizer, char_vectorizer):
    word_matrix = word_vectorizer.transform([item["word_text"] for item in batch])
    char_matrix = char_vectorizer.transform([item["char_text"] for item in batch])
    return hstack([word_matrix, char_matrix], format="csr")


def threshold_predictions(probabilities, classes, english_threshold):
    class_index = {label: idx for idx, label in enumerate(classes)}
    en_idx = class_index["en"]
    non_en_indices = [idx for label, idx in class_index.items() if label != "en"]
    result = []
    for row in probabilities:
        if row[en_idx] >= english_threshold:
            result.append("en")
        else:
            best_non_en = max(non_en_indices, key=lambda idx: row[idx])
            result.append(classes[best_non_en])
    return np.asarray(result, dtype=str)


def measure(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=LABELS.tolist(), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS.tolist())
    en = report["en"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "en_precision": float(en["precision"]),
        "en_recall": float(en["recall"]),
        "en_f1": float(en["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": matrix.tolist(),
        "false_positive_en": int(sum(1 for a, p in zip(y_true, y_pred) if p == "en" and a != "en")),
        "false_negative_en": int(sum(1 for a, p in zip(y_true, y_pred) if a == "en" and p != "en")),
    }


def main():
    print("Scanning fit and validation metadata...")
    fit_info = scan_dataset(FIT_FILE)
    dev_info = scan_dataset(DEV_FILE)
    overlap = fit_info["sentence_texts"] & dev_info["sentence_texts"]
    if overlap:
        raise ValueError(f"Fit/validation overlap detected: {len(overlap)} sentence texts")
    if VALID_LABELS - set(fit_info["label_counts"]):
        raise ValueError(f"Fit split lacks labels: {sorted(VALID_LABELS - set(fit_info['label_counts']))}")

    print("\n=== English Precision Tuning Experiment (Local Data Augmented) ===")
    print(f"Fit token rows: {fit_info['rows']:,}")
    print(f"Fit sentence IDs: {fit_info['sentence_count']:,}")
    print(f"Fit label counts: {dict(fit_info['label_counts'])}")
    print(f"Validation token rows: {dev_info['rows']:,}")
    print(f"Validation sentence IDs: {dev_info['sentence_count']:,}")
    print(f"Validation label counts: {dict(dev_info['label_counts'])}")
    print(f"Exact normalized sentence overlap: {len(overlap)}")
    print(f"Batch size: {BATCH_SIZE}; epochs: {EPOCHS}")

    total_fit = fit_info["rows"]
    class_weights = {
        label: total_fit / (len(VALID_LABELS) * count)
        for label, count in fit_info["label_counts"].items()
    }
    word_vectorizer = HashingVectorizer(
        n_features=WORD_HASH_FEATURES, alternate_sign=False, norm="l2",
        analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
        lowercase=False, dtype=np.float32,
    )
    char_vectorizer = HashingVectorizer(
        n_features=CHAR_HASH_FEATURES, alternate_sign=False, norm="l2",
        analyzer="char", ngram_range=(2, 5), lowercase=True, dtype=np.float32,
    )
    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5, random_state=42,
        average=True, learning_rate="optimal",
    )

    initialized = False
    for epoch in range(EPOCHS):
        print(f"\nTraining epoch {epoch + 1}/{EPOCHS}...")
        processed = 0
        batch_count = 0
        for batch_count, batch in enumerate(
            iter_batches(FIT_FILE, BATCH_SIZE, shuffle_seed=5026 + epoch), start=1
        ):
            X_batch = make_features(batch, word_vectorizer, char_vectorizer)
            y_batch = np.asarray([item["label"] for item in batch], dtype=str)
            sample_weights = np.asarray([class_weights[label] for label in y_batch], dtype=np.float64)
            if not initialized:
                classifier.partial_fit(X_batch, y_batch, classes=LABELS, sample_weight=sample_weights)
                initialized = True
            else:
                classifier.partial_fit(X_batch, y_batch, sample_weight=sample_weights)
            processed += len(batch)
            if batch_count % 25 == 0:
                print(f"  batches: {batch_count}; fit tokens: {processed:,}/{fit_info['rows']:,}")
            del X_batch, y_batch, sample_weights
            if batch_count % 20 == 0:
                gc.collect()
        print(f"Epoch {epoch + 1} completed: {processed:,} token rows in {batch_count:,} batches.")
        gc.collect()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    y_true = []
    probability_rows = []
    dev_rows = []
    print("\nScoring validation split only; official test data is not loaded...")
    for batch_number, batch in enumerate(iter_batches(DEV_FILE, BATCH_SIZE), start=1):
        X_batch = make_features(batch, word_vectorizer, char_vectorizer)
        probs = classifier.predict_proba(X_batch)
        probability_rows.extend(probs.astype(np.float32).tolist())
        y_true.extend(item["label"] for item in batch)
        dev_rows.extend({
            "sentence_id": item["sentence_id"], "sentence": item["sentence"],
            "token_index": item["token_index"], "token": item["token"],
        } for item in batch)
        if batch_number % 20 == 0:
            print(f"  scored validation tokens: {len(y_true):,}/{dev_info['rows']:,}")
        del X_batch, probs

    probabilities = np.asarray(probability_rows, dtype=np.float32)
    classes = classifier.classes_.tolist()
    baseline_pred = np.asarray(classes, dtype=str)[np.argmax(probabilities, axis=1)]
    baseline_metrics = measure(y_true, baseline_pred)

    threshold_results = []
    candidate_predictions = {}
    for threshold in THRESHOLDS:
        preds = threshold_predictions(probabilities, classes, threshold)
        metrics = measure(y_true, preds)
        threshold_results.append({
            "english_threshold": threshold,
            "accuracy": metrics["accuracy"],
            "english_precision": metrics["en_precision"],
            "english_recall": metrics["en_recall"],
            "english_f1": metrics["en_f1"],
            "macro_f1": metrics["macro_f1"],
            "false_positive_english": metrics["false_positive_en"],
            "false_negative_english": metrics["false_negative_en"],
            "confusion_matrix": metrics["confusion_matrix"],
        })
        candidate_predictions[threshold] = (preds, metrics)

    eligible = [r for r in threshold_results if r["english_recall"] >= MIN_EN_RECALL_FOR_RECOMMENDATION]
    if eligible:
        best = max(eligible, key=lambda r: (r["english_precision"], r["macro_f1"], r["accuracy"]))
        selected_threshold = best["english_threshold"]
        selected_pred, selected_metrics = candidate_predictions[selected_threshold]
        selection_note = (
            f"Selected on validation as highest English precision among tested thresholds "
            f"with English recall >= {MIN_EN_RECALL_FOR_RECOMMENDATION:.2f}; ties break by macro F1, then accuracy."
        )
    else:
        selected_threshold = None
        selected_pred = baseline_pred
        selected_metrics = baseline_metrics
        selection_note = (
            "No tested threshold met the minimum English recall requirement; "
            "default argmax retained for further review."
        )

    print("\n=== VALIDATION RESULTS: DEFAULT ARGMAX ===")
    print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"English precision: {baseline_metrics['en_precision']:.4f}")
    print(f"English recall: {baseline_metrics['en_recall']:.4f}")
    print(f"English F1: {baseline_metrics['en_f1']:.4f}")
    print(f"Macro F1: {baseline_metrics['macro_f1']:.4f}")
    print("Confusion matrix; order ['en', 'hi', 'other']:")
    print(np.array(baseline_metrics["confusion_matrix"]))

    print("\n=== ENGLISH THRESHOLD COMPARISON ===")
    print("threshold | accuracy | en_precision | en_recall | en_f1 | macro_f1 | false_en_FP")
    for result in threshold_results:
        print(
            f"{result['english_threshold']:.2f} | {result['accuracy']:.4f} | "
            f"{result['english_precision']:.4f} | {result['english_recall']:.4f} | "
            f"{result['english_f1']:.4f} | {result['macro_f1']:.4f} | "
            f"{result['false_positive_english']}"
        )

    print("\n=== SELECTED VALIDATION OPERATING POINT ===")
    print("English probability threshold:", selected_threshold if selected_threshold is not None else "default argmax")
    print(f"Validation accuracy: {selected_metrics['accuracy']:.4f}")
    print(f"English precision: {selected_metrics['en_precision']:.4f}")
    print(f"English recall: {selected_metrics['en_recall']:.4f}")
    print(f"English F1: {selected_metrics['en_f1']:.4f}")
    print(f"Macro F1: {selected_metrics['macro_f1']:.4f}")
    print(selection_note)

    # Save this development candidate separately; do not alter the test-evaluated model.
    joblib.dump({
        "classifier": classifier,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "labels": classes,
        "english_threshold": selected_threshold,
        "feature_config": {
            "word_hash_features": WORD_HASH_FEATURES,
            "char_hash_features": CHAR_HASH_FEATURES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "fit_file": FIT_FILE.name,
            "validation_file": DEV_FILE.name,
        },
        "threshold_note": selection_note,
    }, MODEL_PATH, compress=1)

    threshold_columns = [
        "english_threshold", "accuracy", "english_precision", "english_recall",
        "english_f1", "macro_f1", "false_positive_english",
        "false_negative_english", "confusion_matrix",
    ]
    with THRESHOLD_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=threshold_columns)
        writer.writeheader()
        for result in threshold_results:
            row = dict(result)
            row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
            writer.writerow(row)

    with ERROR_PATH.open("w", encoding="utf-8", newline="") as file:
        columns = ["sentence_id", "sentence", "token_index", "token", "actual", "default_prediction", "selected_prediction"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row, actual, default, selected in zip(dev_rows, y_true, baseline_pred, selected_pred):
            if actual != selected or actual != default:
                writer.writerow({**row, "actual": actual, "default_prediction": default, "selected_prediction": selected})

    report = {
        "dataset": "COMI-LINGUA development tuning with local manual augmentation",
        "fit_rows": fit_info["rows"],
        "fit_sentence_count": fit_info["sentence_count"],
        "validation_rows": dev_info["rows"],
        "validation_sentence_count": dev_info["sentence_count"],
        "fit_validation_sentence_overlap": len(overlap),
        "class_weights": class_weights,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "baseline_argmax": baseline_metrics,
        "threshold_results": threshold_results,
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": selected_metrics,
        "threshold_selection_note": selection_note,
        "note": (
            "Threshold selection uses development validation only. These selected metrics are not a final test score. "
            "The official COMI-LINGUA test file was not accessed by this script."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved candidate model: {MODEL_PATH}")
    print(f"Saved threshold table: {THRESHOLD_PATH}")
    print(f"Saved validation report: {REPORT_PATH}")
    print(f"Saved validation errors: {ERROR_PATH}")
    print("\nValidation tuning completed; official test data was not accessed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, MemoryError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        if isinstance(error, MemoryError):
            print("MemoryError: share the terminal output before retrying.", file=sys.stderr)
        sys.exit(1)
