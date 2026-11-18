
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"

TRAIN_FILE = DATA_DIR / "train_tokens.csv"
TEST_FILE = DATA_DIR / "test_tokens.csv"
OUTPUT_FILE = DATA_DIR / "train_tokens_clean.csv"


def normalize_sentence(text):
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.casefold().split())


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {path.name}")

        headers = set(reader.fieldnames)
        required = {"sentence_id", "sentence", "token_index", "token", "language"}

        if not required.issubset(headers):
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{sorted(required - headers)}"
            )

        rows = list(reader)
        return reader.fieldnames, rows


def write_csv(path, columns, rows):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    train_columns, train_rows = read_csv(TRAIN_FILE)
    _, test_rows = read_csv(TEST_FILE)

    if not train_rows or not test_rows:
        raise ValueError("Training or test dataset is empty.")

    test_sentences = {
        normalize_sentence(row["sentence"])
        for row in test_rows
    }

    original_train_sentences = {
        normalize_sentence(row["sentence"])
        for row in train_rows
    }

    overlap_before = (
        original_train_sentences & test_sentences
    )

    # Exclude every training token belonging to an overlapping sentence.
    clean_train_rows = [
        row for row in train_rows
        if normalize_sentence(row["sentence"]) not in test_sentences
    ]

    clean_train_sentences = {
        normalize_sentence(row["sentence"])
        for row in clean_train_rows
    }

    overlap_after = clean_train_sentences & test_sentences

    removed_sentence_ids = {
        row["sentence_id"] for row in train_rows
        if normalize_sentence(row["sentence"]) in test_sentences
    }

    if not clean_train_rows:
        raise ValueError(
            "No training rows remain after removing overlap."
        )

    if overlap_after:
        raise ValueError(
            f"Overlap still exists after cleaning: {len(overlap_after)}"
        )

    # Write a separate file; the original train and test files stay intact.
    write_csv(OUTPUT_FILE, train_columns, clean_train_rows)

    print("\n=== Train/Test Overlap Removal ===")
    print(f"Original training token rows: {len(train_rows)}")
    print(f"Training token rows retained: {len(clean_train_rows)}")
    print(f"Training token rows removed: {len(train_rows) - len(clean_train_rows)}")
    print(f"Overlapping sentence texts found: {len(overlap_before)}")
    print(f"Training sentence IDs removed: {len(removed_sentence_ids)}")
    print(f"Test token rows preserved: {len(test_rows)}")
    print(f"Overlapping sentences remaining: {len(overlap_after)}")
    print(f"Clean training file: {OUTPUT_FILE}")

    if overlap_after:
        print("\nERROR: Training/test overlap remains.")
        return 1

    print("\nCleaning successful. Test data was not modified.")
    print("Use train_tokens_clean.csv for training.")
    print("Keep test_tokens.csv for final evaluation.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
