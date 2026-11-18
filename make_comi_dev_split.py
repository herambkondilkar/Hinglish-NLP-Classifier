
import csv
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"

SOURCE_FILE = DATA_DIR / "train_tokens_clean.csv"
FIT_FILE = DATA_DIR / "train_tokens_fit.csv"
DEV_FILE = DATA_DIR / "validation_tokens.csv"

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42


def normalize_sentence(text):
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.casefold().split())


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Clean training file not found: {SOURCE_FILE}"
        )

    # First pass: collect unique sentence texts only.
    unique_sentences = set()

    with SOURCE_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "sentence_id", "sentence", "token_index",
            "token", "language"
        }

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"Unexpected columns in {SOURCE_FILE.name}: "
                f"{reader.fieldnames}"
            )

        fields = reader.fieldnames

        for row in reader:
            sentence = (row.get("sentence") or "").strip()

            if not sentence:
                raise ValueError("Training data contains an empty sentence.")

            unique_sentences.add(normalize_sentence(sentence))

    if len(unique_sentences) < 2:
        raise ValueError("Not enough unique sentences for a split.")

    sentence_list = sorted(unique_sentences)
    random.Random(RANDOM_SEED).shuffle(sentence_list)

    validation_count = max(
        1,
        round(len(sentence_list) * VALIDATION_FRACTION)
    )

    validation_sentences = set(sentence_list[:validation_count])
    fit_sentences = set(sentence_list[validation_count:])

    # Second pass: stream token rows to separate files.
    fit_counts = Counter()
    dev_counts = Counter()
    fit_rows = 0
    dev_rows = 0

    with SOURCE_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, \
         FIT_FILE.open(
            "w", encoding="utf-8", newline=""
         ) as fit_out, \
         DEV_FILE.open(
            "w", encoding="utf-8", newline=""
         ) as dev_out:

        reader = csv.DictReader(source)
        fit_writer = csv.DictWriter(fit_out, fieldnames=fields)
        dev_writer = csv.DictWriter(dev_out, fieldnames=fields)

        fit_writer.writeheader()
        dev_writer.writeheader()

        for row in reader:
            sentence_key = normalize_sentence(row["sentence"])
            label = row["language"].strip().lower()

            if sentence_key in validation_sentences:
                dev_writer.writerow(row)
                dev_counts[label] += 1
                dev_rows += 1
            elif sentence_key in fit_sentences:
                fit_writer.writerow(row)
                fit_counts[label] += 1
                fit_rows += 1
            else:
                raise ValueError(
                    "Sentence could not be assigned to either split."
                )

    if fit_rows == 0 or dev_rows == 0:
        raise ValueError("The split produced an empty output.")

    overlap = fit_sentences & validation_sentences

    print("\n=== COMI-LINGUA Development Split ===")
    print(f"Source unique sentences: {len(unique_sentences)}")
    print(f"Fit sentences: {len(fit_sentences)}")
    print(f"Validation sentences: {len(validation_sentences)}")
    print(f"Fit token rows: {fit_rows}")
    print(f"Validation token rows: {dev_rows}")
    print(f"Fit label counts: {dict(fit_counts)}")
    print(f"Validation label counts: {dict(dev_counts)}")
    print(f"Sentence overlap: {len(overlap)}")
    print(f"\nFit data: {FIT_FILE}")
    print(f"Validation data: {DEV_FILE}")

    if overlap:
        raise ValueError("Unexpected sentence overlap detected.")

    print("\nDevelopment split created. Official test file unchanged.")
    print("Use train_tokens_fit.csv for tuning.")
    print("Use validation_tokens.csv for model comparisons.")
    print("Keep test_tokens.csv reserved for final evaluation.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
