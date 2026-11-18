
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"

TRAIN_FILE = DATA_DIR / "train_tokens.csv"
TEST_FILE = DATA_DIR / "test_tokens.csv"

REQUIRED_COLUMNS = {
    "sentence_id",
    "sentence",
    "token_index",
    "token",
    "language",
    "votes",
    "annotator_count",
    "agreement",
}
VALID_LABELS = {"en", "hi", "other"}
VALID_AGREEMENTS = {"unanimous", "majority"}


def normalize(text):
    return " ".join((text or "").strip().lower().split())


def read_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"No header found in {path.name}")

        headers = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name
        }

        missing = REQUIRED_COLUMNS - set(headers)
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        rows = [
            {
                key: (row.get(headers[key]) or "").strip()
                for key in REQUIRED_COLUMNS
            }
            for row in reader
        ]

    return rows


def validate_file(path):
    rows = read_dataset(path)

    errors = []
    labels = Counter()
    agreements = Counter()
    positions = set()
    sentence_texts = {}
    unique_sentences = set()

    for line_number, row in enumerate(rows, start=2):
        sentence_id = row["sentence_id"]
        sentence = row["sentence"]
        token = row["token"]
        label = row["language"].lower()
        agreement = row["agreement"].lower()

        if not sentence_id or not sentence or not token:
            errors.append(
                f"Line {line_number}: empty ID, sentence, or token."
            )
            continue

        try:
            token_index = int(row["token_index"])
            votes = int(row["votes"])
            annotator_count = int(row["annotator_count"])
        except ValueError:
            errors.append(
                f"Line {line_number}: invalid numeric field."
            )
            continue

        if token_index < 1:
            errors.append(
                f"Line {line_number}: token_index must be >= 1."
            )

        if label not in VALID_LABELS:
            errors.append(
                f"Line {line_number}: invalid label {label!r}."
            )

        if not 2 <= votes <= 3:
            errors.append(
                f"Line {line_number}: invalid vote count {votes}."
            )

        if not 2 <= annotator_count <= 3:
            errors.append(
                f"Line {line_number}: invalid annotator count "
                f"{annotator_count}."
            )

        if votes > annotator_count:
            errors.append(
                f"Line {line_number}: votes exceed annotator count."
            )

        if agreement not in VALID_AGREEMENTS:
            errors.append(
                f"Line {line_number}: invalid agreement {agreement!r}."
            )

        if agreement == "unanimous" and (
            votes != 3 or annotator_count != 3
        ):
            errors.append(
                f"Line {line_number}: inconsistent unanimous record."
            )

        key = (sentence_id, token_index)

        if key in positions:
            errors.append(
                f"Line {line_number}: duplicate position {key}."
            )
        positions.add(key)

        normalized_sentence = normalize(sentence)
        unique_sentences.add(normalized_sentence)

        previous_sentence = sentence_texts.get(sentence_id)

        if (
            previous_sentence is not None
            and previous_sentence != sentence
        ):
            errors.append(
                f"Line {line_number}: sentence ID maps to "
                "inconsistent sentence text."
            )

        sentence_texts[sentence_id] = sentence

        labels[label] += 1
        agreements[agreement] += 1

    print(f"\n=== {path.name} ===")
    print(f"Exported token rows: {len(rows)}")
    print(f"Unique sentence IDs: {len(sentence_texts)}")
    print(f"Unique sentence texts: {len(unique_sentences)}")

    print("\nLabel distribution:")
    for label in sorted(VALID_LABELS):
        print(f"{label}: {labels[label]}")

    print("\nAgreement distribution:")
    for agreement in sorted(VALID_AGREEMENTS):
        print(f"{agreement}: {agreements[agreement]}")

    print(f"\nStructural issues: {len(errors)}")

    for error in errors[:15]:
        print("-", error)

    if len(errors) > 15:
        print(f"...and {len(errors) - 15} more issues.")

    return {
        "rows": len(rows),
        "sentences": unique_sentences,
        "issues": errors,
    }


def main():
    train = validate_file(TRAIN_FILE)
    test = validate_file(TEST_FILE)

    overlap = train["sentences"] & test["sentences"]

    print("\n=== TRAIN/TEST SEPARATION ===")
    print(f"Exact normalized sentences in both splits: {len(overlap)}")

    if overlap:
        print(
            "WARNING: Some complete sentence texts occur in both splits. "
            "Investigate these before claiming independent evaluation."
        )
    else:
        print("No exact sentence-text overlap detected.")

    total_issues = len(train["issues"]) + len(test["issues"])

    print("\n=== FINAL RESULT ===")
    print(f"Combined structural issues: {total_issues}")

    if total_issues:
        print("Validation failed. Do not train yet.")
        return 1

    print("Structural validation passed.")
    print("This does not prove every linguistic label is correct.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
