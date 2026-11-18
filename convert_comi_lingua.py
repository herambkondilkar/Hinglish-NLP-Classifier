
import ast
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "raw" / "comi_lingua"
OUTPUT_DIR = ROOT / "data" / "processed" / "comi_lingua"

TRAIN_FILE = SOURCE_DIR / "LID_train.csv"
TEST_FILE = SOURCE_DIR / "LID_test.csv"

ANNOTATORS = [
    "Annotated by: Annotator 1",
    "Annotated by: Annotator 2",
    "Annotated by: Annotator 3",
]

# Normalize observed short codes to the dataset's three LID labels.
# These aliases are inferred from the downloaded records.
LABEL_ALIASES = {
    "hi": "hi",
    "h": "hi",
    "en": "en",
    "e": "en",
    "ot": "ot",
    "u": "ot",
}

OUTPUT_LABELS = {
    "hi": "hi",
    "en": "en",
    "ot": "other",
}


def normalize_token(value):
    return " ".join(str(value or "").strip().lower().split())


def normalize_label(value):
    raw_label = str(value or "").strip().lower()
    return LABEL_ALIASES.get(raw_label)


def log_issue(issues, filename, row_number, annotator,
              item_number, token, raw_index, reason):
    issues.append({
        "file": filename,
        "csv_row": row_number,
        "annotator": annotator,
        "item_number": item_number,
        "token": str(token or ""),
        "raw_index": str(raw_index),
        "reason": reason,
    })


def read_tags(raw_value, filename, row_number,
              annotator, issues, alias_counts):
    """Parse annotations, normalize known codes, and record bad items."""
    if not raw_value or not raw_value.strip():
        return {}

    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as error:
        log_issue(
            issues, filename, row_number, annotator,
            "", "", "", f"Could not parse annotation list: {error}"
        )
        return {}

    if not isinstance(parsed, list):
        log_issue(
            issues, filename, row_number, annotator,
            "", "", "", "Annotation value was not a list"
        )
        return {}

    result = {}

    for item_number, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            log_issue(
                issues, filename, row_number, annotator,
                item_number, "", "", "Annotation item was not a dictionary"
            )
            continue

        token = str(item.get("key") or "").strip()
        raw_index = item.get("index")
        raw_label = str(item.get("value") or "").strip().lower()
        label = normalize_label(raw_label)

        if raw_index is None or str(raw_index).strip() == "":
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index,
                "Missing token index; skipped without guessing position"
            )
            continue

        try:
            index = int(raw_index)
        except (ValueError, TypeError):
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index,
                "Token index was not an integer"
            )
            continue

        if index < 0:
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index, "Negative token index"
            )
            continue

        if not token:
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index, "Missing token text"
            )
            continue

        if label is None:
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index,
                f"Unrecognized language label: {raw_label!r}"
            )
            continue

        if raw_label != label:
            alias_counts[raw_label] += 1

        if index in result:
            log_issue(
                issues, filename, row_number, annotator,
                item_number, token, raw_index,
                "Duplicate token index in one annotator's record"
            )
            continue

        result[index] = {"token": token, "label": label}

    return result


def read_source_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"No CSV header in {path.name}")

        headers = {
            name.strip().lower(): name
            for name in reader.fieldnames if name
        }

        required = {"sentences"} | {
            name.lower() for name in ANNOTATORS
        }

        missing = required - set(headers)
        if missing:
            raise ValueError(
                f"{path.name} is missing columns {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        return list(reader), headers


def process_file(path, split_name, parse_issues, alias_counts):
    source_rows, headers = read_source_rows(path)
    processed = []
    disagreements = []
    stats = Counter()

    for row_number, row in enumerate(source_rows, start=2):
        stats["source_rows"] += 1
        sentence = (row.get(headers["sentences"]) or "").strip()

        if not sentence:
            stats["empty_sentences"] += 1
            continue

        annotations = []

        for annotator in ANNOTATORS:
            raw = row.get(headers[annotator.lower()], "")
            annotations.append(
                read_tags(
                    raw, path.name, row_number, annotator,
                    parse_issues, alias_counts
                )
            )

        all_indices = sorted({
            index
            for annotation in annotations
            for index in annotation
        })

        for index in all_indices:
            stats["indexed_positions_seen"] += 1

            entries = [
                annotation[index]
                for annotation in annotations
                if index in annotation
            ]

            token_counts = Counter(
                normalize_token(entry["token"])
                for entry in entries
            )

            if not token_counts:
                stats["skipped_positions"] += 1
                continue

            token_key, token_votes = token_counts.most_common(1)[0]

            # Require two annotators to agree on token identity.
            if token_votes < 2:
                stats["skipped_positions"] += 1
                disagreements.append({
                    "split": split_name,
                    "source_row": row_number,
                    "sentence": sentence,
                    "token_index_zero_based": index,
                    "tokens": " | ".join(
                        entry["token"] for entry in entries
                    ),
                    "labels": " | ".join(
                        entry["label"] for entry in entries
                    ),
                    "reason": "No majority agreement on token identity",
                })
                continue

            matching = [
                entry for entry in entries
                if normalize_token(entry["token"]) == token_key
            ]

            label_counts = Counter(
                entry["label"] for entry in matching
            )
            majority_label, votes = label_counts.most_common(1)[0]

            # Require two annotators to agree on the language label.
            if votes < 2:
                stats["skipped_positions"] += 1
                disagreements.append({
                    "split": split_name,
                    "source_row": row_number,
                    "sentence": sentence,
                    "token_index_zero_based": index,
                    "tokens": " | ".join(
                        entry["token"] for entry in matching
                    ),
                    "labels": " | ".join(
                        entry["label"] for entry in matching
                    ),
                    "reason": "No majority agreement on language label",
                })
                continue

            if len(matching) == 3 and votes == 3:
                agreement = "unanimous"
                stats["unanimous_positions"] += 1
            else:
                agreement = "majority"
                stats["majority_positions"] += 1

            processed.append({
                "sentence_id": f"{split_name}_{row_number}",
                "sentence": sentence,
                "token_index": index + 1,
                "token": matching[0]["token"],
                "language": OUTPUT_LABELS[majority_label],
                "votes": votes,
                "annotator_count": len(matching),
                "agreement": agreement,
            })

            if len(label_counts) > 1 or len(token_counts) > 1:
                disagreements.append({
                    "split": split_name,
                    "source_row": row_number,
                    "sentence": sentence,
                    "token_index_zero_based": index,
                    "tokens": " | ".join(
                        entry["token"] for entry in entries
                    ),
                    "labels": " | ".join(
                        entry["label"] for entry in entries
                    ),
                    "reason": "Annotator disagreement; majority label used",
                })

    return processed, disagreements, stats


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_disagreements = []
    all_parse_issues = []
    total_alias_counts = Counter()

    output_columns = [
        "sentence_id", "sentence", "token_index", "token",
        "language", "votes", "annotator_count", "agreement"
    ]

    for path, split_name in [
        (TRAIN_FILE, "train"),
        (TEST_FILE, "test"),
    ]:
        parse_issues = []
        alias_counts = Counter()

        processed, disagreements, stats = process_file(
            path, split_name, parse_issues, alias_counts
        )

        output_path = OUTPUT_DIR / f"{split_name}_tokens.csv"
        write_csv(output_path, processed, output_columns)

        all_disagreements.extend(disagreements)
        all_parse_issues.extend(parse_issues)
        total_alias_counts.update(alias_counts)

        label_counts = Counter(
            row["language"] for row in processed
        )

        print(f"\n=== {split_name.upper()} CONVERSION ===")
        print(f"Source rows: {stats['source_rows']}")
        print(f"Empty sentences: {stats['empty_sentences']}")
        print(f"Indexed positions: {stats['indexed_positions_seen']}")
        print(f"Consensus tokens exported: {len(processed)}")
        print(f"Unanimous positions: {stats['unanimous_positions']}")
        print(f"Majority positions: {stats['majority_positions']}")
        print(f"Skipped positions: {stats['skipped_positions']}")
        print(f"Label distribution: {dict(label_counts)}")
        print(f"Malformed items: {len(parse_issues)}")
        print(f"Alias codes normalized: {dict(alias_counts)}")
        print(f"Output: {output_path}")

    write_csv(
        OUTPUT_DIR / "annotator_disagreements.csv",
        all_disagreements,
        [
            "split", "source_row", "sentence",
            "token_index_zero_based", "tokens", "labels", "reason"
        ],
    )

    write_csv(
        OUTPUT_DIR / "annotation_parse_issues.csv",
        all_parse_issues,
        [
            "file", "csv_row", "annotator", "item_number",
            "token", "raw_index", "reason"
        ],
    )

    print("\n=== CONVERSION FINISHED ===")
    print(f"Alias codes normalized in total: {dict(total_alias_counts)}")
    print(
        "Disagreement report:",
        OUTPUT_DIR / "annotator_disagreements.csv"
    )
    print(
        "Malformed-item report:",
        OUTPUT_DIR / "annotation_parse_issues.csv"
    )
    print(
        "\nOriginal source files were not modified. "
        "Test tokens remain separate from training tokens."
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
