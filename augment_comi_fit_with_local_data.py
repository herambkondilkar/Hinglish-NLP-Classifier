import csv
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "comi_lingua"
LOCAL_FILE = ROOT / "data" / "word_labels.csv"
CHALLENGE_FILE = ROOT / "data" / "real_world_hinglish_eval.csv"
FIT_FILE = DATA_DIR / "train_tokens_fit.csv"
DEV_FILE = DATA_DIR / "validation_tokens.csv"
TEST_FILE = DATA_DIR / "test_tokens.csv"
OUTPUT_FILE = DATA_DIR / "train_tokens_fit_augmented.csv"

FIT_REQUIRED = {
    "sentence_id", "sentence", "token_index", "token", "language",
    "votes", "annotator_count", "agreement",
}
LOCAL_REQUIRED = {"sentence", "token_index", "token", "language"}
VALID_LABELS = {"en", "hi", "other"}
LOCAL_REPEAT_COUNT = 8
RANDOM_SEED = 42


def norm(text):
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.casefold().split())


def get_headers(reader, path, required):
    if not reader.fieldnames:
        raise ValueError(f"No CSV header found: {path}")
    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return headers


def scan_sentence_texts(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = get_headers(reader, path, required_columns)
        texts = set()
        rows = 0
        for row in reader:
            rows += 1
            sentence = (row.get(headers["sentence"]) or "").strip()
            if sentence:
                texts.add(norm(sentence))
        return texts, rows


def scan_fit_metadata(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = get_headers(reader, path, FIT_REQUIRED)
        texts = set()
        sentence_ids = set()
        row_count = 0
        for row in reader:
            row_count += 1
            sid = (row.get(headers["sentence_id"]) or "").strip()
            sentence = (row.get(headers["sentence"]) or "").strip()
            if not sid or not sentence:
                raise ValueError(f"Empty sentence_id/sentence in {path.name}")
            sentence_ids.add(sid)
            texts.add(norm(sentence))
        return reader.fieldnames, headers, texts, sentence_ids, row_count


def main():
    if OUTPUT_FILE.exists():
        print(f"ERROR: Output already exists: {OUTPUT_FILE}")
        print("Rename or back up that file before rerunning this script.")
        return 1

    fit_columns, fit_headers, fit_texts, fit_ids, fit_row_count = scan_fit_metadata(FIT_FILE)
    dev_texts, _ = scan_sentence_texts(DEV_FILE, {"sentence"})
    test_texts, _ = scan_sentence_texts(TEST_FILE, {"sentence"})
    challenge_texts, _ = scan_sentence_texts(CHALLENGE_FILE, {"sentence"})
    protected = fit_texts | dev_texts | test_texts | challenge_texts

    if fit_texts & dev_texts or fit_texts & test_texts:
        raise ValueError("Existing COMI fit split overlaps validation/test; aborting augmentation.")

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(f"Local word-label file not found: {LOCAL_FILE}")

    with LOCAL_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        local_headers = get_headers(reader, LOCAL_FILE, LOCAL_REQUIRED)
        local_rows = list(reader)

    grouped = defaultdict(list)
    original_sentence = {}
    excluded = Counter()

    for row in local_rows:
        sentence = (row.get(local_headers["sentence"]) or "").strip()
        token = (row.get(local_headers["token"]) or "").strip()
        label = (row.get(local_headers["language"]) or "").strip().lower()
        raw_index = (row.get(local_headers["token_index"]) or "").strip()
        key = norm(sentence)

        if not key or not token:
            excluded["empty_sentence_or_token"] += 1
            continue
        if label not in VALID_LABELS:
            excluded["invalid_label"] += 1
            continue
        try:
            index = int(raw_index)
        except ValueError:
            excluded["invalid_index"] += 1
            continue
        if index < 1:
            excluded["invalid_index"] += 1
            continue

        original_sentence.setdefault(key, sentence)
        grouped[key].append({"index": index, "token": token, "language": label})

    eligible_groups = []
    conflicting = 0
    for local_num, (sentence_key, tokens) in enumerate(grouped.items(), start=1):
        sentence = original_sentence[sentence_key]

        if sentence_key in protected:
            if sentence_key in challenge_texts:
                excluded["overlaps_real_world_challenge"] += 1
            elif sentence_key in test_texts:
                excluded["overlaps_comi_test"] += 1
            elif sentence_key in dev_texts:
                excluded["overlaps_comi_validation"] += 1
            else:
                excluded["already_in_comi_fit"] += 1
            continue

        # This augmentation specifically targets Roman-script Hindi/Hinglish.
        if re.search(r"[\u0900-\u097F]", sentence):
            excluded["not_roman_script"] += 1
            continue
        if not any(item["language"] == "hi" for item in tokens):
            excluded["no_hindi_label"] += 1
            continue

        by_index = {}
        token_at = {}
        conflict = False
        for item in tokens:
            pos = item["index"]
            value = (norm(item["token"]), item["language"])
            if pos in by_index and by_index[pos] != value:
                conflict = True
                break
            by_index[pos] = value
            token_at.setdefault(pos, item)

        if conflict or not by_index:
            excluded["conflicting_or_empty_sentence"] += 1
            conflicting += 1
            continue

        indices = sorted(by_index)
        if indices != list(range(1, len(indices) + 1)):
            excluded["non_contiguous_indices"] += 1
            continue

        group_rows = []
        for index in indices:
            item = token_at[index]
            group_rows.append({
                "sentence_id": "",  # assigned uniquely per repeat below
                "sentence": sentence,
                "token_index": index,
                "token": item["token"],
                "language": item["language"],
                "votes": 1,
                "annotator_count": 1,
                "agreement": "local_manual_single_source",
            })
        eligible_groups.append((local_num, sentence_key, group_rows))

    if not eligible_groups:
        print("No eligible Roman-script Hindi/Hinglish local sentences remain after overlap checks.")
        print("Excluded counts:", dict(excluded))
        print("No output file was created.")
        return 1

    # Oversample this small supplemental set and interleave it by sentence group
    # throughout the fit file, not only at the end of each training epoch.
    rng = random.Random(RANDOM_SEED)
    repeated_groups = []
    for repeat in range(1, LOCAL_REPEAT_COUNT + 1):
        for local_num, sentence_key, group_rows in eligible_groups:
            unique_id = f"local_aug_r{repeat:02d}_{local_num:06d}"
            repeated_groups.append([
                {**row, "sentence_id": unique_id}
                for row in group_rows
            ])
    rng.shuffle(repeated_groups)

    fit_sentence_count = len(fit_ids)
    inject_every = max(1, round(fit_sentence_count / len(repeated_groups)))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    base_rows_written = 0
    base_groups_written = 0
    local_rows_written = 0
    local_groups_written = 0
    local_cursor = 0
    output_sentence_texts = set()

    with FIT_FILE.open("r", encoding="utf-8-sig", newline="") as source, \
         OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(source)
        headers = get_headers(reader, FIT_FILE, FIT_REQUIRED)
        writer = csv.DictWriter(output, fieldnames=fit_columns)
        writer.writeheader()

        for sentence_id, group_iter in groupby(
            reader, key=lambda r: (r.get(headers["sentence_id"]) or "").strip()
        ):
            group_rows = list(group_iter)
            writer.writerows(group_rows)
            base_rows_written += len(group_rows)
            base_groups_written += 1
            output_sentence_texts.add(norm(group_rows[0].get(headers["sentence"], "")))

            if base_groups_written % inject_every == 0 and local_cursor < len(repeated_groups):
                extra_group = repeated_groups[local_cursor]
                for extra_row in extra_group:
                    normalized_extra = {k.strip().lower(): k for k in fit_columns}
                    # The current training CSV schema is standardized, but map defensively.
                    mapped = {}
                    for key, value in extra_row.items():
                        mapped[normalized_extra[key]] = value
                    writer.writerow(mapped)
                    local_rows_written += 1
                local_groups_written += 1
                output_sentence_texts.add(norm(extra_group[0]["sentence"]))
                local_cursor += 1

        while local_cursor < len(repeated_groups):
            extra_group = repeated_groups[local_cursor]
            for extra_row in extra_group:
                normalized_extra = {k.strip().lower(): k for k in fit_columns}
                mapped = {normalized_extra[key]: value for key, value in extra_row.items()}
                writer.writerow(mapped)
                local_rows_written += 1
            local_groups_written += 1
            output_sentence_texts.add(norm(extra_group[0]["sentence"]))
            local_cursor += 1

    # Confirm no text was added to validation, official test, or challenge set.
    leaked = output_sentence_texts & (dev_texts | test_texts | challenge_texts)
    if leaked:
        OUTPUT_FILE.unlink(missing_ok=True)
        raise ValueError(
            f"Post-write overlap found ({len(leaked)} sentences). Output was removed."
        )

    print("\n=== Local Romanized Hindi/Hinglish Augmentation ===")
    print(f"Original fit token rows: {fit_row_count:,}")
    print(f"Original fit sentence IDs: {fit_sentence_count:,}")
    print(f"Local annotation rows read: {len(local_rows):,}")
    print(f"Unique local sentences examined: {len(grouped):,}")
    print(f"Eligible distinct Roman-script Hindi/Hinglish sentences: {len(eligible_groups):,}")
    print(f"Repeat factor: {LOCAL_REPEAT_COUNT}x")
    print(f"Supplemental sentence groups inserted: {local_groups_written:,}")
    print(f"Supplemental token rows inserted: {local_rows_written:,}")
    print(f"New fit token rows: {base_rows_written + local_rows_written:,}")
    print(f"Excluded counts: {dict(excluded)}")
    print(f"Conflicting local sentences: {conflicting}")
    print("Exact overlap with validation/test/challenge after augmentation: 0")
    print(f"New fit file: {OUTPUT_FILE}")
    print("Original fit, validation, test, and challenge files were not modified.")
    print("Local rows are marked local_manual_single_source, not multi-annotator consensus.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
