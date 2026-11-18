"""Download COMI-LINGUA's token-level LID train/test CSVs on Windows.

Run from the Hinglish-NLP-Classifier project root:
    python download_comi_lingua.py

Files are saved to data/raw/comi_lingua/. Existing valid files are kept.
"""

import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "raw" / "comi_lingua"
FILES = ("LID_train.csv", "LID_test.csv")
REPOSITORIES = (
    "LingoIITGN/COMI-LINGUA",       # Dataset maintainer's Hugging Face repo
    "Tanushreeeeee/COMI-LINGUA",   # Public mirror of the same dataset
)
CHUNK_SIZE = 1024 * 1024
MIN_VALID_SIZE = 1_000_000


def validate_csv(path: Path) -> bool:
    """Reject Git-LFS pointers, HTML error pages, and incomplete downloads."""
    try:
        if not path.exists() or path.stat().st_size < MIN_VALID_SIZE:
            return False
        with path.open("rb") as f:
            prefix = f.read(512).lstrip()
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return False
        if prefix.lower().startswith((b"<!doctype html", b"<html")):
            return False
        # The corpus CSV contains multiple columns. Read only the header row.
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next((row for row in reader if row and any(c.strip() for c in row)), None)
        return bool(header and len(header) >= 2)
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return False


def download_one(filename: str) -> Path:
    destination = OUTPUT_DIR / filename
    if validate_csv(destination):
        print(f"Already present and looks valid: {destination}")
        return destination

    # Do not leave an incomplete prior file in place.
    destination.unlink(missing_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    for repo in REPOSITORIES:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        print(f"\nDownloading {filename} from {repo}...")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Hinglish-NLP-Classifier-dataset-downloader/1.0",
                "Accept": "text/csv,application/octet-stream,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else None
                downloaded = 0
                last_report = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= 10 * CHUNK_SIZE:
                        if total:
                            print(f"  {downloaded / 1048576:.0f} / {total / 1048576:.0f} MB", flush=True)
                        else:
                            print(f"  Downloaded {downloaded / 1048576:.0f} MB", flush=True)
                        last_report = downloaded

            if not validate_csv(partial):
                raise ValueError(
                    "Downloaded content failed CSV/size checks (possibly an HTML error or Git-LFS pointer)."
                )

            partial.replace(destination)
            print(f"Saved: {destination} ({destination.stat().st_size / 1048576:.1f} MB)")
            return destination
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, csv.Error) as exc:
            errors.append(f"{repo}: {exc}")
            partial.unlink(missing_ok=True)
            print(f"Could not download from {repo}: {exc}")

    raise RuntimeError(
        f"Could not download {filename}. Details:\n  " + "\n  ".join(errors)
    )


def main() -> int:
    print("COMI-LINGUA LID dataset downloader")
    print(f"Destination: {OUTPUT_DIR}")
    print("These are large files; keep the terminal open until completion.\n")
    try:
        saved = [download_one(name) for name in FILES]
    except Exception as exc:
        print(f"\nDOWNLOAD FAILED: {exc}", file=sys.stderr)
        print("No training or test files were merged into your project data.")
        return 1

    print("\nDownload complete. Files kept separate:")
    for path in saved:
        print(f"  {path}")
    print("\nKeep LID_test.csv separate; do not use it for training or tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
