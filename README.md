# Hinglish NLP Classifier

A Python desktop prototype for **token-level language identification** in Hindi-English code-mixed text (Hinglish). It classifies each token as **English**, **Hindi**, or **Other**, including Romanized Hindi, Devanagari Hindi, English words, numerals, and punctuation.

## Highlights

- Desktop GUI built with Python Tkinter.
- Token-level predictions with confidence information.
- Context-aware lexical features, character n-gram features, script and token-shape signals.
- English decision threshold tuned on a development split.
- Export predictions to CSV and copy an analysis summary.
- One-click Windows launcher.
- Training, validation, tuning and evaluation scripts included.

## Demo input

```text
Mujhe Python mein help chahiye
```

Example token labels from the included model:

| Token | Prediction |
|---|---|
| Mujhe | Hindi |
| Python | English |
| mein | Hindi |
| help | English |
| chahiye | Hindi |

## Run on Windows

**Requirements:** Python 3.11+ (64-bit) and Tkinter.

1. Clone or download this repository.
2. Add the trained model checkpoint at `models/comi_lingua_dev_tuned_augmented.joblib`. The binary checkpoint is distributed in the companion GitHub-ready ZIP, but is not stored in this repository yet. See [models/README.md](models/README.md).
3. Open PowerShell in the project folder.
4. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Launch the application:

```powershell
python hinglish_desktop_app.py
```

You can also double-click `Launch_Hinglish_Classifier.bat`.

## Model and evaluation

The desktop application uses the three-class COMI-LINGUA model with an English probability threshold of `0.75`. The threshold was selected on the development split and evaluated on the COMI-LINGUA held-out test split.

| Metric | Result |
|---|---:|
| Test tokens | 106,993 |
| Accuracy | **96.17%** |
| Macro F1 | **95.55%** |
| English precision | 93.65% |
| English recall | 92.13% |
| English F1 | 92.89% |

The full evaluation report is in `models/comi_lingua_augmented_final_test_report.json`. These results measure agreement with the dataset's consensus labels; they do not guarantee equivalent performance on every real-world source, dialect, spelling style, or domain.

## Training and data

Training scripts are included for research reproducibility, but the training corpora and local annotation files are **not** bundled. The COMI-LINGUA dataset can be downloaded from the maintainer's Hugging Face dataset repository using:

```powershell
python download_comi_lingua.py
python convert_comi_lingua.py
```

Further processing, splitting, training, threshold tuning, and evaluation scripts are provided at the project root. Some augmented training scripts also expect local annotation inputs described in their source; those local working files are intentionally not committed. See [data/README.md](data/README.md).

## Dataset attribution

The project uses the COMI-LINGUA Hindi-English language-identification dataset from the Lingo Research Group at IIT Gandhinagar. The dataset is declared under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Dataset files are not redistributed here. Download them from the official source and follow its attribution and citation requirements.

Dataset: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA

Citation:

```bibtex
@inproceedings{sheth-etal-2025-comi,
  title = "{COMI}-{LINGUA}: Expert Annotated Large-Scale Dataset for Multitask {NLP} in {H}indi-{English} Code-Mixing",
  author = "Sheth, Rajvee and Beniwal, Himanshu and Singh, Mayank",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
  year = "2025",
  pages = "7973--7992"
}
```

## Limitations

- This is a research/academic desktop prototype, not a production language-identification service.
- Short, ambiguous tokens, spelling variation, named entities, and domain-specific vocabulary can be difficult to classify.
- The model predicts English/Hindi/Other token labels; it does not perform translation, transliteration, sentiment analysis, or sentence-level semantic understanding.
- The desktop app uses the COMI-LINGUA three-class model. Separate binary English/Hindi experiments are not part of this desktop inference path.
- Some evaluation data and error diagnostics were reviewed during development; do not describe those results as an entirely untouched external benchmark.

## License

No open-source license has been selected for the project code yet. Dataset rights are separate and governed by the dataset's own license. Add a code license only after deciding how you want others to reuse the project.
