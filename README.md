# Hinglish NLP Classifier

A Python desktop prototype for **token-level language identification** in Hindi-English code-mixed text (Hinglish). It classifies each token as **English**, **Hindi**, or **Other**, including Romanized Hindi, Devanagari Hindi, English words, numerals, and punctuation.

## Highlights

- Desktop GUI built with Python Tkinter.
- Token-level predictions with confidence information.
- Context-aware lexical features, character n-gram features, script and token-shape signals.
- English decision threshold tuned on a development split.
- Export predictions to CSV and copy an analysis summary.
- One-click Windows launcher.
- Pre-trained model included, so the desktop app can be tried without downloading the training corpora.

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

## Project structure

```text
Hinglish-NLP-Classifier/
├── hinglish_desktop_app.py
├── Launch_Hinglish_Classifier.bat
├── models/
│   ├── comi_lingua_dev_tuned_augmented.joblib
│   ├── comi_lingua_augmented_final_test_report.json
│   └── comi_lingua_dev_tuning_augmented_report.json
├── data/README.md
├── download_comi_lingua.py
├── convert_comi_lingua.py
├── make_comi_dev_split.py
├── remove_train_test_overlap.py
├── validate_comi_processed.py
├── train_comi_lingua_model.py
├── augment_comi_fit_with_local_data.py
├── tune_comi_english_precision_augmented.py
├── evaluate_comi_lingua_augmented_final.py
└── requirements.txt
```

## Run on Windows

**Requirements:** Python 3.11+ (64-bit) and Tkinter. Tkinter is included with most standard Windows Python installations.

1. Download or clone this repository.
2. Open PowerShell in the project folder.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install the pinned ML dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Launch the application:

```powershell
python hinglish_desktop_app.py
```

You can also double-click `Launch_Hinglish_Classifier.bat`. The launcher automatically uses `.venv` when it exists.

## Model and evaluation

The desktop application uses the included **three-class COMI-LINGUA model** with an English probability threshold of `0.75`. The threshold was selected on the development/validation split and then evaluated on the official held-out COMI-LINGUA test split.

| Metric | Result |
|---|---:|
| Test tokens | 106,993 |
| Accuracy | **96.17%** |
| Macro F1 | **95.55%** |
| English precision | 93.65% |
| English recall | 92.13% |
| English F1 | 92.89% |

The full report is in `models/comi_lingua_augmented_final_test_report.json`. These results measure agreement with the dataset's consensus labels; they do not guarantee the same performance on every real-world source, dialect, spelling style, or domain.

## Training and data

The supplied model is ready for desktop inference. Training scripts are included for research reproducibility, but the training corpora and local annotation files are **not** bundled.

The COMI-LINGUA dataset can be downloaded from the maintainer's Hugging Face dataset repository using:

```powershell
python download_comi_lingua.py
python convert_comi_lingua.py
```

Further processing, splitting, training, threshold tuning, and evaluation scripts are provided at the project root. The augmented training script also expects local annotation inputs described in its source; those private/local working files are intentionally not committed. See `data/README.md`.

## Dataset attribution

The project uses the COMI-LINGUA Hindi-English language-identification dataset from the Lingo Research Group at IIT Gandhinagar. The dataset is declared under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The dataset files are not redistributed here; download them from the official source and follow its attribution and citation requirements.

Dataset: <https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA>

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
- The included model is trained for English/Hindi/Other token labels; it does not perform translation, transliteration, sentiment analysis, or sentence-level semantic understanding.
- The app uses the COMI-LINGUA three-class model. Separate binary English/Hindi experiments are not part of this desktop inference path.

## License

No open-source license has been selected for the project code in this repository yet. Dataset rights remain separate and are governed by the dataset's own license. Add a code license only after deciding how you want others to reuse the project.
