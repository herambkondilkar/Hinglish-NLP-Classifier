"""Desktop UI for the augmented COMI-LINGUA word-level language classifier.

Place this file in the Hinglish-NLP-Classifier project root and run:
    python hinglish_desktop_app.py

Model expected at:
    models/comi_lingua_dev_tuned_augmented.joblib
"""

import csv
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import joblib
    from scipy.sparse import hstack
except ImportError as exc:
    joblib = None
    hstack = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "comi_lingua_dev_tuned_augmented.joblib"
DEFAULT_THRESHOLD = 0.75

TOKEN_PATTERN = re.compile(
    r"@[\w]+|#[\w]+|[A-Za-z]+(?:['’][A-Za-z]+)*|"
    r"[\u0900-\u097F]+|\d+(?:[.:,]\d+)*|[^\w\s]",
    flags=re.UNICODE,
)

LABEL_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "other": "Other",
}

BG = "#0f172a"
PANEL = "#172033"
PANEL_LIGHT = "#1e293b"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
ACCENT = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#fb7185"


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


def predict_sentence(sentence, bundle):
    tokens = TOKEN_PATTERN.findall(sentence)
    if not tokens:
        return []

    word_vectorizer = bundle["word_vectorizer"]
    char_vectorizer = bundle["char_vectorizer"]
    classifier = bundle["classifier"]
    classes = [str(label) for label in bundle["labels"]]
    threshold = float(bundle.get("english_threshold") or DEFAULT_THRESHOLD)

    word_texts = [build_word_features(tokens, i) for i in range(len(tokens))]
    word_matrix = word_vectorizer.transform(word_texts)
    char_matrix = char_vectorizer.transform(tokens)
    features = hstack([word_matrix, char_matrix], format="csr")
    probabilities = classifier.predict_proba(features)

    class_index = {label: idx for idx, label in enumerate(classes)}
    if "en" not in class_index:
        raise ValueError("Loaded model does not contain an English ('en') class.")

    non_english_indices = [idx for label, idx in class_index.items() if label != "en"]
    results = []
    for token, row in zip(tokens, probabilities):
        english_probability = float(row[class_index["en"]])
        if english_probability >= threshold:
            predicted = "en"
        else:
            best_index = max(non_english_indices, key=lambda idx: row[idx])
            predicted = classes[best_index]

        results.append({
            "token": token,
            "predicted": predicted,
            "english_probability": english_probability,
            "selected_probability": float(row[class_index[predicted]]),
            "probabilities": {label: float(row[idx]) for label, idx in class_index.items()},
        })
    return results


class HinglishDesktopApp:
    def __init__(self, root):
        self.root = root
        self.bundle = None
        self.last_results = []
        self.threshold = DEFAULT_THRESHOLD

        self.root.title("Hinglish NLP Classifier")
        self.root.geometry("1080x760")
        self.root.minsize(860, 620)
        self.root.configure(bg=BG)

        self._configure_styles()
        self._build_ui()
        self._load_model()

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Metric.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 18, "bold"))
        style.configure("MetricCaption.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 9))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=29,
            borderwidth=0,
            font=("Consolas", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_LIGHT,
            foreground=TEXT,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padding=8,
        )
        style.map("Treeview", background=[("selected", "#075985")])

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=(22, 18, 22, 18))
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(
            header, text="Hinglish NLP Classifier",
            font=("Segoe UI", 22, "bold"), foreground=TEXT,
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Token-level identification for English, Hindi, and Other tokens",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="Loading model…")
        status_row = ttk.Frame(header)
        status_row.pack(fill="x", pady=(12, 0))
        self.status_label = tk.Label(
            status_row, textvariable=self.status_var,
            bg=PANEL, fg=AMBER, padx=10, pady=7,
            font=("Segoe UI", 9, "bold"), anchor="w",
        )
        self.status_label.pack(fill="x")

        input_panel = tk.Frame(outer, bg=PANEL, padx=14, pady=12)
        input_panel.pack(fill="x", pady=(0, 14))
        tk.Label(
            input_panel, text="Enter a sentence", bg=PANEL, fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            input_panel,
            text="Examples: Mujhe Python mein help chahiye  •  कृपया screenshot upload कर दो",
            bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(3, 8))

        self.input_text = tk.Text(
            input_panel, height=4, wrap="word",
            bg="#0b1220", fg=TEXT, insertbackground=TEXT,
            selectbackground="#075985", relief="flat",
            font=("Segoe UI", 12), padx=10, pady=9,
            undo=True,
        )
        self.input_text.pack(fill="x")
        self.input_text.bind("<Control-Return>", self._analyze_shortcut)

        buttons = ttk.Frame(input_panel)
        buttons.pack(fill="x", pady=(10, 0))
        self.analyze_button = ttk.Button(
            buttons, text="Analyze sentence", style="Primary.TButton",
            command=self.analyze,
        )
        self.analyze_button.pack(side="left")
        ttk.Button(
            buttons, text="Load example", style="Secondary.TButton",
            command=self.load_example,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="Clear", style="Secondary.TButton",
            command=self.clear,
        ).pack(side="left", padx=(8, 0))
        self.threshold_label = tk.Label(
            buttons, text="English threshold: 0.75", bg=PANEL, fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
        )
        self.threshold_label.pack(side="right", padx=(8, 0))

        metrics = ttk.Frame(outer)
        metrics.pack(fill="x", pady=(0, 14))
        self.metric_vars = {
            "tokens": tk.StringVar(value="—"),
            "en": tk.StringVar(value="—"),
            "hi": tk.StringVar(value="—"),
            "other": tk.StringVar(value="—"),
        }
        captions = [
            ("tokens", "TOKENS"),
            ("en", "ENGLISH"),
            ("hi", "HINDI"),
            ("other", "OTHER"),
        ]
        for idx, (key, caption) in enumerate(captions):
            card = tk.Frame(metrics, bg=PANEL, padx=13, pady=10)
            card.grid(row=0, column=idx, sticky="nsew", padx=(0 if idx == 0 else 8, 0))
            metrics.columnconfigure(idx, weight=1, uniform="metric")
            tk.Label(card, textvariable=self.metric_vars[key], bg=PANEL, fg=ACCENT,
                     font=("Segoe UI", 18, "bold")).pack(anchor="w")
            tk.Label(card, text=caption, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 0))

        results_header = ttk.Frame(outer)
        results_header.pack(fill="x", pady=(0, 7))
        ttk.Label(results_header, text="Token-level results", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.summary_var = tk.StringVar(value="Enter a sentence and choose Analyze sentence.")
        ttk.Label(results_header, textvariable=self.summary_var, style="Muted.TLabel").pack(side="right")

        table_frame = tk.Frame(outer, bg=PANEL)
        table_frame.pack(fill="both", expand=True)
        columns = ("index", "token", "label", "en_prob", "label_prob", "hi_prob", "other_prob")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "index": "#", "token": "TOKEN", "label": "PREDICTED LABEL",
            "en_prob": "EN PROB.", "label_prob": "LABEL PROB.",
            "hi_prob": "HI PROB.", "other_prob": "OTHER PROB.",
        }
        widths = {"index": 48, "token": 260, "label": 150, "en_prob": 105,
                  "label_prob": 115, "hi_prob": 105, "other_prob": 115}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w" if col == "token" else "center",
                             stretch=(col == "token"))
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.tag_configure("en", foreground="#7dd3fc")
        self.tree.tag_configure("hi", foreground="#6ee7b7")
        self.tree.tag_configure("other", foreground="#fcd34d")
        self.tree.tag_configure("odd", background="#111b2d")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Button(footer, text="Export results to CSV", command=self.export_results).pack(side="left")
        ttk.Button(footer, text="Copy summary", command=self.copy_summary).pack(side="left", padx=(8, 0))
        ttk.Label(
            footer,
            text="Probabilities are model estimates, not guarantees.",
            style="Muted.TLabel",
        ).pack(side="right")

    def _load_model(self):
        if IMPORT_ERROR is not None:
            self.status_var.set(f"Missing dependency: {IMPORT_ERROR}")
            self.status_label.configure(fg=RED)
            self.analyze_button.configure(state="disabled")
            messagebox.showerror(
                "Python dependency missing",
                f"Could not import required packages.\n\n{IMPORT_ERROR}\n\n"
                "Install the project's existing dependencies before running the app.",
            )
            return

        if not MODEL_PATH.exists():
            self.status_var.set(f"Model not found: {MODEL_PATH}")
            self.status_label.configure(fg=RED)
            self.analyze_button.configure(state="disabled")
            messagebox.showerror(
                "Model file not found",
                "The augmented model file was not found. Expected location:\n\n"
                f"{MODEL_PATH}\n\nMake sure this app is inside your project root and the model exists.",
            )
            return

        try:
            self.bundle = joblib.load(MODEL_PATH)
            required = {"classifier", "word_vectorizer", "char_vectorizer", "labels"}
            missing = required - set(self.bundle)
            if missing:
                raise ValueError(f"Model bundle is missing: {sorted(missing)}")
            self.threshold = float(self.bundle.get("english_threshold") or DEFAULT_THRESHOLD)
            classes = [str(label) for label in self.bundle["labels"]]
            if not {"en", "hi", "other"}.issubset(set(classes)):
                raise ValueError(f"Expected en, hi, and other labels; model contains {classes}")
            self.threshold_label.configure(text=f"English threshold: {self.threshold:.2f}")
            self.status_var.set(
                f"Model loaded  •  Augmented COMI-LINGUA  •  {len(classes)} classes  •  Ready"
            )
            self.status_label.configure(fg=GREEN)
        except MemoryError:
            self.status_var.set("Not enough memory to load the model")
            self.status_label.configure(fg=RED)
            self.analyze_button.configure(state="disabled")
            messagebox.showerror(
                "Memory error",
                "Python could not allocate enough memory to load the model.\n\n"
                "Close other applications, restart Python/VS Code, and try again.",
            )
        except Exception as exc:
            self.status_var.set("Model loading failed")
            self.status_label.configure(fg=RED)
            self.analyze_button.configure(state="disabled")
            messagebox.showerror("Could not load model", str(exc))

    def _analyze_shortcut(self, _event=None):
        self.analyze()
        return "break"

    def load_example(self):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", "Mujhe Python mein help chahiye")
        self.input_text.focus_set()

    def clear(self):
        self.input_text.delete("1.0", "end")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.last_results = []
        for var in self.metric_vars.values():
            var.set("—")
        self.summary_var.set("Enter a sentence and choose Analyze sentence.")
        self.input_text.focus_set()

    def analyze(self):
        if self.bundle is None:
            messagebox.showerror("Model unavailable", "The model has not loaded successfully.")
            return

        sentence = self.input_text.get("1.0", "end-1c").strip()
        if not sentence:
            messagebox.showinfo("Enter a sentence", "Type a Hinglish or English sentence first.")
            self.input_text.focus_set()
            return

        try:
            results = predict_sentence(sentence, self.bundle)
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return

        self.last_results = results
        for item in self.tree.get_children():
            self.tree.delete(item)

        counts = {"en": 0, "hi": 0, "other": 0}
        for idx, row in enumerate(results, start=1):
            label = row["predicted"]
            counts[label] = counts.get(label, 0) + 1
            tags = [label]
            if idx % 2 == 0:
                tags.append("odd")
            self.tree.insert(
                "", "end",
                values=(
                    idx,
                    row["token"],
                    LABEL_NAMES.get(label, label),
                    f"{row['english_probability']:.1%}",
                    f"{row['selected_probability']:.1%}",
                    f"{row['probabilities'].get('hi', 0.0):.1%}",
                    f"{row['probabilities'].get('other', 0.0):.1%}",
                ),
                tags=tuple(tags),
            )

        self.metric_vars["tokens"].set(str(len(results)))
        self.metric_vars["en"].set(str(counts.get("en", 0)))
        self.metric_vars["hi"].set(str(counts.get("hi", 0)))
        self.metric_vars["other"].set(str(counts.get("other", 0)))
        self.summary_var.set(
            f"{len(results)} tokens analyzed  •  threshold {self.threshold:.2f}"
        )

    def export_results(self):
        if not self.last_results:
            messagebox.showinfo("Nothing to export", "Analyze a sentence first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save token predictions",
            defaultextension=".csv",
            initialfile="hinglish_predictions.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        columns = [
            "token_index", "token", "predicted_label", "english_probability",
            "selected_label_probability", "hindi_probability",
            "other_probability",
        ]
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                for idx, row in enumerate(self.last_results, start=1):
                    writer.writerow({
                        "token_index": idx,
                        "token": row["token"],
                        "predicted_label": row["predicted"],
                        "english_probability": f"{row['english_probability']:.6f}",
                        "selected_label_probability": f"{row['selected_probability']:.6f}",
                        "hindi_probability": f"{row['probabilities'].get('hi', 0.0):.6f}",
                        "other_probability": f"{row['probabilities'].get('other', 0.0):.6f}",
                    })
            messagebox.showinfo("Export successful", f"Predictions saved to:\n{path}")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))

    def copy_summary(self):
        if not self.last_results:
            messagebox.showinfo("Nothing to copy", "Analyze a sentence first.")
            return

        lines = [f"English threshold: {self.threshold:.2f}", "", "Token | Prediction | EN probability | Selected-label probability"]
        for row in self.last_results:
            lines.append(
                f"{row['token']} | {row['predicted']} | "
                f"{row['english_probability']:.1%} | {row['selected_probability']:.1%}"
            )
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copied", "Prediction summary copied to clipboard.")


def main():
    root = tk.Tk()
    HinglishDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
