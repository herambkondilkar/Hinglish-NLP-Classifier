import unittest
from pathlib import Path
import joblib

from hinglish_desktop_app import MODEL_PATH, predict_sentence


class ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Required model is missing: {MODEL_PATH}")
        cls.bundle = joblib.load(MODEL_PATH)

    def test_hinglish_token_labels(self):
        result = predict_sentence("Mujhe Python mein help chahiye", self.bundle)
        labels = [(item["token"], item["predicted"]) for item in result]
        self.assertEqual(labels, [
            ("Mujhe", "hi"),
            ("Python", "en"),
            ("mein", "hi"),
            ("help", "en"),
            ("chahiye", "hi"),
        ])

    def test_english_sentence(self):
        result = predict_sentence("I need help with my college assignment", self.bundle)
        self.assertTrue(result)
        self.assertTrue(all(item["predicted"] == "en" for item in result))

    def test_devanagari_sentence(self):
        result = predict_sentence("मुझे आज कॉलेज जाना है", self.bundle)
        self.assertTrue(result)
        self.assertTrue(all(item["predicted"] == "hi" for item in result))


if __name__ == "__main__":
    unittest.main()
