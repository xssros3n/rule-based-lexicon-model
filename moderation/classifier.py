"""
Custom FastText Model Wrapper for Hinglish.
"""
import fasttext
import os

class WholeMessageClassifier:
    def __init__(self, model_path: str):
        fasttext.FastText.eprint = lambda x: None
        if os.path.exists(model_path):
            self.model = fasttext.load_model(model_path)
        else:
            self.model = None

    def predict(self, normalized_text: str) -> tuple:
        """
        Returns (label: str, confidence: float)
        """
        if not self.model or not normalized_text.strip():
            return "safe", 1.0

        pred = self.model.predict(normalized_text)
        label = pred[0][0].replace("__label__", "")
        confidence = round(float(pred[1][0]), 4)
        return label, confidence
