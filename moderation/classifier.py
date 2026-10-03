"""
Classifier wrapper around fastText.
"""
import fasttext
import os

class WholeMessageClassifier:
    def __init__(self, model_path: str):
        # Suppress fasttext warnings
        fasttext.FastText.eprint = lambda x: None
        if os.path.exists(model_path):
            self.model = fasttext.load_model(model_path)
        else:
            self.model = None
            print(f"Warning: Model not found at {model_path}")

    def predict(self, normalized_text: str) -> tuple:
        """
        Returns (label: str, confidence: float)
        """
        if not self.model:
            return "safe", 1.0
            
        if not normalized_text.strip():
            return "safe", 1.0

        pred = self.model.predict(normalized_text)
        label = pred[0][0].replace("__label__", "")
        confidence = round(float(pred[1][0]), 4)
        return label, confidence
