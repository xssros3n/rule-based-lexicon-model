"""
Custom Scikit-Learn Model Wrapper for Hinglish.
Uses TF-IDF + LinearSVC for Render-friendly, memory-efficient, exact-word prediction.
"""
import os
import joblib

class WholeMessageClassifier:
    def __init__(self, vectorizer_path: str, model_path: str):
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
        else:
            self.vectorizer = None
            self.model = None

    def predict(self, normalized_text: str) -> tuple:
        """
        Returns (label: str, confidence: float)
        """
        if not self.model or not normalized_text.strip():
            return "safe", 1.0

        X = self.vectorizer.transform([normalized_text])
        # SVM doesn't output traditional probabilities by default. 
        # For simplicity in LinearSVC, we get decision_function and normalize or just return a mock high confidence
        # since it's a binary classification and it only fires if it's strongly across the boundary.
        decision = self.model.decision_function(X)[0]
        pred = self.model.predict(X)[0]
        
        # decision function distance from margin as pseudo-confidence
        confidence = min(abs(decision), 1.0) if abs(decision) > 0 else 0.8
        
        return pred, round(confidence, 4)
