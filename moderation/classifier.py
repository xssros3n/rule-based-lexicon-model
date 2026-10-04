"""
Classifier wrapper mocked out since we use exact match now.
"""
class WholeMessageClassifier:
    def __init__(self, model_path: str):
        pass

    def predict(self, normalized_text: str) -> tuple:
        """
        Returns (label: str, confidence: float)
        Always safe since we bypassed it.
        """
        return "safe", 1.0
