"""
Service orchestrating the moderation pipeline.
"""
from moderation.normalizer import TextNormalizer
from moderation.span_detector import SpanDetector
from moderation.classifier import WholeMessageClassifier
from moderation.sanitizer import Sanitizer
from moderation.policy import PolicyEngine
import os

class ModerationService:
    def __init__(self, model_path: str, lexicon_path: str = None):
        if lexicon_path is None:
            lexicon_path = os.path.join(os.path.dirname(model_path), "profanity_lexicon.json")
            
        self.normalizer = TextNormalizer()
        self.span_detector = SpanDetector(lexicon_path=lexicon_path)
        base_dir = os.path.dirname(model_path)
        self.classifier = WholeMessageClassifier(
            vectorizer_path=os.path.join(base_dir, "vectorizer.joblib"),
            model_path=os.path.join(base_dir, "model.joblib")
        )
        self.sanitizer = Sanitizer()
        self.policy = PolicyEngine()

    def moderate(self, text: str) -> dict:
        """
        End-to-end moderation pipeline.
        """
        # 1. Normalize
        normalized_text, norm_to_orig = self.normalizer.normalize(text)
        
        # 2. Span Detection
        detected_spans = self.span_detector.detect_spans(normalized_text)
        
        # 3. Whole-message Classification (HYBRID Pass 2)
        if len(detected_spans) > 0:
            label = "abusive"
            confidence = 1.0
        else:
            label, confidence = self.classifier.predict(normalized_text)
        
        # 4. Sanitize (Censor)
        censored_text = self.sanitizer.sanitize(text, detected_spans, self.normalizer, norm_to_orig)
        
        # 5. Policy
        decision = self.policy.decide(label, confidence, len(detected_spans) > 0)
        
        if len(detected_spans) > 0:
            decision["allowed"] = False
            decision["action"] = "CENSOR_WARN"
            decision["reason_code"] = "PROFANITY_DETECTED"
        elif label == "abusive" and confidence > 0.85:
            # ML Model caught something very clever!
            decision["allowed"] = False
            decision["action"] = "WARN"
            decision["reason_code"] = "ML_ABUSE_DETECTED"
            
        return {
            "allowed": decision["allowed"],
            "censored_text": censored_text,
            "is_abusive": label in ["abusive", "severe_abusive"],
            "label": label,
            "confidence": confidence,
            "action": decision["action"],
            "reason_code": decision["reason_code"],
            "detected_spans": detected_spans
        }
