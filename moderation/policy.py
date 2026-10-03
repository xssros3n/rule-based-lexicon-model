"""
Policy engine — separates ML classification from action decisions.
Thresholds are configurable via environment variables.
"""
import os

class PolicyEngine:
    def __init__(self):
        self.confidence_warn = float(os.getenv("CONFIDENCE_WARN", "0.5"))
        self.confidence_delete = float(os.getenv("CONFIDENCE_DELETE", "0.8"))
        self.confidence_block = float(os.getenv("CONFIDENCE_BLOCK", "0.9"))

    def decide(self, label: str, confidence: float, has_spans: bool) -> dict:
        if label == "safe" and not has_spans:
            return {"allowed": True, "action": "allow", "reason_code": None}

        if has_spans and label == "safe":
             return {"allowed": True, "action": "warn", "reason_code": "abusive_span_detected"}

        if label == "severe_abusive":
            if confidence >= self.confidence_block:
                return {"allowed": False, "action": "block", "reason_code": "severe_high_conf"}
            elif confidence >= self.confidence_delete:
                return {"allowed": False, "action": "delete", "reason_code": "severe_medium_conf"}
            else:
                return {"allowed": False, "action": "review", "reason_code": "severe_low_conf"}

        if label == "abusive":
            if confidence >= self.confidence_delete:
                return {"allowed": False, "action": "delete", "reason_code": "abusive_high_conf"}
            elif confidence >= self.confidence_warn:
                # Ordinary profanity should normally result in CENSOR + WARN
                return {"allowed": True, "action": "censor_warn", "reason_code": "ABUSIVE_LANGUAGE"}
            else:
                return {"allowed": True, "action": "review", "reason_code": "abusive_low_conf"}

        return {"allowed": True, "action": "review", "reason_code": "unknown_label"}
