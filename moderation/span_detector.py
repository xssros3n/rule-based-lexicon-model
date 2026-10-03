"""
Deterministic span detector using a lexicon.
"""
import re
import json
import os

class SpanDetector:
    def __init__(self, lexicon=None, lexicon_path=None):
        if lexicon is not None:
            self.lexicon = lexicon
        elif lexicon_path and os.path.exists(lexicon_path):
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
        else:
            # Default minimal lexicon for demonstration
            self.lexicon = {
                "bc": "medium",
                "mc": "medium",
                "chutiya": "medium",
                "fuck": "high",
                "bitch": "medium",
                "bastard": "medium"
            }
            
        # Build regex patterns. We look for word boundaries to avoid Scunthorpe problems.
        # But we also want to catch spaced out letters like "b c" 
        # For simplicity and speed, we first search exact matches of lexicon words,
        # then we search for spaced-out versions.
        
        self.patterns = []
        for word, severity in self.lexicon.items():
            # Exact word match with word boundaries
            exact_pattern = r'\b' + re.escape(word) + r'\b'
            self.patterns.append((re.compile(exact_pattern), word, severity))
            
            # Obfuscated pattern: allow optional non-word chars (like ., -, spaces) between letters
            # e.g., 'b' + r'[.\-\s]*' + 'c'
            if len(word) >= 2:
                obfuscated = r'[.\-\s@_]*'.join(re.escape(char) for char in word)
                # Still enforce word boundaries around the obfuscated word
                obfuscated_pattern = r'\b' + obfuscated + r'\b'
                # We compile it
                self.patterns.append((re.compile(obfuscated_pattern), word, severity))

    def detect_spans(self, normalized_text: str) -> list:
        """
        Find spans in normalized text.
        Returns a list of dicts:
        [{'text': word, 'start': start_idx, 'end': end_idx, 'severity': sev, 'category': 'abusive'}]
        """
        spans = []
        for pattern, base_word, severity in self.patterns:
            for match in pattern.finditer(normalized_text):
                start, end = match.span()
                # Check if this span is already covered by a longer match?
                # For simplicity, we just add all matches and merge them later.
                spans.append({
                    "text": base_word,
                    "start": start,
                    "end": end,
                    "severity": severity,
                    "category": "abusive"
                })
        
        # Sort spans by start index
        spans.sort(key=lambda x: (x["start"], -x["end"]))
        
        # Filter subsumed spans
        filtered_spans = []
        for span in spans:
            if not filtered_spans:
                filtered_spans.append(span)
            else:
                last_span = filtered_spans[-1]
                if span["start"] >= last_span["start"] and span["end"] <= last_span["end"]:
                    # This span is entirely inside the last one, ignore
                    continue
                filtered_spans.append(span)
                
        return filtered_spans
