"""
Censoring engine. Merges spans and masks original text.
"""
class Sanitizer:
    def __init__(self, masking_char='*'):
        self.masking_char = masking_char

    def sanitize(self, original_text: str, norm_spans: list, normalizer, norm_to_orig: list) -> str:
        """
        Takes original text, spans detected in normalized text, and applies masking.
        """
        if not norm_spans or not original_text:
            return original_text

        # 1. Map spans to original text
        orig_spans = []
        for span in norm_spans:
            start, end = normalizer.map_span(span["start"], span["end"], norm_to_orig)
            if start != -1 and end != -1:
                orig_spans.append((start, end))
                
        if not orig_spans:
            return original_text

        # 2. Merge overlapping original spans
        orig_spans.sort(key=lambda x: x[0])
        merged_spans = []
        for start, end in orig_spans:
            if not merged_spans:
                merged_spans.append([start, end])
            else:
                last_start, last_end = merged_spans[-1]
                if start <= last_end:
                    merged_spans[-1][1] = max(last_end, end)
                else:
                    merged_spans.append([start, end])

        # 3. Apply masking
        # Convert text to list of characters for easier manipulation
        text_chars = list(original_text)
        
        for start, end in merged_spans:
            # We want to replace non-whitespace characters in the span with '*'
            # Default masking policy: word length N -> N asterisks
            # But let's keep punctuation or space as is?
            # E.g. "b c" -> "* *" or "**"? The user asks for:
            # word length 2 -> **, "tu kaha jaa rha hai **" (from "bc")
            # For simplicity, we just replace all characters in the span except whitespace with '*'
            for i in range(start, end):
                if i < len(text_chars) and not text_chars[i].isspace():
                    text_chars[i] = self.masking_char
                    
        return "".join(text_chars)
