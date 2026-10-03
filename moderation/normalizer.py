"""
Text normalization that preserves mapping between original and normalized text.
Critical for exact span detection and censorship.
"""
import unicodedata
import re

# Zero-width and invisible characters to strip during normalization
ZERO_WIDTH_CHARS = set([
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
    '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
    '\ufeff', '\u00ad', '\u034f', '\u061c', '\u17b4',
    '\u17b5', '\u180e',
])

class TextNormalizer:
    def __init__(self):
        pass

    def normalize(self, text: str):
        """
        Normalizes text while maintaining a mapping back to original indices.
        Returns:
            normalized_text (str): The cleaned text.
            norm_to_orig (list): A list where norm_to_orig[i] gives the index
                                 in `text` corresponding to the i-th char in `normalized_text`.
        """
        if not text:
            return "", []

        normalized_chars = []
        norm_to_orig = []

        # We do a simpler lowercase + zero-width stripping + basic whitespace collapse.
        # NFKC changes string length drastically and makes mapping complex.
        # We do mapping char by char.
        
        last_was_space = False
        
        # Leetspeak mapping
        leet_map = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a', '$': 's', '!': 'i'
        }

        for i, char in enumerate(text):
            # Skip zero-width chars
            if char in ZERO_WIDTH_CHARS:
                continue
            
            # Collapse whitespace
            if char.isspace():
                if last_was_space:
                    continue
                normalized_chars.append(' ')
                norm_to_orig.append(i)
                last_was_space = True
            else:
                # Lowercase the character and apply leetspeak mapping
                lowered = char.lower()
                lowered = leet_map.get(lowered, lowered)
                
                for lc in lowered:
                    normalized_chars.append(lc)
                    norm_to_orig.append(i)
                last_was_space = False

        normalized_text = "".join(normalized_chars)
        return normalized_text, norm_to_orig

    def map_span(self, norm_start: int, norm_end: int, norm_to_orig: list) -> tuple:
        """
        Given start and end in normalized text, find start and end in original text.
        """
        if not norm_to_orig or norm_start >= len(norm_to_orig):
            return -1, -1
        
        orig_start = norm_to_orig[norm_start]
        # norm_end is exclusive. 
        if norm_end > len(norm_to_orig):
            norm_end = len(norm_to_orig)
        
        if norm_end == 0:
            return orig_start, orig_start

        # For the end index, we want the index just after the last character in the span.
        orig_end = norm_to_orig[norm_end - 1] + 1
        
        return orig_start, orig_end