"""
Ruche Language Filter - Language detection and filtering
"""
import re
from typing import Dict, Optional, List
from collections import Counter


# Common words for language detection
LANGUAGE_PATTERNS = {
    "fr": [
        "le", "la", "les", "un", "une", "des", "est", "sont", "été", "être",
        "avec", "pour", "dans", "sur", "ce", "cette", "ces", "ces", "nous",
        "vous", "ils", "elles", "je", "tu", "il", "elle", "que", "qui", "quoi",
        "comment", "pourquoi", "quand", "où", "mais", "donc", "car", "aussi",
        "plus", "comme", "fait", "bien", "peut", "deux", "temps", "alors"
    ],
    "en": [
        "the", "a", "an", "is", "are", "was", "were", "been", "being",
        "with", "for", "in", "on", "this", "that", "these", "those", "we",
        "you", "they", "he", "she", "it", "what", "how", "why", "when", "where",
        "but", "or", "and", "also", "more", "like", "just", "can", "one", "time"
    ],
    "de": [
        "der", "die", "das", "ein", "eine", "ist", "sind", "war", "wurde",
        "mit", "für", "in", "auf", "dieser", "diese", "dieses", "wir", "ihr",
        "sie", "ich", "du", "was", "wie", "warum", "wann", "wo", "aber", "oder"
    ],
    "es": [
        "el", "la", "los", "las", "un", "una", "es", "son", "está", "estar",
        "con", "para", "en", "este", "esta", "estos", "estas", "nosotros",
        "vosotros", "ellos", "yo", "tú", "qué", "cómo", "por qué", "cuándo"
    ]
}


def detect_language(text: str) -> str:
    """
    Detect language of text.
    
    Returns: 'fr', 'en', 'de', 'es', or 'unknown'
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    scores = {}
    
    for lang, lang_words in LANGUAGE_PATTERNS.items():
        matches = len(words & set(lang_words))
        scores[lang] = matches
    
    if not scores:
        return "unknown"
    
    best_lang = max(scores, key=scores.get)
    
    # Require minimum score to avoid false positives
    if scores[best_lang] < 2:
        return "unknown"
    
    return best_lang


def is_language(text: str, target_lang: str) -> bool:
    """
    Check if text is in target language.
    
    Args:
        text: Text to check
        target_lang: Target language code ('fr', 'en', etc.)
    
    Returns:
        True if text is in target language
    """
    detected = detect_language(text)
    return detected == target_lang


def filter_by_language(texts: List[str], target_lang: str) -> List[Dict]:
    """
    Filter list of texts by language.
    
    Returns list of dicts with text and detected language.
    """
    results = []
    
    for text in texts:
        lang = detect_language(text)
        results.append({
            "text": text,
            "language": lang,
            "is_target": lang == target_lang
        })
    
    return results


class LanguageFilter:
    """Language filter for Ruche pipeline"""
    
    def __init__(self, target_lang: str = "fr"):
        self.target_lang = target_lang
        self.stats = {
            "total_checked": 0,
            "accepted": 0,
            "rejected": 0
        }
    
    def check(self, text: str) -> bool:
        """Check if text is in target language"""
        self.stats["total_checked"] += 1
        
        detected = detect_language(text)
        is_target = detected == self.target_lang
        
        if is_target:
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
        
        return is_target
    
    def get_stats(self) -> dict:
        return self.stats.copy()


# CLI test
if __name__ == "__main__":
    texts = [
        "Bonjour, comment allez-vous?",
        "Hello, how are you doing today?",
        "Guten Tag, wie geht es Ihnen?",
        "Hola, cómo estás?",
        "This is some English text about programming in Python.",
        "Ceci est un texte français sur la programmation Python."
    ]
    
    for text in texts[:4]:
        lang = detect_language(text)
        print(f"'{text[:30]}...' → {lang}")
    
    print(f"\nFilter test (target: fr):")
    lf = LanguageFilter("fr")
    for text in texts:
        ok = lf.check(text)
        print(f"  {ok}: {text[:30]}...")
    print(f"Stats: {lf.get_stats()}")
