"""
Ruche Quality Filter - Heuristic scoring without AI
Score between 0.0 and 1.0
"""
import re
from typing import Dict, Optional


def quality_score(text: str, config: Dict) -> float:
    """
    Calculate quality score based on heuristics.
    
    Args:
        text: Text to score
        config: Category config with keywords, min_words, etc.
    
    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0
    words = text.split()
    
    # 1. Length (0.0 - 0.25)
    min_words = config.get("min_words", 80)
    score += min(len(words) / max(min_words * 2, 1), 0.25)
    
    # 2. Vocabulary richness (0.0 - 0.20)
    if words:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        score += min(unique_ratio * 0.20 / 0.4, 0.20)
    
    # 3. Keyword presence (0.0 - 0.30)
    keywords = config.get("keywords", [])
    if keywords:
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        score += min(keyword_hits * 0.10, 0.30)
    
    # 4. No spam detection (0.0 or 0.15)
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.4:  # Not too many repetitions
            score += 0.15
    
    # 5. Language match (0.0 or 0.10)
    target_lang = config.get("lang", "fr")
    detected = detect_language_simple(text)
    if detected == target_lang:
        score += 0.10
    
    return min(score, 1.0)


def detect_language_simple(text: str) -> str:
    """
    Simple language detection without external library.
    
    Returns: 'fr', 'en', or 'unknown'
    """
    if not text:
        return "unknown"
    
    text_lower = text.lower()
    
    # French indicators
    fr_words = ["le", "la", "les", "un", "une", "des", "est", "sont", "avec", "pour", "dans", "sur", "ce", "cette", "nous", "vous", "ils", "elles", "je", "tu", "il", "elle", "que", "qui", "quoi", "comment", "pourquoi", "quand", "où"]
    fr_count = sum(1 for w in fr_words if w in text_lower)
    
    # English indicators
    en_words = ["the", "a", "an", "is", "are", "with", "for", "in", "on", "this", "that", "we", "you", "they", "he", "she", "it", "what", "how", "why", "when", "where"]
    en_count = sum(1 for w in en_words if w in text_lower)
    
    if fr_count > en_count + 2:
        return "fr"
    elif en_count > fr_count + 2:
        return "en"
    
    return "unknown"


def is_spam(text: str) -> bool:
    """
    Detect if text is spam.
    
    Returns True if likely spam.
    """
    words = text.split()
    
    # Too many repetitions
    if words and len(set(words)) / len(words) < 0.2:
        return True
    
    # Common spam patterns
    spam_patterns = [
        r"buy now",
        r"click here",
        r"limited time",
        r"act now",
        r"free money",
        r"winner",
        r"congratulations",
        r"\$\d+",  # Dollar amounts
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def has_code_blocks(text: str) -> bool:
    """Check if text contains code blocks"""
    code_indicators = [
        "def ",
        "function",
        "class ",
        "import ",
        "from ",
        "const ",
        "let ",
        "var ",
        "if (",
        "for (",
        "while (",
        "return ",
        "console.log",
        "print(",
    ]
    
    return any(indicator in text for indicator in code_indicators)


def filter_by_quality(text: str, config: Dict) -> Optional[Dict]:
    """
    Main filter function.
    
    Returns dict with:
        - accept: bool
        - score: float
        - reason: str
    """
    # Skip empty
    if not text or len(text.strip()) < 20:
        return {"accept": False, "score": 0, "reason": "too_short"}
    
    # Skip spam
    if is_spam(text):
        return {"accept": False, "score": 0, "reason": "spam"}
    
    # Calculate score
    score = quality_score(text, config)
    
    # Check minimum score
    min_score = config.get("min_score", 0.65)
    
    if score >= min_score:
        return {
            "accept": True,
            "score": score,
            "reason": "passed"
        }
    else:
        return {
            "accept": False,
            "score": score,
            "reason": f"score_below_min ({score:.2f} < {min_score})"
        }


# CLI test
if __name__ == "__main__":
    import json
    
    # Test
    test_config = {
        "keywords": ["python", "linux"],
        "min_words": 80,
        "min_score": 0.65,
        "lang": "en"
    }
    
    test_texts = [
        "This is a short text.",  # Too short, low score
        "Python is a great programming language for data science and machine learning. Linux is an open source operating system used by millions of developers worldwide.",  # Good
        "Buy now limited time offer!!! Free money winner congratulations!",  # Spam
    ]
    
    for text in test_texts:
        result = filter_by_quality(text, test_config)
        print(f"Text: {text[:50]}...")
        print(f"Result: {json.dumps(result)}\n")
