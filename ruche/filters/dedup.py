"""
Ruche Dedup Filter - SimHash for fast deduplication
64-bit fingerprint for near-duplicate detection
"""
import hashlib
from typing import Set, List, Optional
import numpy as np


class SimHash:
    """64-bit SimHash for deduplication"""
    
    def __init__(self, features: List[str] = None):
        self.features = features or []
    
    def compute(self, text: str, top_n: int = 100) -> int:
        """
        Compute 64-bit fingerprint from text.
        
        Args:
            text: Input text
            top_n: Number of features to consider
        
        Returns:
            64-bit integer fingerprint
        """
        words = text.lower().split()[:top_n]
        
        # Hash each word to 64-bit
        bits = [0] * 64
        
        for word in words:
            # Get 64-bit hash of word
            h = int(hashlib.md5(word.encode()).hexdigest()[:16], 16)
            
            # XOR each bit position
            for i in range(64):
                if h & (1 << i):
                    bits[i] += 1
                else:
                    bits[i] -= 1
        
        # Build fingerprint: 1 if positive, 0 if negative
        fingerprint = 0
        for i, b in enumerate(bits):
            if b > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Calculate Hamming distance between two hashes"""
        return bin(hash1 ^ hash2).count('1')


class DedupFilter:
    """Deduplication filter using SimHash"""
    
    def __init__(self, threshold: float = 0.95):
        """
        Args:
            threshold: Similarity threshold (0.0-1.0). 
                       Higher = more strict (more unique)
        """
        self.threshold = threshold
        self.seen_hashes: Set[int] = set()
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_added": 0
        }
    
    def check(self, text: str) -> bool:
        """
        Check if text is unique.
        
        Args:
            text: Text to check
        
        Returns:
            True if unique (not duplicate), False if duplicate
        """
        self.stats["total_processed"] += 1
        
        if not text:
            return True
        
        # Compute fingerprint
        fingerprint = SimHash().compute(text)
        
        # Check for near-duplicates
        for seen in self.seen_hashes:
            # Similarity = 1 - (hamming / 64)
            distance = SimHash().hamming_distance(fingerprint, seen)
            similarity = 1 - (distance / 64)
            
            if similarity >= self.threshold:
                self.stats["duplicates_found"] += 1
                return False
        
        # It's unique
        self.seen_hashes.add(fingerprint)
        self.stats["unique_added"] += 1
        return True
    
    def add(self, text: str):
        """Manually add text to seen set"""
        fingerprint = SimHash().compute(text)
        self.seen_hashes.add(fingerprint)
    
    def reset(self):
        """Clear all seen hashes"""
        self.seen_hashes.clear()
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_added": 0
        }
    
    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return self.stats.copy()


def batch_deduplicate(texts: List[str], threshold: float = 0.95) -> List[str]:
    """
    Batch deduplication utility.
    
    Args:
        texts: List of texts
        threshold: Similarity threshold
    
    Returns:
        List of unique texts (first occurrence kept)
    """
    dedup = DedupFilter(threshold)
    unique = []
    
    for text in texts:
        if dedup.check(text):
            unique.append(text)
    
    return unique


# CLI test
if __name__ == "__main__":
    # Test
    dedup = DedupFilter(threshold=0.90)
    
    texts = [
        "This is a unique text about Python programming.",
        "This is a unique text about Python programming.",  # Duplicate
        "Python is a great programming language.",
        "Machine learning with Python is powerful.",  # Different
        "Python is a great programming language.",  # Very similar to above
    ]
    
    for text in texts:
        is_unique = dedup.check(text)
        print(f"Unique: {is_unique} | Text: {text[:50]}...")
    
    print(f"\nStats: {dedup.get_stats()}")
