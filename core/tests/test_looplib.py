#!/usr/bin/env python3
"""Test script for looplib"""
import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.looplib.writer import LoopWriter, LoopBatch
from core.looplib.reader import LoopReader
from core.looplib.validator import validate_loop, quick_check


def test_roundtrip():
    """Test write → read → validate"""
    print("[TEST] Creating test.loop...")
    
    writer = LoopWriter("/home/l-vs/soe/datasets/raw/test.loop", 
                       {"category": "test", "lang": "fr", "source": "unit_test"})
    
    # Add some batches
    for i in range(3):
        ids = np.array([1, 2, 3, 4, 5, i*10, i*10+1, i*10+2], dtype=np.int32)
        mask = np.ones_like(ids)
        labels = ids.copy()
        
        writer.add_batch(LoopBatch(ids, labels, mask))
    
    size = writer.save()
    print(f"[TEST] Created: {size} bytes")
    
    # Validate
    print("[TEST] Validating...")
    result = validate_loop("/home/l-vs/soe/datasets/raw/test.loop")
    print(f"[TEST] Valid: {result['valid']}")
    print(f"[TEST] Stats: {result['stats']}")
    
    # Read
    print("[TEST] Reading...")
    reader = LoopReader("/home/l-vs/soe/datasets/raw/test.loop")
    print(f"[TEST] Batches: {len(reader)}")
    
    meta = reader.read_metadata()
    print(f"[TEST] Metadata: {meta.category}, {meta.lang}")
    
    # Iterate
    for i, batch in enumerate(reader):
        print(f"[TEST] Batch {i}: {len(batch.input_ids)} tokens")
    
    print("[TEST] All tests passed!")
    return True


if __name__ == "__main__":
    test_roundtrip()
