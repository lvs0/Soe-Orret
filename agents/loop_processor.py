#!/usr/bin/env python3
"""LOOP-PROCESSOR - Process and validate .loop files"""
import os
import sys
import json
from pathlib import Path
from core.looplib.validator import validate_loop
from core.looplib.reader import LoopReader

SOE_DIR = Path(os.path.expanduser("~/soe"))
RAW_DIR = SOE_DIR / "datasets" / "raw"
PROCESSED_DIR = SOE_DIR / "datasets" / "processed"

def process_all():
    """Process all .loop files in raw/"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for loop_file in RAW_DIR.glob("*.loop"):
        result = process_one(loop_file)
        results.append(result)
    
    return results

def process_one(path: Path) -> dict:
    """Process single .loop file"""
    print(f"Processing {path.name}...")
    
    # Validate
    validation = validate_loop(str(path))
    
    if not validation["valid"]:
        print(f"  ❌ Invalid: {validation['errors']}")
        return {"file": path.name, "valid": False, "errors": validation["errors"]}
    
    # Read metadata
    try:
        reader = LoopReader(str(path))
        meta = reader.read_metadata()
        
        result = {
            "file": path.name,
            "valid": True,
            "batches": len(reader),
            "stats": validation["stats"],
            "metadata": {
                "category": meta.category,
                "lang": meta.lang,
                "source": meta.source
            }
        }
        
        # Move to processed
        dest = PROCESSED_DIR / path.name
        path.rename(dest)
        
        print(f"  ✅ Valid: {result['batches']} batches")
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {"file": path.name, "valid": False, "errors": [str(e)]}

def main():
    """CLI entry"""
    if len(sys.argv) > 1:
        # Single file
        result = process_one(Path(sys.argv[1]))
        print(json.dumps(result, indent=2))
    else:
        # All files
        results = process_all()
        print(f"\n✅ Processed {len(results)} files")
        for r in results:
            status = "✅" if r.get("valid") else "❌"
            print(f"  {status} {r['file']}")

if __name__ == "__main__":
    main()
