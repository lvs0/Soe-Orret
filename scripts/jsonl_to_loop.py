#!/usr/bin/env python3
"""
Convert JSONL to .loop - without external dependencies
"""
import os
import sys
import json
import struct
from pathlib import Path

SOE_DIR = Path(os.path.expanduser("~/soe"))
RAW_DIR = SOE_DIR / "datasets" / "raw"

MAGIC = b"LOOP"
FOOTER = b"POOLEND\x00"
HEADER_SIZE = 64


def jsonl_to_loop(jsonl_path: str, output_name: str = None) -> bool:
    """Convert JSONL file to .loop format"""
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found")
        return False
    
    output_name = output_name or jsonl_path.stem
    output_path = RAW_DIR / f"{output_name}.loop"
    
    # Read all lines
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try JSON parse
                data = json.loads(line)
                if isinstance(data, dict):
                    # Extract text field
                    text = data.get("text") or data.get("content") or data.get("instruction", "")
                    if "response" in data:
                        text += f"\n\nResponse: {data['response']}"
                    texts.append(text[:500])  # Truncate
                else:
                    texts.append(str(data)[:500])
            except json.JSONDecodeError:
                # Plain text
                texts.append(line[:500])
    
    if not texts:
        print(f"No valid data in {jsonl_path}")
        return False
    
    print(f"Processing {len(texts)} examples from {jsonl_path.name}")
    
    # Write .loop file
    with open(output_path, "wb") as f:
        # Header
        header = bytearray(HEADER_SIZE)
        header[:4] = MAGIC
        struct.pack_into("<H", header, 4, 1)  # version
        struct.pack_into("<Q", header, 10, len(texts))  # n_batches
        f.write(header)
        
        # Data
        data_bytes = b''
        for t in texts:
            encoded = t.encode('utf-8')[:200] + b'\x00'
            data_bytes += encoded
        
        # Index
        index_data = bytearray(len(texts) * 16)
        offset = 0
        for i in range(len(texts)):
            batch_bytes = texts[i].encode('utf-8')[:200]
            struct.pack_into("<QQ", index_data, i * 16, offset, len(batch_bytes))
            offset += len(batch_bytes)
        
        f.write(data_bytes)
        f.write(index_data)
        
        # Footer
        metadata = {
            "source": jsonl_path.name,
            "n_examples": len(texts),
            "converted": "jsonl-to-loop"
        }
        meta_json = json.dumps(metadata)
        meta_bytes = meta_json.encode('utf-8')
        f.write(FOOTER)
        f.write(struct.pack("<Q", len(meta_bytes)))
        f.write(meta_bytes)
    
    print(f"✅ Created: {output_path.name}")
    return True


def convert_all():
    """Convert all JSONL files in raw/"""
    jsonl_files = list(RAW_DIR.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found")
        return
    
    for jf in jsonl_files:
        jsonl_to_loop(jf)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        jsonl_to_loop(sys.argv[1])
    else:
        convert_all()
