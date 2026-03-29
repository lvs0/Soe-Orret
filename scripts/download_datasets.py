#!/usr/bin/env python3
"""
Dataset Downloader - Fetch datasets from HuggingFace
Run from machine with HF access (or Colab)
"""
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets")
    sys.exit(1)

SOE_DIR = Path(os.path.expanduser("~/soe"))
RAW_DIR = SOE_DIR / "datasets" / "raw"

# Known high-quality datasets (French/Code/General)
DATASETS = {
    # French language
    "french-datasets/multilingual-reward-bench": {
        "subset": "code-python",
        "target": 500,
    },
    
    # Code datasets
    "open-instruct/codeparrot": {
        "target": 1000,
    },
    "bigcode/instruction-training": {
        "subset": "python",
        "target": 2000,
    },
    
    # General instruction
    "open-instruct/generic": {
        "target": 2000,
    },
    "argilla/ultrafeedback-binarized": {
        "subset": "train",
        "target": 2000,
    },
    
    # Reddit conversations
    "Bossologist/reddit-conversations-processed": {
        "target": 1000,
    },
    "leozadda/redditConversations4Bot": {
        "target": 500,
    },
    
    # Math/Reasoning
    "open-instruct/mathinstruct": {
        "target": 1000,
    },
    
    # French-specific
    "nvidia/Nemotron-Personas-France": {
        "target": 1000,
    },
    
    # Q&A in French
    "gaia/french-qa": {
        "target": 500,
    },
}


def download_dataset(name: str, config: dict = None) -> bool:
    """Download a dataset and convert to .loop"""
    config = config or {}
    print(f"\n📥 Downloading: {name}")
    
    try:
        # Load dataset
        ds = load_dataset(name, split="train")
        print(f"  Loaded: {len(ds)} examples")
        
        # Take subset
        target = config.get("target", 500)
        if len(ds) > target:
            ds = ds.select(range(target))
        
        # Process and save as JSONL (easier to convert)
        output = RAW_DIR / f"{name.replace('/', '-')}.jsonl"
        
        with open(output, "w", encoding="utf-8") as f:
            for item in ds:
                # Convert to text
                if "text" in item:
                    text = item["text"]
                elif "content" in item:
                    text = item["content"]
                elif "instruction" in item and "response" in item:
                    text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"
                elif "prompt" in item and "completion" in item:
                    text = f"Prompt: {item['prompt']}\nCompletion: {item['completion']}"
                else:
                    # Try to JSON dump
                    import json
                    text = json.dumps(item)
                
                f.write(text + "\n")
        
        print(f"  ✅ Saved: {output.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def download_all():
    """Download all configured datasets"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for name, config in DATASETS.items():
        result = download_dataset(name, config)
        results.append({"dataset": name, "success": result})
    
    # Summary
    success = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}")
    print(f"✅ Downloaded {success}/{len(results)} datasets")
    print(f"Location: {RAW_DIR}")
    print(f"{'='*50}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Download specific
        name = sys.argv[1]
        download_dataset(name, {"target": 1000})
    else:
        download_all()
