# SOE — Système Orret Experiment

## Structure
```
~/soe/
├── core/
│   └── looplib/       # Format .loop v1.0 — binary columnar for fine-tuning
│       ├── SPEC.md     # Format specification
│       └── looplib.py  # Python implementation
├── ruche/             # Data collection system
│   └── ruche.py       # CLI scraper + quality scorer
├── datasets/          # .loop files go here
├── models/            # Fine-tuned models
└── colab/
    └── soe_finetune.ipynb  # Google Colab notebook
```

## Installation
```bash
# Setup Python venv
python3 -m venv ~/.soe/.venv
~/.soe/.venv/bin/pip install zstandard numpy

# Test .loop format
~/.soe/.venv/bin/python3 ~/soe/core/looplib/looplib.py create data.jsonl output.loop
```

## .loop Format
Binary columnar format for LLM fine-tuning:
- **Header**: 22 bytes (MAGIC + version + schema_hash + nb_batches + compression)
- **Index**: 8 bytes per batch offset
- **Batches**: length-prefixed, Zstd compressed columnar data
- **Footer**: CRC32 + MAGIC end marker

```python
from looplib import LoopWriter, LoopDataset, LoopConfig

# Write
writer = LoopWriter("data.loop", LoopConfig())
writer.write_batch(rows)
writer.close()

# Read streaming
with LoopDataset("data.loop") as ds:
    for batch in ds.stream(batch_size=256):
        for row in batch:
            print(row["prompt"], row["quality_score"])
```

## Ruche — Data Collector
```bash
# Demo mode
python3 ruche.py --category soe_demo

# Scrape URL
python3 ruche.py --category ai_theory --url "https://arxiv.org/abs/2305.18223"

# Fetch arXiv papers
python3 ruche.py --category ml_papers --arxiv 2305.18223 2310.06825

# HuggingFace datasets
python3 ruche.py --category datasets --hf facebook/math_SPM
```

## Fine-tuning sur Google Colab
1. Upload `.loop` files to Google Drive
2. Open `colab/soe_finetune.ipynb` in Colab
3. Run all cells
4. Model saved to Drive → download GGUF for local inference

## Models Disponibles
| Provider | Model | Status |
|----------|-------|--------|
| Modal H100 | MiniMax-M2.5 | ✅ Primary |
| Groq | Llama 3.3 70B | ✅ Fallback |
| Ollama local | deepseek-r1:1.5b | ✅ Backup |

## API Endpoints (X250 local)
- NEXUS: http://localhost:8765
- MEDICAIN: http://localhost:8766
- SOE Orret App: http://localhost:8777
- WorldView: http://localhost:8773
