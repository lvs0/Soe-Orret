# SOE .loop Format — Specification v1.0

## Overview
.loop is a binary columnar format optimized for LLM fine-tuning data pipelines.
It replaces JSONL/CSV with a format that is:
- **10x faster** I/O (zero-copy reads via memory mapping)
- **70% smaller** (Zstd compression)
- **Streaming-ready** (batch-based, no full load required)
- **Self-describing** (schema embedded, versioned)

## File Structure

```
┌─────────────────────────────────┐
│ HEADER (64 bytes fixed)          │
│  - Magic: "LOOP" (4 bytes)      │
│  - Version: uint16              │
│  - Schema hash: uint64          │
│  - Num batches: uint64          │
│  - Compression: uint8           │
│  - Flags: uint8                 │
│  - Reserved: 40 bytes          │
├─────────────────────────────────┤
│ BATCH INDEX (8 bytes × N)       │
│  - Offset of each batch         │
├─────────────────────────────────┤
│ BATCH 1 (compressed)            │
│  - num_rows: uint64            │
│  - columns: Arrow IPC format    │
│    • input_ids (int32 array)   │
│    • attention_mask (int8 arr) │
│    • labels (int32 array)       │
│    • metadata (UTF-8 JSON)      │
├─────────────────────────────────┤
│ BATCH 2...                      │
├─────────────────────────────────┤
│ FOOTER                          │
│  - CRC64 checksum               │
│  - Magic end: "EOLO"            │
└─────────────────────────────────┘
```

## Compression
- Method 0: None (raw)
- Method 1: Zstd (default, best ratio/speed)
- Method 2: LZ4 (fastest)
- Method 3: Snappy (legacy compat)

## Schema
Each .loop file has a fixed schema defined at creation:
```python
{
  "prompt": {"type": "string", "tokenized": false},
  "response": {"type": "string", "tokenized": false},
  "input_ids": {"type": "int32", "tokenized": true, "dim": 1},
  "attention_mask": {"type": "int8", "tokenized": true, "dim": 1},
  "labels": {"type": "int32", "tokenized": true, "dim": 1},
  "category": {"type": "string"},
  "source": {"type": "string"},
  "quality_score": {"type": "float32"},
  "created_at": {"type": "int64"}
}
```

## Batching Strategy
- Target batch size: 8MB compressed per batch
- Sequence packing: sort by length, group similar lengths
- Max sequence length: 8192 tokens (configurable)
- Packing efficiency target: >85% context utilization

## Commands
```bash
# Create .loop from JSONL
loop create input.jsonl output.loop --compress zstd

# Inspect .loop
loop info output.loop

# Split for train/eval
loop split input.loop --ratio 0.9

# Merge multiple .loop files
loop merge file1.loop file2.loop output.loop
```

## Python API
```python
from looplib import LoopDataset, LoopWriter

# Read
ds = LoopDataset("data.loop")
for batch in ds.stream(batch_size=512):
    print(batch["input_ids"].shape)

# Write
writer = LoopWriter("output.loop", schema=schema)
writer.write_batch(rows)
writer.close()
```
