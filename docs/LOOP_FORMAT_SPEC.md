# LOOP_FORMAT_SPEC.md - Format .loop

## Overview
Format binaire séquentiel pour stocker des données d'entraînement LLM.

## Structure

### Header (9 bytes)
- Magic: `LOOP` (4 bytes)
- Version: uint8 (1 byte)
- Metadata length: uint32 (4 bytes)

### Metadata
JSON utf-8 payload.

### Entries
- CRC32: 4 bytes (checksum)
- Length: 4 bytes (entry size)
- Data: JSON utf-8

## API

```python
from core.looplib import LoopWriter, LoopReader

# Write
w = LoopWriter('data.loop')
w.write_header({'source': 'github', 'date': '2026-03-24'})
w.write_entry({'text': 'hello', 'lang': 'en'})
w.close()

# Read
r = LoopReader('data.loop')
for entry in r:
    print(entry)
r.close()
```

## Validation
```bash
loop validate data.loop
```
