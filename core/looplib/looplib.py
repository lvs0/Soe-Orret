"""
SOE .loop v1.0 — Binary Columnar Format for LLM Fine-tuning
Header: 22 bytes (explicit struct)
"""
import struct, json, mmap, zlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Tuple
from enum import IntEnum

MAGIC = b"LOOP"
VERSION = 1

class Compression(IntEnum):
    NONE = 0; ZSTD = 1; LZ4 = 2; SNAPPY = 3

DEFAULT_SCHEMA = {
    "input_ids":      {"dtype": "int32",  "tokenized": True},
    "attention_mask": {"dtype": "int8",   "tokenized": True},
    "labels":         {"dtype": "int32",  "tokenized": True},
    "category":       {"dtype": "str"},
    "source":         {"dtype": "str"},
    "quality_score":  {"dtype": "float32"},
    "timestamp":      {"dtype": "int64"},
}

@dataclass
class LoopConfig:
    schema:            Dict[str, Any] = field(default_factory=lambda: DEFAULT_SCHEMA.copy())
    compression:       int = Compression.ZSTD
    batch_size_target: int = 8 * 1024 * 1024
    max_seq_len:      int = 8192

# Header layout (22 bytes):
# [0:4] MAGIC(4) + [4:6] ver(2) + [6:8] schema_hi(2)
# [8:16] nb_batches(8) + [16:20] schema_lo(4) + [20] comp(1) + [21] pad(1)
def _hdr_pack(ver: int, nb: int, schema_json: str, comp: int) -> bytes:
    h = zlib.crc32(schema_json.encode()) & 0xFFFFFFFF
    return (
        MAGIC +
        struct.pack("<H", ver) +               # 2 bytes (pos 4-5)
        struct.pack("<H", (h >> 16) & 0xFFFF) +  # 2 bytes (pos 6-7)
        struct.pack("<Q", nb) +                # 8 bytes (pos 8-15)
        struct.pack("<I", h & 0xFFFFFFFF) +    # 4 bytes (pos 16-19) ← NOTE: <I> (4 bytes)
        struct.pack("<BB", comp & 0xFF, 0)      # 2 bytes (pos 20-21)
    )  # total = 4+2+2+8+4+2 = 22

def _hdr_unpack(data: bytes) -> Tuple[int, int, int, int]:
    ver    = struct.unpack("<H", data[4:6])[0]
    shi    = struct.unpack("<H", data[6:8])[0]
    nb     = struct.unpack("<Q", data[8:16])[0]
    slo    = struct.unpack("<I", data[16:20])[0]
    comp   = data[20]
    h = ((shi << 16) | slo) & 0xFFFFFFFF
    return ver, nb, h, comp

# ─── Writer ─────────────────────────────────────────────────
class LoopWriter:
    def __init__(self, path: str, config: Optional[LoopConfig] = None):
        self.path = Path(path)
        self.cfg = config or LoopConfig()
        self.schema_json = json.dumps(self.cfg.schema, sort_keys=True)
        self.batches: List[bytes] = []   # stored compressed batches
        self.rows_written = 0
        self._use_zstd = (self.cfg.compression == Compression.ZSTD)
        # Reserve header + space for index entries (filled later)
        self._header_end = 22  # header is always 22 bytes

    def _compress(self, data: bytes) -> bytes:
        if not self._use_zstd:
            return data
        try:
            import zstandard as Zstd
            return Zstd.ZstdCompressor(level=3).compress(data)
        except Exception:
            return data

    def write_batch(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        packed = self._pack_rows(rows)
        compressed = self._compress(packed)
        # Length prefix (8 bytes) + compressed data
        self.batches.append(struct.pack("<Q", len(compressed)) + compressed)
        self.rows_written += len(rows)

    def _pack_rows(self, rows: List[Dict[str, Any]]) -> bytes:
        if not rows:
            return b""
        cols = {}
        for key in rows[0].keys():
            vals = [r.get(key, None) for r in rows]
            dtype = self.cfg.schema.get(key, {}).get("dtype", "str")
            if dtype == "int32":
                cols[key] = np.array([int(v) if v is not None else 0 for v in vals], dtype=np.int32).tobytes()
            elif dtype == "int8":
                cols[key] = np.array([int(v) if v is not None else 0 for v in vals], dtype=np.int8).tobytes()
            elif dtype == "int64":
                cols[key] = np.array([int(v) if v is not None else 0 for v in vals], dtype=np.int64).tobytes()
            elif dtype == "float32":
                cols[key] = np.array([float(v) if v is not None else 0.0 for v in vals], dtype=np.float32).tobytes()
            else:
                encoded = json.dumps([str(v) if v is not None else "" for v in vals]).encode("utf-8")
                cols[key] = encoded  # just the JSON bytes — no length prefix
        result = struct.pack("<I", len(rows)) + struct.pack("<I", len(cols))
        for key, data in cols.items():
            result += struct.pack("<I", len(key.encode())) + key.encode()
            result += struct.pack("<Q", len(data)) + data
        return result

    def close(self):
        nb = len(self.batches)
        index_size = nb * 8
        data_start = self._header_end + index_size  # where first batch starts

        with open(self.path, "wb") as f:
            # Write header
            f.write(_hdr_pack(VERSION, nb, self.schema_json, self.cfg.compression))
            # Compute and write index entries
            pos = data_start
            for batch in self.batches:
                f.write(struct.pack("<Q", pos))
                pos += len(batch)
            # Write batches
            for batch in self.batches:
                f.write(batch)
            # Footer
            all_data = b"".join(self.batches)
            crc = zlib.crc32(all_data) & 0xFFFFFFFF
            f.write(struct.pack("<I", crc) + MAGIC)

        kb = self.path.stat().st_size // 1024
        print(f"⟨loop⟩ {self.rows_written} rows, {nb} batches → {self.path} ({kb}KB)")

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

# ─── Reader ────────────────────────────────────────────────
class LoopDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.cfg = LoopConfig()  # Use default schema for reading
        f = open(path, "rb")
        self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        ver, nb, self.schema_hash, self.comp = _hdr_unpack(self._mmap[:22])
        self.num_batches = nb
        self._offsets = [
            struct.unpack("<Q", self._mmap[22 + i*8 : 22 + (i+1)*8])[0]
            for i in range(nb)
        ]

    def stream(self, batch_size: int = 256) -> Iterator[List[Dict[str, Any]]]:
        for off in self._offsets:
            blen = struct.unpack("<Q", self._mmap[off:off+8])[0]
            data = self._mmap[off+8:off+8+blen]
            if self.comp == Compression.ZSTD:
                try:
                    import zstandard as Zstd
                    data = Zstd.ZstdDecompressor().decompress(data)
                except Exception:
                    pass
            rows = self._unpack_rows(data)
            for i in range(0, len(rows), batch_size):
                yield rows[i:i+batch_size]

    def _unpack_rows(self, data: bytes) -> List[Dict[str, Any]]:
        pos = 0
        num_rows = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
        num_cols = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
        cols = {}
        for _ in range(num_cols):
            klen = struct.unpack("<I", data[pos:pos+4])[0]; pos += 4
            key = data[pos:pos+klen].decode(); pos += klen
            dlen = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
            cols[key] = data[pos:pos+dlen]; pos += dlen
        rows = []
        for i in range(num_rows):
            row = {}
            for key, col_bytes in cols.items():
                dtype = self.cfg.schema.get(key, {}).get("dtype", "str")
                if dtype == "str":
                    row[key] = json.loads(col_bytes.decode())[i]
                else:
                    row[key] = np.frombuffer(col_bytes, dtype=dtype)[i]
            rows.append(row)
        return rows

    def __len__(self): return self.num_batches
    def close(self):
        self._mmap.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

# ─── CLI ───────────────────────────────────────────────────
def create_loop(jsonl_path: str, loop_path: str, category: str = "general"):
    import time
    cfg = LoopConfig(compression=Compression.ZSTD)
    writer = LoopWriter(loop_path, cfg)
    batch, size = [], 0
    with open(jsonl_path) as f:
        for line in f:
            if not (line := line.strip()): continue
            row = json.loads(line)
            row["category"] = category
            row["timestamp"] = int(time.time())
            row.setdefault("quality_score", 0.8)
            batch.append(row)
            size += len(json.dumps(row))
            if size >= cfg.batch_size_target:
                writer.write_batch(batch)
                batch, size = [], 0
        if batch:
            writer.write_batch(batch)
    writer.close()

def inspect_loop(path: str):
    with LoopDataset(path) as ds:
        print(f"⟨loop⟩ {path}")
        print(f"  Version:     v{VERSION}")
        print(f"  Batches:     {ds.num_batches}")
        print(f"  Compression: {Compression(ds.comp).name}")
        print(f"  Size:        {Path(path).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("SOE .loop v1 — create | info"); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "create":
        create_loop(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "general")
    elif cmd == "info":
        inspect_loop(sys.argv[2])
    else:
        print(f"Unknown: {cmd}")
