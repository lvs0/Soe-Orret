"""
memory/aria.py
Aria — 5-layer SQLite memory system.
Layers: episodic, semantic, procedural, sensory, affective.
Each layer is a separate SQLite table with shared primary key space.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Any, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    layer: str
    key: str
    value: str
    timestamp: float
    tags: List[str]


class LayeredMemory:
    """
    5-layer SQLite memory:
    1. episodic  — event records, timestamps, contexts
    2. semantic  — facts, concepts, propositional knowledge
    3. procedural — how-to, steps, recipes
    4. sensory  — raw impressions, embeddings, image/audio references
    5. affective — valence, arousal, sentiment signals
    """

    LAYERS = ["episodic", "semantic", "procedural", "sensory", "affective"]

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        for layer in self.LAYERS:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {layer} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT DEFAULT '[]'
                )
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{layer}_key ON {layer}(key)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{layer}_timestamp ON {layer}(timestamp)")
        self.conn.commit()

    def store(self, layer: str, key: str, value: Any, tags: Optional[List[str]] = None) -> int:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown layer: {layer}. Must be one of {self.LAYERS}")
        tags = tags or []
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT INTO {layer} (key, value, timestamp, tags) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value), time.time(), json.dumps(tags)),
        )
        self.conn.commit()
        return cur.lastrowid

    def retrieve(self, layer: str, key: str) -> Optional[List[MemoryEntry]]:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown layer: {layer}")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT key, value, timestamp, tags FROM {layer} WHERE key = ? ORDER BY timestamp DESC",
            (key,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        return [
            MemoryEntry(layer=layer, key=r[0], value=json.loads(r[1]), timestamp=r[2], tags=json.loads(r[3]))
            for r in rows
        ]

    def search(self, layer: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown layer: {layer}")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT key, value, timestamp, tags FROM {layer} WHERE value LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = cur.fetchall()
        return [
            MemoryEntry(layer=layer, key=r[0], value=json.loads(r[1]), timestamp=r[2], tags=json.loads(r[3]))
            for r in rows
        ]

    def recent(self, layer: str, limit: int = 20) -> List[MemoryEntry]:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown layer: {layer}")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT key, value, timestamp, tags FROM {layer} ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            MemoryEntry(layer=layer, key=r[0], value=json.loads(r[1]), timestamp=r[2], tags=json.loads(r[3]))
            for r in rows
        ]

    def delete(self, layer: str, key: str) -> int:
        if layer not in self.LAYERS:
            raise ValueError(f"Unknown layer: {layer}")
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {layer} WHERE key = ?", (key,))
        self.conn.commit()
        return cur.rowcount

    def close(self):
        self.conn.close()

    def __repr__(self) -> str:
        return f"Aria(db={self.db_path}, layers={self.LAYERS})"
