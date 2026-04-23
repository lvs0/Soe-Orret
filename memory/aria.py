"""5-layer hierarchical SQLite memory system (Aria)."""

import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    key: str
    content: str
    layer: int
    priority: float = 0.5
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class AriaMemory:
    """5-layer hierarchical SQLite memory system.

    Layers:
        0: Working memory (immediate, ephemeral)
        1: Short-term (session-level)
        2: Medium-term (day-level)
        3: Long-term (week/month)
        4: Archive (permanent)
    """

    LAYER_NAMES = {0: "working", 1: "short_term", 2: "medium_term", 3: "long_term", 4: "archive"}

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(Path.home() / ".soe" / "aria_memory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    layer INTEGER NOT NULL CHECK(layer BETWEEN 0 AND 4),
                    priority REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_access REAL NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_layer ON memory(layer)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_memory_access ON memory(last_access)")
            conn.commit()
            conn.close()

    def store(self, key: str, content: str, layer: int = 0, priority: float = 0.5, metadata: dict | None = None) -> int:
        """Store a memory entry.

        Args:
            key: Memory key
            content: Memory content
            layer: Target layer (0-4)
            priority: Access priority (0.0-1.0)
            metadata: Optional metadata dict

        Returns:
            Row ID of inserted/updated entry
        """
        if not 0 <= layer <= 4:
            raise ValueError(f"Layer must be 0-4, got {layer}")

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            now = time.time()
            meta_json = json.dumps(metadata or {})

            existing = c.execute("SELECT id FROM memory WHERE key = ?", (key,)).fetchone()
            if existing:
                c.execute("""
                    UPDATE memory SET content=?, layer=?, priority=?, last_access=?, metadata=?
                    WHERE key=?
                """, (content, layer, priority, now, meta_json, key))
                row_id = existing[0]
            else:
                c.execute("""
                    INSERT INTO memory (key, content, layer, priority, access_count, last_access, created_at, metadata)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                """, (key, content, layer, priority, now, now, meta_json))
                row_id = c.lastrowid

            conn.commit()
            conn.close()
            return row_id

    def retrieve(self, key: str, layer: int | None = None) -> MemoryEntry | None:
        """Retrieve a memory entry by key.

        Args:
            key: Memory key
            layer: Optional layer filter

        Returns:
            MemoryEntry if found, None otherwise
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            if layer is not None:
                row = c.execute("SELECT * FROM memory WHERE key=? AND layer=?", (key, layer)).fetchone()
            else:
                row = c.execute("SELECT * FROM memory WHERE key=?", (key,)).fetchone()

            conn.close()

            if row:
                return self._row_to_entry(row)
            return None

    def query(self, pattern: str = "%", layer: int | None = None, limit: int = 50) -> list[MemoryEntry]:
        """Query memories by key pattern.

        Args:
            pattern: SQL LIKE pattern
            layer: Optional layer filter
            limit: Max results

        Returns:
            List of matching MemoryEntries
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            if layer is not None:
                rows = c.execute(
                    "SELECT * FROM memory WHERE key LIKE ? AND layer=? ORDER BY priority DESC, last_access DESC LIMIT ?",
                    (pattern, layer, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM memory WHERE key LIKE ? ORDER BY priority DESC, last_access DESC LIMIT ?",
                    (pattern, limit)
                ).fetchall()

            conn.close()
            return [self._row_to_entry(r) for r in rows]

    def promote(self, key: str, target_layer: int | None = None) -> bool:
        """Promote memory to higher layer (lower number).

        Args:
            key: Memory key
            target_layer: Optional specific target, otherwise +1 layer

        Returns:
            True if promoted, False if not found or at layer 0
        """
        entry = self.retrieve(key)
        if not entry:
            return False

        if target_layer is None:
            new_layer = max(0, entry.layer - 1)
        else:
            new_layer = target_layer

        self.store(key, entry.content, layer=new_layer, priority=entry.priority, metadata=entry.metadata)
        return True

    def demote(self, key: str, target_layer: int | None = None) -> bool:
        """Demote memory to lower layer (higher number).

        Args:
            key: Memory key
            target_layer: Optional specific target, otherwise +1 layer

        Returns:
            True if demoted, False if not found or at layer 4
        """
        entry = self.retrieve(key)
        if not entry:
            return False

        if target_layer is None:
            new_layer = min(4, entry.layer + 1)
        else:
            new_layer = target_layer

        self.store(key, entry.content, layer=new_layer, priority=entry.priority, metadata=entry.metadata)
        return True

    def touch(self, key: str) -> bool:
        """Update access timestamp and count."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "UPDATE memory SET access_count=access_count+1, last_access=? WHERE key=?",
                (time.time(), key)
            )
            updated = c.rowcount > 0
            conn.commit()
            conn.close()
            return updated

    def delete(self, key: str, layer: int | None = None) -> bool:
        """Delete memory entries."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            if layer is not None:
                c.execute("DELETE FROM memory WHERE key=? AND layer=?", (key, layer))
            else:
                c.execute("DELETE FROM memory WHERE key=?", (key,))
            deleted = c.rowcount > 0
            conn.commit()
            conn.close()
            return deleted

    def stats(self) -> dict[int, int]:
        """Count entries per layer."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            stats = {i: 0 for i in range(5)}
            for row in c.execute("SELECT layer, COUNT(*) as cnt FROM memory GROUP BY layer"):
                stats[row[0]] = row[1]
            conn.close()
            return stats

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            key=row["key"],
            content=row["content"],
            layer=row["layer"],
            priority=row["priority"],
            access_count=row["access_count"],
            last_access=row["last_access"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"] or "{}")
        )
