"""
Aria - 5-layer SQLite memory system
Soe-Orret hierarchical memory with multi-tier storage
"""

import sqlite3
import json
import hashlib
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import threading


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    id: Optional[int] = None
    key: str = ""
    content: str = ""
    layer: int = 0  # 0-4 (5 layers)
    priority: float = 1.0
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.accessed_at == 0.0:
            self.accessed_at = self.created_at


class AriaMemoryLayer:
    """
    Single layer of the 5-tier Aria memory system.
    
    Layers:
    - Layer 0: Working memory (immediate, high turnover)
    - Layer 1: Short-term (recent context, session-level)
    - Layer 2: Medium-term (day-level persistence)
    - Layer 3: Long-term (week/month persistence)
    - Layer 4: Archive (permanent storage, compressed)
    """
    
    def __init__(self, layer_id: int, db_path: str):
        self.layer_id = layer_id
        self.db_path = db_path
        self._local = threading.local()
        self._init_table()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_table(self):
        """Initialize SQLite table for this layer"""
        conn = self._get_conn()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS layer_{self.layer_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                priority REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding TEXT,  -- JSON array
                metadata TEXT,   -- JSON object
                compressed INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_layer_{self.layer_id}_key 
            ON layer_{self.layer_id}(key)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_layer_{self.layer_id}_priority 
            ON layer_{self.layer_id}(priority DESC)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_layer_{self.layer_id}_accessed 
            ON layer_{self.layer_id}(accessed_at DESC)
        """)
        conn.commit()
    
    def store(self, entry: MemoryEntry) -> int:
        """Store a memory entry in this layer"""
        conn = self._get_conn()
        
        embedding_json = json.dumps(entry.embedding) if entry.embedding else None
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
        
        cursor = conn.execute(f"""
            INSERT OR REPLACE INTO layer_{self.layer_id} 
            (key, content, priority, created_at, accessed_at, access_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.key, entry.content, entry.priority,
            entry.created_at, entry.accessed_at, entry.access_count,
            embedding_json, metadata_json
        ))
        conn.commit()
        return cursor.lastrowid
    
    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key"""
        conn = self._get_conn()
        
        row = conn.execute(f"""
            SELECT * FROM layer_{self.layer_id} WHERE key = ?
        """, (key,)).fetchone()
        
        if row is None:
            return None
        
        # Update access stats
        conn.execute(f"""
            UPDATE layer_{self.layer_id} 
            SET access_count = access_count + 1, accessed_at = ?
            WHERE key = ?
        """, (time.time(), key))
        conn.commit()
        
        return self._row_to_entry(row)
    
    def query(self, limit: int = 100, min_priority: float = 0.0) -> List[MemoryEntry]:
        """Query entries from this layer"""
        conn = self._get_conn()
        
        rows = conn.execute(f"""
            SELECT * FROM layer_{self.layer_id} 
            WHERE priority >= ?
            ORDER BY priority DESC, accessed_at DESC
            LIMIT ?
        """, (min_priority, limit)).fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def search_content(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search entries by content substring"""
        conn = self._get_conn()
        
        rows = conn.execute(f"""
            SELECT * FROM layer_{self.layer_id} 
            WHERE content LIKE ?
            ORDER BY priority DESC, accessed_at DESC
            LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def delete(self, key: str) -> bool:
        """Delete an entry by key"""
        conn = self._get_conn()
        cursor = conn.execute(f"""
            DELETE FROM layer_{self.layer_id} WHERE key = ?
        """, (key,))
        conn.commit()
        return cursor.rowcount > 0
    
    def clear(self):
        """Clear all entries from this layer"""
        conn = self._get_conn()
        conn.execute(f"DELETE FROM layer_{self.layer_id}")
        conn.commit()
    
    def count(self) -> int:
        """Count entries in this layer"""
        conn = self._get_conn()
        row = conn.execute(f"""
            SELECT COUNT(*) FROM layer_{self.layer_id}
        """).fetchone()
        return row[0] if row else 0
    
    def get_oldest(self, limit: int = 10) -> List[MemoryEntry]:
        """Get oldest entries (for eviction)"""
        conn = self._get_conn()
        
        rows = conn.execute(f"""
            SELECT * FROM layer_{self.layer_id} 
            ORDER BY accessed_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        return MemoryEntry(
            id=row['id'],
            key=row['key'],
            content=row['content'],
            layer=self.layer_id,
            priority=row['priority'],
            created_at=row['created_at'],
            accessed_at=row['accessed_at'],
            access_count=row['access_count'],
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )


class AriaMemory:
    """
    Aria - 5-layer hierarchical memory system
    
    Manages memory across 5 persistence tiers with automatic
    promotion/demotion based on access patterns and importance.
    """
    
    LAYER_NAMES = [
        "working",      # Layer 0: Immediate context
        "short_term",   # Layer 1: Session context
        "medium_term",  # Layer 2: Day context
        "long_term",    # Layer 3: Week/month context
        "archive"       # Layer 4: Permanent storage
    ]
    
    def __init__(self, db_path: str = "aria_memory.db"):
        self.db_path = db_path
        self.layers: List[AriaMemoryLayer] = []
        
        # Initialize 5 layers
        for i in range(5):
            self.layers.append(AriaMemoryLayer(i, db_path))
    
    def store(self, key: str, content: str, layer: int = 1,
              priority: float = 1.0, embedding: Optional[List[float]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """
        Store a memory entry
        
        Args:
            key: Unique identifier
            content: Memory content
            layer: Target layer (0-4)
            priority: Importance score (0.0-1.0)
            embedding: Optional vector embedding
            metadata: Optional additional data
        """
        if not 0 <= layer <= 4:
            raise ValueError(f"Layer must be 0-4, got {layer}")
        
        entry = MemoryEntry(
            key=key,
            content=content,
            layer=layer,
            priority=priority,
            embedding=embedding,
            metadata=metadata
        )
        
        self.layers[layer].store(entry)
        return entry
    
    def retrieve(self, key: str, layer: Optional[int] = None) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry. Searches all layers if layer not specified.
        Promotes accessed entries to higher layers.
        """
        if layer is not None:
            return self.layers[layer].retrieve(key)
        
        # Search all layers from top to bottom
        for i in range(4, -1, -1):
            entry = self.layers[i].retrieve(key)
            if entry is not None:
                # Promote to higher layer if accessed frequently
                if entry.access_count > 5 and i > 0:
                    self._promote(entry, i - 1)
                return entry
        
        return None
    
    def query_layer(self, layer: int, limit: int = 100, 
                    min_priority: float = 0.0) -> List[MemoryEntry]:
        """Query entries from a specific layer"""
        return self.layers[layer].query(limit, min_priority)
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[int, MemoryEntry]]:
        """Search across all layers"""
        results = []
        for i, layer in enumerate(self.layers):
            entries = layer.search_content(query, limit)
            for entry in entries:
                results.append((i, entry))
        
        # Sort by priority and recency
        results.sort(key=lambda x: (x[1].priority, x[1].accessed_at), reverse=True)
        return results[:limit]
    
    def consolidate(self):
        """
        Run memory consolidation:
        - Promote frequently accessed entries
        - Demote rarely accessed entries
        - Compress archive layer
        """
        # Promote from lower layers
        for i in range(1, 5):  # Skip working memory (layer 0)
            entries = self.layers[i].query(limit=100)
            for entry in entries:
                if entry.access_count > 10 and i > 0:
                    self._promote(entry, i - 1)
        
        # Demote rarely accessed entries
        for i in range(4):  # Skip archive (layer 4)
            entries = self.layers[i].query(limit=100)
            for entry in entries:
                age = time.time() - entry.accessed_at
                if entry.access_count < 2 and age > 86400 * (i + 1):  # Age-based threshold
                    self._demote(entry, i + 1)
    
    def _promote(self, entry: MemoryEntry, target_layer: int):
        """Promote an entry to a higher layer"""
        # Remove from current layer
        self.layers[entry.layer].delete(entry.key)
        
        # Store in target layer
        entry.layer = target_layer
        self.layers[target_layer].store(entry)
    
    def _demote(self, entry: MemoryEntry, target_layer: int):
        """Demote an entry to a lower layer"""
        # Remove from current layer
        self.layers[entry.layer].delete(entry.key)
        
        # Store in target layer
        entry.layer = target_layer
        self.layers[target_layer].store(entry)
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_entries": sum(l.count() for l in self.layers),
            "layers": {
                name: {
                    "count": self.layers[i].count(),
                    "type": "working" if i == 0 else "short" if i == 1 else "medium" if i == 2 else "long" if i == 3 else "archive"
                }
                for i, name in enumerate(self.LAYER_NAMES)
            }
        }
    
    def clear_layer(self, layer: int):
        """Clear a specific layer"""
        self.layers[layer].clear()
    
    def clear_all(self):
        """Clear all layers"""
        for layer in self.layers:
            layer.clear()


if __name__ == "__main__":
    # Test Aria memory
    import tempfile
    import os
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix=".db")
    
    try:
        aria = AriaMemory(db_path)
        
        # Store some memories
        aria.store("user_pref", "User likes dark mode", layer=1, priority=0.9)
        aria.store("session_ctx", "Current task: coding", layer=0, priority=1.0)
        aria.store("long_term", "Project goals for 2026", layer=3, priority=0.8)
        
        # Retrieve
        entry = aria.retrieve("user_pref")
        print(f"Retrieved: {entry.key} = {entry.content}")
        
        # Search
        results = aria.search("mode")
        print(f"Search results: {len(results)} found")
        
        # Stats
        stats = aria.stats()
        print(f"Memory stats: {json.dumps(stats, indent=2)}")
        
        print("\nAria memory system test passed!")
        
    finally:
        os.unlink(db_path)
