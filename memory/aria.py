"""
Aria - 5-layer SQLite memory system

Layer architecture:
- L1: Working memory (ephemeral, session-based)
- L2: Short-term memory (recent context, hours)
- L3: Medium-term memory (daily context, days)
- L4: Long-term memory (persistent knowledge, months)
- L5: Archive (compressed historical data)
"""

import sqlite3
import json
import zlib
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: Optional[int] = None
    layer: int = 1  # 1-5
    key: str = ""
    value: Any = None
    metadata: Dict[str, Any] = None
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[str] = None


class AriaMemory:
    """
    5-layer SQLite memory system for agent context management.
    """
    
    LAYER_NAMES = {
        1: "working",
        2: "short_term",
        3: "medium_term",
        4: "long_term",
        5: "archive"
    }
    
    DEFAULT_TTLS = {
        1: timedelta(hours=1),      # Working: 1 hour
        2: timedelta(hours=24),     # Short-term: 24 hours
        3: timedelta(days=7),       # Medium-term: 7 days
        4: timedelta(days=90),      # Long-term: 90 days
        5: None                     # Archive: permanent
    }
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer INTEGER NOT NULL CHECK(layer BETWEEN 1 AND 5),
                key TEXT NOT NULL,
                value BLOB NOT NULL,
                metadata BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                UNIQUE(layer, key)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer ON memory_entries(layer)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires ON memory_entries(expires_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_key ON memory_entries(key)
        """)
        
        conn.commit()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        return zlib.compress(json.dumps(value, default=str).encode('utf-8'))
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        return json.loads(zlib.decompress(data).decode('utf-8'))
    
    def _compute_expiry(self, layer: int) -> Optional[str]:
        """Compute expiration timestamp for a layer."""
        ttl = self.DEFAULT_TTLS.get(layer)
        if ttl is None:
            return None
        expiry = datetime.utcnow() + ttl
        return expiry.isoformat()
    
    def store(self, layer: int, key: str, value: Any, metadata: Optional[Dict] = None, 
              ttl: Optional[timedelta] = None) -> bool:
        """
        Store a value in the specified memory layer.
        
        Args:
            layer: Memory layer (1-5)
            key: Unique key for the entry
            value: Data to store
            metadata: Optional metadata dictionary
            ttl: Optional custom TTL (overrides default)
            
        Returns:
            True if stored successfully
        """
        if not 1 <= layer <= 5:
            raise ValueError(f"Layer must be 1-5, got {layer}")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        serialized_value = self._serialize(value)
        serialized_metadata = self._serialize(metadata) if metadata else None
        
        if ttl is not None:
            expires_at = (datetime.utcnow() + ttl).isoformat()
        else:
            expires_at = self._compute_expiry(layer)
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO memory_entries 
                (layer, key, value, metadata, expires_at, access_count, created_at)
                VALUES (?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            """, (layer, key, serialized_value, serialized_metadata, expires_at))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error storing memory: {e}")
            return False
    
    def retrieve(self, layer: int, key: str) -> Optional[MemoryEntry]:
        """
        Retrieve a value from the specified memory layer.
        
        Args:
            layer: Memory layer (1-5)
            key: Key to retrieve
            
        Returns:
            MemoryEntry if found and not expired, None otherwise
        """
        if not 1 <= layer <= 5:
            raise ValueError(f"Layer must be 1-5, got {layer}")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Clean expired entries first
        self._cleanup_expired()
        
        cursor.execute("""
            SELECT * FROM memory_entries 
            WHERE layer = ? AND key = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        """, (layer, key))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        # Update access stats
        cursor.execute("""
            UPDATE memory_entries 
            SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (row['id'],))
        conn.commit()
        
        return MemoryEntry(
            id=row['id'],
            layer=row['layer'],
            key=row['key'],
            value=self._deserialize(row['value']),
            metadata=self._deserialize(row['metadata']) if row['metadata'] else None,
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            access_count=row['access_count'] + 1,
            last_accessed=datetime.utcnow().isoformat()
        )
    
    def search(self, layer: Optional[int] = None, key_pattern: Optional[str] = None,
               limit: int = 100) -> List[MemoryEntry]:
        """
        Search for memory entries.
        
        Args:
            layer: Optional layer filter
            key_pattern: Optional key pattern (SQL LIKE)
            limit: Maximum results
            
        Returns:
            List of matching MemoryEntry objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._cleanup_expired()
        
        query = """
            SELECT * FROM memory_entries 
            WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        """
        params = []
        
        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)
        
        if key_pattern is not None:
            query += " AND key LIKE ?"
            params.append(f"%{key_pattern}%")
        
        query += " ORDER BY last_accessed DESC NULLS LAST LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [
            MemoryEntry(
                id=row['id'],
                layer=row['layer'],
                key=row['key'],
                value=self._deserialize(row['value']),
                metadata=self._deserialize(row['metadata']) if row['metadata'] else None,
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                access_count=row['access_count'],
                last_accessed=row['last_accessed']
            )
            for row in rows
        ]
    
    def promote(self, key: str, from_layer: int, to_layer: int) -> bool:
        """
        Promote a memory entry to a higher layer.
        
        Args:
            key: Key to promote
            from_layer: Source layer
            to_layer: Target layer (must be >= from_layer)
            
        Returns:
            True if promoted successfully
        """
        if to_layer < from_layer:
            raise ValueError("Can only promote to equal or higher layers")
        
        entry = self.retrieve(from_layer, key)
        if entry is None:
            return False
        
        # Store in new layer
        success = self.store(to_layer, key, entry.value, entry.metadata)
        
        # Remove from old layer if promotion succeeded
        if success and to_layer != from_layer:
            self.delete(from_layer, key)
        
        return success
    
    def delete(self, layer: int, key: str) -> bool:
        """Delete a memory entry."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM memory_entries WHERE layer = ? AND key = ?
        """, (layer, key))
        conn.commit()
        
        return cursor.rowcount > 0
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM memory_entries 
            WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
        """)
        conn.commit()
    
    def get_layer_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each layer."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._cleanup_expired()
        
        stats = {}
        for layer in range(1, 6):
            cursor.execute("""
                SELECT COUNT(*) as count, 
                       AVG(access_count) as avg_access,
                       MAX(created_at) as newest,
                       MIN(created_at) as oldest
                FROM memory_entries WHERE layer = ?
            """, (layer,))
            
            row = cursor.fetchone()
            stats[layer] = {
                'name': self.LAYER_NAMES[layer],
                'count': row['count'] or 0,
                'avg_access': row['avg_access'] or 0,
                'newest': row['newest'],
                'oldest': row['oldest']
            }
        
        return stats
    
    def consolidate(self):
        """
        Consolidate memories: promote frequently accessed entries
        and archive old entries from lower layers.
        """
        stats = self.get_layer_stats()
        
        # Promote frequently accessed short-term to medium-term
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT key, access_count FROM memory_entries 
            WHERE layer = 2 AND access_count > 5
        """)
        
        for row in cursor.fetchall():
            self.promote(row['key'], 2, 3)
        
        # Archive old long-term memories
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        cursor.execute("""
            SELECT key FROM memory_entries 
            WHERE layer = 4 AND created_at < ?
        """, (cutoff,))
        
        for row in cursor.fetchall():
            self.promote(row['key'], 4, 5)
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Example usage
    with AriaMemory("/tmp/aria_test.db") as memory:
        # Store in different layers
        memory.store(1, "current_task", {"action": "processing", "step": 3})
        memory.store(2, "session_context", {"user": "alice", "preferences": {"theme": "dark"}})
        memory.store(3, "daily_summary", {"tasks_completed": 5, "meetings": 2})
        memory.store(4, "user_profile", {"name": "Alice", "role": "developer"})
        memory.store(5, "historical_data", {"version": "1.0", "archived": True})
        
        # Retrieve
        entry = memory.retrieve(1, "current_task")
        print(f"Working memory: {entry.value if entry else 'Not found'}")
        
        # Search
        results = memory.search(key_pattern="session")
        print(f"Search results: {len(results)} entries")
        
        # Stats
        stats = memory.get_layer_stats()
        for layer, info in stats.items():
            print(f"Layer {layer} ({info['name']}): {info['count']} entries")
        
        # Cleanup
        import os
        os.remove("/tmp/aria_test.db")
