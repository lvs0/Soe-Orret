"""
Aria - Mémoire hiérarchique à 5 couches SQLite
Système de mémoire événementielle avec persistance SQLite.
"""

import sqlite3
import json
import time
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading


@dataclass
class MemoryEntry:
    """Entrée de mémoire avec métadonnées."""
    id: Optional[str] = None
    layer: int = 0  # 0-4 (couche hiérarchique)
    content: str = ""
    embedding: Optional[List[float]] = None
    timestamp: float = 0.0
    ttl: Optional[float] = None  # Time-to-live en secondes
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    parent_id: Optional[str] = None
    priority: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Génère un ID unique basé sur le contenu et le timestamp."""
        data = f"{self.content}:{self.timestamp}:{time.time_ns()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'entrée en dictionnaire."""
        return {
            'id': self.id,
            'layer': self.layer,
            'content': self.content,
            'embedding': json.dumps(self.embedding) if self.embedding else None,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'tags': json.dumps(self.tags),
            'metadata': json.dumps(self.metadata),
            'parent_id': self.parent_id,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Crée une entrée depuis un dictionnaire."""
        return cls(
            id=data['id'],
            layer=data['layer'],
            content=data['content'],
            embedding=json.loads(data['embedding']) if data['embedding'] else None,
            timestamp=data['timestamp'],
            ttl=data['ttl'],
            tags=json.loads(data['tags']) if data['tags'] else [],
            metadata=json.loads(data['metadata']) if data['metadata'] else {},
            parent_id=data['parent_id'],
            priority=data['priority']
        )
    
    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class AriaMemory:
    """
    Système de mémoire hiérarchique Aria avec 5 couches.
    
    Couches:
    - Layer 0: Cache immédiat (TTL court, accès rapide)
    - Layer 1: Working memory (contexte actif)
    - Layer 2: Short-term (événements récents)
    - Layer 3: Long-term (connaissances persistantes)
    - Layer 4: Archive (historique complet)
    """
    
    LAYER_NAMES = {
        0: "cache",
        1: "working",
        2: "short_term",
        3: "long_term",
        4: "archive"
    }
    
    DEFAULT_TTL = {
        0: 60,        # 1 minute
        1: 3600,      # 1 heure
        2: 86400,     # 1 jour
        3: None,      # Pas d'expiration
        4: None       # Pas d'expiration
    }
    
    def __init__(self, db_path: str = "aria_memory.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Obtient une connexion thread-safe."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialise le schéma de la base de données."""
        conn = self._get_connection()
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                layer INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                timestamp REAL NOT NULL,
                ttl REAL,
                tags TEXT,
                metadata TEXT,
                parent_id TEXT,
                priority REAL DEFAULT 0.0,
                FOREIGN KEY (parent_id) REFERENCES memory_entries(id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer ON memory_entries(layer)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent ON memory_entries(parent_id)
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                timestamp REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, link_type),
                FOREIGN KEY (source_id) REFERENCES memory_entries(id),
                FOREIGN KEY (target_id) REFERENCES memory_entries(id)
            )
        """)
        
        conn.commit()
    
    def store(self, entry: MemoryEntry, auto_layer: bool = False) -> str:
        """
        Stocke une entrée en mémoire.
        
        Args:
            entry: L'entrée à stocker
            auto_layer: Si True, détermine automatiquement la couche
            
        Returns:
            ID de l'entrée stockée
        """
        if auto_layer:
            entry.layer = self._determine_layer(entry)
        
        if entry.ttl is None and entry.layer in self.DEFAULT_TTL:
            entry.ttl = self.DEFAULT_TTL[entry.layer]
        
        conn = self._get_connection()
        data = entry.to_dict()
        
        conn.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (id, layer, content, embedding, timestamp, ttl, tags, metadata, parent_id, priority)
            VALUES (:id, :layer, :content, :embedding, :timestamp, :ttl, :tags, :metadata, :parent_id, :priority)
        """, data)
        
        conn.commit()
        return entry.id
    
    def _determine_layer(self, entry: MemoryEntry) -> int:
        """Détermine automatiquement la couche appropriée."""
        # Logique simple basée sur la priorité et le contenu
        if entry.priority > 0.8:
            return 1  # Working memory pour haute priorité
        elif entry.priority > 0.5:
            return 2  # Short-term
        elif len(entry.content) > 1000:
            return 4  # Archive pour contenu long
        else:
            return 3  # Long-term par défaut
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Récupère une entrée par son ID."""
        conn = self._get_connection()
        
        row = conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?",
            (entry_id,)
        ).fetchone()
        
        if row is None:
            return None
        
        entry = MemoryEntry.from_dict(dict(row))
        
        if entry.is_expired():
            self.delete(entry_id)
            return None
        
        return entry
    
    def query(self, layer: Optional[int] = None, 
              tags: Optional[List[str]] = None,
              since: Optional[float] = None,
              limit: int = 100) -> List[MemoryEntry]:
        """
        Recherche des entrées selon critères.
        
        Args:
            layer: Filtrer par couche
            tags: Filtrer par tags (doit contenir tous)
            since: Timestamp minimum
            limit: Nombre max de résultats
            
        Returns:
            Liste des entrées correspondantes
        """
        conn = self._get_connection()
        
        query_parts = ["SELECT * FROM memory_entries WHERE 1=1"]
        params = []
        
        if layer is not None:
            query_parts.append("AND layer = ?")
            params.append(layer)
        
        if since is not None:
            query_parts.append("AND timestamp >= ?")
            params.append(since)
        
        query_parts.append("ORDER BY priority DESC, timestamp DESC")
        query_parts.append(f"LIMIT {limit}")
        
        rows = conn.execute(" ".join(query_parts), params).fetchall()
        
        entries = []
        for row in rows:
            entry = MemoryEntry.from_dict(dict(row))
            if not entry.is_expired():
                # Vérification des tags en mémoire
                if tags is None or all(tag in entry.tags for tag in tags):
                    entries.append(entry)
        
        return entries
    
    def search_similar(self, embedding: List[float], 
                       layer: Optional[int] = None,
                       top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """
        Recherche par similarité (distance cosinus simplifiée).
        
        Args:
            embedding: Vecteur de référence
            layer: Couche optionnelle
            top_k: Nombre de résultats
            
        Returns:
            Liste de (entrée, score_similarité)
        """
        # Récupère les entrées avec embeddings
        entries = self.query(layer=layer, limit=1000)
        entries_with_emb = [e for e in entries if e.embedding is not None]
        
        if not entries_with_emb:
            return []
        
        # Calcul de similarité cosinus
        import numpy as np
        ref = np.array(embedding)
        ref_norm = np.linalg.norm(ref)
        
        scores = []
        for entry in entries_with_emb:
            emb = np.array(entry.embedding)
            emb_norm = np.linalg.norm(emb)
            
            if emb_norm > 0 and ref_norm > 0:
                similarity = np.dot(ref, emb) / (ref_norm * emb_norm)
                scores.append((entry, float(similarity)))
        
        # Trie par similarité décroissante
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def link(self, source_id: str, target_id: str, 
             link_type: str = "related", strength: float = 1.0):
        """Crée un lien entre deux entrées."""
        conn = self._get_connection()
        
        conn.execute("""
            INSERT OR REPLACE INTO memory_links 
            (source_id, target_id, link_type, strength, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (source_id, target_id, link_type, strength, time.time()))
        
        conn.commit()
    
    def get_linked(self, entry_id: str, 
                   link_type: Optional[str] = None) -> List[Tuple[MemoryEntry, str, float]]:
        """
        Récupère les entrées liées.
        
        Returns:
            Liste de (entrée, type_lien, force)
        """
        conn = self._get_connection()
        
        query = """
            SELECT l.*, e.* FROM memory_links l
            JOIN memory_entries e ON l.target_id = e.id
            WHERE l.source_id = ?
        """
        params = [entry_id]
        
        if link_type:
            query += " AND l.link_type = ?"
            params.append(link_type)
        
        rows = conn.execute(query, params).fetchall()
        
        results = []
        for row in rows:
            entry = MemoryEntry.from_dict({
                'id': row['target_id'],
                'layer': row['layer'],
                'content': row['content'],
                'embedding': row['embedding'],
                'timestamp': row['timestamp'],
                'ttl': row['ttl'],
                'tags': row['tags'],
                'metadata': row['metadata'],
                'parent_id': row['parent_id'],
                'priority': row['priority']
            })
            results.append((entry, row['link_type'], row['strength']))
        
        return results
    
    def delete(self, entry_id: str) -> bool:
        """Supprime une entrée et ses liens."""
        conn = self._get_connection()
        
        conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", 
                    (entry_id, entry_id))
        
        cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
        conn.commit()
        
        return cursor.rowcount > 0
    
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées. Retourne le nombre supprimé."""
        conn = self._get_connection()
        now = time.time()
        
        # Trouve les expirés
        rows = conn.execute(
            "SELECT id, timestamp, ttl FROM memory_entries WHERE ttl IS NOT NULL"
        ).fetchall()
        
        expired_ids = [
            row['id'] for row in rows 
            if now - row['timestamp'] > row['ttl']
        ]
        
        for eid in expired_ids:
            self.delete(eid)
        
        return len(expired_ids)
    
    def promote(self, entry_id: str, new_layer: int) -> bool:
        """Change la couche d'une entrée (promotion/démotion)."""
        entry = self.retrieve(entry_id)
        if entry is None:
            return False
        
        entry.layer = new_layer
        entry.ttl = self.DEFAULT_TTL.get(new_layer)
        self.store(entry)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques sur la mémoire."""
        conn = self._get_connection()
        
        stats = {
            'total_entries': 0,
            'by_layer': {},
            'total_links': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        # Compte par couche
        for layer in range(5):
            count = conn.execute(
                "SELECT COUNT(*) FROM memory_entries WHERE layer = ?",
                (layer,)
            ).fetchone()[0]
            stats['by_layer'][self.LAYER_NAMES[layer]] = count
            stats['total_entries'] += count
        
        # Liens
        stats['total_links'] = conn.execute(
            "SELECT COUNT(*) FROM memory_links"
        ).fetchone()[0]
        
        # Timestamps
        oldest = conn.execute(
            "SELECT MIN(timestamp) FROM memory_entries"
        ).fetchone()[0]
        newest = conn.execute(
            "SELECT MAX(timestamp) FROM memory_entries"
        ).fetchone()[0]
        
        if oldest:
            stats['oldest_entry'] = datetime.fromtimestamp(oldest).isoformat()
        if newest:
            stats['newest_entry'] = datetime.fromtimestamp(newest).isoformat()
        
        return stats
    
    def clear_layer(self, layer: int) -> int:
        """Vide complètement une couche."""
        conn = self._get_connection()
        
        # Supprime d'abord les liens
        conn.execute("""
            DELETE FROM memory_links 
            WHERE source_id IN (SELECT id FROM memory_entries WHERE layer = ?)
            OR target_id IN (SELECT id FROM memory_entries WHERE layer = ?)
        """, (layer, layer))
        
        # Puis les entrées
        cursor = conn.execute(
            "DELETE FROM memory_entries WHERE layer = ?",
            (layer,)
        )
        conn.commit()
        
        return cursor.rowcount


def demo():
    """Démonstration du système de mémoire Aria."""
    print("=" * 50)
    print("Soe-Orret: Aria Memory Demo")
    print("=" * 50)
    
    # Utilise une DB temporaire
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    aria = AriaMemory(db_path)
    
    print(f"\nCouches de mémoire:")
    for i, name in aria.LAYER_NAMES.items():
        ttl = aria.DEFAULT_TTL[i]
        ttl_str = f"{ttl}s" if ttl else "∞"
        print(f"  Layer {i}: {name} (TTL: {ttl_str})")
    
    print("\nStockage d'entrées:")
    
    # Cache (layer 0)
    entry1 = MemoryEntry(layer=0, content="Résultat de calcul rapide: 42", priority=0.9)
    id1 = aria.store(entry1)
    print(f"  [Cache] ID: {id1[:8]}...")
    
    # Working memory (layer 1)
    entry2 = MemoryEntry(layer=1, content="Contexte conversationnel actif", 
                         tags=["context", "active"], priority=0.8)
    id2 = aria.store(entry2)
    print(f"  [Working] ID: {id2[:8]}...")
    
    # Long-term (layer 3)
    entry3 = MemoryEntry(layer=3, content="Connaissance persistante sur les systèmes distribués",
                         tags=["knowledge", "distributed"], priority=0.5)
    id3 = aria.store(entry3)
    print(f"  [Long-term] ID: {id3[:8]}...")
    
    # Création de liens
    aria.link(id2, id3, "references", 0.9)
    print(f"  Lien créé: {id2[:8]}... → {id3[:8]}...")
    
    print("\nRequête par couche:")
    for layer in range(5):
        entries = aria.query(layer=layer)
        if entries:
            print(f"  Layer {layer}: {len(entries)} entrée(s)")
            for e in entries:
                print(f"    - {e.content[:40]}...")
    
    print("\nRecherche par tag:")
    tagged = aria.query(tags=["knowledge"])
    for e in tagged:
        print(f"  Found: {e.content[:50]}...")
    
    print("\nEntrées liées:")
    linked = aria.get_linked(id2)
    for entry, link_type, strength in linked:
        print(f"  → {entry.content[:40]}... (type={link_type}, strength={strength:.2f})")
    
    print("\nStatistiques:")
    stats = aria.get_stats()
    print(f"  Total entrées: {stats['total_entries']}")
    print(f"  Total liens: {stats['total_links']}")
    for name, count in stats['by_layer'].items():
        print(f"  - {name}: {count}")
    
    # Cleanup
    import os
    os.unlink(db_path)
    
    print("\n" + "=" * 50)
    print("Demo terminée!")
    print("=" * 50)


if __name__ == "__main__":
    demo()
