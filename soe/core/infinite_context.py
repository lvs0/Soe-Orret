"""
Système de Contexte Illimité pour SOE-Orret
Optimisé pour Lenovo X250 - Contexte dynamique avec compression et RAG
"""
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import random
import re

logger = logging.getLogger(__name__)

# Imports optionnels — chargement différé
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from sklearn.cluster import MiniBatchKMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


@dataclass
class ContextSegment:
    """Un segment de contexte avec métadonnées."""
    content: str
    importance: float  # 0.0-1.0
    timestamp: float
    tokens_estimate: int
    embedding: Optional[List[float]] = None
    compressed: bool = False


class InfiniteContextManager:
    """
    Gestionnaire de contexte illimité pour SOE.
    
    Stratégies:
    1. Compression sémantique des anciens messages
    2. RAG hiérarchique depuis mémoire ARIA
    3. Token recycling (réutilisation embeddings)
    4. Cache LRU pour segments fréquents
    5. Fenêtre glissante adaptative
    """
    
    def __init__(
        self,
        max_window_tokens: int = 4096,
        compression_threshold: float = 0.3,
        cache_size: int = 1000,
        base_path: str = "~/soe/memory/context",
        skip_embedding_rebuild: bool = False,
    ):
        self.max_window = max_window_tokens
        self.compression_threshold = compression_threshold
        self.path = Path(base_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Cache LRU pour segments fréquents
        self.cache: OrderedDict[str, ContextSegment] = OrderedDict()
        self.cache_size = cache_size
        
        # Segments actifs
        self.active_segments: List[ContextSegment] = []
        
        # Statistiques
        self.stats = {
            "total_segments": 0,
            "compressed_segments": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        # === NOUVEAU : Attributs pour embeddings + FAISS ===
        self._embedding_model = None          # SentenceTransformer (lazy)
        self._faiss_index = None              # faiss.IndexHNSWFlat (lazy)
        self._embedding_cache: Dict[str, np.ndarray] = {}  # Cache embeddings
        self._embedding_dim = 384             # all-MiniLM-L6-v2
        # Note: embeddings seront calculés depuis _get_embedding_model()
        self.skip_embedding_rebuild = skip_embedding_rebuild  # Problème-007/A1 throttle

        # Charger l'état (après initialisation des attributs FAISS)
        self._load_state()

    def _estimate_tokens(self, text: str) -> int:
        """Estimation rapide du nombre de tokens."""
        return int(len(text.split()) * 1.3)
    
    def _old_compress(self, segment: ContextSegment) -> ContextSegment:
        """
        Compression simple par troncation (méthode legacy fallback).
        """
        original_length = len(segment.content)
        target_length = int(original_length * self.compression_threshold)
        
        # Garder le début et la fin, résumer le milieu
        if original_length > target_length:
            keep_start = target_length // 3
            keep_end = target_length // 3
            middle = " [...] "
            compressed = (
                segment.content[:keep_start] + 
                middle + 
                segment.content[-keep_end:]
            )
            segment.content = compressed
            segment.compressed = True
            segment.tokens_estimate = self._estimate_tokens(compressed)
            self.stats["compressed_segments"] += 1
        
        return segment

    def _compress_segment(self, segment: ContextSegment) -> ContextSegment:
        """
        Compression sémantique (essaye summarization, fallback troncature).
        """
        if segment.compressed:
            return segment
        
        # Essayer compression sémantique (K-means)
        if _SKLEARN_AVAILABLE and _NUMPY_AVAILABLE:
            try:
                compressed_content = self._summarize_extractive(segment, self.compression_threshold)
                if compressed_content and len(compressed_content) > 0:
                    segment.content = compressed_content
                    segment.compressed = True
                    segment.tokens_estimate = self._estimate_tokens(compressed_content)
                    self.stats["compressed_segments"] += 1
                    return segment
            except Exception:
                pass  # fallback silencieux
        
        # Fallback: troncature simple
        return self._old_compress(segment)

    def add_segment(self, content: str, importance: float = 0.5) -> str:
        """
        Ajoute un segment au contexte.
        Retourne l'ID du segment.
        """
        # Clé stable = hash du contenu seul (pas du timestamp)
        # → même contenu ajouté à des moments différents = même segment (déduplication)
        segment_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Calculer l'embedding (si disponible)
        embedding = None
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            emb_array = self._embed_text(content)
            # Convertir numpy array en liste pour type hint List[float]
            embedding = emb_array.tolist() if emb_array is not None else None

        segment = ContextSegment(
            content=content,
            importance=importance,
            timestamp=time.time(),
            tokens_estimate=self._estimate_tokens(content),
            embedding=embedding  # NOUVEAU (liste de floats)
        )
        
        # Vérifier cache
        if segment_id in self.cache:
            self.stats["cache_hits"] += 1
            # Mettre à jour LRU
            self.cache.move_to_end(segment_id)
            return segment_id
        
        self.stats["cache_misses"] += 1
        self.cache[segment_id] = segment
        
        # Limiter taille cache
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        self.active_segments.append(segment)
        self.stats["total_segments"] += 1

        # Ajouter à l'index FAISS si embedding disponible
        if embedding is not None:
            faiss_idx = self._ensure_faiss_index()
            if faiss_idx is not None:
                try:
                    import numpy as np
                    faiss_idx.add(np.array([embedding], dtype=np.float32))
                except Exception:
                    pass  # Silencieux, sera rattrapé à la sauvegarde

        # Compression si nécessaire
        self._optimize_window()
        
        self._save_state()
        return segment_id
    
    def _optimize_window(self):
        """
        Optimise la fenêtre de contexte.
        Compresse les segments moins importants si dépasse max_window.
        """
        total_tokens = sum(s.tokens_estimate for s in self.active_segments)
        
        if total_tokens <= self.max_window:
            return
        
        # Trier par importance (croissante)
        sorted_segments = sorted(
            self.active_segments, 
            key=lambda s: s.importance
        )
        
        # Compresser les moins importants
        tokens_to_remove = total_tokens - self.max_window
        removed = 0
        
        for segment in sorted_segments:
            if removed >= tokens_to_remove:
                break
            
            if not segment.compressed and segment.importance < 0.7:
                old_tokens = segment.tokens_estimate
                self._compress_segment(segment)
                removed += old_tokens - segment.tokens_estimate
    
    def get_context_window(self, query: str = "") -> str:
        """
        Construit la fenêtre de contexte optimale.
        Inclut les segments actifs + contexte pertinent depuis mémoire.
        """
        # Construire depuis segments actifs
        context_parts = []
        
        # Trier par timestamp (plus récent d'abord)
        sorted_segments = sorted(
            self.active_segments,
            key=lambda s: s.timestamp,
            reverse=True
        )
        
        for segment in sorted_segments:
            context_parts.append(segment.content)
        
        return "\n\n".join(context_parts)
    
    def retrieve_relevant(self, query: str, k: int = 5) -> List[ContextSegment]:
        """
        Récupère segments pertinents (sémantique prioritaire, fallback lexical).
        """
        if not self.active_segments:
            return []
        
        # Essayer recherche sémantique
        semantic_results = self._retrieve_by_semantic(query, k)
        if semantic_results:
            return semantic_results
        
        # Fallback lexical (ancien comportement)
        return self._retrieve_by_keywords(query, k)

    def _retrieve_by_keywords(self, query: str, k: int) -> List[ContextSegment]:
        """Recherche par mots-clés (scoring: keywords + importance + fraîcheur)."""
        query_lower = query.lower()
        scored = []
        
        for segment in self.active_segments:
            score = 0.0
            content_lower = segment.content.lower()
            
            # Score par mots-clés
            for word in query_lower.split():
                if word in content_lower:
                    score += 0.1
            
            # Score par importance
            score += segment.importance * 0.3
            
            # Score par fraîcheur
            age = time.time() - segment.timestamp
            freshness = max(0, 1 - age / 86400)  # décroît sur 24h
            score += freshness * 0.2
            
            scored.append((score, segment))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def _retrieve_by_semantic(self, query: str, k: int) -> List[ContextSegment]:
        """Recherche sémantique via FAISS."""
        faiss_idx = self._ensure_faiss_index()
        if faiss_idx is None or faiss_idx.ntotal == 0:
            logger.debug("FAISS index unavailable or empty — returning []")
            return []
        
        q_emb = self._embed_text(query)
        if q_emb is None:
            logger.debug("Embedding failed — returning []")
            return []
        
        import numpy as np
        query_vector = np.array([q_emb], dtype=np.float32)
        
        try:
            distances, indices = faiss_idx.search(query_vector, min(k, faiss_idx.ntotal))
        except Exception as e:
            logger.debug(f"FAISS search failed: {e} — returning []")
            return []
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.active_segments):
                results.append(self.active_segments[idx])
        return results

    def _get_embedding_model(self):
        """Lazy-load du modèle sentence-transformers."""
        if self._embedding_model is not None:
            return self._embedding_model
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.eval()  # mode inference
            self._embedding_model = model
            return model
        except Exception:
            return None

    def _ensure_faiss_index(self):
        """Crée ou charge l'index FAISS."""
        if self._faiss_index is not None:
            return self._faiss_index
        
        if not _FAISS_AVAILABLE:
            return None
        
        try:
            index_path = self.path / "faiss.index"
            if index_path.exists():
                self._faiss_index = faiss.read_index(str(index_path))
            else:
                import numpy as np
                self._faiss_index = faiss.IndexHNSWFlat(self._embedding_dim, 16)
                self._faiss_index.hnsw.efConstruction = 200
                self._faiss_index.hnsw.efSearch = 50
            return self._faiss_index
        except Exception:
            self._faiss_index = None
            return None

    def _embed_text(self, text: str):
        """Encode un texte en embedding (avec cache en mémoire)."""
        if not text or not text.strip():
            return None
        
        # Cache hit
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        model = self._get_embedding_model()
        if model is None:
            return None
        
        try:
            emb = model.encode(
                [text],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )[0]
            self._embedding_cache[text] = emb
            return emb
        except Exception:
            return None

    def _summarize_extractive(self, segment: ContextSegment, target_ratio: float = 0.3) -> str:
        """
        Résumé extractif par clustering K-means des embeddings de phrases.
        Retourne les phrases représentatives (centroïdes), dans l'ordre original.
        """
        if not _SKLEARN_AVAILABLE or not _NUMPY_AVAILABLE or not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return self._old_compress(segment).content
        
        # Tokenizer par phrases (ponctuation)
        sentences = re.split(r'(?<=[.!?])\s+', segment.content.strip())
        
        if len(sentences) <= 3:
            return segment.content  # Pas assez de phrases
        
        # Échantillonnage pour performance : max 20 phrases
        if len(sentences) > 20:
            start = sentences[:5]
            end = sentences[-5:]
            middle = sentences[5:-5]
            sampled_middle = random.sample(middle, min(10, len(middle)))
            sample_sentences = start + sampled_middle + end
        else:
            sample_sentences = sentences
        
        # Encoder chaque phrase
        embeddings = []
        valid_sentences = []
        for sent in sample_sentences:
            if len(sent) < 5:  # Ignorer phrases trop courtes
                continue
            emb = self._embed_text(sent)
            if emb is not None:
                embeddings.append(emb)
                valid_sentences.append(sent)
        
        if len(embeddings) < 3:
            return self._old_compress(segment).content
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        n_sentences = len(valid_sentences)
        n_clusters = max(2, int(n_sentences * target_ratio))
        
        try:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=min(100, len(embeddings)))
            kmeans.fit(embeddings_array)
            
            selected_indices = set()
            for centroid in kmeans.cluster_centers_:
                distances = np.linalg.norm(embeddings_array - centroid, axis=1)
                closest = np.argmin(distances)
                selected_indices.add(closest)
            
            selected_indices = sorted(selected_indices)
            selected_sentences = [valid_sentences[i] for i in selected_indices]
            summary = '. '.join(selected_sentences)
            if not summary.endswith('.'):
                summary += '.'
            return summary
        except Exception:
            return self._old_compress(segment).content

    def build_optimal_window(
        self, 
        query: str, 
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Construit la fenêtre de contexte optimale pour une query.
        Combine segments actifs + segments pertinents.
        """
        max_tok = max_tokens or self.max_window
        
        # Récupérer segments pertinents
        relevant = self.retrieve_relevant(query, k=10)
        
        # Combiner avec segments récents
        recent = sorted(
            self.active_segments,
            key=lambda s: s.timestamp,
            reverse=True
        )[:5]
        
        # Dédupliquer
        all_segments = []
        seen = set()
        for s in relevant + recent:
            if s.content not in seen:
                all_segments.append(s)
                seen.add(s.content)
        
        # Construire contexte dans limite tokens
        context_parts = []
        current_tokens = 0
        
        for segment in all_segments:
            if current_tokens + segment.tokens_estimate > max_tok:
                break
            context_parts.append(segment.content)
            current_tokens += segment.tokens_estimate
        
        return self._seal_window(
            context_parts=context_parts,
            all_segments=all_segments,
            included_indices=seen,
            max_tok=max_tok,
        )
    
    def _seal_window(
        self,
        context_parts: List[str],
        all_segments: List[ContextSegment],
        included_indices: set,
        max_tok: int,
    ) -> str:
        """
        Scelle la fenêtre de contexte : assemble, vérifie le budget, log le verdict.
        Garantie (contrat) : actual_tokens <= max_tok OU avertissement émis.
        """
        result = "\n\n".join(context_parts)
        actual_tokens = self._estimate_tokens(result)
        included_count = len(context_parts)
        excluded_count = len(all_segments) - included_count

        if actual_tokens <= max_tok:
            verdict = "OK"
        elif actual_tokens <= int(max_tok * 1.1):
            verdict = "OVER_BUDGET_WARN"
            logger.warning(
                "[SOE/004] window over budget: %d vs %d "
                "(+%d, %d segs, %d excluded)"
                % (actual_tokens, max_tok, actual_tokens - max_tok,
                   included_count, excluded_count)
            )
        else:
            verdict = "OVER_BUDGET_FAIL"
            logger.error(
                "[SOE/004] window SEVERELY over budget: %d vs %d "
                "(+%d) -- check tokens_estimate accuracy"
                % (actual_tokens, max_tok, actual_tokens - max_tok)
            )

        logger.info(
            "[SOE/004] window sealed: %d segs, "
            "%d actual tokens, limit %d, verdict=%s"
            % (included_count, actual_tokens, max_tok, verdict)
        )
        return result

    def _save_state(self):
        """Sauvegarde l'état du gestionnaire + index FAISS."""
        state = {
            "stats": self.stats,
            "active_segments": [
                {
                    "content": s.content,
                    "importance": s.importance,
                    "timestamp": s.timestamp,
                    "tokens_estimate": s.tokens_estimate,
                    "compressed": s.compressed
                    # Note: on ne sauvegarde pas 'embedding' (trop gros, reconstruit au chargement)
                }
                for s in self.active_segments[-100:]  # Derniers 100
            ]
        }

        with open(self.path / "context_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Sauvegarde index FAISS si disponible
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            try:
                import faiss
                index_path = self.path / "faiss.index"
                faiss.write_index(self._faiss_index, str(index_path))
            except Exception:
                pass  # Silencieux


    def _load_state(self):
        """Charge l'état et reconstruit embeddings/FAISS."""
        state_file = self.path / "context_state.json"
        if not state_file.exists():
            return

        with open(state_file, "r") as f:
            state = json.load(f)

        self.stats = state.get("stats", self.stats)

        # Reconstruire segments
        for s_data in state.get("active_segments", []):
            segment = ContextSegment(
                content=s_data["content"],
                importance=s_data["importance"],
                timestamp=s_data["timestamp"],
                tokens_estimate=s_data["tokens_estimate"],
                compressed=s_data.get("compressed", False),
                embedding=None  # sera calculé ci-après si dépendances dispo
            )
            self.active_segments.append(segment)

        # Reconstruire embeddings + index FAISS si dépendances disponibles
        if _FAISS_AVAILABLE and _SENTENCE_TRANSFORMERS_AVAILABLE and not self.skip_embedding_rebuild:
            self._ensure_faiss_index()
            if self._faiss_index is not None and self.active_segments:
                import numpy as np
                total = len(self.active_segments)
                logger.info(f"[SOE/A1] _load_state: rebuilding embeddings for {total} segments")
                embeddings = []
                for i, seg in enumerate(self.active_segments):
                    emb_array = self._embed_text(seg.content)
                    if emb_array is not None:
                        seg.embedding = emb_array.tolist()
                        embeddings.append(emb_array)
                    if (i + 1) % 20 == 0 or i == total - 1:
                        logger.info(
                            f"[SOE/A1] _load_state: processed {i + 1}/{total} "
                            f"segments ({len(embeddings)} embeddings built so far)"
                        )
                if embeddings:
                    self._faiss_index.add(np.array(embeddings, dtype=np.float32))
                logger.info(f"[SOE/A1] _load_state: FAISS index rebuilt, {len(embeddings)} vectors added")
        elif self.skip_embedding_rebuild:
            logger.warning(
                "[SOE/A1] _load_state: skip_embedding_rebuild=True — "
                "FAISS index NOT rebuilt. Semantic search disabled until next save/reload without skip."
            )

        # Rebuild LRU cache from loaded segments (Bug B fix)
        for seg in self.active_segments:
            seg_id = hashlib.sha256(seg.content.encode()).hexdigest()[:16]
            self.cache[seg_id] = seg
            self.cache.move_to_end(seg_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get_stats(self) -> Dict:
        """Retourne les statistiques."""
        return {
            **self.stats,
            "active_segments": len(self.active_segments),
            "total_tokens": sum(s.tokens_estimate for s in self.active_segments),
            "cache_size": len(self.cache),
            "compression_ratio": (
                self.stats["compressed_segments"] / max(self.stats["total_segments"], 1)
                if self.stats["total_segments"] > 0 else 0
            ),
            "faiss_available": self._faiss_index is not None,
            "faiss_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
            "embedding_cache_size": len(self._embedding_cache),
        }
    
    def clear(self):
        """Efface tous les segments."""
        self.active_segments.clear()
        self.cache.clear()
        self._embedding_cache.clear()
        self.stats = {
            "total_segments": 0,
            "compressed_segments": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._save_state()
        # Note: FAISS index non réinitialisé ici — on le laisse en mémoire
        # pour éviter re-build coûteux. Option: self._faiss_index.reset() si besoin


# Test
if __name__ == "__main__":
    manager = InfiniteContextManager(max_window_tokens=2048)
    
    # Ajouter des segments
    for i in range(20):
        content = f"Ceci est le segment numéro {i}. " * (10 + i % 5)
        manager.add_segment(content, importance=0.5 + (i % 3) * 0.2)
    
    print("Stats:", manager.get_stats())
    
    # Récupérer contexte
    query = "segment important"
    window = manager.build_optimal_window(query)
    print(f"Contexte pour '{query}': {len(window)} caractères")
