"""
Mémoire Hiérarchique SOE-Orret — Architecture ARIA (5 couches).
L1 Working Memory  : contexte actif (~4K tokens, RAM pure)
L2 Episodic Memory : expériences passées avec scoring de cohérence
L3 Semantic Memory : connaissances factuelles (FAISS)
L4 Procedural      : séquences d'actions et patterns
L5 World Model     : modèle interne du monde / self-awareness
"""
import json, time, hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

@dataclass
class Episode:
    id:                str
    timestamp:         float
    user_input:        str
    ai_response:       str
    outcome:           str = "unknown"
    coherence_issues:  List[str] = field(default_factory=list)
    lessons_learned:   List[str] = field(default_factory=list)
    importance:        float = 0.5

class WorkingMemory:
    """L1 — Contexte actif de la conversation courante."""
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self._items = []
        self._tokens = 0

    def add(self, role: str, content: str):
        est = len(content.split()) * 1.3
        self._items.append({"role": role, "content": content})
        self._tokens += est
        while self._tokens > self.max_tokens and len(self._items) > 1:
            removed = self._items.pop(0)
            self._tokens -= len(removed["content"].split()) * 1.3

    def get_context(self):
        return [{"role": i["role"], "content": i["content"]} for i in self._items]

    def clear(self):
        self._items.clear()
        self._tokens = 0

class EpisodicMemory:
    """L2 — Expériences passées avec analyse de cohérence."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "episodes.jsonl"
        self.episodes: List[Episode] = self._load()

    def _load(self):
        eps = []
        if self.file.exists():
            with open(self.file) as f:
                for line in f:
                    if line.strip():
                        try:
                            eps.append(Episode(**json.loads(line)))
                        except Exception:
                            pass
        return eps

    def store(self, episode: Episode):
        self.episodes.append(episode)
        with open(self.file, "a") as f:
            f.write(json.dumps(asdict(episode), ensure_ascii=False) + "\n")

    def recall(self, query: str, n=5) -> List[Episode]:
        return sorted(self.episodes,
                      key=lambda e: e.timestamp * 0.3 + e.importance * 0.7,
                      reverse=True)[:n]

    def analyze_coherence(self) -> Dict:
        if not self.episodes:
            return {"coherence_avg": 1.0, "patterns": [], "lessons": []}
        all_issues = []
        for ep in self.episodes:
            all_issues.extend(ep.coherence_issues)
        freq = {}
        for issue in all_issues:
            freq[issue] = freq.get(issue, 0) + 1
        recurring = [i for i, c in freq.items() if c >= 2]
        all_lessons = []
        for ep in self.episodes:
            all_lessons.extend(ep.lessons_learned)
        return {
            "n_episodes": len(self.episodes),
            "recurring_patterns": recurring,
            "lessons": list(dict.fromkeys(all_lessons))[:10],
        }

class SemanticMemory:
    """L3 — Mémoire sémantique avec FAISS + sentence-transformers."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.facts = []
        self.embedder = None
        self.index    = None
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            idx_path = self.path / "semantic.faiss"
            if idx_path.exists():
                self.index = faiss.read_index(str(idx_path))
            else:
                self.index = faiss.IndexFlatIP(384)
            # Charger les faits
            ff = self.path / "facts.jsonl"
            if ff.exists():
                with open(ff) as f:
                    for line in f:
                        if line.strip():
                            self.facts.append(json.loads(line))
        except ImportError:
            pass  # Dégradation gracieuse

    def store_fact(self, fact: str, source: str = ""):
        entry = {"fact": fact, "source": source, "timestamp": time.time()}
        self.facts.append(entry)
        with open(self.path / "facts.jsonl", "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.embedder and self.index:
            import numpy as np, faiss
            vec = self.embedder.encode([fact[:512]], normalize_embeddings=True)
            self.index.add(vec.astype("float32"))
            faiss.write_index(self.index, str(self.path / "semantic.faiss"))

    def search(self, query: str, k=3) -> List[dict]:
        if not (self.embedder and self.index and self.index.ntotal > 0):
            return []
        import numpy as np
        q_vec = self.embedder.encode([query[:512]], normalize_embeddings=True)
        scores, ids = self.index.search(q_vec.astype("float32"), min(k, self.index.ntotal))
        return [self.facts[i] for i in ids[0] if 0 <= i < len(self.facts)]

class OrretMemorySystem:
    """Gestionnaire unifié de la mémoire hiérarchique ARIA."""
    def __init__(self, base_path="~/soe/memory"):
        self.base = Path(base_path).expanduser()
        self.working  = WorkingMemory(max_tokens=4096)
        self.episodic = EpisodicMemory(str(self.base / "episodic"))
        self.semantic = SemanticMemory(str(self.base / "semantic"))
        print(f"[ARIA] {len(self.episodic.episodes)} épisodes chargés")

    def retrieve_context(self, query: str) -> str:
        parts = []
        eps = self.episodic.recall(query, n=3)
        if eps:
            parts.append("## Expériences passées pertinentes")
            for ep in eps:
                lesson = ep.lessons_learned[0] if ep.lessons_learned else ""
                parts.append(f"- [{ep.outcome}] '{ep.user_input[:60]}' → {lesson}")
        analysis = self.episodic.analyze_coherence()
        if analysis["recurring_patterns"]:
            parts.append("## Points de vigilance")
            for p in analysis["recurring_patterns"][:2]:
                parts.append(f"- {p}")
        # L3 : faits sémantiques
        facts = self.semantic.search(query, k=2)
        if facts:
            parts.append("## Connaissances mémorisées")
            for f in facts:
                parts.append(f"- {f['fact'][:100]}")
        return "\n".join(parts)

    def record(self, user_input: str, response: str, outcome="success",
               coherence_issues=None, lessons=None):
        ep = Episode(
            id=hashlib.sha256(f"{user_input}{time.time()}".encode()).hexdigest()[:16],
            timestamp=time.time(),
            user_input=user_input,
            ai_response=response,
            outcome=outcome,
            coherence_issues=coherence_issues or [],
            lessons_learned=lessons or [],
            importance=min(0.5 + len(response) / 2000, 1.0),
        )
        self.episodic.store(ep)
        self.working.add("assistant", response)
