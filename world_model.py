"""
SOE - World Model & Context System
Le modèle du monde qui donne du "sens" aux prédictions.

Inspiré de:
- Cognitive Maps (Tolman)
- World Models (Ha & Schmidhuber)
- Predictive Coding (Friston)
"""
import numpy as np
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque
import networkx as nx


@dataclass
class Concept:
    """Un concept dans le modèle du monde."""
    id: str
    name: str
    embedding: np.ndarray
    connections: Dict[str, float]  # concept_id -> weight
    properties: Dict[str, Any]
    created_at: float
    last_access: float
    activation_level: float = 0.0


@dataclass
class Episode:
    """Une episode mémorisée."""
    id: str
    content: str
    embedding: np.ndarray
    importance: float
    timestamp: float
    concepts: List[str]  # IDs des concepts liés
    outcome: Optional[str] = None


class WorldModel:
    """
    World Model - Représentation interne du monde.
    
    Fonctionnalités:
    1. Mémoire sémantique (concepts + relations)
    2. Mémoire épisodique (expériences vécues)
    3. Raisonnement causal
    4. Prédiction grounded
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Graphe des concepts (comme le néocortex)
        self.concepts: Dict[str, Concept] = {}
        
        # Mémoire épisodique
        self.episodes: List[Episode] = []
        self.episode_buffer = deque(maxlen=1000)
        
        # État courant
        self.current_concepts: Set[str] = set()
        self.context_embedding = np.zeros(embedding_dim)
        
        # Métriques
        self.stats = {
            "concepts_created": 0,
            "episodes_stored": 0,
            "reasoning_steps": 0,
            "predictions_made": 0
        }
        
        # Métadonnées globales
        self.metadata = {
            "user_name": "Lévy",
            "preferences": {},
            "conversation_count": 0,
            "facts_learned": []
        }
    
    def add_concept(self, name: str, embedding: np.ndarray, 
                   properties: Dict = None) -> str:
        """Ajoute un nouveau concept."""
        # Hash unique pour le concept
        concept_id = hashlib.md5(name.encode()).hexdigest()[:12]
        
        if concept_id in self.concepts:
            self.concepts[concept_id].last_access = time.time()
            self.concepts[concept_id].activation_level += 0.1
            return concept_id
        
        concept = Concept(
            id=concept_id,
            name=name,
            embedding=embedding[:self.embedding_dim] if len(embedding) >= self.embedding_dim else 
                       np.pad(embedding, (0, self.embedding_dim - len(embedding))),
            connections={},
            properties=properties or {},
            created_at=time.time(),
            last_access=time.time(),
            activation_level=0.5
        )
        
        self.concepts[concept_id] = concept
        self.stats["concepts_created"] += 1
        
        # Connecter au concept courant le plus similaire
        self._connect_similar_concepts(concept)
        
        return concept_id
    
    def _connect_similar_concepts(self, new_concept: Concept):
        """Connecte le nouveau concept aux concepts similaires."""
        if not self.concepts:
            return
        
        for cid, concept in self.concepts.items():
            if cid == new_concept.id:
                continue
            
            # Similarité cosinus
            sim = self._cosine_similarity(new_concept.embedding, concept.embedding)
            
            if sim > 0.5:  # Seuil de connection
                # Connection bidirectionnelle
                self.concepts[cid].connections[new_concept.id] = sim
                new_concept.connections[cid] = sim
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcule la similarité cosinus."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def add_episode(self, content: str, embedding: np.ndarray,
                   importance: float = 0.5, concepts: List[str] = None,
                   outcome: str = None):
        """Ajoute une episode à la mémoire."""
        episode_id = hashlib.md5(content[:100].encode()).hexdigest()[:12]
        
        episode = Episode(
            id=episode_id,
            content=content[:500],  # Limiter la taille
            embedding=embedding[:self.embedding_dim] if len(embedding) >= self.embedding_dim else
                       np.pad(embedding, (0, self.embedding_dim - len(embedding))),
            importance=importance,
            timestamp=time.time(),
            concepts=concepts or [],
            outcome=outcome
        )
        
        self.episodes.append(episode)
        self.episode_buffer.append(episode)
        self.stats["episodes_stored"] += 1
        
        # Mise à jour du contexte global
        self.context_embedding = self.context_embedding * 0.9 + embedding * 0.1
    
    def retrieve(self, query_embedding: np.ndarray, 
                 top_k: int = 5, use_episodic: bool = True,
                 use_semantic: bool = True) -> Dict[str, List]:
        """
        Récupère les souvenirs pertinents.
        """
        results = {"concepts": [], "episodes": []}
        
        if use_semantic and self.concepts:
            # Trouver les concepts similaires
            concept_scores = []
            for cid, concept in self.concepts.items():
                sim = self._cosine_similarity(query_embedding, concept.embedding)
                concept_scores.append((cid, concept, sim))
            
            concept_scores.sort(key=lambda x: x[2], reverse=True)
            
            for cid, concept, sim in concept_scores[:top_k]:
                concept.last_access = time.time()
                results["concepts"].append({
                    "id": cid,
                    "name": concept.name,
                    "similarity": float(sim),
                    "properties": concept.properties,
                    "connections": list(concept.connections.keys())[:3]
                })
        
        if use_episodic and self.episodes:
            # Trouver les épisodes similaires
            episode_scores = []
            for ep in self.episodes:
                sim = self._cosine_similarity(query_embedding, ep.embedding)
                # Bonus pour importance et récence
                recency = 1.0 / (1.0 + (time.time() - ep.timestamp) / 86400)
                score = sim * 0.5 + ep.importance * 0.3 + recency * 0.2
                episode_scores.append((ep, score))
            
            episode_scores.sort(key=lambda x: x[1], reverse=True)
            
            for ep, score in episode_scores[:top_k]:
                results["episodes"].append({
                    "id": ep.id,
                    "content": ep.content[:200] + "..." if len(ep.content) > 200 else ep.content,
                    "score": float(score),
                    "timestamp": ep.timestamp,
                    "importance": ep.importance,
                    "concepts": ep.concepts
                })
        
        return results
    
    def reason(self, query: str, context: Dict) -> Dict:
        """
        Raisonnement causal sur le modèle du monde.
        """
        self.stats["reasoning_steps"] += 1
        
        reasoning = {
            "query": query,
            "steps": [],
            "conclusion": None,
            "confidence": 0.0
        }
        
        # Étape 1: Identifier les concepts actifs
        active_concepts = self._get_active_concepts(query)
        
        # Étape 2: Explorer les connections
        for concept_id in active_concepts:
            concept = self.concepts.get(concept_id)
            if not concept:
                continue
            
            # Suivre les connections
            for connected_id in list(concept.connections.keys())[:3]:
                connected = self.concepts.get(connected_id)
                if connected:
                    reasoning["steps"].append({
                        "from": concept.name,
                        "to": connected.name,
                        "strength": concept.connections[connected_id]
                    })
        
        # Étape 3: Générer une conclusion
        if reasoning["steps"]:
            reasoning["conclusion"] = f"Relation trouvée: {reasoning['steps'][0]['from']} -> {reasoning['steps'][0]['to']}"
            reasoning["confidence"] = 0.6
        else:
            reasoning["conclusion"] = "Pas de raisonnement possible avec le modèle actuel"
            reasoning["confidence"] = 0.1
        
        return reasoning
    
    def _get_active_concepts(self, query: str) -> List[str]:
        """Identifie les concepts actifs pour une requête."""
        # Simple keyword matching pour l'instant
        # Plus tard: utiliser les embeddings
        active = []
        
        query_lower = query.lower()
        for cid, concept in self.concepts.items():
            if concept.name.lower() in query_lower:
                active.append(cid)
        
        return active[:5]
    
    def predict(self, current_context: Dict) -> Dict:
        """
        Prédit le prochain état basé sur le modèle du monde.
        """
        self.stats["predictions_made"] += 1
        
        prediction = {
            "next_topic": None,
            "likely_intent": None,
            "confidence": 0.0,
            "reasoning": ""
        }
        
        # Analyser les patterns récents
        recent_episodes = list(self.episode_buffer)[-10:]
        
        if len(recent_episodes) >= 2:
            # Patterns de conversation
            intents = [ep.outcome for ep in recent_episodes if ep.outcome]
            
            if intents:
                from collections import Counter
                most_common = Counter(intents).most_common(1)
                prediction["likely_intent"] = most_common[0][0]
                prediction["confidence"] = 0.5
        
        return prediction
    
    def update_metadata(self, key: str, value: Any):
        """Met à jour les métadonnées."""
        if key in self.metadata:
            if isinstance(self.metadata[key], list):
                self.metadata[key].append(value)
            else:
                self.metadata[key] = value
        else:
            self.metadata[key] = value
    
    def get_context_string(self, query: str, embedding: np.ndarray) -> str:
        """
        Construit une chaîne de contexte pour le prompt.
        """
        parts = []
        
        # Récupérer les souvenirs pertinents
        retrieved = self.retrieve(embedding, top_k=3)
        
        # Concepts
        if retrieved["concepts"]:
            parts.append("## Connaissances")
            for c in retrieved["concepts"]:
                parts.append(f"- {c['name']}: {c.get('properties', {}).get('description', '')}")
        
        # Épisodes
        if retrieved["episodes"]:
            parts.append("\n## Passé pertinent")
            for ep in retrieved["episodes"]:
                parts.append(f"- {ep['content'][:100]}")
        
        # Métadonnées utilisateur
        if self.metadata.get("user_name"):
            parts.append(f"\n## Utilisateur: {self.metadata['user_name']}")
        
        return "\n".join(parts)
    
    def get_status(self) -> Dict:
        """Retourne le statut du modèle du monde."""
        return {
            "concepts": len(self.concepts),
            "episodes": len(self.episodes),
            "metadata": self.metadata,
            "stats": self.stats
        }


class WorldModelIntegration:
    """Intégration World Model + Synapse Engine."""
    
    def __init__(self):
        self.world_model = WorldModel(embedding_dim=128)
    
    def process(self, user_input: str, model_output: str) -> Dict:
        """Traite une interaction complète."""
        # Ici on utiliserait les embeddings réels
        # Pour l'instant, simulation simple
        fake_embedding = np.random.randn(128)
        
        # Ajouter les concepts de la conversation
        for word in user_input.split()[:3]:
            if len(word) > 3:
                self.world_model.add_concept(
                    word, 
                    fake_embedding,
                    {"source": "user_input"}
                )
        
        # Ajouter l'episode
        self.world_model.add_episode(
            f"User: {user_input}\nAssistant: {model_output}",
            fake_embedding,
            importance=0.6,
            outcome="conversation"
        )
        
        # Mettre à jour le compteur
        self.world_model.metadata["conversation_count"] += 1
        
        return {
            "world_model": self.world_model.get_status(),
            "context": self.world_model.get_context_string(user_input, fake_embedding)
        }
    
    def retrieve_for_prompt(self, prompt: str) -> str:
        """Récupère le contexte pertinent pour un prompt."""
        fake_embedding = np.random.randn(128)
        return self.world_model.get_context_string(prompt, fake_embedding)


def test_world_model():
    """Test du modèle du monde."""
    wm = WorldModel(embedding_dim=64)
    
    # Ajouter des concepts
    wm.add_concept("Python", np.array([0.1, 0.2, 0.3] * 21)[:64], {"type": "language"})
    wm.add_concept("Programmation", np.array([0.15, 0.25, 0.35] * 21)[:64], {"type": "domain"})
    wm.add_concept("Linux", np.array([0.2, 0.1, 0.4] * 21)[:64], {"type": "os"})
    
    # Ajouter une episode
    wm.add_episode(
        "L'utilisateur a demandé comment installer Python",
        np.array([0.1, 0.2, 0.3] * 21)[:64],
        importance=0.7,
        outcome="installation_help"
    )
    
    # Retrieval
    query_emb = np.array([0.1, 0.2, 0.3] * 21)[:64]
    results = wm.retrieve(query_emb, top_k=2)
    
    print("🧠 World Model Test")
    print(f"  Concepts: {len(wm.concepts)}")
    print(f"  Episodes: {len(wm.episodes)}")
    print(f"  Retrieved concepts: {len(results['concepts'])}")
    print(f"  Retrieved episodes: {len(results['episodes'])}")
    
    return wm


if __name__ == "__main__":
    test_world_model()
    print("\n✓ World Model prêt pour SOE")