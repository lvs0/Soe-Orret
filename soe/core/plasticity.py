"""
SOE-Orret Core Module - Plasticité Neurale Artificielle
Basé sur les spécifications de Lévy Verpoort Scherpereel
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SynapticEvent:
    """Un événement qui mérite d'être mémorisé (renforcement synaptique)."""
    prompt: str
    response: str
    reinforcement: float  # -1.0 (mauvais) à +1.0 (excellent)
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    emotional_weight: float = 0.5  # Importance émotionnelle


class NeuralPlasticityEngine:
    """
    Moteur de plasticité neurale pour Orret.
    
    Principe : après chaque session significative, des micro-updates LoRA
    sont appliquées pour renforcer les patterns appris.
    C'est l'équivalent de la consolidation de la mémoire pendant le sommeil.
    """
    
    def __init__(
        self,
        model=None,
        lora_rank: int = 4,  # Petit rang pour micro-updates
        learning_rate: float = 1e-5,  # Très bas pour ne pas perdre les bases
        consolidation_threshold: int = 50,  # Nb événements avant consolidation
        storage_path: str = "~/soe/memory/plasticity",
    ):
        self.model = model
        self.lora_rank = lora_rank
        self.lr = learning_rate
        self.threshold = consolidation_threshold
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self.pending_events: List[SynapticEvent] = []
        self.consolidation_count = 0
        
        # Charger les événements non consolidés
        self._load_pending()
        print(f"[Plasticité] {len(self.pending_events)} événements en attente")
    
    def record(self, event: SynapticEvent):
        """Enregistre un événement synaptique."""
        self.pending_events.append(event)
        
        # Sauvegarde incrémentale
        with open(self.path / "pending.jsonl", "a") as f:
            f.write(json.dumps({
                "prompt": event.prompt[:500],
                "response": event.response[:500],
                "reinforcement": event.reinforcement,
                "timestamp": event.timestamp,
                "tags": event.tags,
                "emotional_weight": event.emotional_weight,
            }) + "\n")
        
        # Consolider si seuil atteint
        if len(self.pending_events) >= self.threshold:
            self.consolidate()
    
    def consolidate(self):
        """
        Consolide les événements en micro-updates LoRA.
        C'est l'équivalent de la consolidation mémoire nocturne.
        """
        if not self.pending_events:
            return
        
        print(f"[Plasticité] Consolidation de {len(self.pending_events)} événements...")
        
        # Filtrer par poids émotionnel et reinforcement
        positive = [e for e in self.pending_events if e.reinforcement > 0.2]
        negative = [e for e in self.pending_events if e.reinforcement < -0.2]
        
        if not positive and not negative:
            print("[Plasticité] Aucun événement significatif. Pas d'update.")
            return
        
        # Sauvegarder pour le prochain cycle
        consolidation_data = {
            "timestamp": time.time(),
            "positive_examples": [
                {"prompt": e.prompt, "response": e.response,
                 "weight": e.reinforcement * e.emotional_weight}
                for e in positive[:20]  # Top 20
            ],
            "negative_examples": [
                {"prompt": e.prompt, "response": e.response,
                 "weight": abs(e.reinforcement)}
                for e in negative[:10]  # Top 10
            ],
            "consolidation_id": self.consolidation_count,
        }
        
        output_file = self.path / f"consolidation_{self.consolidation_count:04d}.json"
        with open(output_file, "w") as f:
            json.dump(consolidation_data, f, indent=2)
        
        print(f"[Plasticité] Consolidation #{self.consolidation_count} sauvegardée: {output_file}")
        
        # Réinitialiser
        self.pending_events = []
        self.consolidation_count += 1
        
        # Nettoyer le fichier pending
        (self.path / "pending.jsonl").unlink(missing_ok=True)
    
    def _load_pending(self):
        """Charge les événements non consolidés."""
        pending_file = self.path / "pending.jsonl"
        if pending_file.exists():
            with open(pending_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.pending_events.append(SynapticEvent(**data))
    
    def reset(self):
        """Reset neural - efface tous les adapters LoRA personnalisés."""
        print("[Plasticité] Reset neural en cours...")
        self.pending_events = []
        self.consolidation_count = 0
        
        # Supprimer les fichiers de consolidation
        for f in self.path.glob("consolidation_*.json"):
            f.unlink()
        (self.path / "pending.jsonl").unlink(missing_ok=True)
        
        print("[Plasticité] Reset terminé. Mémoire épisodique effacée.")


# Test simple
if __name__ == "__main__":
    engine = NeuralPlasticityEngine()
    
    # Simuler quelques événements
    for i in range(5):
        event = SynapticEvent(
            prompt=f"Question {i}",
            response=f"Réponse {i}",
            reinforcement=0.8,
            tags=["test"]
        )
        engine.record(event)
    
    print(f"Événements en attente: {len(engine.pending_events)}")
