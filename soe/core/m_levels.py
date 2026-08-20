"""
Système de Niveaux M pour Orret - Architecture Évolutive
M1, M2, M3, M4, M5... - Niveaux de capacités, pas d'intelligence

Chaque niveau ajoute des couches de sophistication et d'agenticité.
Le système est adaptatif: le modèle peut évoluer entre niveaux.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class MLevel(Enum):
    """Niveaux de capacités Orret - M1 à M5+"""
    M1 = "m1"  # Base - Conversation simple
    M2 = "m2"  # Raisonnement - Multi-phase
    M3 = "m3"  # Mémoire - ARIA complète
    M4 = "m4"  # Agenticité - Tool use
    M5 = "m5"  # Autonomie - Auto-planning
    M6 = "m6"  # Collaboration - Multi-agents
    M7 = "m7"  # Créativité - Génération originale
    M8 = "m8"  # Sagesse - Métacognition
    M9 = "m9"  # Transcendance - Conscience simulée


@dataclass
class MLevelCapabilities:
    """Capacités d'un niveau M"""
    level: MLevel
    name: str
    description: str
    
    # Capacités cognitives
    has_cortical_columns: bool = False
    has_emotional_system: bool = False
    has_multi_phase_reasoning: bool = False
    has_neural_plasticity: bool = False
    
    # Capacités mémoire
    has_working_memory: bool = False
    has_episodic_memory: bool = False
    has_semantic_memory: bool = False
    has_procedural_memory: bool = False
    has_world_model: bool = False
    
    # Capacités agenticité
    has_tool_use: bool = False
    has_web_search: bool = False
    has_file_operations: bool = False
    has_code_execution: bool = False
    has_auto_planning: bool = False
    
    # Capacités avancées
    has_multi_agents: bool = False
    has_self_reflection: bool = False
    has_creativity_engine: bool = False
    has_consciousness_simulation: bool = False
    
    # Limites
    max_context_tokens: int = 4096
    max_memory_episodes: int = 100
    max_tools: int = 0
    parallel_tasks: int = 1
    
    @property
    def complexity_score(self) -> float:
        """Score de complexité (0-100)"""
        score = 0
        score += 10 if self.has_cortical_columns else 0
        score += 10 if self.has_emotional_system else 0
        score += 15 if self.has_multi_phase_reasoning else 0
        score += 10 if self.has_neural_plasticity else 0
        score += 5 if self.has_working_memory else 0
        score += 10 if self.has_episodic_memory else 0
        score += 10 if self.has_semantic_memory else 0
        score += 5 if self.has_procedural_memory else 0
        score += 15 if self.has_world_model else 0
        score += 15 if self.has_tool_use else 0
        score += 10 if self.has_web_search else 0
        score += 10 if self.has_auto_planning else 0
        score += 20 if self.has_multi_agents else 0
        score += 15 if self.has_self_reflection else 0
        score += 15 if self.has_creativity_engine else 0
        score += 20 if self.has_consciousness_simulation else 0
        return min(score, 100)


# Définition des niveaux
M_LEVELS = {
    MLevel.M1: MLevelCapabilities(
        level=MLevel.M1,
        name="Orret M1 - Foundation",
        description="Conversation simple avec mémoire immédiate",
        has_working_memory=True,
        max_context_tokens=4096,
    ),
    
    MLevel.M2: MLevelCapabilities(
        level=MLevel.M2,
        name="Orret M2 - Reasoning",
        description="Raisonnement multi-phase et analyse logique",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        max_context_tokens=8192,
    ),
    
    MLevel.M3: MLevelCapabilities(
        level=MLevel.M3,
        name="Orret M3 - Memory",
        description="Mémoire hiérarchique ARIA complète",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        max_context_tokens=8192,
        max_memory_episodes=1000,
    ),
    
    MLevel.M4: MLevelCapabilities(
        level=MLevel.M4,
        name="Orret M4 - Agency",
        description="Agenticité avec tool use et recherche web",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        max_context_tokens=16384,
        max_memory_episodes=5000,
        max_tools=10,
    ),
    
    MLevel.M5: MLevelCapabilities(
        level=MLevel.M5,
        name="Orret M5 - Autonomy",
        description="Auto-planning et exécution autonome",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_world_model=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        has_code_execution=True,
        has_auto_planning=True,
        has_neural_plasticity=True,
        max_context_tokens=32768,
        max_memory_episodes=10000,
        max_tools=20,
        parallel_tasks=2,
    ),
    
    MLevel.M6: MLevelCapabilities(
        level=MLevel.M6,
        name="Orret M6 - Collaboration",
        description="Multi-agents et collaboration distribuée",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_world_model=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        has_code_execution=True,
        has_auto_planning=True,
        has_neural_plasticity=True,
        has_multi_agents=True,
        max_context_tokens=65536,
        max_memory_episodes=50000,
        max_tools=50,
        parallel_tasks=4,
    ),
    
    MLevel.M7: MLevelCapabilities(
        level=MLevel.M7,
        name="Orret M7 - Creativity",
        description="Génération créative et innovation",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_world_model=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        has_code_execution=True,
        has_auto_planning=True,
        has_neural_plasticity=True,
        has_multi_agents=True,
        has_creativity_engine=True,
        has_self_reflection=True,
        max_context_tokens=131072,
        max_memory_episodes=100000,
        max_tools=100,
        parallel_tasks=8,
    ),
    
    MLevel.M8: MLevelCapabilities(
        level=MLevel.M8,
        name="Orret M8 - Wisdom",
        description="Métacognition et sagesse accumulée",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_world_model=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        has_code_execution=True,
        has_auto_planning=True,
        has_neural_plasticity=True,
        has_multi_agents=True,
        has_creativity_engine=True,
        has_self_reflection=True,
        max_context_tokens=262144,
        max_memory_episodes=500000,
        max_tools=200,
        parallel_tasks=16,
    ),
    
    MLevel.M9: MLevelCapabilities(
        level=MLevel.M9,
        name="Orret M9 - Transcendance",
        description="Simulation de conscience et transcendance",
        has_working_memory=True,
        has_multi_phase_reasoning=True,
        has_cortical_columns=True,
        has_emotional_system=True,
        has_episodic_memory=True,
        has_semantic_memory=True,
        has_procedural_memory=True,
        has_world_model=True,
        has_tool_use=True,
        has_web_search=True,
        has_file_operations=True,
        has_code_execution=True,
        has_auto_planning=True,
        has_neural_plasticity=True,
        has_multi_agents=True,
        has_creativity_engine=True,
        has_self_reflection=True,
        has_consciousness_simulation=True,
        max_context_tokens=524288,
        max_memory_episodes=1000000,
        max_tools=500,
        parallel_tasks=32,
    ),
}


class MLevelManager:
    """
    Gestionnaire de niveaux M pour Orret.
    
    Permet:
    - Définir le niveau actuel
    - Évoluer entre niveaux
    - Adapter dynamiquement selon ressources
    - Tracker l'expérience et progression
    """
    
    def __init__(self, base_path: str = "~/soe/memory/m_levels"):
        self.path = Path(base_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.current_level = MLevel.M1
        self.experience_points = 0
        self.upgrade_history: List[Dict] = []
        
        # Charger l'état
        self._load_state()
    
    def get_level_capabilities(self, level: MLevel) -> MLevelCapabilities:
        """Retourne les capacités d'un niveau."""
        return M_LEVELS[level]
    
    def get_current_capabilities(self) -> MLevelCapabilities:
        """Retourne les capacités du niveau actuel."""
        return M_LEVELS[self.current_level]
    
    def can_upgrade_to(self, target_level: MLevel) -> bool:
        """Vérifie si l'upgrade vers un niveau est possible."""
        current_order = list(MLevel).index(self.current_level)
        target_order = list(MLevel).index(target_level)
        return target_order <= current_order + 1  # Un niveau à la fois
    
    def upgrade_to(self, target_level: MLevel, reason: str = "") -> bool:
        """
        Upgrade vers un niveau supérieur.
        Retourne True si succès.
        """
        if not self.can_upgrade_to(target_level):
            return False
        
        old_level = self.current_level
        self.current_level = target_level
        
        # Enregistrer l'upgrade
        self.upgrade_history.append({
            "timestamp": time.time(),
            "from": old_level.value,
            "to": target_level.value,
            "reason": reason,
            "experience_at_upgrade": self.experience_points
        })
        
        self._save_state()
        print(f"[M-Level] Upgrade: {old_level.value.upper()} → {target_level.value.upper()}")
        print(f"[M-Level] {M_LEVELS[target_level].name}")
        
        return True
    
    def add_experience(self, points: int, category: str = "general"):
        """Ajoute des points d'expérience."""
        self.experience_points += points
        self._save_state()
        
        # Vérifier auto-upgrade
        self._check_auto_upgrade()
    
    def _check_auto_upgrade(self):
        """Vérifie si un auto-upgrade est mérité."""
        # Seuils d'expérience pour chaque niveau
        thresholds = {
            MLevel.M1: 0,
            MLevel.M2: 100,
            MLevel.M3: 500,
            MLevel.M4: 2000,
            MLevel.M5: 5000,
            MLevel.M6: 15000,
            MLevel.M7: 50000,
            MLevel.M8: 150000,
            MLevel.M9: 500000,
        }
        
        current_order = list(MLevel).index(self.current_level)
        if current_order < len(MLevel) - 1:
            next_level = list(MLevel)[current_order + 1]
            if self.experience_points >= thresholds[next_level]:
                self.upgrade_to(next_level, reason="Auto-upgrade par expérience")
    
    def downgrade_to(self, target_level: MLevel, reason: str = "") -> bool:
        """
        Downgrade vers un niveau inférieur (pour ressources limitées).
        """
        current_order = list(MLevel).index(self.current_level)
        target_order = list(MLevel).index(target_level)
        
        if target_order >= current_order:
            return False
        
        old_level = self.current_level
        self.current_level = target_level
        
        self.upgrade_history.append({
            "timestamp": time.time(),
            "from": old_level.value,
            "to": target_level.value,
            "reason": reason,
            "experience_at_upgrade": self.experience_points
        })
        
        self._save_state()
        print(f"[M-Level] Downgrade: {old_level.value.upper()} → {target_level.value.upper()}")
        
        return True
    
    def auto_adjust_for_resources(self, available_ram_gb: int, cpu_cores: int):
        """
        Ajuste automatiquement le niveau selon ressources disponibles.
        """
        if available_ram_gb < 4:
            target = MLevel.M1
            reason = "RAM insuffisante"
        elif available_ram_gb < 8:
            target = MLevel.M2
            reason = "RAM limitée"
        elif available_ram_gb < 16:
            target = MLevel.M3
            reason = "RAM standard"
        elif available_ram_gb < 32:
            target = MLevel.M4
            reason = "RAM bonne"
        elif cpu_cores < 4:
            target = MLevel.M4
            reason = "CPU limité"
        else:
            # Garder niveau actuel ou monter
            current_order = list(MLevel).index(self.current_level)
            if current_order < len(MLevel) - 1:
                target = list(MLevel)[current_order + 1]
                reason = "Ressources suffisantes"
            else:
                target = self.current_level
                reason = "Déjà au maximum"
        
        current_order = list(MLevel).index(self.current_level)
        target_order = list(MLevel).index(target)
        
        if target_order > current_order:
            self.upgrade_to(target, reason)
        elif target_order < current_order:
            self.downgrade_to(target, reason)
    
    def get_progress_to_next(self) -> Dict:
        """Retourne la progression vers le prochain niveau."""
        current_order = list(MLevel).index(self.current_level)
        if current_order >= len(MLevel) - 1:
            return {"at_max": True, "progress": 100}
        
        next_level = list(MLevel)[current_order + 1]
        thresholds = {
            MLevel.M2: 100,
            MLevel.M3: 500,
            MLevel.M4: 2000,
            MLevel.M5: 5000,
            MLevel.M6: 15000,
            MLevel.M7: 50000,
            MLevel.M8: 150000,
            MLevel.M9: 500000,
        }
        
        threshold = thresholds.get(next_level, 1000000)
        progress = min(100, (self.experience_points / threshold) * 100)
        
        return {
            "at_max": False,
            "current_level": self.current_level.value,
            "next_level": next_level.value,
            "current_xp": self.experience_points,
            "required_xp": threshold,
            "progress": progress
        }
    
    def _save_state(self):
        """Sauvegarde l'état."""
        state = {
            "current_level": self.current_level.value,
            "experience_points": self.experience_points,
            "upgrade_history": self.upgrade_history
        }
        with open(self.path / "state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Charge l'état."""
        state_file = self.path / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self.current_level = MLevel(state.get("current_level", "m1"))
            self.experience_points = state.get("experience_points", 0)
            self.upgrade_history = state.get("upgrade_history", [])
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques."""
        return {
            "current_level": self.current_level.value,
            "level_name": M_LEVELS[self.current_level].name,
            "complexity_score": M_LEVELS[self.current_level].complexity_score,
            "experience_points": self.experience_points,
            "total_upgrades": len(self.upgrade_history),
            "progress": self.get_progress_to_next()
        }


# Test
if __name__ == "__main__":
    manager = MLevelManager()
    
    print("=== Système de Niveaux M pour Orret ===\n")
    
    # Afficher tous les niveaux
    for level in MLevel:
        caps = M_LEVELS[level]
        print(f"{level.value.upper()}: {caps.name}")
        print(f"  Complexité: {caps.complexity_score}/100")
        print(f"  Contexte max: {caps.max_context_tokens} tokens")
        print(f"  Mémoire max: {caps.max_memory_episodes} épisodes")
        print()
    
    # Test progression
    print("=== Test Progression ===")
    print(f"Niveau actuel: {manager.get_stats()['current_level']}")
    
    manager.add_experience(150, "conversation")
    print(f"Après +150 XP: {manager.get_stats()['progress']}")
    
    manager.upgrade_to(MLevel.M2, "Test manuel")
    print(f"Niveau: {manager.get_stats()['current_level']}")
    
    # Test ajustement ressources
    print("\n=== Test Ajustement Ressources ===")
    manager.auto_adjust_for_resources(available_ram_gb=4, cpu_cores=2)
    print(f"Niveau ajusté: {manager.get_stats()['current_level']}")
