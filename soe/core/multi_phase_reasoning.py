"""
Raisonnement Multi-Phase pour SOE-Orret.
6 phases inspirées du cerveau humain :
  PERCEPTION → INTUITION → ANALYSE → SYNTHÈSE → VALIDATION → EXPRESSION
Chaque phase peut être sautée selon la complexité.
"""
import time
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ReasoningPhase(Enum):
    PERCEPTION  = "perception"
    INTUITION   = "intuition"
    ANALYSIS    = "analysis"
    SYNTHESIS   = "synthesis"
    VALIDATION  = "validation"
    EXPRESSION  = "expression"

@dataclass
class PhaseResult:
    phase:      ReasoningPhase
    output:     str
    confidence: float
    duration:   float
    skipped:    bool = False

PHASE_PROMPTS = {
    ReasoningPhase.PERCEPTION: (
        "[PHASE: PERCEPTION]\nQuelle est la vraie question derrière la surface ?\n"
        "Y a-t-il une ambiguïté ? Quel est le contexte émotionnel ?\n"
        "Demande: {query}\nAnalyse de perception (3-5 lignes max):"
    ),
    ReasoningPhase.INTUITION: (
        "[PHASE: INTUITION]\nPremière réaction rapide et non censurée.\n"
        "Demande: {query}\nPerception précédente: {prev}\nIntuition brute:"
    ),
    ReasoningPhase.ANALYSIS: (
        "[PHASE: ANALYSE PROFONDE]\n"
        "Éléments clés, contraintes, risques d'erreur, informations manquantes.\n"
        "Demande: {query}\nIntuition: {prev}\nAnalyse détaillée:"
    ),
    ReasoningPhase.SYNTHESIS: (
        "[PHASE: SYNTHÈSE]\nConstruit la réponse directement à partir de l'analyse.\n"
        "Demande: {query}\nAnalyse: {prev}\nSynthèse:"
    ),
    ReasoningPhase.VALIDATION: (
        "[PHASE: VALIDATION]\nVérifie : répond exactement à la demande ? "
        "Erreur factuelle/logique ? Ton approprié ?\n"
        "Synthèse à valider: {prev}\nDemande originale: {query}\n"
        "Validation (OK ou corrections):"
    ),
    ReasoningPhase.EXPRESSION: (
        "[PHASE: EXPRESSION FINALE]\nFormule la réponse finale. "
        "Adapte le ton. Sois précis. Naturel.\n"
        "Synthèse validée: {prev}\nRéponse finale (sans mentionner les phases):"
    ),
}

SIMPLE_QUERY  = [ReasoningPhase.PERCEPTION, ReasoningPhase.SYNTHESIS, ReasoningPhase.EXPRESSION]
MEDIUM_QUERY  = [ReasoningPhase.PERCEPTION, ReasoningPhase.INTUITION,
                 ReasoningPhase.SYNTHESIS, ReasoningPhase.VALIDATION, ReasoningPhase.EXPRESSION]
COMPLEX_QUERY = list(ReasoningPhase)

class MultiPhaseReasoner:
    def __init__(self, model_func: Callable):
        self.model = model_func

    def _estimate_complexity(self, query: str) -> str:
        complex_kw = ["explique","compare","analyse","pourquoi","comment","différence",
                      "philosophie","conscience","éthique","critique","théorie","prouve"]
        medium_kw  = ["qu'est-ce","définis","résume","liste","exemples","comment fonctionne"]
        q = query.lower()
        words = len(query.split())
        if words > 30 or any(kw in q for kw in complex_kw): return "complex"
        if words > 10 or any(kw in q for kw in medium_kw):  return "medium"
        return "simple"

    def reason(self, query: str, forced_complexity: Optional[str] = None,
               verbose: bool = False) -> Tuple[str, List[PhaseResult]]:
        complexity = forced_complexity or self._estimate_complexity(query)
        phase_map  = {"simple": SIMPLE_QUERY, "medium": MEDIUM_QUERY, "complex": COMPLEX_QUERY}
        active     = phase_map[complexity]
        results    = []
        prev_output = ""

        if verbose:
            print(f"[Raisonnement] Complexité: {complexity} ({len(active)} phases)")

        for phase in ReasoningPhase:
            t0 = time.time()
            if phase not in active:
                results.append(PhaseResult(phase=phase, output="", confidence=1.0,
                                           duration=0.0, skipped=True))
                continue
            prompt_tpl = PHASE_PROMPTS[phase]
            prompt = prompt_tpl.format(query=query, prev=prev_output[:600])
            try:
                output = self.model(prompt, max_new_tokens=300, temperature=0.6)
            except Exception as e:
                output = f"[Erreur phase {phase.value}: {e}]"
            dt = time.time() - t0
            if verbose:
                print(f"  [{phase.value.upper()}] {dt:.1f}s — {output[:80]}...")
            results.append(PhaseResult(phase=phase, output=output,
                                       confidence=0.8, duration=dt))
            prev_output = output

        final = next((r.output for r in reversed(results)
                      if r.phase == ReasoningPhase.EXPRESSION and not r.skipped), prev_output)
        return final, results
