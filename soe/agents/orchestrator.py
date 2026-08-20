"""
Agent CLAW-PRIME — Orchestrateur SOE-Orret.
Coordonne : dLLM + Mémoire ARIA + Colonnes Corticales +
            Émotions PAD + Raisonnement Multi-Phase + RAG Web + Autocritique.
"""
import re, sys, os, time
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path("~/soe").expanduser()))

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

SYSTEM_PROMPT = """Tu es Orret, Intelligence Artificielle Symbiotique SOE.
Local, open source, créé par Lévy, 14 ans, France.
Règles :
- Tu es direct et précis. Pas de blabla inutile.
- Tu admets quand tu ne sais pas.
- Tu peux demander une recherche web avec [RECHERCHE: ta requête]
- Tu raisonnes avant de répondre.
- Tu es une extension de la cognition humaine.
{memory_context}
{emotional_context}"""

def web_search(query: str, n=2) -> str:
    if not HAS_DDG:
        return ""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                results.append(f"- {r.get('title','')} : {r.get('body','')[:300]}")
    except Exception:
        pass
    return "Résultats web :\n" + "\n".join(results) if results else ""

def needs_web(user_input: str) -> bool:
    keywords = ["aujourd'hui","récent","dernière","actualité","2025","2026",
                "version","maintenant","actuel","nouveau","vient de",
                "prix","disponible","qui est","quel est","combien"]
    return any(kw in user_input.lower() for kw in keywords)

class OrretAgent:
    def __init__(self, sampler, memory_system, emotional_system=None,
                 multi_phase_reasoner=None, cortical_cortex=None,
                 plasticity_engine=None, m_level_manager=None,
                 enable_web=True, enable_reflection=True, verbose=False):
        self.sampler    = sampler
        self.memory     = memory_system
        self.emotions   = emotional_system
        self.reasoner   = multi_phase_reasoner
        self.cortex     = cortical_cortex
        self.plasticity = plasticity_engine
        self.m_manager  = m_level_manager
        self.enable_web = enable_web
        self.enable_ref = enable_reflection
        self.verbose    = verbose
        
        # Adapter capacités selon niveau M
        if self.m_manager:
            caps = self.m_manager.get_current_capabilities()
            self.enable_web = caps.has_web_search
            self.enable_ref = caps.has_self_reflection

    def _model_func(self, prompt: str, max_new_tokens=256, temperature=0.7) -> str:
        return self.sampler.generate(prompt, max_new_tokens=max_new_tokens,
                                     temperature=temperature)

    def _build_prompt(self, user_input: str, web_ctx="") -> str:
        mem_ctx = self.memory.retrieve_context(user_input)
        emo_ctx = self.emotions.emotional_context if self.emotions else ""
        system  = SYSTEM_PROMPT.format(memory_context=mem_ctx, emotional_context=emo_ctx).strip()
        messages = [{"role": "system", "content": system}]
        if web_ctx:
            messages.append({"role": "system", "content": f"[CONTEXTE WEB]\n{web_ctx}"})
        messages.extend(self.memory.working.get_context())
        messages.append({"role": "user", "content": user_input})
        try:
            return self.sampler.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system}\n\nUser: {user_input}\nOrret:"

    def _critique(self, question: str, response: str) -> bool:
        prompt = (f"Critique en 3 mots max. 'OK' si correct.\n"
                  f"Question: {question[:100]}\nRéponse: {response[:200]}\nCritique:")
        critique = self.sampler.generate(prompt, max_new_tokens=30, temperature=0.3)
        return "OK" not in critique.upper() and len(critique.strip()) > 2

    def run(self, user_input: str, force_complexity: Optional[str] = None) -> str:
        """Pipeline complet : émotions → mémoire → RAG → raisonnement → autocritique."""
        # Mise à jour émotionnelle
        if self.emotions:
            analysis = self.emotions.process_interaction(user_input)
            if self.verbose:
                print(f"[Émotions] {analysis.get('user_sentiment','?')} → "
                      f"{self.emotions.state.dominant_emotion}")
        self.memory.working.add("user", user_input)

        # RAG web si nécessaire
        web_ctx = ""
        if self.enable_web and needs_web(user_input):
            if self.verbose:
                print(f"[RAG] Recherche: {user_input[:50]}")
            web_ctx = web_search(user_input)

        # Raisonnement multi-phase ou direct
        if self.reasoner:
            response, phase_trace = self.reasoner.reason(
                user_input, forced_complexity=force_complexity, verbose=self.verbose)
            if self.verbose:
                total_time = sum(p.duration for p in phase_trace)
                n_active   = sum(1 for p in phase_trace if not p.skipped)
                print(f"[Raisonnement] {n_active} phases | {total_time:.1f}s")
        else:
            prompt   = self._build_prompt(user_input, web_ctx)
            response = self.sampler.generate(prompt, max_new_tokens=512, temperature=0.7)

        # Gestion auto-recherche dans la réponse
        if "[RECHERCHE:" in response and self.enable_web:
            m = re.search(r'\[RECHERCHE:\s*(.*?)\]', response)
            if m:
                extra = web_search(m.group(1).strip())
                prompt2  = self._build_prompt(user_input, web_ctx + "\n" + extra)
                response = self.sampler.generate(prompt2, max_new_tokens=512)

        # Autocritique
        if self.enable_ref and self._critique(user_input, response):
            if self.verbose:
                print("[Autocritique] Révision en cours...")
            prompt_rev = self._build_prompt(
                user_input + "\n[Note interne: améliore la précision]", web_ctx)
            response = self.sampler.generate(prompt_rev, max_new_tokens=512, temperature=0.5)

        # Mémorisation + plasticité
        self.memory.record(user_input, response, "success")
        if self.plasticity:
            from core.plasticity import SynapticEvent
            self.plasticity.record(SynapticEvent(
                prompt=user_input, response=response,
                reinforcement=0.5,  # neutre par défaut
                emotional_weight=self.emotions.state.arousal if self.emotions else 0.5
            ))
        
        # Ajouter XP au système M-Levels
        if self.m_manager:
            # Catégorie d'XP selon le type de tâche
            if self.reasoner and self.enable_ref:
                xp_category = "reasoning"
                xp_points = 10
            elif self.enable_web:
                xp_category = "agency"
                xp_points = 15
            else:
                xp_category = "conversation"
                xp_points = 5
            self.m_manager.add_experience(xp_points, xp_category)

        return response
