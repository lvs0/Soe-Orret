"""
Architecture à Colonnes Corticales pour SOE-Orret.
Inspiration : Jeff Hawkins, "A Thousand Brains" (2021).
Chaque colonne est un module spécialisé. La réponse finale
émerge du consensus entre colonnes, comme dans le cerveau.
"""
from typing import List, Dict, Tuple, Callable, Optional

class CorticalColumn:
    """Une colonne spécialisée dans un domaine."""
    def __init__(self, name: str, domain: str, weight: float = 1.0):
        self.name = name
        self.domain = domain
        self.weight = weight
        self.confidence_history: List[float] = []

    def update_weight(self, feedback: float):
        self.weight = max(0.1, min(2.0, self.weight + feedback * 0.05))
        self.confidence_history.append(feedback)

class CorticalCortex:
    """
    Cortex d'Orret : ensemble de colonnes spécialisées.
    8 colonnes : LOGIC, LANG, CODE, EMOTION, MEMORY, SOCIAL, CREATIVE, CRITIC.
    """
    COLUMNS = [
        ("LOGIC",    "logical_reasoning",   1.0),
        ("LANG",     "language",            1.2),
        ("CODE",     "programming",         1.0),
        ("EMOTION",  "emotional",           0.8),
        ("MEMORY",   "memory_retrieval",    1.0),
        ("SOCIAL",   "social_psychology",   0.7),
        ("CREATIVE", "creativity",          0.9),
        ("CRITIC",   "self_critique",       1.1),
    ]

    def __init__(self, model_func: Callable):
        self.model = model_func
        self.columns = {
            name: CorticalColumn(name, domain, weight)
            for name, domain, weight in self.COLUMNS
        }

    def _route_to_columns(self, query: str) -> List[str]:
        q = query.lower()
        active = ["CRITIC", "LANG"]
        if any(w in q for w in ["calcul","math","logique","pourquoi","si","prouve"]):
            active.append("LOGIC")
        if any(w in q for w in ["code","python","bash","bug","script","programme"]):
            active.append("CODE")
        if any(w in q for w in ["ressens","amour","triste","heureux","emotion","peur"]):
            active.extend(["EMOTION","SOCIAL"])
        if any(w in q for w in ["souviens","rappelle","avant","déjà","session"]):
            active.append("MEMORY")
        if any(w in q for w in ["crée","invente","imagine","original","histoire"]):
            active.append("CREATIVE")
        return list(set(active))

    def process(self, query: str, context: str = "") -> Tuple[str, Dict]:
        """Traitement multi-colonnes avec synthèse pondérée."""
        active_columns = self._route_to_columns(query)
        column_responses = {}
        for col_name in active_columns:
            col = self.columns[col_name]
            prompt = (
                f"[COLONNE {col_name} - {col.domain}]\n"
                f"Contexte: {context[:400]}\n"
                f"Question: {query}\n"
                f"Réponds selon ta spécialité ({col.domain}). "
                f"Sois concis (2-3 phrases max).\nRéponse:"
            )
            try:
                response = self.model(prompt, max_new_tokens=150, temperature=0.6)
            except Exception:
                response = "[Colonne indisponible]"
            column_responses[col_name] = {"response": response, "weight": col.weight}

        synthesis_parts = [
            f"[{n}] {d['response']}"
            for n, d in column_responses.items() if n != "LANG"
        ]
        synthesis_prompt = (
            f"Tu es la colonne SYNTHÈSE d'Orret.\n"
            f"Question originale: {query}\n"
            f"Analyses des modules:\n{chr(10).join(synthesis_parts)}\n"
            f"Synthèse finale (naturelle, sans mention des modules):"
        )
        final_response = self.model(synthesis_prompt, max_new_tokens=400, temperature=0.7)
        return final_response, {"active_columns": active_columns, "column_responses": column_responses}
