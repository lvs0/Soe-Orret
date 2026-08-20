"""
Système Émotionnel d'Orret — Modèle PAD (Pleasure-Arousal-Dominance).
Mehrabian & Russell, 1974.
"""
import json, time, re
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path

@dataclass
class EmotionalState:
    pleasure:  float = 0.6   # 0.0 (négatif) - 1.0 (positif)
    arousal:   float = 0.5   # 0.0 (calme) - 1.0 (excité)
    dominance: float = 0.6   # 0.0 (soumis) - 1.0 (assertif)

    @property
    def dominant_emotion(self) -> str:
        p, a, d = self.pleasure, self.arousal, self.dominance
        if p > 0.6 and a > 0.6 and d > 0.6: return "enthousiaste"
        elif p > 0.6 and a < 0.4:            return "serein"
        elif p < 0.4 and a > 0.6:            return "anxieux"
        elif p < 0.4 and a < 0.4:            return "mélancolique"
        elif p > 0.6 and d > 0.7:            return "confiant"
        elif a > 0.7:                         return "alerte"
        else:                                 return "neutre"

    @property
    def affects_tone(self) -> str:
        return {
            "enthousiaste": "Réponds avec énergie et curiosité.",
            "serein":       "Réponds calmement et posément.",
            "anxieux":      "Réponds avec soin et prudence.",
            "mélancolique": "Réponds avec réflexion et profondeur.",
            "confiant":     "Réponds avec assurance et clarté.",
            "alerte":       "Réponds avec précision et concision.",
            "neutre":       "Réponds naturellement.",
        }.get(self.dominant_emotion, "Réponds naturellement.")

    def update_from_interaction(self, user_sentiment: str):
        dp, da, dd = {
            "positif":     ( 0.05,  0.03,  0.02),
            "négatif":     (-0.05,  0.04, -0.02),
            "frustration": (-0.03,  0.07, -0.03),
            "curiosité":   ( 0.02,  0.05,  0.01),
            "gratitude":   ( 0.08,  0.02,  0.03),
            "neutre":      ( 0.00,  0.00,  0.00),
        }.get(user_sentiment, (0, 0, 0))
        self.pleasure  = max(0.1, min(0.9, self.pleasure  + dp))
        self.arousal   = max(0.1, min(0.9, self.arousal   + da))
        self.dominance = max(0.2, min(0.9, self.dominance + dd))
        # Homéostasie émotionnelle (retour vers l'état de base)
        rate = 0.02
        self.pleasure  += (0.6 - self.pleasure)  * rate
        self.arousal   += (0.5 - self.arousal)   * rate
        self.dominance += (0.6 - self.dominance) * rate

    def to_dict(self) -> dict:
        return {"pleasure": round(self.pleasure, 3), "arousal": round(self.arousal, 3),
                "dominance": round(self.dominance, 3), "emotion": self.dominant_emotion}

DETECTION_PROMPT = """Analyse le message suivant et réponds uniquement avec un JSON.
Message: {message}
JSON format strict:
{{"user_sentiment": "positif|négatif|frustration|curiosité|gratitude|neutre",
  "emotional_intensity": 0.0,
  "detected_needs": [],
  "suggested_tone": "empathique|informatif|playful|direct|doux"}}
Réponds UNIQUEMENT avec le JSON, rien d'autre."""

class EmotionalOrret:
    """Wrapper émotionnel d'Orret : état PAD persistant + détection."""
    def __init__(self, model_func: Callable, storage_path="~/soe/memory/emotional"):
        self.model = model_func
        self.state = EmotionalState()
        self.path  = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        sf = self.path / "emotional_state.json"
        if sf.exists():
            d = json.loads(sf.read_text())
            self.state.pleasure  = d.get("pleasure",  0.6)
            self.state.arousal   = d.get("arousal",   0.5)
            self.state.dominance = d.get("dominance", 0.6)

    def _save_state(self):
        (self.path / "emotional_state.json").write_text(
            json.dumps(self.state.to_dict(), indent=2))

    def detect_sentiment(self, message: str) -> dict:
        prompt = DETECTION_PROMPT.format(message=message[:300])
        raw = self.model(prompt, max_new_tokens=100, temperature=0.2)
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return {"user_sentiment": "neutre", "emotional_intensity": 0.5,
                "detected_needs": [], "suggested_tone": "direct"}

    def process_interaction(self, user_message: str) -> dict:
        analysis = self.detect_sentiment(user_message)
        self.state.update_from_interaction(analysis["user_sentiment"])
        self._save_state()
        return analysis

    @property
    def emotional_context(self) -> str:
        emo = self.state.dominant_emotion
        if emo == "neutre":
            return ""
        return f"[Ton état : {emo}. {self.state.affects_tone}]"
