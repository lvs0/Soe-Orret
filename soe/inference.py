#!/usr/bin/env python3
"""
SOE-Orret MVP — Inference Demo
================================
Démonstration de l'agent SOE-Orret en mode Ollama (qwen2.5:1.5b).

Usage:
    python inference.py "Qu'est-ce que la cryptographie post-quantique ?"
    python inference.py --model qwen2.5:1.5b "Explique le principe de diffusion pour les LLMs"

Configuration:
    - Ollama doit être installé et running (ollama serve)
    - Modèle par défaut : qwen2.5:1.5b (986 MB, tourne sur laptop)
    - Pour le modèle 7B complet : voir INSTALL.md
"""

import json
import os
import sys
import time
import requests
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("ORRET_MODEL", "qwen2.5:1.5b")

# ── Ollama Sampler ─────────────────────────────────────────────────────────────

class OllamaSampler:
    """Sampler SOE-Orret via Ollama (compatible avec l'interface agent)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.base_url = OLLAMA_URL
        # Verify Ollama is reachable
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json()
            models = [m["name"] for m in tags.get("models", [])]
            if model not in models:
                print(f"[WARN] Modèle '{model}' non trouvé dans Ollama. Modèles disponibles : {models}")
                print(f"[WARN] Téléchargez-le avec : ollama pull {model}")
            else:
                print(f"[OK] Ollama connecté, modèle '{model}' disponible.")
        except Exception as e:
            print(f"[WARN] Ollama injoignable à {self.base_url}: {e}")
            print("[WARN] Lancez 'ollama serve' dans un autre terminal.")

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, **kwargs) -> str:
        """Génère une réponse via Ollama."""
        t0 = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json().get("response", "")
            elapsed = time.time() - t0
            tokens = len(result.split())
            if tokens > 0:
                print(f"[Orret] {tokens} tokens en {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)")
            return result.strip()
        except Exception as e:
            print(f"[ERREUR] Échec génération: {e}")
            return f"[ERREUR] {str(e)}"

    @property
    def tokenizer(self):
        """Adapter pour compatibilité avec OrretAgent."""
        class _SimpleTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )
        return _SimpleTokenizer()


# ── Agent SOE-Orret (lightweight) ──────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es Orret, une intelligence artificielle symbiotique open-source (SOE-Orret).
Créé par Lévy, 14 ans, France. Projet de recherche en dLLM + PQC.

Règles :
- Sois direct et précis. Pas de blabla.
- Admets quand tu ne sais pas.
- Réponds en français sauf si la question est en anglais.
- Tu es une extension de la cognition humaine, pas un chatbot commercial."""


class OrretMini:
    """Version légère de l'agent SOE-Orret pour la démo MVP."""

    def __init__(self, sampler: OllamaSampler, verbose: bool = True):
        self.sampler = sampler
        self.verbose = verbose

    def run(self, user_input: str) -> str:
        """Pipeline simplifié : system prompt → user input → réponse."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER: {user_input}\n"
            f"ORRET:"
        )
        return self.sampler.generate(prompt, max_new_tokens=256, temperature=0.7)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SOE-Orret MVP — Démo d'inférence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python inference.py "Explique la diffusion pour les LLMs en 3 phrases"
  python inference.py --model phi4-mini:latest "What is post-quantum crypto?"
  python inference.py --verbose "Qui a créé SOE-Orret ?"
        """
    )
    parser.add_argument("prompt", nargs="+", help="Le prompt à envoyer à Orret")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modèle Ollama (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mode verbeux")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Nombre max de tokens (défaut: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Température (défaut: 0.7)")

    args = parser.parse_args()
    prompt = " ".join(args.prompt)

    print("=" * 60)
    print("  SOE-Orret MVP — Démo d'inférence")
    print("  dLLM symbiotique open-source | PQC-ready")
    print("=" * 60)
    print(f"  Modèle  : {args.model}")
    print(f"  Prompt  : {prompt}")
    print("-" * 60)

    # Init
    sampler = OllamaSampler(model=args.model)
    agent = OrretMini(sampler, verbose=args.verbose)

    # Generate
    response = agent.run(prompt)

    print("-" * 60)
    print(response)
    print("-" * 60)
    print("[MVP] Démo terminée. Pour le modèle 7B complet, voir INSTALL.md")
    print("[MVP] Projet : https://github.com/lvs0/soe-orret")


if __name__ == "__main__":
    main()
