# SOE-Orret

**Symbiotic Operating Environment — Block Diffusion Language Model**

[![Status](https://img.shields.io/badge/status-MVP%20démonstrateur-blue)](https://github.com/lvs0/soe-orret)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

SOE-Orret est un **modèle de langage à diffusion par blocs (dLLM)** open-source conçu pour fonctionner sur un laptop. Il combine une architecture agent symbiotique (raisonnement multi-phase, mémoire ARIA, émotions PAD, plasticité neurale) avec une couche de sécurité post-quantique (ML-KEM-1024 + ML-DSA-65 via Polygone).

**Créé par Lévy, 14 ans, France.** Projet de recherche en dLLM + PQC.

---

## Pourquoi SOE-Orret ?

Les modèles autoregressifs (GPT, LLaMA) génèrent du texte token par token, de gauche à droite. Les modèles de diffusion par blocs (dLLM) génèrent des blocs entiers de tokens en parallèle, avec :
- **Contexte bidirectionnel** — chaque token voit tous les autres
- **Révision possible** — les tokens peu confiants peuvent être re-générés
- **Parallélisme par bloc** — K tokens par étape au lieu d'1
- **Résistance au "reversal curse"** — peut raisonner dans les deux sens

SOE-Orret adapte Qwen2.5-7B-Instruct en dLLM via la méthode A2D (Fast-dLLM v2, NeurIPS 2025), et ajoute une architecture agent unique :
- Raisonnement 6 phases (Perception → Intuition → Analyse → Synthèse → Validation → Expression)
- Mémoire ARIA 5 couches (Working / Épisodique / Sémantique / Procédurale / Émotionnelle)
- Système émotionnel PAD (Pleasure-Arousal-Dominance)
- Plasticité neurale avec événements synaptiques
- Niveaux M (M1→M9) adaptatifs aux ressources
- Sécurité PQC intégrée (chiffrement post-quantique)

---

## État du projet

**MVP démontrable** — inférence fonctionnelle via Ollama (qwen2.5:1.5b testé, 7B en cours).

| Composant | Statut |
|---|---|
| Inférence dLLM (block diffusion) | 🟡 En cours (Fast-dLLM v2 adaptation) |
| Agent 6-phase reasoning | 🟡 Backbone écrit, à tester bout-en-bout |
| Mémoire ARIA 5 couches | 🟡 Backbone écrit, à tester |
| API REST (OpenAI-compatible) | 🟡 Serveur FastAPI écrit, non testé |
| Intégration PQC (Polygone) | 🟡 Module écrit, à brancher |
| Tests | ❌ 0 tests |
| Benchmarks | ❌ 0 benchmarks |
| Paper arXiv | ❌ À rédiger |

**Prochaine étape** : adaptation Fast-dLLM v2 → SOE-Orret 7B, benchmarks HumanEval/French-BLEU.

---

## Installation rapide

```bash
# 1. Cloner
git clone https://github.com/lvs0/soe-orret
cd soe-orret

# 2. Installer Ollama + modèle
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b

# 3. Lancer l'inférence
python soe/inference.py "Explique la diffusion pour les LLMs en 3 phrases."
```

Pour le setup complet (modèle 7B, API REST), voir [INSTALL.md](INSTALL.md).

---

## Architecture

```
USER INPUT
    │
    ▼
┌─────────────────────────┐
│   SOE ORCHESTRATOR      │
│   (6-Phase Cognitive)   │
└──┬──────┬──────┬───────┘
   │      │      │
   ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────────┐
│SOE-O1│ │ARIA  │ │dLLM      │
│Reason│ │Memory│ │Inference │
│6-ph. │ │5-lay.│ │Block-Diff│
└──────┘ └──────┘ └──────────┘
   │
   ▼
┌──────────────────────┐
│  POLYGONE (PQC)      │
│  ML-KEM + ML-DSA     │
└──────────────────────┘
```

---

## Recherche

- [RESEARCH_NOTES_dLLM.md](RESEARCH_NOTES_dLLM.md) — État de l'art dLLM 2025-2026
- [AUDIT_SOE.md](AUDIT_SOE.md) — Audit technique complet du code
- [ARCHITECTURE.md](soe/ARCHITECTURE.md) — Design document

## Projets liés

- [Polygone Network](https://github.com/lvs0/Polygone-Network) — Protocole P2P post-quantique
- Fast-dLLM v2 (NVIDIA/MIT) — Méthode d'adaptation AR→dLLM

## Citation

```bibtex
@misc{soe-orret2026,
  title={SOE-Orret: A Post-Quantum Symbiotic Operating Environment for Edge Language Models},
  author={Verpoort Scherpereel, L{\'e}vy},
  year={2026},
  howpublished={\url{https://github.com/lvs0/soe-orret}},
}
```

## Licence

MIT — voir [LICENSE](LICENSE).
