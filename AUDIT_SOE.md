# AUDIT SOE-Orret — État du projet pour publication recherche

**Date** : 2026-08-20
**Auditeur** : Ruflo (mission SOE)
**Référentiel** : ARCHITECTURE.md + guides SOE_ORRET_GUIDE_FULL.txt / SOE_ORRET_EXTENDED_FULL.txt (≈104 KB de spec)
**Barre** : projet de recherche publiable (ICML/NeurIPS-comparable, pas hobbyiste)

---

## TL;DR (verdict honnête)

- **Architecture documentée, code partiellement implémenté.** On a 4 058 LOC Python sur 1 750 attendues (donc **+130 %** par rapport à ARCHITECTURE.md, mais c'est trompeur : une grosse partie est du code pléthorique / mort / dupliqué).
- **Aucun test unitaire, aucun benchmark, aucune démo qui tourne bout-en-bout.** Pas de `pytest`, pas de CI.
- **Le sampler "dLLM" actuel est un faux dLLM** : il utilise `AutoModelForCausalLM` (autoregressif), pas un vrai modèle de diffusion masqué. C'est l'écart scientifique #1 à combler pour passer de "projet sympa" à "papier de recherche publiable".
- **Le modèle Qwen2.5-7B-Instruct sur disque est INCOMPLET** : 2/4 shards présents (~6.6 GB), pas de `tokenizer.json`, pas de `model.safetensors.index.json`. Impossible à charger via `transformers`. Téléchargement à reprendre.
- **Pas de `__init__.py` à la racine, pas de `pyproject.toml` / `setup.py`, pas de README, pas de LICENSE.** Le repo n'est pas package-able ni installable.
- **Pas de remote Git configuré.** `git remote -v` retourne vide. On ne peut pas push.

→ **Conclusion** : l'épine dorsale conceptuelle est là (5 fichiers core + 2 memory + 1 agent + 1 sampler), mais le projet n'est **ni exécutable, ni testable, ni publiable** en l'état. Il faut combler 5 trous critiques listés en section 5 avant d'écrire la moindre ligne de paper.

---

## 1. Inventaire du code existant (16 fichiers .py, 4 058 LOC)

| Fichier | LOC | Rôle réel | État |
|---|---|---|---|
| `inference/engine/dllm_sampler.py` | 114 | `BlockDiffusionSampler` — **mais utilise `AutoModelForCausalLM`, pas un vrai dLLM**. Masque tokens, fixe confiants. Nom trompeur. | **Code-mort-de-design** : l'algorithme décrit dans le docstring (block diffusion bidirectionnel) n'est pas ce que le code fait. |
| `inference/api/server.py` | 319 | FastAPI OpenAI-compatible. Sélectionne le mode (Ollama / GGUF / dLLM) au startup. Endpoints `/v1/chat/completions`, `/soe/health`, `/soe/m_level/upgrade`, `/soe/openclaw`. Routage Fireworks/OpenRouter avec daily token counter. | **Vivant** : module principal, mais ne tourne pas sans dépendances. |
| `api/server.py` | 457 | **Doublon** de `inference/api/server.py`. Import des `run_server`, `SoeOrretHandler` non définis ici. Brisera l'import. | **Code mort** : à supprimer. |
| `api/__init__.py` | 3 | Ré-export depuis `api/server` (qui ne marche pas). | **Code mort** : cascade d'import cassé. |
| `agent/orchestrator.py` | 455 | **Doublon** de `agents/orchestrator.py`. Orchestrateur générique (Task/Agent/Orchestrator) threadé. Non utilisé par le vrai agent SOE. | **Code mort** : à supprimer. |
| `agents/orchestrator.py` | 163 | **Le vrai** `OrretAgent` (CLAW-PRIME) : émotions → mémoire → RAG DuckDuckGo → raisonnement multi-phase → autocritique → plasticité → M-Levels XP. | **Vivant** : cœur du système. |
| `core/cortical_columns.py` | 89 | Colonnes corticales (Jeff Hawkins "A Thousand Brains"). 1 module. | **À tester** : existe mais aucune preuve qu'il tourne. |
| `core/emotional_system.py` | 114 | Modèle PAD (Pleasure-Arousal-Dominance, Mehrabian & Russell 1974). | **À tester**. |
| `core/infinite_context.py` | 677 | **Le plus gros fichier (677 LOC)**. Compression + RAG pour contexte illimité. Optimisé Lenovo X250. | **À tester** : non référencé dans `agents/orchestrator.py` — **probablement mort**. |
| `core/m_levels.py` | 521 | Système M1→M9 (capabilities, pas intelligence). XP / niveaux. | **À tester** : référencé par `inference/api/server.py` et `agents/orchestrator.py`. |
| `core/multi_phase_reasoning.py` | 111 | Raisonnement 6 phases (Perception→Intuition→Analysis→Synthesis→Validation→Expression). | **À tester** : référencé par `agents/orchestrator.py`. |
| `core/plasticity.py` | 158 | Plasticité neurale (événements synaptiques). | **À tester** : référencé par `agents/orchestrator.py`. |
| `core/polygone_integration.py` | 273 | Intégration Polygone (AES-256-GCM, Shamir SSS, audit). | **À tester** : non branché dans le pipeline principal. |
| `memory/aria.py` | 410 | Aria — mémoire SQLite 5 couches (Working/Episodic/Semantic/Procedural/Emotional). | **À tester** : référencé par `memory/hierarchy.py`. |
| `memory/hierarchy.py` | 188 | API haut-niveau de la mémoire ARIA. | **À tester** : référencé par `inference/api/server.py`. |
| `memory/__init__.py` | 3 | Ré-export `LayeredMemory`, `MemoryEntry`. | **Vivant**. |

### Synthèse code-mort-vs-vivant

- **Vivant (importé par le pipeline principal)** : `agents/orchestrator.py`, `core/m_levels.py`, `core/multi_phase_reasoning.py`, `core/plasticity.py`, `core/emotional_system.py`, `memory/aria.py`, `memory/hierarchy.py`, `inference/api/server.py`, `inference/engine/dllm_sampler.py`, `memory/__init__.py`.
- **Mort (doublons / jamais importés)** : `api/server.py`, `api/__init__.py`, `agent/orchestrator.py` (avec un `s`), `core/infinite_context.py` (677 LOC fantômes).
- **À tester** : tout le reste. Aucun test = aucune preuve.

**Bruit supprimable** : ≈1 200 LOC de code mort (les 3 fichiers doublons + infinite_context).

---

## 2. Comparaison ARCHITECTURE.md ↔ réalité

ARCHITECTURE.md (111 lignes) annonce ~1 750 LOC propres et modulaires. **Réalité : 4 058 LOC dont ~1 200 de code mort + ~700 dans un fichier non branché.** Le ratio signal/bruit est mauvais.

| Module annoncé dans ARCHITECTURE.md | Statut |
|---|---|
| `core/types.py` (180 LOC) | **MANQUANT** — pas de fichier `types.py`. |
| `memory/aria.py` (200 LOC) | ✅ Existe, 410 LOC (≈2× prévu). |
| `memory/hierarchy.py` (180 LOC) | ✅ Existe, 188 LOC. |
| `reasoning/engine.py` (120 LOC) | **MANQUANT** — pas de sous-dossier `reasoning/`. La logique est dans `core/multi_phase_reasoning.py` (111 LOC). |
| `reasoning/predictive_coding.py` (180 LOC) | **MANQUANT** — Free Energy Principle référencé nulle part. |
| `reasoning/metacognition.py` (120 LOC) | **MANQUANT** — autocritique faite inline dans `agents/orchestrator.py` (~5 LOC). |
| `inference/engine.py` (140 LOC) | **MANQUANT** — la logique est dans `inference/engine/dllm_sampler.py` (114 LOC, mais c'est un faux dLLM). |
| `agents/manager.py` (70 LOC) | **MANQUANT** — `agents/orchestrator.py` joue ce rôle (163 LOC). |
| `tools/registry.py` (80 LOC) | **MANQUANT** — registre d'outils non implémenté. |
| `orchestration/soe_orchestrator.py` (200 LOC) | **MANQUANT** — orchestrateur central non implémenté. `agents/orchestrator.py` fait office. |
| `security/polygone.py` (100 LOC) | **MANQUANT** — sécurité dans `core/polygone_integration.py` (273 LOC) mais pas isolée. |
| `training/dllm.py` (60 LOC) | **MANQUANT** — aucun module d'entraînement. |
| `api/server.py` (120 LOC) | ⚠️ Existe en 2 versions (319 + 457 LOC), conflit d'import. |

**12 modules annoncés sur 13 sont manquants ou mal localisés.** ARCHITECTURE.md est un design idéalisé, pas un miroir du code.

---

## 3. Entry point

- **Aucun `__main__.py` à la racine de `soe/`.**
- **Aucun `main.py`, `app.py`, `cli.py`.**
- **Aucun `pyproject.toml` / `setup.py`.**
- Entry points partiels : `inference/api/server.py` lance uvicorn sous `if __name__ == "__main__"`. C'est le seul.
- `python -m soe` ne marchera pas. `pip install -e .` non plus.

→ Le projet n'est **pas package-able**. Pour le rendre exécutable, il faut au minimum :
1. `soe/__init__.py` (vide suffit)
2. `soe/__main__.py` qui dispatch vers `inference.api.server`
3. `pyproject.toml` minimal

---

## 4. Tests

- **Aucun fichier `test_*.py` / `*_test.py` / `conftest.py`** trouvé dans le repo.
- `.pytest_cache/` existe (répertoire résiduel d'une exécution pytest antérieure, sans source).
- Aucune CI (`.github/workflows/` absent).
- Aucun benchmark (HumanEval, GSM8K, MMLU, French-BLEU — rien).

→ **0 tests, 0 benchmarks, 0 CI.** Critère rédhibitoire pour un paper.

---

## 5. Modèle Qwen2.5-7B-Instruct — état disque

```
soe/models/base/qwen2.5-7b-instruct/
├── config.json                    663 B    ✅
├── generation_config.json         243 B    ✅
├── merges.txt                     1.6 MB   ✅
├── .gitattributes                 1.5 KB   ✅
├── LICENSE                        11 KB    ✅
├── model-00001-of-00004.safetensors   3.95 GB  ✅
├── model-00002-of-00004.safetensors   2.67 GB  ✅
├── model-00003-of-00004.safetensors   —       ❌ MANQUANT
├── model-00004-of-00004.safetensors   —       ❌ MANQUANT
├── model.safetensors.index.json   —       ❌ MANQUANT (obligatoire pour sharded load)
├── tokenizer.json                 —       ❌ MANQUANT
├── vocab.json                     —       ❌ MANQUANT
├── special_tokens_map.json        —       ❌ MANQUANT
└── tokenizer_config.json          —       ❌ MANQUANT
```

**Manque ≈6.5 GB de poids (parts 3 et 4) + tous les fichiers de tokenizer.** `transformers.AutoModelForCausalLM.from_pretrained()` crashera immédiatement.

**Action** : `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir soe/models/base/qwen2.5-7b-instruct --include "*.safetensors" "*.json" "tokenizer*"`. Bloqueur pour toute démo.

---

## 6. Git / repo hygiene

- ✅ `.git/` présent, 1 commit : `afcc1cd SOE-Orret + Polygone: mission long-terme Ruflo démarre`.
- ❌ **`git remote -v` vide** — pas de remote, on ne peut pas push.
- ❌ Pas de `.gitignore` à la racine de `soe/` (existe seulement dans `soe/.gitignore`).
- ⚠️ Les `models/*.safetensors` ne sont PAS dans `.gitignore` mais ne sont PAS trackés (6.6 GB). À exclure explicitement sinon un `git add .` accidentel tuera le repo.
- ❌ Pas de `LICENSE` à la racine.
- ❌ Pas de `README.md`.
- ❌ Pas de `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.

---

## 7. Lacunes pour passer de "projet sympa" à "papier de recherche publiable"

Cinq trous critiques, classés par impact scientifique décroissant :

### 🔴 TROU 1 — Le sampler n'est PAS un vrai dLLM (critique, bloquant)
Le nom `BlockDiffusionSampler` vend de la diffusion masquée. Le code charge `AutoModelForCausalLM` (transformer causal standard) et fait du masque-then-refine en re-passant le modèle N fois. **Ce n'est pas un dLLM**, c'est de l'itération autoregressive avec masque. Un relecteur ICML détruira le papier en 2 minutes.

**Pour publier** : il faut soit (a) fine-tuner Qwen2.5-7B en masked diffusion (LLaDA-style : remplacement de l'objectif de pré-entraînement), soit (b) partir d'un vrai dLLM publié (LLaDA-8B, MDLM, DiffuLLaMA) et l'utiliser tel quel. Sans (a) ou (b), il n'y a pas de papier dLLM.

### 🔴 TROU 2 — Aucun entraînement, aucun fine-tuning
Aucun `training/dllm.py` (annoncé dans ARCHITECTURE.md mais absent). Aucun script d'entraînement. Aucun dataset de fine-tuning. Aucun GPU script. La promesse "dLLM 7B" n'est tenue par aucun artefact reproductible.

**Pour publier** : un script `train_dllm.py` + un dataset (WikiText-103, FineWeb-Edu) + une commande de reproduction + courbes loss.

### 🟠 TROU 3 — Aucun benchmark
Aucun script d'évaluation. Aucune comparaison vs autoregressif baseline. Le paper n'a aucun chiffre.

**Pour publier** : benchmarks standards (lambada, hellaswag, ARC, GSM8K) + benchmarks French (FLUE) si positionnement FR. Tableau comparatif autoregressif vs dLLM avec seuils d'incertitude.

### 🟠 TROU 4 — Pas d'isolation des 4 piliers de l'architecture
ARCHITECTURE.md annonce 4 piliers (SOE-O1, SOE-Orret, Synapse, Format Enchante). **Synapse** (P2P) et **Format Enchante** (compression) sont absents du code. La promesse architecturale n'est pas tenue.

**Pour publier** : soit retirer 2 piliers du papier (focus sur SOE-O1 + SOE-Orret uniquement), soit implémenter des stubs documentés.

### 🟡 TROU 5 — Pas de spécification formelle
Aucun `SPEC.md` à jour. Les guides (104 KB) sont narratifs, pas formels. Pour un paper, il faut une section "Method" précise (équations, hyperparamètres, pseudo-code).

---

## 8. Plan de remédiation (par ordre de leverage pour publication)

| # | Action | Temps | Statut |
|---|---|---|---|
| 1 | Re-télécharger Qwen2.5-7B-Instruct complet (~15 GB) | 30 min | À FAIRE — bloquant démo |
| 2 | Décider vrai dLLM : (a) fine-tuner Qwen2.5-7B en MDLM, ou (b) embarquer LLaDA-8B, ou (c) revendiquer Block-Diffusion-on-AR comme contribution (nécessite ablation sérieuse) | 1 journée réflexion | À FAIRE — décision critique |
| 3 | Implémenter `soe/__main__.py`, `pyproject.toml`, `LICENSE` (MIT), `README.md` | 1 h | À FAIRE |
| 4 | Supprimer code mort (≈1 200 LOC) : `api/server.py`, `api/__init__.py`, `agent/orchestrator.py`, `core/infinite_context.py` (ou le brancher) | 30 min | À FAIRE |
| 5 | Écrire 5–10 tests pytest sur le pipeline émotion→mémoire→réponse | 2 h | À FAIRE |
| 6 | Benchmark HumanEval-50 (ou autre) en mode dLLM vs AR baseline | 4 h | À FAIRE |
| 7 | Rédiger `SPEC.md` formel (équations dLLM, hyperparamètres, ablation) | 1 journée | À FAIRE |
| 8 | Setup remote Git (GitHub), push initial | 30 min | À FAIRE — bloquant |

---

## 9. Ce que ce projet A comme atout (à protéger)

- **Le nom et la vision** : "dLLM symbiotique open-source créé par un dev de 14 ans en France" — c'est une vraie niche. Cartesia et Inception jouent en haut de gamme, personne ne joue sur le segment "laptop + open + PQC + RAG + dLLM".
- **Le code 6-phase reasoning + émotions + plasticité** : c'est un embryon d'agent architecture différenciant. Pas un clone de LangChain.
- **L'intégration Polygone (PQC)** : très peu de projets open-source combinent LLM + PQC en pratique.
- **La cohérence conceptuelle** : 4 piliers qui se tiennent.

→ Ne pas se disperser. **Focus = (vrai) dLLM + benchmarks + paper**. Le reste peut attendre.

---

## 10. États valides (format Hermes)

- **DONE** : audit lu, code-mort identifié, trous critiques listés.
- **PARTIAL** : pas de remote Git, pas de tests, pas exécuté, pas benchmarké.
- **BLOCKED** : démo MVP bloquée par Qwen2.5-7B incomplet + sampler non-dLLM.
- **FAILED** : aucun.
- **NEEDS_RESEARCH** : décision "fine-tuner Qwen vs embarquer LLaDA-8B" — impact scientifique majeur.

---

*Signé Ruflo, mission SOE, 2026-08-20.*
