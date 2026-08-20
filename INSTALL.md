# INSTALL.md — SOE-Orret

## Prérequis

- **Python 3.10+** (testé avec 3.13)
- **Ollama** (pour l'inférence locale)
- **~1 GB RAM** (mode 1.5B) ou **~16 GB RAM** (mode 7B complet)
- **Linux / macOS** (Windows via WSL2)

## Installation rapide (mode 1.5B, 5 minutes)

```bash
# 1. Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Télécharger le modèle
ollama pull qwen2.5:1.5b

# 3. Cloner le repo
git clone https://github.com/lvs0/soe-orret
cd soe-orret

# 4. Lancer la démo
python soe/inference.py "Bonjour Orret, qui es-tu ?"
```

## Installation complète (mode 7B, ~30 minutes)

### Étape 1 : Modèle Qwen2.5-7B-Instruct complet

```bash
# Option A : via Ollama (recommandé)
ollama pull qwen2.5:7b-instruct

# Option B : via Hugging Face (transformers, nécessite ~15 GB disque)
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/base/qwen2.5-7b-instruct
```

### Étape 2 : Environnement Python

```bash
# Avec uv (recommandé)
uv venv
uv pip install -r requirements.txt

# Ou avec pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Étape 3 : Lancer l'API REST

```bash
# Mode Ollama (recommandé pour laptop)
USE_OLLAMA=true OLLAMA_MODEL=qwen2.5:7b-instruct \
  python soe/inference/api/server.py

# L'API est accessible sur http://localhost:8080
# Documentation : http://localhost:8080/docs
```

### Étape 4 : Tester

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orret-dllm-7b",
    "messages": [{"role": "user", "content": "Explique la cryptographie post-quantique en 2 phrases."}]
  }'
```

## Mode dLLM complet (Fast-dLLM v2, en développement)

```bash
# Télécharger les poids Fast-dLLM v2
hf download Efficient-Large-Model/Fast_dLLM_v2_7B --local-dir models/fast-dllm-v2

# Installer les dépendances GPU
uv pip install torch transformers accelerate

# Lancer avec le sampler block diffusion
ORRET_MODEL_PATH=models/fast-dllm-v2 python soe/inference/api/server.py
```

## Dépannage

| Problème | Solution |
|---|---|
| "Ollama injoignable" | `ollama serve` dans un autre terminal |
| "Modèle non trouvé" | `ollama pull qwen2.5:1.5b` |
| "No space left on device" | Libérer de l'espace disque (le modèle 7B fait ~15 GB) |
| "CUDA out of memory" | Mode CPU uniquement : `CUDA_VISIBLE_DEVICES=""` |
| ImportError: transformers | `uv pip install transformers torch` |
