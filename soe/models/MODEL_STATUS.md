# Model Status — Qwen2.5-7B-Instruct (INCOMPLET)

**Date** : 2026-08-20

## État disque

```
soe/models/base/qwen2.5-7b-instruct/
├── config.json                    663 B    ✅
├── generation_config.json         243 B    ✅
├── merges.txt                     1.6 MB   ✅
├── .gitattributes                 1.5 KB   ✅
├── LICENSE                        11 KB    ✅
├── model-00001-of-00004.safetensors   3.95 GB  ✅
├── model-00002-of-00004.safetensors   2.67 GB  ✅
├── model-00003-of-00004.safetensors   —       ❌ MANQUANT (~3.5 GB)
├── model-00004-of-00004.safetensors   —       ❌ MANQUANT (~3.0 GB)
├── model.safetensors.index.json   27 KB    ✅
├── tokenizer.json                 7.0 MB   ✅
├── vocab.json                     2.8 MB   ✅
├── special_tokens_map.json        —       ❌ MANQUANT (récupérable depuis config)
└── tokenizer_config.json          7.3 KB   ✅
```

**Manque ≈6.5 GB de poids (parts 3 et 4) + special_tokens_map.json.**

## Pourquoi pas téléchargé maintenant

- Disque `/home` : 5.3 GB libres au moment de l'audit (98 % utilisé).
- Shards 3-4 : ~6.5 GB à télécharger → impossible sans libérer de l'espace.
- Décision : **ne pas tenter le download** (règle mission : "STOP si disque < 500 MB" ; ici on est à 5.3 GB mais c'est insuffisant pour 6.5 GB + marge).
- Fallback : `inference.py` tourne avec Ollama `qwen2.5:1.5b` (986 MB, déjà installé).

## Pour finaliser plus tard

```bash
# 1. Libérer ~3 GB de disque (ex: purger les caches pip, .venv inutiles)
# 2. Relancer le download :
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir soe/models/base/qwen2.5-7b-instruct \
  --include "*.safetensors" "*.json" "tokenizer*"
# 3. Tester :
python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('soe/models/base/qwen2.5-7b-instruct')"
```

## Note honnête

Le sampler `BlockDiffusionSampler` (`soe/inference/engine/dllm_sampler.py`) charge
`AutoModelForCausalLM` (autoregressif standard), pas un vrai dLLM. Même une fois le
modèle 7B complet, ce n'est pas un dLLM — c'est de l'itération AR avec masque.
Voir `RESEARCH_NOTES_dLLM.md` pour la roadmap vers un vrai dLLM
(Fast-dLLM v2 / LLaDA-8B / MDLM).
