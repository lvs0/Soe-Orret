# RAPPORT HEBDOMADAIRE — Mission SOE-Orret + Polygone
## Semaine du 20 août 2026

**Rédigé par** : Ruflo (agent mission SOE)
**Date** : 2026-08-20

---

## Résumé

Mission lancée et 7/8 tâches majeures complétées en une session. Les deux audits sont écrits, les deux états de l'art sont documentés, le MVP SOE-Orret tourne (Ollama qwen2.5:1.5b, 3 prompts testés), les repos sont propres, la landing page est prête.

---

## Ce qui a été fait

### ✅ AUDIT_SOE.md
- 16 fichiers Python (4 058 LOC) inventoriés et classifiés
- Code mort identifié : ~1 200 LOC (3 fichiers doublons + infinite_context non branché)
- **Lacune critique #1** : `BlockDiffusionSampler` n'est PAS un vrai dLLM (utilise `AutoModelForCausalLM`, pas de diffusion)
- **Lacune critique #2** : Qwen2.5-7B sur disque INCOMPLET (2/4 shards, pas de tokenizer)
- 5 trous critiques listés pour passer de "projet sympa" à "papier de recherche publiable"
- 0 tests, 0 benchmarks, 0 CI, pas de remote Git

### ✅ AUDIT_POLYGONE.md
- **162 tests passent, 0 failures** (plus que les 109 revendiqués)
- ML-KEM-1024 (FIPS 203) ✅ — `crates/core/src/crypto/kem.rs`
- ML-DSA-65 (FIPS 204) ✅ — `crates/core/src/crypto/sign.rs`
- SPHINCS+ (FIPS 205) ❌ — non implémenté
- Shamir SS (4-of-7), AES-256-GCM, BLAKE3, Zeroize — tout testé
- Remote Git configuré : `https://github.com/lvs0/Polygone-Network`
- Code Rust propre, workspace cargo, licence AGPL-3.0

### ✅ RESEARCH_NOTES_dLLM.md
- Papers couverts : MDLM (NeurIPS 2024), DiffuLLaMA (ICLR 2025), LLaDA-8B (2025), Fast-dLLM v2 (NeurIPS 2025), Dream 7B, MMaDA (NeurIPS 2025)
- **Fast-dLLM v2 identifié comme la référence pour SOE-Orret** : même modèle de base (Qwen2.5-7B-Instruct), même approche (A2D), poids publics sur Hugging Face
- Comparaison technique dLLM vs AR (bidirectionnel, parallèle, révisable)
- Concurrents : Inception Labs ($50M, Mercury), Cartesia (SSM, pas dLLM)
- Positionnement SOE-Orret : niche "dLLM open-source edge + agent + PQC"

### ✅ RESEARCH_NOTES_PQC.md
- Standards NIST FIPS 203/204/205 documentés (publiés août 2024)
- Librairies : liboqs (C), pqcrypto (Rust, utilisé par Polygone)
- Concurrents : Nillion (MPC, $25M), Zama (FHE, $73M), Sunscreen (FHE compiler)
- Positionnement Polygone : SEUL protocole P2P open-source avec KEM + DSA + Shamir fragmentation
- Synergie SOE-Orret + Polygone : agent IA autonome avec identité cryptographique PQC

### ✅ Démo MVP SOE-Orret
- `soe/inference.py` écrit (170 LOC) : interface Ollama + agent OrretMini
- Testé avec 3 prompts sur qwen2.5:1.5b (Ollama) :
  1. "Qu'est-ce que SOE-Orret ?" → ✅ Réponse cohérente
  2. "Explique le principe de la diffusion masquée" → ✅ Réponse détaillée (139 tokens)
  3. "Why is post-quantum cryptography important for AI agents?" → ✅ Réponse en anglais
- Vitesse : 0.7–6.0 tok/s sur CPU (Lenovo X250)
- Bloqueur identifié : disque plein (1 GB libre), modèle 7B incomplet

### ✅ Repos propres
- `.gitignore` créé (exclut models/, safetensors, caches)
- `README.md` complet (4 500 bytes) : badges, pourquoi, architecture, quick start, citation BibTeX
- `INSTALL.md` : 2 modes (1.5B rapide, 7B complet), dépannage
- `LICENSE` : MIT

### ✅ Landing page
- `index.html` : design dark tech, responsive
- Sections : Hero, Problème/Solution, Architecture (diagramme ASCII stylisé), Features, État du projet (table), Démarrage rapide, Recherche, Footer

---

## Bloqueurs

| Bloqueur | Impact | Solution |
|---|---|---|
| Disque plein (1 GB libre) | Ne peut pas télécharger Qwen2.5-7B complet | Libérer espace ou utiliser disque externe |
| Sampler actuel = faux dLLM | Bloquant pour publication | Utiliser Fast-dLLM v2 comme base |
| Pas de remote Git sur SOE | Ne peut pas push | Créer repo GitHub `lvs0/soe-orret` |
| Pas de GPU | Lent (0.7 tok/s sur 1.5B) | Ollama + CPU acceptable pour MVP |

---

## Prochaines étapes (semaine prochaine)

1. **Libérer espace disque** (supprimer caches, vieux modèles, ou utiliser /mnt/transcend)
2. **Télécharger Fast-dLLM v2 7B** depuis Hugging Face → remplacer le sampler
3. **Benchmark** : comparer AR baseline vs dLLM sur HumanEval-50 ou French-BLEU
4. **Écrire 5-10 tests** pytest sur le pipeline agent
5. **Créer repo GitHub** + push initial
6. **Rédiger paper** (5-10 pages, format ICML/NeurIPS)
7. **Ajouter SPHINCS+** dans Polygone (FIPS 205, ~150 LOC)

---

## Métriques

- **Fichiers créés** : 10 (AUDIT_SOE.md, AUDIT.md, RESEARCH_NOTES_dLLM.md, RESEARCH_NOTES_PQC.md, inference.py, .gitignore, README.md, INSTALL.md, LICENSE, index.html)
- **Commits** : 6
- **Tests Polygone** : 162 passants / 0 failures
- **Lignes écrites** : ~8 000 (audits + research notes + code + docs)
- **Modèle testé** : qwen2.5:1.5b (986 MB, Ollama)

---

*Prochaine mise à jour : après téléchargement du modèle 7B et benchmark.*
