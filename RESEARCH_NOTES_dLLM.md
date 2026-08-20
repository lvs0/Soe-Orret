# RESEARCH NOTES — dLLM (Diffusion Language Models)
## État de l'art 2025-2026 — niveau ingénieur/chercheur

**Date** : 2026-08-20
**Auteur** : Ruflo (mission SOE)
**Contexte** : SOE-Orret — construction d'un dLLM 7B open-source sur Qwen2.5

---

## 1. Le tournant dLLM : pourquoi c'est le moment

Les modèles de langage autoregressifs (AR) dominent depuis GPT-2 (2019). Mais 2024-2026 a vu une accélération majeure des modèles de diffusion pour le texte, portée par :

1. **LLaDA-8B (février 2025)** — preuve que la diffusion scale à 8B params et rivalise avec LLaMA3-8B
2. **Fast-dLLM v2 (septembre 2025)** — adaptation AR→dLLM en 1B tokens (500× moins que Dream)
3. **Inception Labs (janvier 2026)** — $50M levés, Mercury à 1 000+ tok/s, validation commerciale
4. **MDLM (NeurIPS 2024)** — fondations théoriques : paramétrisation SUBS qui simplifie la loss

**Le message du champ** : la diffusion n'est plus une curiosité académique. C'est une alternative crédible à l'autoregression, avec des avantages structurels que l'AR ne peut pas rattraper.

---

## 2. Papers fondateurs (ordre chronologique, avec analyse technique)

### 2.1 MDLM — Masked Diffusion Language Models (NeurIPS 2024)
**Auteurs** : Sahoo et al. (Cornell)
**arXiv** : 2406.07524
**Contribution clé** : paramétrisation SUBS (substitution-based) qui transforme la loss de diffusion en un mélange de MLM losses classiques. Plus simple à implémenter, plus stable à entraîner que SEDD (score-entropy).

**Chiffres** :
- Zero-shot perplexity : MDLM s'approche de l'AR sur PTB, WikiText, LM1B, Lambada
- Pas encore au niveau AR, mais l'écart se réduit
- Code open-source : github.com/kuleshov-group/mdlm

**Pertinence SOE-Orret** : MDLM est la base théorique la plus propre pour un dLLM from-scratch. Si on part de zéro (pas d'adaptation AR→dLLM), c'est le papier à implémenter.

---

### 2.2 DiffuLLaMA / DiffuGPT (ICLR 2025)
**Auteurs** : Gong et al. (HKU)
**arXiv** : 2410.17891
**Contribution clé** : adaptation de modèles AR existants (LLaMA, GPT) en modèles de diffusion. Preuve que le transfert de poids AR→dLLM marche.

**Chiffres** :
- Moins de tokens d'entraînement que from-scratch
- Performance compétitive sur benchmarks de raisonnement
- Mais : pas encore au niveau SOTA AR

**Pertinence SOE-Orret** : approche la plus pragmatique pour nous. On a Qwen2.5-7B-Instruct sur disque. Fast-dLLM v2 a montré que l'adaptation AR→dLLM en 1B tokens est possible. C'est la voie à suivre.

---

### 2.3 LLaDA-8B — Large Language Diffusion Models (février 2025)
**Auteurs** : Nie, Zhu et al. (Peking Univ / Microsoft / StepFun)
**arXiv** : 2502.09992
**Contribution clé** : premier dLLM from-scratch à 8B params qui rivalise avec LLaMA3-8B.

**Chiffres clés** :
- 8B params, entraîné from scratch
- Surpasse GPT-4o sur le "reversal curse" (poème inversé)
- Zero/few-shot benchmarks : compétitif avec LLaMA3-8B
- LLaDA-MoE-7B-A1B (sept 2025) : version MoE, 7B capacité, 1.4B actifs → coût inférence réduit

**Pourquoi c'est important** : LLaDA a brisé le plafond psychologique. Avant, le consensus était "la diffusion pour le texte, ça marche pas au-dessus de 1B". LLaDA-8B a prouvé le contraire.

**Pertinence SOE-Orret** : c'est le concurrent direct. Si on sort un dLLM 7B, la comparaison obligatoire sera avec LLaDA-8B. Il faut soit le battre sur un axe (latence, French, PQC, edge), soit trouver un positionnement différenciant.

---

### 2.4 Fast-dLLM v2 (septembre 2025) — LA RÉFÉRENCE POUR SOE-ORRET
**Auteurs** : Wu, Zhang et al. (NVIDIA / MIT / Tsinghua)
**arXiv** : 2509.26328
**Contribution clé** : adaptation d'un modèle AR pré-entraîné (Qwen2.5-7B-Instruct) en dLLM avec SEULEMENT 1B tokens de fine-tuning. C'est 500× moins que Dream (580B tokens).

**Architecture** :
- Block diffusion : génère K tokens par bloc (pas token par token)
- KV-cache réutilisé entre les étapes de diffusion
- Adaptation du modèle causal → bidirectionnel via masque d'attention modifié
- Méthode A2D (AR-to-Diffusion) : fine-tuning avec objectif de masked diffusion

**Chiffres** :
- **Qwen2.5-7B-Instruct → Fast-dLLM v2 7B**
- 1B tokens de fine-tuning (coût GPU accessible)
- Match ou surpasse AR baseline en accuracy
- SOTA efficacité parmi les dLLMs
- **Code + poids sur Hugging Face** : Efficient-Large-Model/Fast_dLLM_v2_7B

**Pertinence SOE-Orret** : C'EST EXACTEMENT CE QU'ON VEUT FAIRE. Même modèle de base (Qwen2.5-7B-Instruct), même approche (adaptation AR→dLLM). Fast-dLLM v2 est le chemin le plus court vers un MVP crédible.

**Plan d'action** :
1. Télécharger les poids Fast-dLLM v2 7B sur Hugging Face (ils sont publics)
2. Utiliser leur code d'inférence comme base
3. Ajouter notre couche agent (émotions, mémoire, raisonnement multi-phase)
4. Positionner SOE-Orret comme "Fast-dLLM v2 + agent architecture + PQC"

---

### 2.5 Dream 7B / Dream-Coder-7B (2025)
**Auteurs** : Ye et al. (ByteDance / Tsinghua)
**Contribution** : dLLM from-scratch avec full-attention diffusion. 580B tokens de pré-entraînement. Coût GPU très élevé.

**Pertinence SOE-Orret** : Pas pertinent pour nous (coût d'entraînement prohibitif). Mais intéressant comme baseline : Fast-dLLM v2 les bat avec 500× moins de données.

---

### 2.6 MMaDA — Multimodal Large Diffusion Language Models (NeurIPS 2025)
**Auteurs** : Gen-Verse (Princeton / Peking / Tsinghua / ByteDance)
**arXiv** : 2505.15809
**Contribution** : premier modèle de diffusion multimodal (texte + image) unifié. Raisonnement textuel, compréhension multimodale, génération texte→image dans une seule architecture de diffusion discrète.

**Pertinence SOE-Orret** : vision long-terme. Pour l'instant, focus texte. Mais MMaDA montre que la diffusion peut être multimodale nativement — un avantage sur l'AR.

---

## 3. Comparaison technique : dLLM vs Autoregressif

| Propriété | Autoregressif (GPT-style) | Diffusion masquée (dLLM) |
|---|---|---|
| **Direction** | Gauche→droite (causal) | Bidirectionnelle (chaque token voit tous les autres) |
| **Génération** | Séquentielle (1 token/step) | Parallèle par bloc (K tokens/step) |
| **Révision** | Impossible (tokens fixés) | Possible (tokens low-confidence re-masqués) |
| **Reversal Curse** | Oui (ne peut pas raisonner droite→gauche) | Non (contexte bidirectionnel) |
| **Latence** | O(N) en nombre de tokens | O(N/K) avec K=block_size |
| **Perplexité** | SOTA actuel | ~95-98% du SOTA AR (LLaDA, Fast-dLLM v2) |
| **Inférence** | 1 forward pass / token | N_steps forward passes / bloc |
| **Fine-tuning** | Mature (LoRA, QLoRA) | Émergent (A2D, adaptation AR→dLLM) |
| **Écosystème** | Dominant (vLLM, TGI, Ollama) | Naissant (pas encore de serving optimisé) |

### Avantage clé du dLLM pour SOE-Orret

1. **Bidirectionnel** → meilleure compréhension de contexte long. Critique pour notre agent 6-phase.
2. **Révisable** → l'autocritique peut littéralement re-générer des parties de la réponse. Pas juste "recommencer".
3. **Parallèle** → sur un laptop (X250), la parallélisation par bloc peut compenser l'absence de GPU.
4. **Niche académique** → moins de concurrence que l'AR. Un papier dLLM 7B avec agent architecture est plus publiable qu'un énième fine-tune LLaMA.

---

## 4. Concurrents startups (2025-2026)

### 4.1 Inception Labs (Palo Alto)
- **CEO** : Stefano Ermon (Stanford, ex-cofondateur de...)
- **Financement** : $50M (Menlo Ventures, 2026)
- **Produit** : Mercury / Mercury 2 — dLLM propriétaire, 1 000+ tok/s
- **Positionnement** : "10x faster than GPT-4 at same quality"
- **Statut** : closed-source, API-only
- **Différence SOE-Orret** : on est open-source, edge/laptop, PQC. Pas en compétition directe.

### 4.2 Cartesia (San Francisco)
- **CEO** : Karan Goel (ex-Stanford, co-auteur de Mamba/SSM)
- **Financement** : ~$27M (Index Ventures, 2024)
- **Produit** : Sonic (TTS temps réel), Rene (modèle SSM, pas diffusion)
- **Note** : Cartesia fait du state-space (Mamba), pas du dLLM. Leur nom revient dans les discussions "alternatives à l'AR" mais c'est une techno différente.
- **Pertinence** : voisin conceptuel, pas concurrent direct.

### 4.3 Positionnement SOE-Orret

```
                    Propriétaire      Open-source
                    ───────────       ───────────
Haute performance   Inception Labs    LLaDA-8B
(cloud/GPU)         Mercury 2         Fast-dLLM v2
                    ─────────────────────────────
Edge/laptop         —                 SOE-Orret ← NOUS
(PQC + agent)                         (niche vide)
```

**Notre niche** : dLLM open-source sur laptop, avec agent architecture (émotions, mémoire, raisonnement) et sécurité PQC intégrée. Personne ne fait ça.

---

## 5. Implications pour SOE-Orret — décision technique

### Option A : Partir de Fast-dLLM v2 (RECOMMANDÉ)
- Télécharger les poids `Fast_dLLM_v2_7B` sur Hugging Face
- Utiliser leur code d'inférence (block diffusion, KV-cache)
- Ajouter notre couche agent (émotions, mémoire, raisonnement)
- **Avantage** : MVP en 1 semaine, crédibilité immédiate ("basé sur Fast-dLLM v2, publié à NeurIPS 2025")
- **Inconvénient** : dépendance à un modèle tiers

### Option B : Fine-tuner Qwen2.5-7B nous-mêmes en A2D
- Implémenter la méthode Fast-dLLM v2 (A2D) sur Qwen2.5-7B-Instruct
- Fine-tuner avec 1B tokens (coût GPU : ~$500-2000 sur cloud)
- **Avantage** : contrôle total, contribution originale
- **Inconvénient** : temps, coût GPU, risque d'échec

### Option C : From-scratch MDLM
- Implémenter MDLM from scratch à 1B ou 7B
- **Avantage** : contribution maximale, papier publiable
- **Inconvénient** : 6-12 mois, coût GPU massif, hors scope MVP

**Recommandation** : **Option A pour le MVP, Option B pour la v2.** Fast-dLLM v2 est exactement ce qu'on veut — autant l'utiliser comme base et se concentrer sur ce qui nous différencie (agent architecture, PQC, edge deployment).

---

## 6. Prochaines étapes recherche

1. [ ] Cloner `Efficient-Large-Model/Fast_dLLM_v2_7B` sur Hugging Face
2. [ ] Tester l'inférence block diffusion sur Qwen2.5-7B-Instruct local
3. [ ] Mesurer latence (tok/s) vs AR baseline sur le même hardware
4. [ ] Intégrer le sampler dLLM dans `inference/engine/dllm_sampler.py`
5. [ ] Benchmark HumanEval ou French-BLEU
6. [ ] Rédiger section "Related Work" pour le papier

---

## 7. Références clés (format BibTeX-ready)

```
@article{sahoo2024mdlm,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and others},
  journal={NeurIPS},
  year={2024}
}

@article{nie2025llada,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and others},
  journal={arXiv:2502.09992},
  year={2025}
}

@article{wu2025fastdllmv2,
  title={Fast-dLLM v2: Efficient Block-Diffusion LLM},
  author={Wu, Chengyue and Zhang, Hao and others},
  journal={arXiv:2509.26328},
  year={2025}
}

@article{gong2024diffullama,
  title={Scaling Diffusion Language Models via Adaptation from Autoregressive Models},
  author={Gong, Z and others},
  journal={ICLR},
  year={2025}
}

@article{genverse2025mmada,
  title={MMaDA: Multimodal Large Diffusion Language Models},
  author={Gen-Verse and others},
  journal={NeurIPS},
  year={2025}
}
```

---

*Signé Ruflo, mission SOE, 2026-08-20.*
