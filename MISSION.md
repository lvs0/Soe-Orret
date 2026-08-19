# SOE-Orret + Polygone — Mission Long-Terme (Ruflo)

## 🎯 Objectif (réaliste)
Transformer SOE-Orret et Polygone en **projets de recherche open-source légitimes** qui :
1. Ont un repo propre avec démo qui marche
2. Sont publiés sur arXiv / Hugging Face
3. Sont couverts par la presse tech FR (Numerama, Next INpact, Les Échos Tech)
4. Reçoivent 1+ collaboration recherche ou financement (< €500K)
5. Sont cités par d'autres projets ou chercheurs

**Pas l'objectif** : rivaliser avec Mistral/AI21/Cohere (impossible sans compute).
**L'objectif** : devenir un **joueur reconnu** dans le niche dLLM-PQC-FR.

---

## 📋 TASKS (loop jusqu'à cancel)

### 1. AUDIT SOE-Orret
- [ ] Lire tous les fichiers de ~/Projets/SOE/
- [ ] Identifier ce qui marche (code exécutable ?)
- [ ] Identifier ce qui manque (spec, bench, démo, doc)
- [ ] Écrire AUDIT_REPORT.md
- [ ] Commit + push

### 2. AUDIT Polygone
- [ ] Lire ~/Projets/Polygone-v2/
- [ ] Vérifier 109 tests passent
- [ ] Identifier niveau PQC (Kyber, Dilithium, sphincs+)
- [ ] Identifier ce qui manque vs standards NIST
- [ ] Écrire AUDIT_REPORT.md
- [ ] Commit + push

### 3. RECHERCHE — dLLM (2025-2026)
- [ ] LLaDA (LLaDA-8B, 2025) — paper + code
- [ ] Fast-dLLM v2 (RND1, 2025) — méthode A2D
- [ ] MDLM (Masked Diffusion LM, 2023-2025)
- [ ] DiffuLLaMA, MMaDA, etc.
- [ ] Concurrent startups : Cartesia, Inception, etc.
- [ ] Écrire RESEARCH_NOTES_dLLM.md
- [ ] Commit + push

### 4. RECHERCHE — PQC
- [ ] NIST PQC standards (FIPS 203/204/205)
- [ ] Kyber (ML-KEM), Dilithium (ML-DSA), sphincs+ (SLH-DSA)
- [ ] Libs : liboqs, Open Quantum Safe
- [ ] Concurrent : Nillion, Zama, Sunscreen
- [ ] Écrire RESEARCH_NOTES_PQC.md
- [ ] Commit + push

### 5. PRODUCTION SOE-Orret MVP
- [ ] Setup Python env (conda/uv)
- [ ] Charger Qwen2.5-7B-Instruct via Ollama
- [ ] Documenter conversion A2D minimale
- [ ] Écrire inference loop dLLM
- [ ] 1 démo CLI : `soe generate "..."`
- [ ] 1 benchmark HumanEval-50 ou French-BLEU
- [ ] Dockerfile
- [ ] Commit + push

### 6. PRODUCTION Polygone
- [ ] Vérifier intégration Kyber/Dilithium
- [ ] 1 démo réseau local
- [ ] Bench vs libs existantes (liboqs)
- [ ] Documentation protocole
- [ ] Commit + push

### 7. BRANDING
- [ ] Domaine soe-orret.ai (à acheter ou réserver)
- [ ] Domaine polygone.network
- [ ] Landing page (HTML/CSS statique)
- [ ] Logo sobre (text-based, pas d'images)
- [ ] Bio + contact + links

### 8. PUBLICATION
- [ ] Rédiger papier technique SOE-Orret (5-10 pages)
- [ ] Rédiger papier technique Polygone (5-10 pages)
- [ ] Soumettre arXiv (cs.AI, cs.CR)
- [ ] Upload modèles sur Hugging Face

### 9. COMMUNAUTÉ
- [ ] Post "Show HN: SOE-Orret – open-source dLLM on a laptop"
- [ ] Post r/MachineLearning
- [ ] Post r/LocalLLaMA
- [ ] Thread Twitter/Mastodon
- [ ] Email à 5-10 chercheurs PQC + dLLM

### 10. STRATÉGIE PRESSE
- [ ] Liste journalistes FR/EN tech (Numerama, Next INpact, Les Échos, Wired, The Verge, Ars Technica)
- [ ] Templates email/pitch
- [ ] DM templates
- [ ] Stratégie "lancement" (date, jour, heure)

---

## 📁 LIVRABLES

- `~/Projets/SOE/AUDIT_REPORT.md`
- `~/Projets/Polygone-v2/AUDIT_REPORT.md`
- `~/Projets/SOE/RESEARCH_NOTES_dLLM.md`
- `~/Projets/SOE/RESEARCH_NOTES_PQC.md`
- `~/Projets/SOE/MVP/` (code qui marche)
- `~/Projets/SOE/paper.tex` (papier arXiv)
- `~/Projets/Polygone-v2/paper.tex`
- `~/Projets/SOE/landing/` (HTML)
- `~/Projets/SOE/STATE.md` (progression, mise à jour à chaque tick)

---

## 📊 RAPPORTS

- **Quotidien** : STATE.md mis à jour (lu par toi à la demande)
- **Hebdomadaire** : message dans Telegram Home channel "Mission SOE — semaine X"
- **Sur demande** : tu peux me demander "Ruflo, où t'en es ?" et je te ponds un recap

---

## 🚀 KICKOFF

- **Démarrage** : 20 août 2026
- **Premier livrable** : AUDIT_REPORT.md SOE-Orret sous 4h
- **Cancel** : `ruflo cancel mission-soe` ou tu me dis "stop"

---

## 💡 PRINCIPES

1. **Pas de bullshit** : chaque livrable est un truc qui marche, pas un claim.
2. **Pas de scope creep** : reste focus MVP, pas de fancy features.
3. **Commit early, commit often** : chaque tick = un commit.
4. **Backup quotidien** : STATE.md + git.
5. **Tu restes le patron** : tu peux me dire d'arrêter, de pivoter, de focus.

---

*"Ce qui est maintenant prouvé n'était autrefois qu'imaginé." — William Blake*
