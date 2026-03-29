# SOE — Système Orret

>Système d'IA open source, local-first, construit from scratch.

## 📦 Modules

| Module | Description |
|--------|-------------|
| `core/looplib` | Format .loop - données binaires columnar |
| `ruche` | Pipeline de collecte de données |
| `models/orret` | Modèles Orret fine-tunés |
| `agents` | Sous-agents autonomes |

## 🚀 Démarrage Rapide

```bash
# Créer un fichier .loop
from soe.core.looplib import LoopWriter
writer = LoopWriter("data.loop", {"category": "coding", "lang": "fr"})
writer.add_batch(tokens, labels, mask)
writer.save()
```

## 📊 Statut

- **Phase**: Bootstrap
- **Dataset**: En cours de création
- **Modèle**: Orret v0.1 à entraîner

## 🌐 Ressources

- SOE Brand Bible: `http://192.168.1.104:8081`
- MeshCloud: `http://192.168.1.104:11440`
- Ollama: `http://localhost:11434`

---
*SOE — L'intelligence artificielle doit être libre.*
