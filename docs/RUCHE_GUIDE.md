# RUCHE_GUIDE.md - Data Collection

## Overview
Ruche est l'orchestrateur de collecte de données du projet SOE.

## Configuration
Edit `ruche/ruche_config.yaml`:
```yaml
sources:
  github:
    enabled: true
    topics: [python, machine-learning]
    max_issues: 50
```

## Sources Disponibles
- **github**: Issues GitHub par topic
- **wikipedia**: Articles par catégorie
- **stackoverflow**: Questions par tag
- **arxiv**: Papers par catégorie

## Utilisation
```bash
cd ~/soe
python3 -c "from ruche.ruche import Ruche; Ruche().run()"
```

## Output
Données brutes dans `datasets/raw/*.loop`
