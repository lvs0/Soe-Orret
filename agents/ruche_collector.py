#!/usr/bin/env python3
"""RUCHE-COLLECTOR - Data collection agent"""
import os
import sys
import yaml
import asyncio
from pathlib import Path
from datetime import datetime
from ruche.filters.quality import filter_by_quality
from ruche.filters.dedup import DedupFilter
from core.looplib.writer import LoopWriter, LoopBatch
import numpy as np

SOE_DIR = Path(os.path.expanduser("~/soe"))
RAW_DIR = SOE_DIR / "datasets" / "raw"

class RucheCollector:
    """Collect data for categories"""
    
    def __init__(self):
        self.config = self._load_config()
        self.dedup = DedupFilter(threshold=0.90)
        self.stats = {"collected": 0, "errors": []}
    
    def _load_config(self):
        config_path = SOE_DIR / "ruche" / "ruche_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def collect(self, category: str) -> bool:
        """Collect data for one category"""
        cat_config = self.config["categories"].get(category)
        if not cat_config:
            print(f"Unknown category: {category}")
            return False
        
        # TODO: Replace with real scrapers
        # For now, use placeholder data
        samples = self._get_samples(category)
        
        output = RAW_DIR / f"{category}.loop"
        writer = LoopWriter(str(output), {
            "category": category,
            "lang": cat_config.get("lang", "fr"),
            "source": "ruche-collector",
            "collected_at": datetime.now().isoformat()
        })
        
        for text in samples:
            result = filter_by_quality(text, cat_config)
            if result["accept"] and self.dedup.check(text):
                # Simple tokenization (first N chars as IDs)
                ids = np.array([ord(c) % 65536 for c in text[:100]], dtype=np.int32)
                mask = np.ones_like(ids)
                writer.add_batch(LoopBatch(ids, ids, mask))
                self.stats["collected"] += 1
        
        writer.save()
        print(f"Collected {self.stats['collected']} examples for {category}")
        return True
    
    def _get_samples(self, category: str) -> list:
        """Get sample data - replace with real scrapers"""
        samples = {
            "coding_fr": [
                "Python async with asyncio enables concurrent programming. Use async def for coroutines and await to suspend execution.",
                "FastAPI is a modern Python web framework with automatic OpenAPI docs. It supports dependency injection and type hints.",
                "Docker containers package applications with dependencies. Use 'docker build' to create images, 'docker run' to start containers.",
                "Linux shell: grep for search, awk for processing, sed for editing, find for files. Pipes connect commands.",
                "Git: git init, git add, git commit, git push, git pull. Branch with git checkout -b.",
                "REST APIs return JSON. Use requests library: requests.get(url, headers=headers).",
                "SSH keys for passwordless login: ssh-keygen creates pair, add public key to remote ~/.ssh/authorized_keys.",
                "Cron syntax: minute hour day month weekday. '0 2 * * *' = 2 AM daily.",
                "Python virtual environments: python -m venv env, source env/bin/activate.",
                "Kubernetes: kubectl for manage pods, services, deployments. YAML config files."
            ],
            "instructions_fr": [
                "Pour installer Python sur Windows, téléchargez depuis python.org et lancez l'installateur.",
                "Créer un environnement virtuel: python -m venv nom_environnement puis activation.",
                "Commandes Linux de base: ls (liste), cd (répertoire), mkdir (dossier), rm (supprimer).",
                "Configurer SSH sur Ubuntu: sudo apt install openssh-server, puis modifier /etc/ssh/sshd_config.",
                "Docker: docker run -d -p 80:80 nginx pour lancer un container.",
                "Git: git clone url, git add fichier, git commit -m 'message', git push origin main.",
                "Installer pip: python -m ensurepip ou python -m pip install --upgrade pip.",
                "Updater Python: sudo apt update && sudo apt upgrade python3.",
                "Configurer firewall: sudo ufw allow 22, sudo ufw enable.",
                "Monitorer processus: ps aux, top, htop."
            ]
        }
        return samples.get(category, ["Sample text."])


def main():
    """CLI entry"""
    if len(sys.argv) < 2:
        # Collect all
        collector = RucheCollector()
        for cat in collector.config["categories"]:
            collector.collect(cat)
    else:
        # Collect specific
        collector = RucheCollector()
        collector.collect(sys.argv[1])
    
    print(f"\n✅ Collection complete: {collector.stats}")


if __name__ == "__main__":
    main()
