#!/usr/bin/env python3
"""Complete SOE bootstrap - NO external dependencies except stdlib"""
import os
import sys
import json
import struct
from pathlib import Path
from datetime import datetime
import shutil

SOE_DIR = Path(os.path.expanduser("~/soe"))
RAW_DIR = SOE_DIR / "datasets" / "raw"
PROCESSED_DIR = SOE_DIR / "datasets" / "processed"

# ─────────────────────────────────────────────────────────────
# SIMPLE LOOP FORMAT (no zstandard, just gzip for demo)
# ─────────────────────────────────────────────────────────────

MAGIC = b"LOOP"
FOOTER = b"POOLEND\x00"
HEADER_SIZE = 64

def write_loop_simple(path: str, data: list, metadata: dict):
    """Write simple .loop file"""
    with open(path, "wb") as f:
        # Header
        header = bytearray(HEADER_SIZE)
        header[:4] = MAGIC
        struct.pack_into("<H", header, 4, 1)  # version
        struct.pack_into("<Q", header, 10, len(data))  # n_batches
        f.write(header)
        
        # Build data first to know offsets
        data_bytes = b''.join(d.encode('utf-8')[:200] + b'\x00' for d in data)
        
        # Index (16 bytes per entry)
        index_data = bytearray(len(data) * 16)
        offset = 0
        for i in range(len(data)):
            batch_bytes = data[i].encode('utf-8')[:200]
            struct.pack_into("<QQ", index_data, i * 16, offset, len(batch_bytes))
            offset += len(batch_bytes)
        
        # Write index position then data
        index_offset = f.tell()
        f.write(data_bytes)
        f.write(index_data)
        
        # Footer
        meta_json = json.dumps(metadata)
        meta_bytes = meta_json.encode('utf-8')
        f.write(FOOTER)
        f.write(struct.pack("<Q", len(meta_bytes)))
        f.write(meta_bytes)

def read_loop_simple(path: str) -> dict:
    """Read simple .loop file"""
    with open(path, "rb") as f:
        # Header
        magic = f.read(4)
        if magic != MAGIC:
            return None
        
        version, n_batches = struct.unpack("<HQ", f.read(10))
        
        # Skip to data (simplified - just read everything after header)
        f.seek(HEADER_SIZE + n_batches * 16)
        
        data = []
        for _ in range(n_batches):
            batch = f.read(200).decode('utf-8', errors='ignore').strip('\x00')
            if batch:
                data.append(batch)
        
        # Read metadata
        f.seek(-16, 2)
        footer = f.read(8)
        if footer == FOOTER:
            meta_size = struct.unpack("<Q", f.read(8))[0]
            f.seek(-16 - meta_size, 2)
            metadata = json.loads(f.read(meta_size))
        else:
            metadata = {}
        
        return {"version": version, "batches": data, "metadata": metadata}


# ─────────────────────────────────────────────────────────────
# RUCHE FILTERS (simplified, no numpy/zstandard)
# ─────────────────────────────────────────────────────────────

def quality_check(text: str, config: dict) -> bool:
    """Simple quality check"""
    words = text.split()
    if len(words) < 5:
        return False
    # Length score
    min_words = config.get("min_words", 20)
    return len(words) >= min_words


# ─────────────────────────────────────────────────────────────
# MAIN BOOTSTRAP
# ─────────────────────────────────────────────────────────────

SAMPLE_DATA = {
    "coding_fr": [
        "Python async programming allows concurrent execution using asyncio and await keywords.",
        "FastAPI is a modern Python web framework with automatic OpenAPI documentation generation.",
        "Docker containerization packages applications with dependencies for consistent deployment.",
        "Linux terminal commands: grep for search, awk for processing, sed for editing, find for files.",
        "Git version control: init, add, commit, push, pull. Branch with checkout -b.",
        "REST APIs return JSON data. Use requests library for HTTP calls.",
        "SSH keys provide secure passwordless authentication between machines.",
        "Cron jobs schedule tasks with minute hour day month weekday syntax.",
        "Python virtual environments isolate package dependencies using venv module.",
        "Kubernetes orchestrates containers with pods, services, and deployments."
    ],
    "instructions_fr": [
        "Pour installer Python sur Windows, téléchargez l'installateur depuis python.org.",
        "Créer un environnement virtuel: python -m venv nom puis source bin/activate.",
        "Commandes Linux de base: ls liste les fichiers, cd change de répertoire.",
        "Configurer SSH sur Ubuntu: sudo apt install openssh-server.",
        "Docker: docker run -d -p 80:80 nginx lance un container nginx.",
        "Git: git clone url clone un dépôt, git addステージ, git commit enregistre.",
        "Installer pip: python -m ensurepip ou python -m pip install --upgrade pip.",
        "Updater Python: sudo apt update && sudo apt upgrade python3.",
        "Configurer firewall: sudo ufw allow port ouvre un port.",
        "Monitorer processus: ps aux affiche tous les processus."
    ],
    "linux_sysadmin": [
        "man7.org provides Linux man pages for all command line utilities.",
        "Ubuntu help documentation covers server setup and security configurations.",
        "Arch Wiki is the best resource for Linux troubleshooting and advanced setups."
    ],
    "science_ia_fr": [
        "L'intelligence artificielle模仿 human intelligence through machine learning algorithms.",
        "Les réseaux de neurones sont inspirés du cerveau humain avec couches d neurones.",
        "Le machine learning permet aux ordinateurs d'apprendre à partir de données sans programmation explicite."
    ],
    "wikipedia_fr": [
        "Python est un langage de programmation créé par Guido van Rossum en 1991.",
        "Linux est un système d'exploitation open source créé par Linus Torvalds en 1991.",
        "L'intelligence artificielle est un ensemble de techniques permettant aux machines de simuler l'intelligence humaine."
    ]
}


def run_bootstrap():
    """Run complete SOE bootstrap"""
    print("=" * 50)
    print("SOE BOOTSTRAP v0.1")
    print("=" * 50)
    
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    collected = 0
    
    for category, samples in SAMPLE_DATA.items():
        print(f"\n📦 Collecting: {category}")
        
        config = {"min_words": 5}
        valid_samples = [s for s in samples if quality_check(s, config)]
        
        # Write .loop file
        output = RAW_DIR / f"{category}.loop"
        metadata = {
            "category": category,
            "lang": "fr",
            "source": "bootstrap-v0.1",
            "collected_at": datetime.now().isoformat(),
            "n_examples": len(valid_samples)
        }
        
        write_loop_simple(str(output), valid_samples, metadata)
        
        print(f"  ✅ {len(valid_samples)} examples → {output.name}")
        collected += len(valid_samples)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"✅ BOOTSTRAP COMPLETE!")
    print(f"  Collected: {collected} examples")
    print(f"  Files: {len(SAMPLE_DATA)} categories")
    print(f"  Location: {RAW_DIR}")
    print("=" * 50)
    
    # List files
    print("\nGenerated .loop files:")
    for f in RAW_DIR.glob("*.loop"):
        size = f.stat().st_size
        print(f"  {f.name} ({size} bytes)")
    
    return True


if __name__ == "__main__":
    run_bootstrap()
