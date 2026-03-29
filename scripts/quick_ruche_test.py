#!/usr/bin/env python3
"""Quick Ruche test - generate sample .loop file"""
import sys
sys.path.insert(0, '/home/l-vs/soe')

from core.looplib.writer import LoopWriter, LoopBatch
from ruche.filters.quality import filter_by_quality
from ruche.filters.dedup import DedupFilter
import numpy as np
import yaml

# Load config
with open('/home/l-vs/soe/ruche/ruche_config.yaml') as f:
    config = yaml.safe_load(f)

cat = config['categories']['coding_fr']

# Sample data (mock - replace with real scrapers)
samples = [
    "Python async programming allows for concurrent execution using asyncio. The await keyword suspends execution until the coroutine completes.",
    "FastAPI is a modern Python web framework that automatically generates OpenAPI documentation. It supports dependency injection and async responses.",
    "Docker containerization packages applications with their dependencies. Build with 'docker build', run with 'docker run', and manage with 'docker compose'.",
    "Linux terminal commands: grep for search, awk for text processing, sed for stream editing, and find for file search.",
    "Machine learning libraries in Python include scikit-learn for traditional ML, TensorFlow and PyTorch for deep learning.",
    "Git version control: git init to start, git add to stage, git commit to save, git push to upload, git pull to download changes.",
    "REST APIs return JSON data. Use requests.get() for GET, requests.post() for POST, with headers for authentication.",
    "Virtual environments isolate Python packages: python -m venv env creates one, source env/bin/activate activates it.",
    "SSH keys provide secure authentication. Generate with ssh-keygen, add to ~/.ssh/authorized_keys for passwordless login.",
    "Cron jobs schedule tasks: '0 2 * * *' runs at 2 AM daily. Use crontab -e to edit the schedule."
]

# Setup
dedup = DedupFilter(threshold=0.90)
writer = LoopWriter('/home/l-vs/soe/datasets/raw/coding_fr.loop', 
                   {'category': 'coding_fr', 'lang': 'fr', 'source': 'ruche-mock'})

# Process
collected = 0
for text in samples:
    result = filter_by_quality(text, cat)
    if result['accept'] and dedup.check(text):
        tokens = list(range(len(text)))  # Mock tokens
        ids = np.array(tokens[:50], dtype=np.int32)[:50]
        mask = np.ones(50, dtype=np.int32)
        writer.add_batch(LoopBatch(ids, ids, mask))
        collected += 1

# Save
size = writer.save()
print(f"Created coding_fr.loop: {size} bytes, {collected} examples")
