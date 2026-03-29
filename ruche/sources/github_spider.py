"""
GitHub Spider - Collecte issues et code depuis GitHub
Source: GitHub REST API
"""
import requests
import json
from pathlib import Path
from datetime import datetime
import time

# Topics from config
TOPICS = ['python', 'machine-learning', 'llm', 'ai', 'transformers']
MAX_ISSUES = 50

def collect_github(config: dict = None):
    """Collecte depuis GitHub API"""
    topics = config.get('topics', TOPICS) if config else TOPICS
    max_per_topic = config.get('max_issues', MAX_ISSUES) if config else MAX_ISSUES
    
    output_file = Path.home() / 'soe' / 'datasets' / 'raw' / f'github_{datetime.now().strftime("%Y%m%d")}.loop'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    from core.looplib import LoopWriter
    writer = LoopWriter(str(output_file))
    writer.write_header({
        'source': 'github',
        'collected': datetime.now().isoformat(),
        'topics': topics
    })
    
    total = 0
    for topic in topics:
        print(f"Fetching topic: {topic}")
        issues = fetch_topic_issues(topic, max_per_topic)
        for issue in issues:
            writer.write_entry(issue)
            total += 1
        time.sleep(1)  # Rate limit
    
    writer.close()
    print(f"✅ GitHub: {total} entries written to {output_file}")
    return {'entries': total, 'file': str(output_file)}

def fetch_topic_issues(topic: str, max_count: int) -> list:
    """Fetch issues from topic search"""
    url = f"https://api.github.com/search/issues"
    params = {
        'q': f'topic:{topic} is:issue created:>2024-01-01',
        'sort': 'created',
        'order': 'desc',
        'per_page': min(max_count, 100)
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            entries = []
            for item in data.get('items', [])[:max_count]:
                entries.append({
                    'type': 'github_issue',
                    'topic': topic,
                    'title': item.get('title', ''),
                    'body': item.get('body', ''),
                    'url': item.get('html_url', ''),
                    'state': item.get('state', ''),
                    'created_at': item.get('created_at', ''),
                    'comments': item.get('comments', 0),
                    'labels': [l['name'] for l in item.get('labels', [])]
                })
            return entries
        else:
            print(f"Error {resp.status_code}: {resp.text[:100]}")
            return []
    except Exception as e:
        print(f"Exception: {e}")
        return []

if __name__ == '__main__':
    collect_github()