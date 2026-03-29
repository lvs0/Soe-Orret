"""
StackOverflow Spider - Collecte questions depuis StackExchange API
Source: Stack Overflow API
"""
import requests
from pathlib import Path
from datetime import datetime

TAGS = ['python', 'machine-learning', 'nlp', 'llm', 'artificial-intelligence']

def collect_stackoverflow(config: dict = None):
    """Collecte depuis Stack Overflow"""
    tags = config.get('tags', TAGS) if config else TAGS
    pages = config.get('pages', 5) if config else 5
    
    output_file = Path.home() / 'soe' / 'datasets' / 'raw' / f'stackoverflow_{datetime.now().strftime("%Y%m%d")}.loop'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    from core.looplib import LoopWriter
    writer = LoopWriter(str(output_file))
    writer.write_header({
        'source': 'stackoverflow',
        'tags': tags,
        'collected': datetime.now().isoformat()
    })
    
    total = 0
    for tag in tags:
        questions = fetch_tag_questions(tag, pages)
        for q in questions:
            writer.write_entry(q)
            total += 1
    
    writer.close()
    print(f"✅ StackOverflow: {total} entries")
    return {'entries': total, 'file': str(output_file)}

def fetch_tag_questions(tag: str, pages: int) -> list:
    """Fetch questions for a tag"""
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        'order': 'desc',
        'sort': 'activity',
        'tagged': tag,
        'site': 'stackoverflow',
        'pagesize': 100
    }
    
    entries = []
    for _ in range(min(pages, 5)):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('items', []):
                    entries.append({
                        'type': 'stackoverflow_question',
                        'tag': tag,
                        'title': item.get('title', ''),
                        'body': item.get('body_markdown', '')[:2000],  # Truncate
                        'score': item.get('score', 0),
                        'answer_count': item.get('answer_count', 0),
                        'view_count': item.get('view_count', 0),
                        'link': item.get('link', ''),
                        'created': item.get('creation_date', '')
                    })
                if not data.get('has_more'):
                    break
                params['page'] = params.get('page', 1) + 1
            else:
                break
        except Exception as e:
            print(f"Error: {e}")
            break
    return entries

if __name__ == '__main__':
    collect_stackoverflow()