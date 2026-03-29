"""
Wikipedia Spider - Collecte articles depuis Wikipedia
Source: Wikipedia API
"""
import requests
import json
from pathlib import Path
from datetime import datetime

CATEGORIES = ['Science', 'Technology', 'Artificial_intelligence', 'Machine_learning']
LANG = 'en'

def collect_wikipedia(config: dict = None):
    """Collecte depuis Wikipedia"""
    lang = config.get('languages', [LANG])[0] if config else LANG
    categories = config.get('categories', CATEGORIES) if config else CATEGORIES
    
    output_file = Path.home() / 'soe' / 'datasets' / 'raw' / f'wikipedia_{datetime.now().strftime("%Y%m%d")}.loop'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    from core.looplib import LoopWriter
    writer = LoopWriter(str(output_file))
    writer.write_header({
        'source': 'wikipedia',
        'lang': lang,
        'categories': categories,
        'collected': datetime.now().isoformat()
    })
    
    total = 0
    for cat in categories:
        articles = fetch_category_articles(cat, lang)
        for article in articles:
            writer.write_entry(article)
            total += 1
    
    writer.close()
    print(f"✅ Wikipedia: {total} entries written to {output_file}")
    return {'entries': total, 'file': str(output_file)}

def fetch_category_articles(category: str, lang: str = 'en') -> list:
    """Fetch articles from a category"""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category}',
        'cmlimit': 50,
        'format': 'json'
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            members = data.get('query', {}).get('categorymembers', [])
            
            entries = []
            for member in members:
                if member['title'].startswith('Category:'):
                    continue
                # Fetch summary
                summary = fetch_article_summary(member['title'], lang)
                entries.append({
                    'type': 'wikipedia_article',
                    'category': category,
                    'title': member['title'],
                    'page_id': member['pageid'],
                    'summary': summary
                })
            return entries
    except Exception as e:
        print(f"Error: {e}")
    return []

def fetch_article_summary(title: str, lang: str = 'en') -> str:
    """Fetch article summary"""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('extract', '')
    except:
        pass
    return ''

if __name__ == '__main__':
    collect_wikipedia()