"""
ArXiv Spider - Collecte papers depuis ArXiv
Source: ArXiv API
"""
import requests
from pathlib import Path
from datetime import datetime

CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL']

def collect_arxiv(config: dict = None):
    """Collecte depuis ArXiv"""
    categories = config.get('categories', CATEGORIES) if config else CATEGORIES
    max_papers = config.get('max_papers', 50) if config else 50
    
    output_file = Path.home() / 'soe' / 'datasets' / 'raw' / f'arxiv_{datetime.now().strftime("%Y%m%d")}.loop'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    from core.looplib import LoopWriter
    writer = LoopWriter(str(output_file))
    writer.write_header({
        'source': 'arxiv',
        'categories': categories,
        'collected': datetime.now().isoformat()
    })
    
    total = 0
    for cat in categories:
        papers = fetch_category_papers(cat, max_papers)
        for paper in papers:
            writer.write_entry(paper)
            total += 1
    
    writer.close()
    print(f"✅ ArXiv: {total} papers")
    return {'entries': total, 'file': str(output_file)}

def fetch_category_papers(category: str, max_count: int) -> list:
    """Fetch papers from category"""
    url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f'cat:{category}',
        'start': 0,
        'max_results': min(max_count, 100),
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            
            entries = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                authors = [a.find('{http://www.w3.org/2005/Atom}name').text 
                          for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                link = entry.find('{http://www.w3.org/2005/Atom}id').text
                
                entries.append({
                    'type': 'arxiv_paper',
                    'category': category,
                    'title': title.replace('\\n', ' ').strip(),
                    'summary': summary.replace('\\n', ' ').strip()[:3000],
                    'authors': authors,
                    'published': published,
                    'link': link
                })
            return entries
    except Exception as e:
        print(f"Error: {e}")
    return []

if __name__ == '__main__':
    collect_arxiv()