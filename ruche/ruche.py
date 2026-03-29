"""
Ruche - Orchestrateur de collecte de données
Spider principal qui coordonne les sources
"""
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ruche')

class Ruche:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent / 'ruche_config.yaml'
        self.config = self._load_config()
        self.output_dir = Path.home() / 'soe' / 'datasets' / 'raw'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)
            
    def run(self, sources: list = None):
        """Lance la collecte sur les sources spécifiées ou toutes"""
        sources = sources or list(self.config.get('sources', {}).keys())
        results = {}
        
        for source_name in sources:
            if source_name not in self.config.get('sources', {}):
                logger.warning(f"Unknown source: {source_name}")
                continue
                
            logger.info(f"Collecting from: {source_name}")
            try:
                result = self._collect_source(source_name)
                results[source_name] = result
            except Exception as e:
                logger.error(f"Error collecting {source_name}: {e}")
                results[source_name] = {'error': str(e)}
                
        self._save_run_log(results)
        return results
        
    def _collect_source(self, source_name: str):
        """Collecte depuis une source spécifique"""
        config = self.config['sources'].get(source_name, {})
        
        if source_name == 'github':
            from .sources.github_spider import collect_github
            return collect_github(config)
        elif source_name == 'wikipedia':
            from .sources.wikipedia_spider import collect_wikipedia
            return collect_wikipedia(config)
        elif source_name == 'stackoverflow':
            from .sources.stackoverflow_spider import collect_stackoverflow
            return collect_stackoverflow(config)
        elif source_name == 'arxiv':
            from .sources.arxiv_spider import collect_arxiv
            return collect_arxiv(config)
        else:
            raise ValueError(f"Unknown source: {source_name}")
            
    def _save_run_log(self, results: dict):
        """Log le run dans un fichier"""
        log_file = Path.home() / 'soe' / 'logs' / 'ruche_runs.log'
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {json.dumps(results)}\n")

if __name__ == '__main__':
    import sys
    ruche = Ruche()
    sources = sys.argv[1:] if len(sys.argv) > 1 else None
    ruche.run(sources)