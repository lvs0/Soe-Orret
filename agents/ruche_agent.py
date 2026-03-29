"""
Agent RUCHE-COLLECTOR - Collecte de données
Trigger: Cron quotidien + appel depuis CLAW-PRIME
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ruche.ruche import Ruche

class RucheAgent:
    def __init__(self):
        self.name = "RUCHE-COLLECTOR"
        self.state_file = Path.home() / 'soe' / '.agent_state.json'
        
    def run(self, sources: list = None):
        """Exécute la collecte"""
        self._log(f"Starting collection: {sources}")
        
        ruche = Ruche()
        results = ruche.run(sources)
        
        # Sauvegarde état
        self._save_state({'last_run': datetime.now().isoformat(), 'results': results})
        self._log(f"Collection complete: {len(results)} sources")
        
        return results
        
    def _log(self, msg: str):
        log_file = Path.home() / 'soe' / 'logs' / 'claw_activity.log'
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {self.name} | {msg}\n")
            
    def _save_state(self, state: dict):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

if __name__ == '__main__':
    agent = RucheAgent()
    sources = sys.argv[1:] if len(sys.argv) > 1 else None
    agent.run(sources)