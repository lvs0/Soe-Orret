"""
Agent LOOP-PROCESSOR - Nettoyage, déduplication, packing des .loop
Trigger: Après chaque run RUCHE-COLLECTOR
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

class LoopProcessorAgent:
    def __init__(self):
        self.name = "LOOP-PROCESSOR"
        self.raw_dir = Path.home() / 'soe' / 'datasets' / 'raw'
        self.processed_dir = Path.home() / 'soe' / 'datasets' / 'processed'
        
    def run(self):
        """Traitement des fichiers .loop bruts"""
        self._log("Starting loop processing")
        
        raw_files = list(self.raw_dir.glob('*.loop'))
        processed_count = 0
        
        for loop_file in raw_files:
            try:
                self._process_file(loop_file)
                processed_count += 1
            except Exception as e:
                self._log(f"Error processing {loop_file}: {e}")
                
        self._log(f"Processed {processed_count} files")
        return {'processed': processed_count}
        
    def _process_file(self, filepath: Path):
        """Nettoie et déduplique un fichier .loop"""
        # Read, dedup, write to processed
        from core.looplib import LoopReader, LoopWriter
        
        reader = LoopReader(str(filepath))
        processed_path = self.processed_dir / filepath.name
        
        writer = LoopWriter(str(processed_path))
        writer.write_header(reader.metadata)
        
        seen = set()
        for entry in reader:
            # Simple dedup based on content hash
            entry_hash = hash(json.dumps(entry, sort_keys=True))
            if entry_hash not in seen:
                seen.add(entry_hash)
                writer.write_entry(entry)
                
        writer.close()
        reader.close()
        
        # Generate stats
        self._generate_stats(processed_path)
        
    def _generate_stats(self, filepath: Path):
        """Génère les stats pour un fichier .loop"""
        from core.looplib import LoopReader
        
        reader = LoopReader(str(filepath))
        stats = {
            'file': filepath.name,
            'entries': reader.entries,
            'metadata': reader.metadata
        }
        reader.close()
        
        stats_file = filepath.parent.parent / 'stats' / f"{filepath.stem}_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def _log(self, msg: str):
        log_file = Path.home() / 'soe' / 'logs' / 'claw_activity.log'
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {self.name} | {msg}\n")

if __name__ == '__main__':
    agent = LoopProcessorAgent()
    agent.run()