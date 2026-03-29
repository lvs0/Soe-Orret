"""
CLAW-PRIME Autonomous Runner
Auto-exécute les tâches SOE selon le planning
"""
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

TASKS = {
    'looplib': 'Test looplib',
    'ruche': 'Run data collection',
    'processor': 'Process datasets',
    'monitor': 'System check',
    'docs': 'Update docs'
}

def run_task(task_name: str):
    """Exécute une tâche"""
    print(f"Running: {task_name}")
    
    if task_name == 'monitor':
        from agents.monitor_agent import SystemMonitorAgent
        agent = SystemMonitorAgent()
        return agent.run()
    
    elif task_name == 'ruche':
        from ruche.ruche import Ruche
        ruche = Ruche()
        return ruche.run()
    
    elif task_name == 'looplib':
        from core.looplib import LoopWriter, LoopReader
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.loop', delete=False) as f:
            w = LoopWriter(f.name)
            w.write_header({'test': 'autonomous'})
            w.write_entry({'task': 'looplib_test'})
            w.close()
            r = LoopReader(f.name)
            r.close()
            os.unlink(f.name)
        return {'status': 'ok'}
    
    return {'status': 'unknown_task'}

def daily_routine():
    """Routine quotidienne"""
    log_file = Path.home() / 'soe' / 'logs' / 'claw_activity.log'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    results = []
    
    # System monitor (always)
    results.append(run_task('monitor'))
    
    # Ruche (si jour approprié)
    if datetime.now().hour == 2:  # 2h du mat
        results.append(run_task('ruche'))
    
    # Looplib test (quotidien)
    results.append(run_task('looplib'))
    
    # Log
    with open(log_file, 'a') as f:
        f.write(f"{timestamp} | CLAW-PRIME | Daily routine: {len(results)} tasks completed\\n")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) > 1:
        task = sys.argv[1]
        run_task(task)
    else:
        daily_routine()