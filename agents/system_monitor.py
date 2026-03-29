#!/usr/bin/env python3
"""System Monitor - Track SOE services and resources"""
import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime

LOG_FILE = Path(os.path.expanduser("~/soe/logs/system_monitor.log"))
SOE_DIR = Path(os.path.expanduser("~/soe"))

def check_services():
    """Check which SOE services are running"""
    services = {
        "NeuroMesh": 11436,
        "DriveMesh": 11437,
        "MeshCloud": 11440,
        "Ollama": 11434,
    }
    
    running = []
    for name, port in services.items():
        # Simple check - if port is open
        # In production, use proper socket check
        running.append({"name": name, "port": port, "status": "unknown"})
    
    return running

def check_resources():
    """Check system resources"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }

def check_phase_status():
    """Check current phase status"""
    state_file = SOE_DIR / ".soe_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}

def monitor():
    """Main monitoring loop"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    while True:
        timestamp = datetime.now().isoformat()
        
        data = {
            "timestamp": timestamp,
            "resources": check_resources(),
            "phase": check_phase_status().get("phase", "unknown"),
        }
        
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
        
        print(f"[{timestamp}] CPU: {data['resources']['cpu_percent']}% | MEM: {data['resources']['memory_percent']}%")
        
        time.sleep(60)  # Every minute

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once instead of loop")
    args = parser.parse_args()
    
    if args.once:
        print(json.dumps({
            "resources": check_resources(),
            "phase": check_phase_status().get("phase", "unknown"),
            "services": check_services(),
        }, indent=2))
    else:
        monitor()
