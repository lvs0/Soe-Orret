"""
Agent SYSTEM-MONITOR - Surveillance SNP1
Trigger: Toutes les 15 minutes (cron)
Surveille: CPU%, RAM%, disque, température, état Ollama, processus SOE
"""
import psutil
import json
from pathlib import Path
from datetime import datetime

class SystemMonitorAgent:
    def __init__(self):
        self.name = "SYSTEM-MONITOR"
        self.thresholds = {
            'cpu_percent': 80,
            'ram_percent': 90,
            'disk_percent': 85
        }
        
    def run(self):
        """Vérifie l'état du système"""
        status = self._check_system()
        self._log_status(status)
        
        # Alerte si seuils dépassés
        alerts = self._check_alerts(status)
        if alerts:
            self._send_alert(alerts)
            
        return status
        
    def _check_system(self):
        """Collecte les métriques système"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self._get_cpu_temp(),
            'ollama_running': self._check_ollama(),
            'soe_processes': self._count_soe_processes(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _get_cpu_temp(self):
        """Température CPU (si dispo)"""
        try:
            temps = psutil.sensors_temperatures()
            if 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except:
            pass
        return None
        
    def _check_ollama(self):
        """Vérifie si Ollama est running"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        return result == 0
        
    def _count_soe_processes(self):
        """Compte les processus SOE"""
        count = 0
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('soe' in str(c).lower() for c in proc.info['cmdline']):
                    count += 1
            except:
                pass
        return count
        
    def _check_alerts(self, status: dict):
        """Vérifie les alertes"""
        alerts = []
        if status['ram_percent'] > self.thresholds['ram_percent']:
            alerts.append(f"RAM: {status['ram_percent']}% > {self.thresholds['ram_percent']}%")
        if status['disk_percent'] > self.thresholds['disk_percent']:
            alerts.append(f"Disk: {status['disk_percent']}% > {self.thresholds['disk_percent']}%")
        return alerts
        
    def _send_alert(self, alerts: list):
        """Envoie alerte via Telegram"""
        # Placeholder - à implémenter avec token Telegram
        self._log(f"ALERT: {alerts}")
        
    def _log_status(self, status: dict):
        log_file = Path.home() / 'soe' / 'logs' / 'system_monitor.log'
        with open(log_file, 'a') as f:
            f.write(f"{status['timestamp']} | CPU:{status['cpu_percent']}% RAM:{status['ram_percent']}% DISK:{status['disk_percent']}%\n")
            
    def _log(self, msg: str):
        log_file = Path.home() / 'soe' / 'logs' / 'claw_activity.log'
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} | {self.name} | {msg}\n")

if __name__ == '__main__':
    agent = SystemMonitorAgent()
    agent.run()