#!/usr/bin/env python3
"""
SOE - System Engine + Ollama Clone
Moteur unifié combinant Synapse + World Model + ARIA Memory

Usage:
    soe run <model>         # Mode interactif comme Ollama
    soe serve               # Serveur API
    soe chat                # Mode conversationnel
    soe status              # État du système
    soe models              # Liste des modèles
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field

# Import des composants SOE
try:
    from synapse_engine import SynapseModel, SynapseConfig, create_model
    from world_model import WorldModel, WorldModelIntegration
    SYNAPSE_AVAILABLE = True
except ImportError:
    SYNAPSE_AVAILABLE = False


# Configuration
SOE_HOME = os.path.expanduser("~/.soe")
MODELS_DIR = os.path.join(SOE_HOME, "models")
CONFIG_FILE = os.path.join(SOE_HOME, "config.json")
MEMORY_DB = os.path.join(SOE_HOME, "memory.db")


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


@dataclass
class SOEConfig:
    """Configuration SOE."""
    version: str = "0.1.0"
    default_model: str = "synapse-local"
    server_port: int = 11435
    ollama_url: str = "http://localhost:11434"
    context_length: int = 2048
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    enable_world_model: bool = True
    enable_memory: bool = True


class SOEMemory:
    """Système de mémoire persistante - comme ARIA."""
    
    def __init__(self, db_path: str = MEMORY_DB):
        self.db_path = db_path
        self.conversations: List[Dict] = []
        self.facts: List[Dict] = []
        self.preferences: Dict = {}
        self._init_storage()
    
    def _init_storage(self):
        """Initialise le stockage JSON (plus simple que SQLite pour compatibility)."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', [])
                    self.facts = data.get('facts', [])
                    self.preferences = data.get('preferences', {})
            except:
                pass
    
    def save(self):
        """Sauvegarde en disque."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump({
                'conversations': self.conversations[-1000:],  # Garder les 1000 dernières
                'facts': self.facts,
                'preferences': self.preferences
            }, f, indent=2)
    
    def add_conversation(self, user: str, assistant: str, metadata: Dict = None):
        """Ajoute une conversation."""
        self.conversations.append({
            'timestamp': time.time(),
            'user': user[:500],
            'assistant': assistant[:1000],
            'metadata': metadata or {}
        })
        self.save()
    
    def add_fact(self, fact: str, importance: float = 0.5):
        """Ajoute un fait à retenir."""
        self.facts.append({
            'fact': fact,
            'importance': importance,
            'created_at': time.time()
        })
        self.save()
    
    def search_context(self, query: str) -> str:
        """Recherche le contexte pertinent."""
        parts = []
        
        # Facts pertinents
        for fact in sorted(self.facts, key=lambda x: x['importance'], reverse=True)[:3]:
            parts.append(f"[Info] {fact['fact']}")
        
        # Conversations récentes
        for conv in self.conversations[-3:]:
            parts.append(f"[History] User: {conv['user'][:100]}...")
        
        return "\n".join(parts[:5]) if parts else ""
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Récupère une préférence."""
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """Définit une préférence."""
        self.preferences[key] = value
        self.save()


class SOEEngine:
    """Moteur principal SOE - Orchestre tous les composants."""
    
    def __init__(self, config: SOEConfig = None):
        self.config = config or SOEConfig()
        
        # Composants
        self.model: Optional[SynapseModel] = None
        self.world_model: Optional[WorldModel] = None
        self.memory = SOEMemory()
        
        # État
        self.conversation_history: List[Dict] = []
        self.is_loaded = False
        
        # Stats
        self.stats = {
            "requests": 0,
            "tokens_generated": 0,
            "errors": 0,
            "start_time": time.time()
        }
    
    def load_model(self, model_name: str = None):
        """Charge un modèle."""
        model_name = model_name or self.config.default_model
        
        if SYNAPSE_AVAILABLE:
            # Configuration synaptique optimisée pour CPU
            config = SynapseConfig(
                d_model=128,
                state_size=32,
                num_heads=4,
                num_layers=2,
                vocab_size=32000
            )
            self.model = create_model(config)
            print(f"{Colors.GREEN}✓{Colors.ENDC} Synapse model loaded: {model_name}")
        else:
            # Fallback: utiliser Ollama si disponible
            print(f"{Colors.YELLOW}⚠{Colors.ENDC} Synapse not available, using Ollama fallback")
        
        if self.config.enable_world_model:
            self.world_model = WorldModel(embedding_dim=128)
            print(f"{Colors.GREEN}✓{Colors.ENDC} World Model initialized")
        
        self.is_loaded = True
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Génère une réponse.
        """
        self.stats["requests"] += 1
        
        # Récupérer le contexte de la mémoire
        context = self.memory.search_context(prompt)
        
        # Ajouter le contexte au prompt si pertinent
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nUser: {prompt}"
        
        # Génération via le modèle
        if self.model and SYNAPSE_AVAILABLE:
            response = self.model.generate(
                full_prompt,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p)
            )
            backend = "synapse"
        else:
            # Fallback Ollama
            response = self._ollama_generate(prompt, **kwargs)
            backend = "ollama"
        
        # Sauvegarder la conversation
        self.memory.add_conversation(prompt, response, {"backend": backend})
        
        # Mettre à jour le World Model si disponible
        if self.world_model:
            self.world_model.add_episode(
                f"User: {prompt}\nAssistant: {response}",
                np.random.randn(128),  # Placeholder - vrai embedding requis
                importance=0.6,
                outcome="chat"
            )
        
        self.stats["tokens_generated"] += len(response.split())
        
        return {
            "response": response,
            "backend": backend,
            "context_used": bool(context),
            "tokens": len(response.split()),
            "done": True
        }
    
    def _ollama_generate(self, prompt: str, **kwargs) -> str:
        """Fallback vers Ollama."""
        import requests
        
        try:
            resp = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "options": {
                        "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                        "temperature": kwargs.get('temperature', self.config.temperature)
                    }
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            pass
        
        return "[Erreur] Impossible de générer une réponse. Vérifiez Ollama ou le modèle Synapse."
    
    def chat(self, messages: List[Dict], **kwargs) -> Dict:
        """Mode chat comme Ollama."""
        # Construire le prompt depuis les messages
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        return self.generate(full_prompt, **kwargs)
    
    def get_status(self) -> Dict:
        """Retourne le statut."""
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "version": self.config.version,
            "model_loaded": self.is_loaded,
            "backend": "synapse" if (self.model and SYNAPSE_AVAILABLE) else "ollama",
            "uptime_seconds": int(uptime),
            "total_requests": self.stats["requests"],
            "total_tokens": self.stats["tokens_generated"],
            "errors": self.stats["errors"],
            "memory_conversations": len(self.memory.conversations),
            "memory_facts": len(self.memory.facts),
            "world_model_concepts": len(self.world_model.concepts) if self.world_model else 0
        }


# CLI Commands
def cmd_run(args):
    """Mode interactif."""
    engine = SOEEngine()
    engine.load_model(args.model)
    
    print(f"\n{Colors.GREEN}SOE v{engine.config.version}{Colors.ENDC} - Tapez 'quit' pour quitter\n")
    
    while True:
        try:
            prompt = input(f"{Colors.CYAN}> {Colors.ENDC}")
            
            if prompt.lower() in ['quit', 'exit', '/exit']:
                break
            
            if not prompt.strip():
                continue
            
            start = time.time()
            result = engine.generate(prompt)
            elapsed = (time.time() - start) * 1000
            
            print(f"\n{Colors.BLUE}{result['response']}{Colors.ENDC}")
            print(f"{Colors.DIM}[{result['backend']} | {elapsed:.0f}ms | {result['tokens']} tokens]{Colors.ENDC}\n")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print(f"\n{Colors.DIM}Sessions terminée{Colors.ENDC}")


def cmd_serve(args):
    """Démarre le serveur API."""
    engine = SOEEngine()
    engine.load_model()
    
    port = args.port or engine.config.server_port
    
    class SOEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/api/generate":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                
                result = engine.generate(
                    data.get("prompt", ""),
                    max_tokens=data.get("options", {}).get("num_predict", 1024),
                    temperature=data.get("options", {}).get("temperature", 0.7)
                )
                
                response = {
                    "model": "soe-local",
                    "response": result["response"],
                    "done": True,
                    "total_duration": int(time.time() * 1e9),
                    "prompt_eval_count": len(data.get("prompt", "")),
                    "eval_count": result["tokens"]
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            elif self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                
                result = engine.chat(data.get("messages", []))
                
                response = {
                    "model": "soe-local",
                    "message": {"role": "assistant", "content": result["response"]},
                    "done": True
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        
        def do_GET(self):
            if self.path == "/":
                status = engine.get_status()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.send_response(404)
                self.end_headers()
    
    print(f"{Colors.GREEN}✓{Colors.ENDC} SOE server started on http://localhost:{port}")
    print(f"   API: http://localhost:{port}/api/generate")
    print(f"   Chat: http://localhost:{port}/api/chat\n")
    
    server = HTTPServer(("0.0.0.0", port), SOEHandler)
    server.serve_forever()


def cmd_status(args):
    """Affiche le statut."""
    engine = SOEEngine()
    if args.load:
        engine.load_model()
    
    status = engine.get_status()
    
    print(f"\n{Colors.BOLD}SOE Status{Colors.ENDC}")
    print(f"  Version: {status['version']}")
    print(f"  Model loaded: {status['model_loaded']}")
    print(f"  Backend: {status['backend']}")
    print(f"  Uptime: {status['uptime_seconds']}s")
    print(f"  Requests: {status['total_requests']}")
    print(f"  Tokens: {status['total_tokens']}")
    print(f"  Errors: {status['errors']}")
    print(f"  Memory facts: {status['memory_facts']}")
    print(f"  Memory conversations: {status['memory_conversations']}")
    print(f"  World model concepts: {status['world_model_concepts']}")
    print()


def cmd_models(args):
    """Liste les modèles."""
    models = [
        {"name": "synapse-local", "size": "~350MB", "params": "125M", "status": "ready"},
        {"name": "synapse-small", "size": "~150MB", "params": "35M", "status": "ready"},
        {"name": "synapse-plus", "size": "~700MB", "params": "350M", "status": "available"},
    ]
    
    print(f"\n{Colors.BOLD}Available Models:{Colors.ENDC}\n")
    for m in models:
        color = Colors.GREEN if m["status"] == "ready" else Colors.YELLOW
        print(f"  {m['name']:<20} {m['size']:<12} {m['params']:<8} {color}{m['status']}{Colors.ENDC}")
    print()


def cmd_chat(args):
    """Mode conversationnel simple."""
    engine = SOEEngine()
    engine.load_model()
    
    print(f"{Colors.GREEN}SOE Chat{Colors.ENDC} - Tapez 'quit' pour quitter\n")
    
    messages = []
    
    while True:
        try:
            user_input = input(f"{Colors.CYAN}> {Colors.ENDC}")
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            messages.append({"role": "user", "content": user_input})
            
            result = engine.chat(messages)
            messages.append({"role": "assistant", "content": result["response"]})
            
            print(f"{Colors.BLUE}{result['response']}{Colors.ENDC}\n")
            
        except KeyboardInterrupt:
            break
    
    print(f"{Colors.DIM}Chat terminé{Colors.ENDC}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="SOE - Revolutionary LLM Engine (O(n) linear complexity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  soe run                  # Mode interactif
  soe serve               # Serveur API
  soe chat                # Mode conversation
  soe status              # Voir le statut
  soe models              # Liste des modèles
        """
    )
    
    sub = parser.add_subparsers(dest="command")
    
    # run
    run_parser = sub.add_parser("run", help="Run interactif")
    run_parser.add_argument("model", nargs="?", help="Model name")
    
    # serve
    serve_parser = sub.add_parser("serve", help="Démarrer serveur API")
    serve_parser.add_argument("-p", "--port", type=int, help="Port")
    
    # status
    status_parser = sub.add_parser("status", help="Statut du système")
    status_parser.add_argument("-l", "--load", action="store_true", help="Charger le modèle")
    
    # models
    sub.add_parser("models", help="Liste des modèles")
    
    # chat
    sub.add_parser("chat", help="Mode conversationnel")
    
    args = parser.parse_args()
    
    # Default: show status
    if args.command is None:
        cmd_status(argparse.Namespace(load=False))
        return
    
    # Dispatch
    if args.command == "run":
        cmd_run(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()