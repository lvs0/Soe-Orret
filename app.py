#!/usr/bin/env python3
"""
SOE Orret — Application Dashboard
Interface glass morphism pour le système SOE/Orret
结合 ARIA + SOMA + .loop + Ruche

Usage: python3 app.py [--port 8765]
"""
import os
import sys
import json
import time
import subprocess
import threading
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

SOE_HOME = os.path.expanduser("~/.soe")
APP_PORT = 8765

# ============================================================
# ROUTage API
# ============================================================
class SOEAPI:
    def __init__(self):
        self.memory_file = os.path.join(SOE_HOME, "memory.json")
        self.state_file = os.path.join(SOE_HOME, "state.json")
        self.loops_dir = os.path.join(SOE_HOME, "datasets")
        self.models_dir = os.path.join(SOE_HOME, "models")
        self.ruche_dir = os.path.join(SOE_HOME, "ruche")
        self._ensure_dirs()
        self._load_state()

    def _ensure_dirs(self):
        for d in [SOE_HOME, self.loops_dir, self.models_dir]:
            os.makedirs(d, exist_ok=True)

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "version": "1.0.0", "name": "Orret", "codename": "SOE",
                "uptime": time.time(), "cycles": 0, "loops_count": 0,
                "memory_entries": 0, "active_agents": [],
                "last_update": None, "status": "online"
            }

    def _save_state(self):
        self.state["last_update"] = time.time()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    # ── Routes API ──────────────────────────────────────────
    def route(self, path, method="GET"):
        if path == "/api/status":
            return self._status()
        if path == "/api/memory":
            return self._memory()
        if path == "/api/loops":
            return self._loops()
        if path == "/api/ruche":
            return self._ruche()
        if path == "/api/models":
            return self._models()
        if path == "/api/agents":
            return self._agents()
        if path == "/api/nexus":
            return self._nexus()
        if path == "/api/blueprint":
            return {"version": "1.0", "title": "ARIA + SOMA Blueprint", "status": "ready"}
        return None

    def _status(self):
        self.state["cycles"] += 1
        self._save_state()
        uptime = time.time() - self.state.get("uptime", time.time())
        return {
            "name": self.state["name"],
            "version": self.state["version"],
            "status": self.state["status"],
            "uptime": int(uptime),
            "cycles": self.state["cycles"],
            "codename": self.state.get("codename", "Orret"),
            "ports": {
                "dashboard": 8765, "nexus": 8765, "medicain": 8766,
                "worldview": 8773, "notifications": 8770
            }
        }

    def _memory(self):
        entries = []
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file) as f:
                    data = json.load(f)
                    entries = data.get("entries", [])[-50:]
            except: pass
        return {"entries": entries, "count": len(entries)}

    def _loops(self):
        loops = []
        if os.path.exists(self.loops_dir):
            for f in os.listdir(self.loops_dir):
                if f.endswith('.loop'):
                    fp = os.path.join(self.loops_dir, f)
                    loops.append({
                        "name": f, "size": os.path.getsize(fp),
                        "modified": int(os.path.getmtime(fp))
                    })
        self.state["loops_count"] = len(loops)
        return {"loops": loops, "count": len(loops)}

    def _ruche(self):
        # État de la ruche
        return {
            "status": "idle", "sources": 0, "collected": 0,
            "last_run": None, "categories": []
        }

    def _models(self):
        models = []
        if os.path.exists(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith(('.gguf', '.bin', '.safetensors')):
                    fp = os.path.join(self.models_dir, f)
                    models.append({"name": f, "size": os.path.getsize(fp)})
        return {"models": models, "count": len(models)}

    def _agents(self):
        return {"agents": self.state.get("active_agents", []), "count": 0}

    def _nexus(self):
        return {"nexus_online": True, "worldview_online": True}


# ============================================================
# HANDLER HTTP
# ============================================================
API = SOEAPI()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Silent

    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API routes
        if path.startswith("/api/"):
            result = API.route(path)
            if result is not None:
                self.send_json(result)
            else:
                self.send_json({"error": "Not found"}, 404)
            return

        # SPA - serve index for all non-file paths
        if path == "/" or not os.path.splitext(path)[1]:
            self.serve_file("/index.html", "text/html")
            return

        ext = os.path.splitext(path)[1]
        mime_types = {
            ".html": "text/html", ".js": "application/javascript",
            ".css": "text/css", ".json": "application/json",
            ".png": "image/png", ".jpg": "image/jpeg",
            ".svg": "image/svg+xml", ".ico": "image/x-icon"
        }
        mime = mime_types.get(ext, "application/octet-stream")
        self.serve_file(path, mime)

    def serve_file(self, path, mime):
        fp = os.path.join(APP_DIR, path.lstrip("/"))
        if os.path.exists(fp) and os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    do_POST = do_GET  #简化


APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def run_server(port):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"⟨SOE⟩ Dashboard Orret → http://localhost:{port}")
        print(f"⟨SOE⟩ Sur WiFi    → http://192.168.1.104:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.port)
