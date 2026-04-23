"""
api/server.py
Soe-Orret REST API server.
Exposes endpoints for sampling, memory ops, and orchestration.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

from agent.orchestrator import Orchestrator, TaskSpec
from memory.aria import LayeredMemory


class SoeOrretHandler(BaseHTTPRequestHandler):
    """HTTP handler for Soe-Orret API."""

    orchestrator: Optional[Orchestrator] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"status": "ok", "service": "soe-orret"})
        elif parsed.path == "/status":
            self._json(200, self.orchestrator.status() if self.orchestrator else {})
        elif parsed.path == "/layers":
            self._json(200, {"layers": LayeredMemory.LAYERS})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/sample":
            self._handle_sample()
        elif parsed.path == "/memory/store":
            self._handle_memory_store()
        else:
            self._json(404, {"error": "not found"})

    def _handle_sample(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body)
            prompt = payload.get("prompt", "")
            num_steps = payload.get("num_steps", 16)
            block_size = payload.get("block_size", 32)
            spec = TaskSpec(prompt=prompt, num_steps=num_steps, block_size=block_size)
            result = self.orchestrator.execute(spec)
            self._json(200, {
                "task_id": result.task_id,
                "status": result.status,
                "output": result.output,
                "duration_ms": result.duration_ms,
                "error": result.error,
            })
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _handle_memory_store(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body)
            layer = payload.get("layer", "semantic")
            key = payload.get("key", "")
            value = payload.get("value", "")
            tags = payload.get("tags", [])
            if not key:
                self._json(400, {"error": "key required"})
                return
            rowid = self.orchestrator.memory.store(layer, key, value, tags)
            self._json(200, {"rowid": rowid})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, code: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[SoeOrretAPI] {args[0]}")


def run_server(port: int = 8765, host: str = "0.0.0.0", db_path: str = ":memory:"):
    """Start the Soe-Orret API server."""
    orchestrator = Orchestrator(memory_db=db_path)
    SoeOrretHandler.orchestrator = orchestrator
    server = HTTPServer((host, port), SoeOrretHandler)
    print(f"Soe-Orret API running on http://{host}:{port}")
    server.serve_forever()
