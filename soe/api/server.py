"""
Soe-Orret API Server - RESTful API for the agent ecosystem
"""

import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, Callable
from urllib.parse import parse_qs, urlparse
import threading
import logging
from datetime import datetime


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the API."""
    
    routes: Dict[str, Dict[str, Callable]] = {}
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def _send_response(self, status: int, data: Any):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, status: int, message: str):
        """Send error response."""
        self._send_response(status, {'error': message, 'status': status})
    
    def _get_body(self) -> Optional[Dict]:
        """Parse request body as JSON."""
        content_length = self.headers.get('Content-Length')
        if not content_length:
            return None
        
        try:
            body = self.rfile.read(int(content_length))
            return json.loads(body.decode('utf-8'))
        except (json.JSONDecodeError, ValueError):
            return None
    
    def _get_query_params(self) -> Dict[str, Any]:
        """Parse query parameters."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        # Flatten single-value lists
        return {k: v[0] if len(v) == 1 else v for k, v in params.items()}
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        handler = self.routes.get('GET', {}).get(path)
        
        if handler:
            try:
                params = self._get_query_params()
                result = handler(params)
                self._send_response(200, result)
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, f"Route not found: {path}")
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        handler = self.routes.get('POST', {}).get(path)
        
        if handler:
            body = self._get_body()
            if body is None and self.headers.get('Content-Length'):
                self._send_error(400, "Invalid JSON body")
                return
            
            try:
                result = handler(body or {})
                self._send_response(201, result)
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, f"Route not found: {path}")
    
    def do_PUT(self):
        """Handle PUT requests."""
        path = urlparse(self.path).path
        handler = self.routes.get('PUT', {}).get(path)
        
        if handler:
            body = self._get_body()
            if body is None:
                self._send_error(400, "Invalid JSON body")
                return
            
            try:
                result = handler(body)
                self._send_response(200, result)
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, f"Route not found: {path}")
    
    def do_DELETE(self):
        """Handle DELETE requests."""
        path = urlparse(self.path).path
        handler = self.routes.get('DELETE', {}).get(path)
        
        if handler:
            try:
                result = handler()
                self._send_response(200, result)
            except Exception as e:
                self._send_error(500, str(e))
        else:
            self._send_error(404, f"Route not found: {path}")


class SoeOrretServer:
    """
    RESTful API server for the Soe-Orret agent ecosystem.
    Provides endpoints for agents, tasks, memory, and sampling.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Component references (to be set)
        self.orchestrator = None
        self.memory = None
        self.diffuser = None
        
        # Setup routes
        self._setup_routes()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SoeOrretServer")
    
    def _setup_routes(self):
        """Configure API routes."""
        APIHandler.routes = {
            'GET': {
                '/health': self._health_check,
                '/status': self._get_status,
                '/agents': self._list_agents,
                '/tasks': self._list_tasks,
                '/memory': self._search_memory,
            },
            'POST': {
                '/agents': self._create_agent,
                '/tasks': self._create_task,
                '/memory': self._store_memory,
                '/sample': self._generate_sample,
            },
            'PUT': {
                '/agents/:id/heartbeat': self._agent_heartbeat,
                '/tasks/:id/complete': self._complete_task,
            },
            'DELETE': {
                '/agents/:id': self._delete_agent,
                '/tasks/:id': self._cancel_task,
            }
        }
    
    # Route handlers
    def _health_check(self, params: Dict) -> Dict:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '0.1.0'
        }
    
    def _get_status(self, params: Dict) -> Dict:
        """Get system status."""
        status = {
            'server': {
                'running': self._running,
                'host': self.host,
                'port': self.port
            },
            'components': {}
        }
        
        if self.orchestrator:
            status['components']['orchestrator'] = self.orchestrator.get_status()
        
        if self.memory:
            status['components']['memory'] = self.memory.get_layer_stats()
        
        return status
    
    def _list_agents(self, params: Dict) -> Dict:
        """List all agents."""
        if not self.orchestrator:
            return {'agents': []}
        
        agents = [
            {
                'id': a.id,
                'name': a.name,
                'role': a.role,
                'state': a.state.name,
                'capabilities': a.capabilities,
                'current_task': a.current_task
            }
            for a in self.orchestrator.agents.values()
        ]
        return {'agents': agents, 'count': len(agents)}
    
    def _create_agent(self, body: Dict) -> Dict:
        """Create a new agent."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not configured")
        
        agent = self.orchestrator.register_agent(
            name=body.get('name', 'Unnamed'),
            role=body.get('role', 'worker'),
            capabilities=body.get('capabilities', []),
            metadata=body.get('metadata', {})
        )
        
        return {
            'id': agent.id,
            'name': agent.name,
            'role': agent.role,
            'state': agent.state.name,
            'created_at': agent.created_at
        }
    
    def _delete_agent(self) -> Dict:
        """Delete an agent."""
        # Note: Would need to parse ID from path in real implementation
        return {'message': 'Agent deletion endpoint - implement path parsing'}
    
    def _agent_heartbeat(self, body: Dict) -> Dict:
        """Process agent heartbeat."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not configured")
        
        agent_id = body.get('agent_id')
        if not agent_id:
            raise ValueError("agent_id required")
        
        success = self.orchestrator.heartbeat(agent_id)
        return {'acknowledged': success, 'agent_id': agent_id}
    
    def _list_tasks(self, params: Dict) -> Dict:
        """List tasks."""
        if not self.orchestrator:
            return {'tasks': []}
        
        status_filter = params.get('status')
        tasks = []
        
        for task in self.orchestrator.tasks.values():
            if status_filter and task.status.name != status_filter:
                continue
            
            tasks.append({
                'id': task.id,
                'name': task.name,
                'status': task.status.name,
                'priority': task.priority,
                'agent_id': task.agent_id,
                'created_at': task.created_at,
                'completed_at': task.completed_at
            })
        
        return {'tasks': tasks, 'count': len(tasks)}
    
    def _create_task(self, body: Dict) -> Dict:
        """Create a new task."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not configured")
        
        task = self.orchestrator.create_task(
            name=body.get('name', 'Unnamed'),
            description=body.get('description', ''),
            priority=body.get('priority', 5),
            metadata=body.get('metadata', {}),
            dependencies=body.get('dependencies', [])
        )
        
        return {
            'id': task.id,
            'name': task.name,
            'status': task.status.name,
            'priority': task.priority,
            'created_at': task.created_at
        }
    
    def _complete_task(self, body: Dict) -> Dict:
        """Complete a task."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not configured")
        
        task_id = body.get('task_id')
        result = body.get('result')
        
        success = self.orchestrator.complete_task(task_id, result)
        return {'completed': success, 'task_id': task_id}
    
    def _cancel_task(self) -> Dict:
        """Cancel a task."""
        return {'message': 'Task cancellation endpoint - implement path parsing'}
    
    def _search_memory(self, params: Dict) -> Dict:
        """Search memory."""
        if not self.memory:
            return {'entries': []}
        
        layer = params.get('layer')
        if layer:
            layer = int(layer)
        
        pattern = params.get('pattern')
        limit = int(params.get('limit', 100))
        
        entries = self.memory.search(layer=layer, key_pattern=pattern, limit=limit)
        
        return {
            'entries': [
                {
                    'id': e.id,
                    'layer': e.layer,
                    'key': e.key,
                    'value': e.value,
                    'metadata': e.metadata,
                    'created_at': e.created_at,
                    'access_count': e.access_count
                }
                for e in entries
            ],
            'count': len(entries)
        }
    
    def _store_memory(self, body: Dict) -> Dict:
        """Store in memory."""
        if not self.memory:
            raise RuntimeError("Memory not configured")
        
        layer = body.get('layer', 1)
        key = body.get('key')
        value = body.get('value')
        metadata = body.get('metadata')
        
        if not key or value is None:
            raise ValueError("key and value required")
        
        success = self.memory.store(layer, key, value, metadata)
        return {'stored': success, 'layer': layer, 'key': key}
    
    def _generate_sample(self, body: Dict) -> Dict:
        """Generate a sample using the diffuser."""
        if not self.diffuser:
            raise RuntimeError("Diffuser not configured")
        
        shape = body.get('shape', [8, 8])
        # Note: Would need a real model for actual sampling
        # This returns a placeholder
        
        return {
            'sample': 'placeholder',
            'shape': shape,
            'message': 'Sample generation requires a model function'
        }
    
    def start(self):
        """Start the API server."""
        if self._running:
            return
        
        self.server = HTTPServer((self.host, self.port), APIHandler)
        self._running = True
        
        self._thread = threading.Thread(target=self._serve)
        self._thread.daemon = True
        self._thread.start()
        
        self.logger.info(f"Server started on http://{self.host}:{self.port}")
    
    def _serve(self):
        """Server loop."""
        while self._running:
            try:
                self.server.handle_request()
            except Exception as e:
                self.logger.error(f"Server error: {e}")
    
    def stop(self):
        """Stop the API server."""
        self._running = False
        if self.server:
            # Trigger one more request to exit handle_request
            try:
                import urllib.request
                urllib.request.urlopen(
                    f"http://{self.host}:{self.port}/health",
                    timeout=1
                )
            except:
                pass
            self.server.server_close()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        self.logger.info("Server stopped")
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference."""
        self.orchestrator = orchestrator
    
    def set_memory(self, memory):
        """Set the memory reference."""
        self.memory = memory
    
    def set_diffuser(self, diffuser):
        """Set the diffuser reference."""
        self.diffuser = diffuser


if __name__ == "__main__":
    # Example usage
    server = SoeOrretServer(host='localhost', port=8080)
    server.start()
    
    print(f"Server running at http://localhost:8080")
    print("Try: curl http://localhost:8080/health")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nServer stopped")
