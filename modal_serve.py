"""SOE-Orret landing page sur Modal — static HTML server (gratuit).

Déploiement :
    modal deploy modal_serve.py

La landing page est index.html à la racine du repo.
"""

import http.server
import os
import socket

import modal

APP_NAME = "soe-orret"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("index.html", "/app/index.html", copy=True)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    timeout=60,
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8080, startup_timeout=30)
def serve():
    """Serve the landing page as static HTML on port 8080."""
    os.chdir("/app")

    # Simple HTTP server that always serves index.html
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.path = "/index.html"
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    server = http.server.HTTPServer(("0.0.0.0", 8080), Handler)
    server.serve_forever()
