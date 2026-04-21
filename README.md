# Soe-Orret

Soe-Orret — autonomous agent with block diffusion sampling, 5-layer SQLite memory (Aria), and FastAPI orchestrator.

## Structure

```
soe-orret/
├── sampler/
│   └── block_diffuser.py   # 16-step block diffusion sampler
├── memory/
│   └── aria.py             # 5-layer SQLite memory (cache, working, short-term, long-term, archive)
├── agent/
│   └── orchestrator.py     # top-level agent coordinating memory + sampler
├── api/
│   └── server.py           # FastAPI server
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m api.server
# → http://localhost:8000
```

## API

- `GET /` — system info
- `GET /health` — health check
- `POST /generate` — block diffusion generation
- `POST /memory/store` — store in memory layer (0-4)
- `POST /memory/query` — query memory
- `GET /memory/stats` — memory statistics
- `POST /events/emit` — emit event
- `GET /stats` — global statistics
- `GET /agents` — list agents
- `GET /agents/{id}` — agent status

## Architecture

- **BlockDiffuser**: 16-step diffusion with probabilistic noise scheduling
- **AriaMemory**: 5-layer SQLite → cache / working / short-term / long-term / archive
- **Orchestrator**: async event-driven agent coordination

## License

MIT
