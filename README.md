# Soe-Orret

A modular AI system featuring block-based diffusion sampling, hierarchical memory, and agent orchestration.

## Structure

```
soe-orret/
├── sampler/           # Block diffusion sampler (16 steps)
│   ├── __init__.py
│   └── block_diffuser.py
├── memory/            # Aria 5-layer memory system
│   ├── __init__.py
│   └── aria.py
├── agent/             # Agent orchestration
│   ├── __init__.py
│   └── orchestrator.py
├── api/               # FastAPI REST server
│   ├── __init__.py
│   └── server.py
├── requirements.txt
└── README.md
```

## Components

### Sampler (`sampler/block_diffuser.py`)
Block-based diffusion sampler with configurable 16-step schedule.
- DDPM-style forward/reverse diffusion
- Block-wise processing for efficiency
- Multiple beta schedules (linear, cosine, quadratic)

### Memory (`memory/aria.py`)
5-layer hierarchical SQLite memory system:
- Layer 0: Working memory (immediate)
- Layer 1: Short-term (session-level)
- Layer 2: Medium-term (day-level)
- Layer 3: Long-term (week/month)
- Layer 4: Archive (permanent)

Features automatic promotion/demotion based on access patterns.

### Agent (`agent/orchestrator.py`)
Central orchestrator for multi-agent coordination:
- Task queue with priority scheduling
- Dependency resolution
- Role-based agent assignment
- Workflow builder

### API (`api/server.py`)
FastAPI REST API exposing all components:
- `/sampler/sample` - Generate diffusion samples
- `/memory/*` - Store, retrieve, query memories
- `/agent/*` - Submit tasks, manage workflows
- `/system/stats` - System statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run API Server
```bash
cd api
python server.py
```

Or with uvicorn directly:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### API Examples

**Generate samples:**
```bash
curl -X POST http://localhost:8000/sampler/sample \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 2, "height": 64, "width": 64, "num_steps": 16}'
```

**Store memory:**
```bash
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"key": "user_pref", "content": "Dark mode preferred", "layer": 1, "priority": 0.9}'
```

**Submit task:**
```bash
curl -X POST http://localhost:8000/agent/task \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1", "description": "Process data", "role": "executor", "priority": 1.5}'
```

## License

MIT
