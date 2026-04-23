# Soe-Orret

A modular agent ecosystem with diffusion-based sampling and hierarchical memory.

## Architecture

```
soe-orret/
├── sampler/          # Block-based diffusion sampler
│   └── block_diffuser.py
├── memory/           # 5-layer SQLite memory system
│   └── aria.py
├── agent/            # Agent orchestration
│   └── orchestrator.py
└── api/              # RESTful API server
    └── server.py
```

## Components

### Block Diffuser (`sampler/block_diffuser.py`)
- 16-step DDPM-like diffusion process
- Block-based processing for memory efficiency
- Configurable noise schedules and sampling

### Aria Memory (`memory/aria.py`)
5-layer hierarchical memory system:
- **L1**: Working memory (1 hour TTL)
- **L2**: Short-term memory (24 hours TTL)
- **L3**: Medium-term memory (7 days TTL)
- **L4**: Long-term memory (90 days TTL)
- **L5**: Archive (permanent)

### Orchestrator (`agent/orchestrator.py`)
- Agent lifecycle management
- Task distribution and scheduling
- Dependency resolution
- Health monitoring

### API Server (`api/server.py`)
- RESTful endpoints for all components
- CORS enabled
- Health checks and status monitoring

## Quick Start

```python
# Example: Using the diffuser
from sampler.block_diffuser import BlockDiffuser, DiffusionConfig

config = DiffusionConfig(num_steps=16)
diffuser = BlockDiffuser(config)

# Example: Using memory
from memory.aria import AriaMemory

memory = AriaMemory("./memory.db")
memory.store(1, "key", {"data": "value"})
entry = memory.retrieve(1, "key")

# Example: Using orchestrator
from agent.orchestrator import Orchestrator

orch = Orchestrator()
orch.start()
agent = orch.register_agent("Worker", "processor")
task = orch.create_task("Process data", priority=3)

# Example: Starting API server
from api.server import SoeOrretServer

server = SoeOrretServer()
server.set_orchestrator(orch)
server.set_memory(memory)
server.start()
```

## API Endpoints

- `GET /health` - Health check
- `GET /status` - System status
- `GET /agents` - List agents
- `POST /agents` - Create agent
- `GET /tasks` - List tasks
- `POST /tasks` - Create task
- `GET /memory` - Search memory
- `POST /memory` - Store in memory
- `POST /sample` - Generate sample

## License

MIT
