"""
API Server - Serveur FastAPI pour Soe-Orret
Endpoints REST pour interagir avec le système.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Modèles Pydantic

class GenerateRequest(BaseModel):
    """Requête de génération."""
    steps: int = Field(default=16, ge=1, le=100)
    num_blocks: int = Field(default=4, ge=1, le=32)
    block_size: int = Field(default=64, ge=8, le=512)
    condition: Optional[List[float]] = None


class GenerateResponse(BaseModel):
    """Réponse de génération."""
    blocks_generated: int
    config: Dict[str, Any]
    generation_time_ms: float


class StoreRequest(BaseModel):
    """Requête de stockage."""
    content: str = Field(..., min_length=1, max_length=10000)
    layer: int = Field(default=2, ge=0, le=4)
    tags: List[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    """Réponse de stockage."""
    stored: bool
    id: str
    layer: int


class QueryRequest(BaseModel):
    """Requête de recherche."""
    layer: Optional[int] = Field(default=None, ge=0, le=4)
    tags: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    since: Optional[float] = None


class MemoryEntryResponse(BaseModel):
    """Réponse d'entrée mémoire."""
    id: str
    layer: int
    layer_name: str
    content: str
    timestamp: float
    tags: List[str]
    priority: float


class QueryResponse(BaseModel):
    """Réponse de recherche."""
    count: int
    entries: List[MemoryEntryResponse]


class EventRequest(BaseModel):
    """Requête d'événement."""
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal")
    target: Optional[str] = None


class EventResponse(BaseModel):
    """Réponse d'événement."""
    emitted: bool
    event_id: str


class StatsResponse(BaseModel):
    """Réponse de statistiques."""
    uptime_seconds: float
    agents_registered: int
    agents_idle: int
    agents_running: int
    events_processed: int
    events_failed: int
    success_rate: float
    memory_stats: Optional[Dict[str, Any]] = None


class AgentStatusResponse(BaseModel):
    """Réponse de statut d'agent."""
    id: str
    name: str
    state: str
    capabilities: List[str]
    task_count: int
    success_count: int
    success_rate: float
    uptime: float


# État global

class AppState:
    """État partagé de l'application."""
    
    def __init__(self):
        self.orchestrator = None
        self.memory = None
        self.start_time = None
        self.db_path = "aria_memory.db"


app_state = AppState()


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    from agent.orchestrator import Orchestrator, SamplerAgent, MemoryAgent
    
    app_state.start_time = time.time()
    app_state.orchestrator = Orchestrator(max_workers=4)
    
    # Enregistre les agents
    sampler = SamplerAgent(app_state.orchestrator)
    memory = MemoryAgent(app_state.orchestrator, app_state.db_path)
    
    app_state.orchestrator.register_agent(sampler.agent)
    app_state.orchestrator.register_agent(memory.agent)
    
    # Démarre l'orchestrateur
    await app_state.orchestrator.start()
    
    yield
    
    # Shutdown
    await app_state.orchestrator.stop()


# Application FastAPI

app = FastAPI(
    title="Soe-Orret API",
    description="Système d'orchestration événementielle avec diffusion par blocs",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints

@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "name": "Soe-Orret",
        "version": "0.1.0",
        "description": "Système d'orchestration événementielle"
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "uptime": time.time() - app_state.start_time if app_state.start_time else 0
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Génère une séquence de blocs via diffusion.
    """
    from sampler.block_diffuser import BlockDiffuser, DiffusionConfig
    
    start = time.time()
    
    config = DiffusionConfig(
        num_steps=request.steps,
        block_size=request.block_size,
        num_blocks=request.num_blocks
    )
    
    diffuser = BlockDiffuser(config)
    
    # Génère les blocs
    condition = torch.tensor(request.condition) if request.condition else None
    blocks = diffuser.generate_sequence(num_blocks=request.num_blocks, condition=condition)
    
    elapsed = (time.time() - start) * 1000
    
    return GenerateResponse(
        blocks_generated=len(blocks),
        config={
            "steps": config.num_steps,
            "block_size": config.block_size,
            "num_blocks": config.num_blocks
        },
        generation_time_ms=elapsed
    )


@app.post("/memory/store", response_model=StoreResponse)
async def memory_store(request: StoreRequest):
    """
    Stocke une entrée en mémoire.
    """
    from memory.aria import AriaMemory, MemoryEntry
    
    aria = AriaMemory(app_state.db_path)
    
    entry = MemoryEntry(
        layer=request.layer,
        content=request.content,
        tags=request.tags,
        priority=request.priority,
        metadata=request.metadata
    )
    
    entry_id = aria.store(entry)
    
    return StoreResponse(
        stored=True,
        id=entry_id,
        layer=request.layer
    )


@app.post("/memory/query", response_model=QueryResponse)
async def memory_query(request: QueryRequest):
    """
    Recherche des entrées en mémoire.
    """
    from memory.aria import AriaMemory
    
    aria = AriaMemory(app_state.db_path)
    
    entries = aria.query(
        layer=request.layer,
        tags=request.tags if request.tags else None,
        since=request.since,
        limit=request.limit
    )
    
    layer_names = {
        0: "cache",
        1: "working",
        2: "short_term",
        3: "long_term",
        4: "archive"
    }
    
    return QueryResponse(
        count=len(entries),
        entries=[
            MemoryEntryResponse(
                id=e.id,
                layer=e.layer,
                layer_name=layer_names.get(e.layer, "unknown"),
                content=e.content[:200] + "..." if len(e.content) > 200 else e.content,
                timestamp=e.timestamp,
                tags=e.tags,
                priority=e.priority
            )
            for e in entries
        ]
    )


@app.get("/memory/stats")
async def memory_stats():
    """
    Statistiques de la mémoire.
    """
    from memory.aria import AriaMemory
    
    aria = AriaMemory(app_state.db_path)
    return aria.get_stats()


@app.post("/events/emit", response_model=EventResponse)
async def emit_event(request: EventRequest):
    """
    Émet un événement dans le système.
    """
    from agent.orchestrator import Event, EventPriority
    
    priority_map = {
        "critical": EventPriority.CRITICAL,
        "high": EventPriority.HIGH,
        "normal": EventPriority.NORMAL,
        "low": EventPriority.LOW,
        "background": EventPriority.BACKGROUND
    }
    
    priority = priority_map.get(request.priority, EventPriority.NORMAL)
    
    event = Event(
        id=f"",
        type=request.type,
        data=request.data,
        priority=priority,
        target=request.target
    )
    
    success = await app_state.orchestrator.emit(event)
    
    return EventResponse(
        emitted=success,
        event_id=event.id
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Statistiques globales du système.
    """
    from memory.aria import AriaMemory
    
    orch_stats = app_state.orchestrator.get_stats()
    
    # Récupère les stats mémoire
    try:
        aria = AriaMemory(app_state.db_path)
        mem_stats = aria.get_stats()
    except Exception:
        mem_stats = None
    
    return StatsResponse(
        uptime_seconds=orch_stats['uptime_seconds'],
        agents_registered=orch_stats['agents_registered'],
        agents_idle=orch_stats['agents_idle'],
        agents_running=orch_stats['agents_running'],
        events_processed=orch_stats['events_processed'],
        events_failed=orch_stats['events_failed'],
        success_rate=orch_stats['success_rate'],
        memory_stats=mem_stats
    )


@app.get("/agents", response_model=List[AgentStatusResponse])
async def list_agents():
    """
    Liste tous les agents et leur statut.
    """
    agents = []
    for agent_id in app_state.orchestrator.agents:
        status = app_state.orchestrator.get_agent_status(agent_id)
        if status:
            agents.append(AgentStatusResponse(**status))
    return agents


@app.get("/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent(agent_id: str):
    """
    Récupère le statut d'un agent spécifique.
    """
    status = app_state.orchestrator.get_agent_status(agent_id)
    if not status:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentStatusResponse(**status)


# Démarrage

def main():
    """Point d'entrée pour le serveur."""
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
