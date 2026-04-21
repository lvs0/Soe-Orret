"""
API Server - FastAPI-based REST API for Soe-Orret
Exposes sampler, memory, and agent orchestration endpoints
"""

import asyncio
import json
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Soe-Orret modules
import sys
sys.path.insert(0, '/tmp/soe-work')

from sampler.block_diffuser import BlockDiffuser, DiffusionConfig, SimpleNoiseModel
from memory.aria import AriaMemory, MemoryEntry
from agent.orchestrator import Orchestrator, WorkflowBuilder, AgentRole, TaskStatus


# ============================================================================
# Pydantic Models
# ============================================================================

class DiffusionRequest(BaseModel):
    """Request model for diffusion sampling"""
    batch_size: int = Field(default=1, ge=1, le=16)
    channels: int = Field(default=3, ge=1, le=16)
    height: int = Field(default=64, ge=16, le=512)
    width: int = Field(default=64, ge=16, le=512)
    num_steps: int = Field(default=16, ge=1, le=100)
    block_size: int = Field(default=64, ge=16, le=256)
    num_blocks: int = Field(default=8, ge=1, le=32)
    schedule: str = Field(default="linear", pattern="^(linear|cosine|quadratic)$")
    use_blocks: bool = Field(default=False)


class DiffusionResponse(BaseModel):
    """Response model for diffusion sampling"""
    status: str
    shape: List[int]
    num_steps: int
    sample_stats: Dict[str, float]
    message: str


class MemoryStoreRequest(BaseModel):
    """Request model for storing memory"""
    key: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1)
    layer: int = Field(default=1, ge=0, le=4)
    priority: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class MemoryResponse(BaseModel):
    """Response model for memory operations"""
    status: str
    key: str
    layer: int
    message: str


class MemoryQueryResponse(BaseModel):
    """Response model for memory queries"""
    status: str
    count: int
    entries: List[Dict[str, Any]]


class TaskSubmitRequest(BaseModel):
    """Request model for submitting a task"""
    task_id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1)
    role: str = Field(default="general", pattern="^(planner|executor|critic|memory|sampler|general)$")
    priority: float = Field(default=1.0, ge=0.0, le=10.0)
    dependencies: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class TaskResponse(BaseModel):
    """Response model for task operations"""
    status: str
    task_id: str
    task_status: str
    message: str


class WorkflowSubmitRequest(BaseModel):
    """Request model for submitting a workflow"""
    tasks: List[TaskSubmitRequest]


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    status: str
    timestamp: str
    sampler: Dict[str, Any]
    memory: Dict[str, Any]
    orchestrator: Dict[str, Any]


# ============================================================================
# Global State
# ============================================================================

class SoeOrretState:
    """Global state for the Soe-Orret system"""
    
    def __init__(self):
        self.diffuser: Optional[BlockDiffuser] = None
        self.noise_model: Optional[SimpleNoiseModel] = None
        self.memory: Optional[AriaMemory] = None
        self.orchestrator: Optional[Orchestrator] = None
        self.initialized: bool = False
    
    def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return
        
        # Initialize sampler
        config = DiffusionConfig(num_steps=16, block_size=64, num_blocks=8)
        self.diffuser = BlockDiffuser(config)
        self.noise_model = SimpleNoiseModel()
        
        # Initialize memory
        self.memory = AriaMemory(db_path="/tmp/soe-work/aria_memory.db")
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(max_workers=4)
        self._setup_default_agents()
        self.orchestrator.start()
        
        self.initialized = True
        print("Soe-Orret system initialized")
    
    def _setup_default_agents(self):
        """Setup default agents"""
        roles = [
            ("planner_1", "Planner Alpha", AgentRole.PLANNER),
            ("executor_1", "Executor Beta", AgentRole.EXECUTOR),
            ("executor_2", "Executor Gamma", AgentRole.EXECUTOR),
            ("critic_1", "Critic Delta", AgentRole.CRITIC),
            ("general_1", "General Epsilon", AgentRole.GENERAL),
        ]
        
        for agent_id, name, role in roles:
            self.orchestrator.register_agent(agent_id, name, role)
    
    def shutdown(self):
        """Shutdown all components"""
        if self.orchestrator:
            self.orchestrator.stop()
        self.initialized = False


# Global state instance
state = SoeOrretState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    state.initialize()
    yield
    # Shutdown
    state.shutdown()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Soe-Orret API",
    description="REST API for Soe-Orret: Block diffusion sampler, Aria memory, and agent orchestration",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Soe-Orret API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/sampler/sample",
            "/memory/store",
            "/memory/retrieve/{key}",
            "/memory/query/{layer}",
            "/agent/task",
            "/agent/tasks",
            "/agent/agents",
            "/system/stats"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": state.initialized,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Sampler Endpoints
# ============================================================================

@app.post("/sampler/sample", response_model=DiffusionResponse)
async def sample_diffusion(request: DiffusionRequest):
    """
    Generate samples using the block diffuser
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Update config if needed
        if request.num_steps != state.diffuser.config.num_steps:
            config = DiffusionConfig(
                num_steps=request.num_steps,
                block_size=request.block_size,
                num_blocks=request.num_blocks,
                schedule=request.schedule
            )
            diffuser = BlockDiffuser(config)
        else:
            diffuser = state.diffuser
        
        shape = (request.batch_size, request.channels, request.height, request.width)
        
        # Generate samples
        if request.use_blocks:
            samples = diffuser.sample_blocks(state.noise_model, shape)
        else:
            samples = diffuser.sample(state.noise_model, shape)
        
        # Compute stats
        sample_stats = {
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "min": float(samples.min()),
            "max": float(samples.max())
        }
        
        return DiffusionResponse(
            status="success",
            shape=list(shape),
            num_steps=request.num_steps,
            sample_stats=sample_stats,
            message=f"Generated {request.batch_size} samples with {request.num_steps} steps"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {str(e)}")


@app.get("/sampler/config")
async def get_sampler_config():
    """Get current sampler configuration"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    config = state.diffuser.config
    return {
        "num_steps": config.num_steps,
        "block_size": config.block_size,
        "num_blocks": config.num_blocks,
        "schedule": config.schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end
    }


# ============================================================================
# Memory Endpoints
# ============================================================================

@app.post("/memory/store", response_model=MemoryResponse)
async def store_memory(request: MemoryStoreRequest):
    """Store a memory entry"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        entry = state.memory.store(
            key=request.key,
            content=request.content,
            layer=request.layer,
            priority=request.priority,
            metadata=request.metadata
        )
        
        return MemoryResponse(
            status="success",
            key=entry.key,
            layer=entry.layer,
            message=f"Stored memory in layer {entry.layer}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store failed: {str(e)}")


@app.get("/memory/retrieve/{key}")
async def retrieve_memory(key: str):
    """Retrieve a memory entry by key"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    entry = state.memory.retrieve(key)
    
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {key}")
    
    return {
        "status": "success",
        "entry": {
            "key": entry.key,
            "content": entry.content,
            "layer": entry.layer,
            "priority": entry.priority,
            "access_count": entry.access_count,
            "created_at": entry.created_at,
            "accessed_at": entry.accessed_at,
            "metadata": entry.metadata
        }
    }


@app.get("/memory/query/{layer}", response_model=MemoryQueryResponse)
async def query_memory(
    layer: int,
    limit: int = Query(default=100, ge=1, le=1000),
    min_priority: float = Query(default=0.0, ge=0.0, le=1.0)
):
    """Query memories from a specific layer"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not 0 <= layer <= 4:
        raise HTTPException(status_code=400, detail="Layer must be 0-4")
    
    entries = state.memory.query_layer(layer, limit, min_priority)
    
    return MemoryQueryResponse(
        status="success",
        count=len(entries),
        entries=[{
            "key": e.key,
            "content": e.content[:100] + "..." if len(e.content) > 100 else e.content,
            "priority": e.priority,
            "access_count": e.access_count
        } for e in entries]
    )


@app.get("/memory/search")
async def search_memory(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100)
):
    """Search memories across all layers"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    results = state.memory.search(query, limit)
    
    return {
        "status": "success",
        "count": len(results),
        "results": [{
            "layer": layer,
            "key": entry.key,
            "content": entry.content[:100] + "..." if len(entry.content) > 100 else e.content,
            "priority": entry.priority
        } for layer, entry in results]
    }


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "success",
        "stats": state.memory.stats()
    }


@app.post("/memory/consolidate")
async def consolidate_memory(background_tasks: BackgroundTasks):
    """Trigger memory consolidation (runs in background)"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    def do_consolidation():
        state.memory.consolidate()
    
    background_tasks.add_task(do_consolidation)
    
    return {
        "status": "started",
        "message": "Memory consolidation started in background"
    }


# ============================================================================
# Agent Endpoints
# ============================================================================

@app.post("/agent/task", response_model=TaskResponse)
async def submit_task(request: TaskSubmitRequest):
    """Submit a new task to the orchestrator"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        role = AgentRole(request.role)
        
        task = state.orchestrator.submit_task(
            task_id=request.task_id,
            description=request.description,
            role=role,
            priority=request.priority,
            dependencies=request.dependencies,
            metadata=request.metadata
        )
        
        return TaskResponse(
            status="success",
            task_id=task.id,
            task_status=task.status.name,
            message=f"Task {task.id} submitted with status {task.status.name}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")


@app.get("/agent/task/{task_id}")
async def get_task(task_id: str):
    """Get task status and details"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    task = state.orchestrator.get_task(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    return {
        "status": "success",
        "task": task.to_dict()
    }


@app.get("/agent/tasks")
async def list_tasks(
    status: Optional[str] = Query(default=None, pattern="^(PENDING|RUNNING|COMPLETED|FAILED|CANCELLED)$")
):
    """List all tasks, optionally filtered by status"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    task_status = TaskStatus[status] if status else None
    tasks = state.orchestrator.list_tasks(task_status)
    
    return {
        "status": "success",
        "count": len(tasks),
        "tasks": [t.to_dict() for t in tasks]
    }


@app.post("/agent/workflow")
async def submit_workflow(request: WorkflowSubmitRequest):
    """Submit a workflow of multiple tasks"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Convert request tasks to dict format
        task_dicts = []
        for t in request.tasks:
            task_dicts.append({
                "id": t.task_id,
                "description": t.description,
                "role": t.role,
                "priority": t.priority,
                "dependencies": t.dependencies,
                "metadata": t.metadata
            })
        
        tasks = state.orchestrator.create_workflow(task_dicts)
        
        return {
            "status": "success",
            "count": len(tasks),
            "tasks": [t.to_dict() for t in tasks]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow submission failed: {str(e)}")


@app.get("/agent/agents")
async def list_agents(
    role: Optional[str] = Query(default=None, pattern="^(planner|executor|critic|memory|sampler|general)$")
):
    """List all registered agents"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agent_role = AgentRole(role) if role else None
    agents = state.orchestrator.list_agents(agent_role)
    
    return {
        "status": "success",
        "count": len(agents),
        "agents": [{
            "id": a.id,
            "name": a.name,
            "role": a.role.value,
            "capabilities": list(a.capabilities),
            "is_available": a.is_available,
            "total_tasks_completed": a.total_tasks_completed
        } for a in agents]
    }


@app.delete("/agent/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = state.orchestrator.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task {task_id}")
    
    return {
        "status": "success",
        "message": f"Task {task_id} cancelled"
    }


# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/system/stats", response_model=SystemStatsResponse)
async def system_stats():
    """Get complete system statistics"""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return SystemStatsResponse(
        status="success",
        timestamp=datetime.utcnow().isoformat(),
        sampler={
            "num_steps": state.diffuser.config.num_steps,
            "block_size": state.diffuser.config.block_size,
            "num_blocks": state.diffuser.config.num_blocks,
            "schedule": state.diffuser.config.schedule
        },
        memory=state.memory.stats(),
        orchestrator=state.orchestrator.get_stats()
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
