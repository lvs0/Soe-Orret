"""FastAPI REST API for Soe-Orret."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sampler.block_diffuser import BlockDiffuser, SamplerConfig
from memory.aria import AriaMemory
from agent.orchestrator import Orchestrator, TaskRole, TaskStatus


app = FastAPI(title="Soe-Orret API", version="0.1.0")

_memory = None
_orchestrator = None
_diffuser = None


def get_memory() -> AriaMemory:
    global _memory
    if _memory is None:
        _memory = AriaMemory()
    return _memory


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def get_diffuser():
    global _diffuser
    if _diffuser is None:
        _diffuser = BlockDiffuser()
    return _diffuser


# --- Sampler Models ---

class SampleRequest(BaseModel):
    batch_size: int = 1
    height: int = 64
    width: int = 64
    num_steps: int = 16
    block_mode: bool = False


class SampleResponse(BaseModel):
    shape: list[int]
    num_steps: int
    block_mode: bool


@app.post("/sampler/sample", response_model=SampleResponse)
def sample(request: SampleRequest):
    """Generate diffusion samples."""
    cfg = SamplerConfig(num_steps=request.num_steps)
    diffuser = BlockDiffuser(cfg)

    shape = (request.batch_size, 3, request.height, request.width)
    if request.block_mode:
        result = diffuser.block_sample(shape)
    else:
        result = diffuser.sample(shape)

    return SampleResponse(
        shape=list(result.shape),
        num_steps=request.num_steps,
        block_mode=request.block_mode
    )


# --- Memory Models ---

class StoreRequest(BaseModel):
    key: str
    content: str
    layer: int = 0
    priority: float = 0.5
    metadata: dict | None = None


class RetrieveRequest(BaseModel):
    key: str
    layer: int | None = None


@app.post("/memory/store")
def store_memory(request: StoreRequest):
    """Store a memory entry."""
    mem = get_memory()
    row_id = mem.store(request.key, request.content, request.layer, request.priority, request.metadata)
    return {"row_id": row_id, "key": request.key, "layer": request.layer}


@app.post("/memory/retrieve")
def retrieve_memory(request: RetrieveRequest):
    """Retrieve a memory entry."""
    mem = get_memory()
    entry = mem.retrieve(request.key, request.layer)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Key '{request.key}' not found")
    mem.touch(request.key)
    return {
        "key": entry.key,
        "content": entry.content,
        "layer": entry.layer,
        "priority": entry.priority,
        "access_count": entry.access_count,
        "created_at": entry.created_at,
        "metadata": entry.metadata
    }


@app.get("/memory/query/{pattern}")
def query_memory(pattern: str, layer: int | None = None, limit: int = 50):
    """Query memories by key pattern."""
    mem = get_memory()
    results = mem.query(pattern, layer, limit)
    return {
        "count": len(results),
        "results": [
            {"key": e.key, "content": e.content, "layer": e.layer, "priority": e.priority}
            for e in results
        ]
    }


@app.delete("/memory/{key}")
def delete_memory(key: str, layer: int | None = None):
    """Delete a memory entry."""
    mem = get_memory()
    deleted = mem.delete(key, layer)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found")
    return {"deleted": True}


# --- Agent Models ---

class TaskSubmitRequest(BaseModel):
    task_id: str
    description: str
    role: str = "executor"
    priority: float = 1.0
    dependencies: list[str] | None = None


@app.post("/agent/task")
def submit_task(request: TaskSubmitRequest):
    """Submit a new task."""
    orch = get_orchestrator()
    role = TaskRole(request.role)
    task_id = orch.submit_task(request.task_id, request.description, role, request.priority, request.dependencies)
    return {"task_id": task_id}


@app.post("/agent/execute")
def execute_next():
    """Execute the next available task."""
    orch = get_orchestrator()
    executed = orch.execute_next()
    if not executed:
        return {"executed": False, "message": "No tasks available"}
    return {"executed": True}


@app.get("/agent/task/{task_id}")
def get_task(task_id: str):
    """Get task status."""
    orch = get_orchestrator()
    task = orch.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {
        "task_id": task.task_id,
        "description": task.description,
        "role": task.role.value,
        "priority": task.priority,
        "status": task.status.value,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at
    }


@app.get("/agent/tasks")
def list_tasks(status: str | None = None):
    """List all tasks."""
    orch = get_orchestrator()
    st = TaskStatus(status) if status else None
    tasks = orch.list_tasks(st)
    return {
        "count": len(tasks),
        "tasks": [
            {"task_id": t.task_id, "description": t.description, "status": t.status.value, "priority": t.priority}
            for t in tasks
        ]
    }


@app.post("/agent/workflow")
def build_workflow(steps: list[dict]):
    """Build a workflow from step definitions."""
    orch = get_orchestrator()
    task_ids = orch.build_workflow(steps)
    return {"task_ids": task_ids}


# --- System ---

@app.get("/system/stats")
def system_stats():
    """System statistics."""
    mem = get_memory()
    orch = get_orchestrator()
    return {
        "memory_layers": mem.stats(),
        "agent_stats": orch.stats()
    }


@app.get("/")
def root():
    return {
        "name": "Soe-Orret",
        "version": "0.1.0",
        "endpoints": [
            "/sampler/sample",
            "/memory/store",
            "/memory/retrieve",
            "/memory/query/{pattern}",
            "/agent/task",
            "/agent/execute",
            "/agent/tasks",
            "/system/stats"
        ]
    }
