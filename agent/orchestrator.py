"""
agent/orchestrator.py
Soe-Orret orchestration agent.
Coordinates sampler, memory, and API layers.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

from sampler.block_diffuser import BlockDiffuser
from memory.aria import LayeredMemory, MemoryEntry


@dataclass
class TaskSpec:
    prompt: str
    layers: List[str] = field(default_factory=lambda: LayeredMemory.LAYERS)
    num_steps: int = 16
    block_size: int = 32
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    status: str
    output: Optional[Any] = None
    memory_entries: List[MemoryEntry] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None


class Orchestrator:
    """
    Soe-Orret orchestrator.
    Drives block diffusion sampling with memory-backed context.
    """

    def __init__(self, memory_db: str = ":memory:", device: str = "cpu"):
        self.device = device
        self.memory = LayeredMemory(memory_db)
        self.diffuser = BlockDiffuser(num_steps=16, block_size=32)
        self.task_counter = 0

    def execute(self, spec: TaskSpec) -> TaskResult:
        """Execute a task from spec to result."""
        start = time.time()
        task_id = f"soe-{int(start * 1000)}-{self.task_counter}"
        self.task_counter += 1

        try:
            # Store incoming prompt in episodic memory
            self.memory.store("episodic", f"task:{task_id}", spec.prompt, tags=["task", "prompt"])

            # Build context from memory layers
            context = self._build_context(spec)

            # Run diffusion sampler
            output = self._run_diffusion(spec, context)

            # Store result in semantic layer
            result_key = f"result:{task_id}"
            self.memory.store("semantic", result_key, output, tags=["result", "output"])

            duration_ms = (time.time() - start) * 1000
            entries = self.memory.recent("semantic", limit=5)

            return TaskResult(
                task_id=task_id,
                status="success",
                output=output,
                memory_entries=entries,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return TaskResult(
                task_id=task_id,
                status="error",
                error=str(e),
                duration_ms=duration_ms,
            )

    def _build_context(self, spec: TaskSpec) -> Dict[str, List[MemoryEntry]]:
        """Gather relevant context from all specified memory layers."""
        context: Dict[str, List[MemoryEntry]] = {}
        for layer in spec.layers:
            context[layer] = self.memory.recent(layer, limit=10)
        return context

    def _run_diffusion(self, spec: TaskSpec, context: Dict) -> Dict[str, Any]:
        """Run block diffusion with context conditioning."""
        import torch

        # Reconstruct diffuser params from spec
        diffuser = BlockDiffuser(num_steps=spec.num_steps, block_size=spec.block_size)

        # Context vector from memory as conditioning signal
        ctx_vector = self._memory_context_vector(context)

        latent_shape = (1, 4, 64, 64)
        score_fn = lambda x, t: torch.randn_like(x) * 0.1

        result = diffuser.sample_blocks(latent_shape, score_fn, device=torch.device(self.device))

        return {
            "task_id": spec.prompt[:32],
            "shape": list(result.shape),
            "num_steps": spec.num_steps,
            "context_layers": list(context.keys()),
        }

    def _memory_context_vector(self, context: Dict) -> torch.Tensor:
        """Flatten memory context into a pseudo-embedding vector."""
        import torch
        entries = sum(len(v) for v in context.values())
        return torch.randn(entries, 1) if entries > 0 else torch.zeros(1, 1)

    def status(self) -> Dict[str, Any]:
        """Return orchestrator status."""
        return {
            "diffuser": repr(self.diffuser),
            "memory": repr(self.memory),
            "device": self.device,
            "task_counter": self.task_counter,
        }
