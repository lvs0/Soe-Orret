"""Agent orchestrator for Soe-Orret."""

import uuid
import time
import heapq
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRole(Enum):
    EXECUTOR = "executor"
    PLANNER = "planner"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


@dataclass
class Task:
    task_id: str
    description: str
    role: TaskRole = TaskRole.EXECUTOR
    priority: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    result: any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def __lt__(self, other):
        return self.priority > other.priority


class Orchestrator:
    """Central orchestrator for multi-agent coordination.

    Features:
    - Priority task queue (heap-based)
    - Dependency resolution
    - Role-based agent assignment
    - Workflow builder
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._queue: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._handlers: dict[TaskRole, list[Callable]] = {role: [] for role in TaskRole}

    def submit_task(self, task_id: str, description: str, role: TaskRole = TaskRole.EXECUTOR,
                    priority: float = 1.0, dependencies: list[str] | None = None) -> str:
        """Submit a new task.

        Args:
            task_id: Unique task identifier
            description: Task description
            role: Task role
            priority: Higher = more urgent (heap-based)
            dependencies: List of task_ids that must complete first

        Returns:
            task_id
        """
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")

            task = Task(
                task_id=task_id,
                description=description,
                role=role,
                priority=priority,
                dependencies=dependencies or []
            )
            self._tasks[task_id] = task
            heapq.heappush(self._queue, (-priority, task_id))
            return task_id

    def get_next_task(self) -> Task | None:
        """Get highest priority task whose dependencies are met."""
        with self._lock:
            while self._queue:
                _, task_id = heapq.heappop(self._queue)
                task = self._tasks.get(task_id)
                if not task:
                    continue
                if task.status != TaskStatus.PENDING:
                    continue
                deps_met = all(
                    self._tasks.get(dep_id, Task("", "", TaskRole.EXECUTOR, status=TaskStatus.PENDING)).status
                    == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_met:
                    return task
            return None

    def start_task(self, task_id: str) -> bool:
        """Mark task as running."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            return True

    def complete_task(self, task_id: str, result: any = None) -> bool:
        """Mark task as completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.RUNNING:
                return False
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.RUNNING:
                return False
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()
            return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            return True

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.priority, reverse=True)

    def register_handler(self, role: TaskRole, handler: Callable[[Task], any]):
        """Register a handler for a task role."""
        self._handlers[role].append(handler)

    def execute_next(self) -> bool:
        """Execute the next available task.

        Returns:
            True if a task was executed, False if queue empty
        """
        task = self.get_next_task()
        if not task:
            return False

        self.start_task(task.task_id)
        try:
            handlers = self._handlers.get(task.role, [])
            result = None
            for handler in handlers:
                result = handler(task)
            self.complete_task(task.task_id, result)
        except Exception as e:
            self.fail_task(task.task_id, str(e))
        return True

    def build_workflow(self, steps: list[dict]) -> list[str]:
        """Build a workflow from step definitions.

        Args:
            steps: List of {"task_id": str, "role": str, "deps": [task_ids]}

        Returns:
            List of task_ids in dependency order
        """
        task_ids = []
        for step in steps:
            tid = step["task_id"]
            role_str = step.get("role", "executor")
            role = TaskRole(role_str) if isinstance(role_str, str) else role_str
            deps = step.get("deps", [])
            self.submit_task(tid, step.get("description", ""), role=role, priority=step.get("priority", 1.0), dependencies=deps)
            task_ids.append(tid)
        return task_ids

    def stats(self) -> dict:
        counts = {s.value: 0 for s in TaskStatus}
        roles = {r.value: 0 for r in TaskRole}
        for t in self._tasks.values():
            counts[t.status.value] += 1
            roles[t.role.value] += 1
        return {"total": len(self._tasks), "status": counts, "roles": roles, "queue_depth": len(self._queue)}
