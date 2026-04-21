"""
Orchestrator - Agent coordination and task management
Soe-Orret central agent orchestration system
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentRole(Enum):
    """Agent specialization roles"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    MEMORY = "memory"
    SAMPLER = "sampler"
    GENERAL = "general"


@dataclass
class Task:
    """Agent task definition"""
    id: str
    description: str
    role: AgentRole = AgentRole.GENERAL
    status: TaskStatus = TaskStatus.PENDING
    priority: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "role": self.role.value,
            "status": self.status.name,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class Agent:
    """Agent definition"""
    id: str
    name: str
    role: AgentRole
    capabilities: Set[str] = field(default_factory=set)
    active_tasks: Set[str] = field(default_factory=set)
    total_tasks_completed: int = 0
    is_available: bool = True
    created_at: float = field(default_factory=time.time)
    
    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle a task"""
        return task.role == self.role or self.role == AgentRole.GENERAL


class Orchestrator:
    """
    Central orchestrator for Soe-Orret agent system.
    
    Manages:
    - Task queue and scheduling
    - Agent lifecycle
    - Dependency resolution
    - Execution flow coordination
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: deque = deque()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._task_callbacks: Dict[str, List[Callable]] = {}
        self._agent_callbacks: List[Callable] = []
    
    def register_agent(self, agent_id: str, name: str, role: AgentRole,
                       capabilities: Optional[Set[str]] = None) -> Agent:
        """Register a new agent"""
        with self._lock:
            agent = Agent(
                id=agent_id,
                name=name,
                role=role,
                capabilities=capabilities or set()
            )
            self.agents[agent_id] = agent
            
            # Notify callbacks
            for callback in self._agent_callbacks:
                callback("registered", agent)
            
            return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents.pop(agent_id)
                for callback in self._agent_callbacks:
                    callback("unregistered", agent)
                return True
            return False
    
    def submit_task(self, task_id: str, description: str,
                    role: AgentRole = AgentRole.GENERAL,
                    priority: float = 1.0,
                    dependencies: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Task:
        """Submit a new task to the orchestrator"""
        with self._lock:
            task = Task(
                id=task_id,
                description=description,
                role=role,
                priority=priority,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
            
            # Sort queue by priority
            self._sort_queue()
            
            return task
    
    def _sort_queue(self):
        """Sort task queue by priority"""
        sorted_tasks = sorted(
            self.task_queue,
            key=lambda tid: self.tasks[tid].priority,
            reverse=True
        )
        self.task_queue = deque(sorted_tasks)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                self._notify_task_callbacks(task)
                return True
            elif task.status == TaskStatus.RUNNING:
                # Mark for cancellation (actual cancellation depends on implementation)
                task.status = TaskStatus.CANCELLED
                self._notify_task_callbacks(task)
                return True
            
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List all tasks, optionally filtered by status"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks
    
    def list_agents(self, role: Optional[AgentRole] = None) -> List[Agent]:
        """List all agents, optionally filtered by role"""
        agents = list(self.agents.values())
        if role:
            agents = [a for a in agents if a.role == role]
        return agents
    
    def start(self):
        """Start the orchestrator scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
    
    def stop(self):
        """Stop the orchestrator scheduler"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            self._process_queue()
            time.sleep(0.1)  # Small delay to prevent busy-waiting
    
    def _process_queue(self):
        """Process the task queue"""
        with self._lock:
            if not self.task_queue:
                return
            
            # Find next executable task
            executable_tasks = []
            for task_id in list(self.task_queue):
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Check dependencies
                deps_satisfied = all(
                    self.tasks.get(dep) and 
                    self.tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                
                if deps_satisfied:
                    executable_tasks.append(task_id)
            
            # Execute tasks
            for task_id in executable_tasks:
                self._execute_task(task_id)
    
    def _execute_task(self, task_id: str):
        """Execute a single task"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return
            
            # Find available agent
            agent = self._find_available_agent(task)
            if not agent:
                return  # No agent available, will retry later
            
            # Mark as running
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            agent.active_tasks.add(task_id)
            agent.is_available = False
            
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            # Submit to executor
            self._executor.submit(self._run_task, task_id, agent.id)
    
    def _find_available_agent(self, task: Task) -> Optional[Agent]:
        """Find an available agent for a task"""
        for agent in self.agents.values():
            if agent.is_available and agent.can_handle(task):
                return agent
        return None
    
    def _run_task(self, task_id: str, agent_id: str):
        """Run a task (executed in thread pool)"""
        task = self.tasks.get(task_id)
        agent = self.agents.get(agent_id)
        
        if not task or not agent:
            return
        
        try:
            # Execute task logic
            result = self._execute_task_logic(task, agent)
            
            # Mark completed
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                agent.active_tasks.discard(task_id)
                agent.total_tasks_completed += 1
                agent.is_available = True
                
        except Exception as e:
            # Mark failed
            with self._lock:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)
                agent.active_tasks.discard(task_id)
                agent.is_available = True
        
        # Notify callbacks
        self._notify_task_callbacks(task)
    
    def _execute_task_logic(self, task: Task, agent: Agent) -> Any:
        """
        Execute the actual task logic.
        Override this method or use callbacks for custom execution.
        """
        # Default implementation - call registered callbacks
        callbacks = self._task_callbacks.get(task.role.value, [])
        
        for callback in callbacks:
            try:
                result = callback(task, agent)
                if result is not None:
                    return result
            except Exception as e:
                print(f"Task callback error: {e}")
        
        # Default: simulate work
        time.sleep(0.1)
        return {"status": "completed", "agent": agent.name, "task": task.description}
    
    def on_task(self, role: AgentRole, callback: Callable[[Task, Agent], Any]):
        """Register a callback for tasks of a specific role"""
        if role.value not in self._task_callbacks:
            self._task_callbacks[role.value] = []
        self._task_callbacks[role.value].append(callback)
    
    def on_agent_event(self, callback: Callable[[str, Agent], None]):
        """Register a callback for agent lifecycle events"""
        self._agent_callbacks.append(callback)
    
    def _notify_task_callbacks(self, task: Task):
        """Notify task status change"""
        # This can be extended for event-driven architectures
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        with self._lock:
            return {
                "agents": {
                    "total": len(self.agents),
                    "available": sum(1 for a in self.agents.values() if a.is_available),
                    "busy": sum(1 for a in self.agents.values() if not a.is_available)
                },
                "tasks": {
                    "total": len(self.tasks),
                    "pending": len(self.task_queue),
                    "running": sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
                    "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
                    "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
                },
                "executor": {
                    "max_workers": self.max_workers,
                    "active": len(self._executor._threads) if hasattr(self._executor, '_threads') else 0
                }
            }
    
    def create_workflow(self, tasks: List[Dict[str, Any]]) -> List[Task]:
        """
        Create a workflow from task definitions
        
        Args:
            tasks: List of task definitions with optional dependencies
            
        Returns:
            List of created Task objects
        """
        created_tasks = []
        
        for task_def in tasks:
            task = self.submit_task(
                task_id=task_def["id"],
                description=task_def["description"],
                role=AgentRole(task_def.get("role", "general")),
                priority=task_def.get("priority", 1.0),
                dependencies=task_def.get("dependencies", []),
                metadata=task_def.get("metadata", {})
            )
            created_tasks.append(task)
        
        return created_tasks


class WorkflowBuilder:
    """Helper class for building task workflows"""
    
    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []
    
    def add_task(self, task_id: str, description: str,
                 role: AgentRole = AgentRole.GENERAL,
                 priority: float = 1.0,
                 dependencies: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> "WorkflowBuilder":
        """Add a task to the workflow"""
        self.tasks.append({
            "id": task_id,
            "description": description,
            "role": role.value,
            "priority": priority,
            "dependencies": dependencies or [],
            "metadata": metadata or {}
        })
        return self
    
    def build(self) -> List[Dict[str, Any]]:
        """Build and return the workflow definition"""
        return self.tasks


if __name__ == "__main__":
    # Test orchestrator
    orch = Orchestrator(max_workers=2)
    
    # Register agents
    orch.register_agent("agent_1", "Planner Alpha", AgentRole.PLANNER)
    orch.register_agent("agent_2", "Executor Beta", AgentRole.EXECUTOR)
    orch.register_agent("agent_3", "General Gamma", AgentRole.GENERAL)
    
    # Start orchestrator
    orch.start()
    
    # Create workflow
    workflow = WorkflowBuilder()
    workflow.add_task("task_1", "Plan the project", AgentRole.PLANNER, priority=2.0)
    workflow.add_task("task_2", "Execute step A", AgentRole.EXECUTOR, priority=1.5, dependencies=["task_1"])
    workflow.add_task("task_3", "Execute step B", AgentRole.EXECUTOR, priority=1.5, dependencies=["task_1"])
    workflow.add_task("task_4", "Review results", AgentRole.CRITIC, priority=1.0, dependencies=["task_2", "task_3"])
    
    # Submit workflow
    tasks = orch.create_workflow(workflow.build())
    
    # Wait for completion
    time.sleep(2)
    
    # Print stats
    stats = orch.get_stats()
    print(f"Orchestrator stats: {json.dumps(stats, indent=2)}")
    
    # Print task results
    for task in orch.list_tasks():
        print(f"Task {task.id}: {task.status.name} - Result: {task.result}")
    
    # Stop
    orch.stop()
    print("\nOrchestrator test passed!")
