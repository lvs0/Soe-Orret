"""
Agent Orchestrator - Central coordination module for agent operations
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import threading
from queue import Queue
import uuid


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = auto()
    BUSY = auto()
    PAUSED = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class Task:
    """A unit of work for an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    priority: int = 5  # 1-10, lower is higher priority
    status: TaskStatus = TaskStatus.PENDING
    agent_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Agent:
    """Agent definition and state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    role: str = ""
    state: AgentState = AgentState.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    task_history: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """
    Central orchestrator for managing agents and tasks.
    Handles task distribution, agent lifecycle, and coordination.
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: Queue = Queue()
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable[[Task], Any]] = {}
        self._callbacks: Dict[str, List[Callable[[Task], None]]] = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Orchestrator")
    
    def register_agent(self, name: str, role: str, capabilities: List[str] = None,
                       metadata: Dict[str, Any] = None) -> Agent:
        """
        Register a new agent with the orchestrator.
        
        Args:
            name: Agent name
            role: Agent role/description
            capabilities: List of capability strings
            metadata: Additional agent metadata
            
        Returns:
            Created Agent instance
        """
        with self._lock:
            agent = Agent(
                name=name,
                role=role,
                capabilities=capabilities or [],
                metadata=metadata or {}
            )
            self.agents[agent.id] = agent
            self.logger.info(f"Registered agent {agent.id}: {name} ({role})")
            return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.state == AgentState.BUSY and agent.current_task:
                    # Cancel current task
                    self.cancel_task(agent.current_task)
                del self.agents[agent_id]
                self.logger.info(f"Unregistered agent {agent_id}")
                return True
            return False
    
    def create_task(self, name: str, description: str = "", priority: int = 5,
                    metadata: Dict[str, Any] = None, 
                    dependencies: List[str] = None) -> Task:
        """
        Create a new task.
        
        Args:
            name: Task name
            description: Task description
            priority: Priority level (1-10, lower is higher)
            metadata: Task metadata
            dependencies: List of task IDs this task depends on
            
        Returns:
            Created Task instance
        """
        with self._lock:
            task = Task(
                name=name,
                description=description,
                priority=priority,
                metadata=metadata or {},
                dependencies=dependencies or []
            )
            self.tasks[task.id] = task
            self.task_queue.put(task.id)
            self.logger.info(f"Created task {task.id}: {name} (priority {priority})")
            return task
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent."""
        with self._lock:
            if task_id not in self.tasks or agent_id not in self.agents:
                return False
            
            task = self.tasks[task_id]
            agent = self.agents[agent_id]
            
            if agent.state != AgentState.IDLE:
                return False
            
            task.agent_id = agent_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow().isoformat()
            
            agent.state = AgentState.BUSY
            agent.current_task = task_id
            
            self.logger.info(f"Assigned task {task_id} to agent {agent_id}")
            return True
    
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow().isoformat()
            
            if task.agent_id:
                agent = self.agents.get(task.agent_id)
                if agent:
                    agent.state = AgentState.IDLE
                    agent.current_task = None
                    agent.task_history.append(task_id)
                    agent.last_heartbeat = datetime.utcnow().isoformat()
            
            # Trigger callbacks
            self._trigger_callbacks(task)
            
            self.logger.info(f"Completed task {task_id}")
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.utcnow().isoformat()
            
            if task.agent_id:
                agent = self.agents.get(task.agent_id)
                if agent:
                    agent.state = AgentState.ERROR
                    agent.current_task = None
            
            self.logger.error(f"Task {task_id} failed: {error}")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow().isoformat()
            
            if task.agent_id:
                agent = self.agents.get(task.agent_id)
                if agent:
                    agent.state = AgentState.IDLE
                    agent.current_task = None
            
            self.logger.info(f"Cancelled task {task_id}")
            return True
    
    def register_handler(self, task_type: str, handler: Callable[[Task], Any]):
        """Register a handler function for a task type."""
        self._handlers[task_type] = handler
    
    def on_task_complete(self, task_id: str, callback: Callable[[Task], None]):
        """Register a callback for when a task completes."""
        if task_id not in self._callbacks:
            self._callbacks[task_id] = []
        self._callbacks[task_id].append(callback)
    
    def _trigger_callbacks(self, task: Task):
        """Trigger callbacks for a completed task."""
        callbacks = self._callbacks.get(task.id, [])
        for callback in callbacks:
            try:
                callback(task)
            except Exception as e:
                self.logger.error(f"Callback error for task {task.id}: {e}")
    
    def start(self):
        """Start the orchestrator scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        self.logger.info("Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.logger.info("Orchestrator stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop for task distribution."""
        while self._running:
            try:
                self._process_task_queue()
                self._check_agent_health()
                threading.Event().wait(1)  # 1 second polling interval
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
    
    def _process_task_queue(self):
        """Process pending tasks in the queue."""
        # Get pending tasks sorted by priority
        pending = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        pending.sort(key=lambda t: t.priority)
        
        for task in pending:
            # Check dependencies
            deps_satisfied = all(
                self.tasks.get(dep_id) and 
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if not deps_satisfied:
                continue
            
            # Find available agent
            available_agent = self._find_available_agent(task)
            if available_agent:
                self.assign_task(task.id, available_agent.id)
                
                # Execute if handler registered
                task_type = task.metadata.get('type', 'default')
                if task_type in self._handlers:
                    threading.Thread(
                        target=self._execute_task,
                        args=(task.id, self._handlers[task_type])
                    ).start()
    
    def _find_available_agent(self, task: Task) -> Optional[Agent]:
        """Find an available agent for a task."""
        required_caps = task.metadata.get('required_capabilities', [])
        
        for agent in self.agents.values():
            if agent.state == AgentState.IDLE:
                # Check capability match
                if not required_caps or all(cap in agent.capabilities for cap in required_caps):
                    return agent
        return None
    
    def _execute_task(self, task_id: str, handler: Callable[[Task], Any]):
        """Execute a task with the given handler."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        try:
            result = handler(task)
            self.complete_task(task_id, result)
        except Exception as e:
            self.fail_task(task_id, str(e))
    
    def _check_agent_health(self):
        """Check and update agent health status."""
        now = datetime.utcnow()
        for agent in self.agents.values():
            if agent.last_heartbeat:
                last = datetime.fromisoformat(agent.last_heartbeat)
                # Mark as error if no heartbeat for 60 seconds while busy
                if (now - last).seconds > 60 and agent.state == AgentState.BUSY:
                    agent.state = AgentState.ERROR
                    if agent.current_task:
                        self.fail_task(agent.current_task, "Agent heartbeat timeout")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status."""
        with self._lock:
            task_counts = {status.name: 0 for status in TaskStatus}
            for task in self.tasks.values():
                task_counts[task.status.name] += 1
            
            agent_counts = {state.name: 0 for state in AgentState}
            for agent in self.agents.values():
                agent_counts[agent.state.name] += 1
            
            return {
                'agents': {
                    'total': len(self.agents),
                    'by_state': agent_counts,
                    'details': [
                        {
                            'id': a.id,
                            'name': a.name,
                            'state': a.state.name,
                            'current_task': a.current_task
                        }
                        for a in self.agents.values()
                    ]
                },
                'tasks': {
                    'total': len(self.tasks),
                    'by_status': task_counts,
                    'queue_size': self.task_queue.qsize()
                },
                'running': self._running
            }
    
    def heartbeat(self, agent_id: str) -> bool:
        """Receive heartbeat from an agent."""
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            agent.last_heartbeat = datetime.utcnow().isoformat()
            
            # Recover from error state on heartbeat
            if agent.state == AgentState.ERROR:
                agent.state = AgentState.IDLE
            
            return True


# Example task handlers
def example_handler(task: Task) -> Dict[str, Any]:
    """Example task handler that simulates work."""
    import time
    time.sleep(0.5)  # Simulate work
    return {'processed': True, 'task_name': task.name}


def analysis_handler(task: Task) -> Dict[str, Any]:
    """Example analysis task handler."""
    data = task.metadata.get('data', [])
    return {
        'count': len(data),
        'sum': sum(data) if data else 0,
        'average': sum(data) / len(data) if data else 0
    }


if __name__ == "__main__":
    # Example usage
    orch = Orchestrator()
    
    # Register handlers
    orch.register_handler('default', example_handler)
    orch.register_handler('analysis', analysis_handler)
    
    # Register agents
    agent1 = orch.register_agent("Worker-1", "general_worker", ['compute', 'io'])
    agent2 = orch.register_agent("Analyzer", "data_analyzer", ['compute', 'analysis'])
    
    # Start orchestrator
    orch.start()
    
    # Create tasks
    task1 = orch.create_task("Process data", "Process incoming data batch", priority=3,
                             metadata={'type': 'default'})
    task2 = orch.create_task("Analyze metrics", "Analyze system metrics", priority=2,
                             metadata={'type': 'analysis', 'data': [1, 2, 3, 4, 5]})
    task3 = orch.create_task("Generate report", "Generate summary report", priority=5,
                             metadata={'type': 'default'},
                             dependencies=[task1.id, task2.id])
    
    # Let scheduler work
    import time
    time.sleep(3)
    
    # Check status
    status = orch.get_status()
    print(json.dumps(status, indent=2))
    
    # Stop
    orch.stop()
