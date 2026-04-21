"""
Orchestrateur d'Agents - Gestionnaire d'exécution événementielle
Coordonne les agents et gère le flux de travail.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import threading


class AgentState(Enum):
    """États possibles d'un agent."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class EventPriority(Enum):
    """Priorités des événements."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Event:
    """Événement dans le système."""
    id: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    target: Optional[str] = None  # Agent cible ou None pour broadcast
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"evt_{int(time.time() * 1000)}_{id(self)}"


@dataclass
class Task:
    """Tâche assignée à un agent."""
    id: str
    event: Event
    handler: Callable
    state: AgentState = AgentState.IDLE
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class Agent:
    """Agent dans le système."""
    id: str
    name: str
    capabilities: Set[str] = field(default_factory=set)
    handler: Optional[Callable] = None
    state: AgentState = AgentState.IDLE
    current_task: Optional[Task] = None
    task_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    def can_handle(self, event_type: str) -> bool:
        """Vérifie si l'agent peut traiter ce type d'événement."""
        return event_type in self.capabilities


class Orchestrator:
    """
    Orchestrateur d'agents événementiel.
    
    Gère:
    - Enregistrement des agents
    - Routage des événements
    - Exécution des tâches
    - Monitoring et reprise sur erreur
    """
    
    def __init__(self, max_workers: int = 4):
        self.agents: Dict[str, Agent] = {}
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.task_history: deque = deque(maxlen=1000)
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)
        self._running = False
        self._lock = threading.RLock()
        
        # Métriques
        self.events_processed = 0
        self.events_failed = 0
        self.start_time: Optional[float] = None
        
        # Callbacks
        self.on_event: Optional[Callable[[Event], None]] = None
        self.on_task_complete: Optional[Callable[[Task], None]] = None
        self.on_error: Optional[Callable[[Event, Exception], None]] = None
    
    def register_agent(self, agent: Agent) -> bool:
        """Enregistre un agent dans le système."""
        with self._lock:
            if agent.id in self.agents:
                return False
            self.agents[agent.id] = agent
            
            # Enregistre les handlers pour chaque capacité
            for cap in agent.capabilities:
                if cap not in self.event_handlers:
                    self.event_handlers[cap] = []
                if agent.handler:
                    self.event_handlers[cap].append(agent.handler)
            
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Retire un agent du système."""
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents.pop(agent_id)
            
            # Nettoie les handlers
            for cap in agent.capabilities:
                if cap in self.event_handlers and agent.handler:
                    if agent.handler in self.event_handlers[cap]:
                        self.event_handlers[cap].remove(agent.handler)
            
            return True
    
    async def emit(self, event: Event) -> bool:
        """
        Émet un événement dans le système.
        
        Args:
            event: L'événement à émettre
            
        Returns:
            True si l'événement a été mis en queue
        """
        # Vérifie le timeout
        if event.timeout and time.time() - event.timestamp > event.timeout:
            return False
        
        # Callback global
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass
        
        # Met en queue avec priorité
        priority = event.priority.value
        await self.event_queue.put((priority, time.time(), event))
        return True
    
    async def emit_simple(self, event_type: str, data: Dict[str, Any],
                          priority: EventPriority = EventPriority.NORMAL,
                          target: Optional[str] = None) -> bool:
        """Émet un événement simple."""
        event = Event(
            id=f"",
            type=event_type,
            data=data,
            priority=priority,
            target=target
        )
        return await self.emit(event)
    
    def find_agent_for_event(self, event: Event) -> Optional[Agent]:
        """Trouve le meilleur agent pour traiter un événement."""
        with self._lock:
            # Cible spécifique
            if event.target and event.target in self.agents:
                agent = self.agents[event.target]
                if agent.can_handle(event.type) and agent.state == AgentState.IDLE:
                    return agent
            
            # Recherche parmi les agents disponibles
            candidates = [
                agent for agent in self.agents.values()
                if agent.can_handle(event.type) and agent.state == AgentState.IDLE
            ]
            
            if not candidates:
                return None
            
            # Sélectionne celui avec le meilleur ratio succès/tâches
            best = min(candidates, key=lambda a: a.task_count - a.success_count * 2)
            return best
    
    async def _execute_task(self, task: Task) -> None:
        """Exécute une tâche avec gestion d'erreurs."""
        agent = self.agents.get(task.event.target) if task.event.target else None
        
        if agent:
            agent.state = AgentState.RUNNING
            agent.current_task = task
            agent.task_count += 1
        
        task.state = AgentState.RUNNING
        task.started_at = time.time()
        
        try:
            async with self._semaphore:
                # Exécute le handler
                if asyncio.iscoroutinefunction(task.handler):
                    result = await task.handler(task.event)
                else:
                    result = task.handler(task.event)
                
                task.result = result
                task.state = AgentState.COMPLETED
                task.completed_at = time.time()
                
                if agent:
                    agent.success_count += 1
                    agent.state = AgentState.IDLE
                    agent.current_task = None
                
                self.events_processed += 1
                
                if self.on_task_complete:
                    self.on_task_complete(task)
                
        except Exception as e:
            task.error = str(e)
            task.retries += 1
            
            if task.retries < task.max_retries:
                # Retry
                task.state = AgentState.IDLE
                await asyncio.sleep(0.1 * task.retries)  # Backoff exponentiel
                await self._execute_task(task)
            else:
                task.state = AgentState.FAILED
                task.completed_at = time.time()
                self.events_failed += 1
                
                if agent:
                    agent.state = AgentState.IDLE
                    agent.current_task = None
                
                if self.on_error:
                    self.on_error(task.event, e)
        
        finally:
            self.task_history.append(task)
    
    async def _process_events(self) -> None:
        """Boucle principale de traitement des événements."""
        while self._running:
            try:
                # Récupère un événement avec timeout
                try:
                    priority, ts, event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Trouve un agent
                agent = self.find_agent_for_event(event)
                
                if agent is None:
                    # Pas d'agent disponible - remet en queue
                    await asyncio.sleep(0.1)
                    await self.event_queue.put((priority, ts, event))
                    continue
                
                # Crée et exécute la tâche
                event.target = agent.id
                task = Task(
                    id=f"task_{int(time.time() * 1000)}_{id(event)}",
                    event=event,
                    handler=agent.handler
                )
                
                # Exécute en background
                asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                print(f"Error in event loop: {e}")
                await asyncio.sleep(0.1)
    
    async def start(self) -> None:
        """Démarre l'orchestrateur."""
        self._running = True
        self.start_time = time.time()
        
        # Démarre le processeur d'événements
        self._main_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Arrête l'orchestrateur."""
        self._running = False
        
        if hasattr(self, '_main_task'):
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        # Attend que les tâches en cours se terminent
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du système."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'uptime_seconds': uptime,
            'agents_registered': len(self.agents),
            'agents_idle': sum(1 for a in self.agents.values() if a.state == AgentState.IDLE),
            'agents_running': sum(1 for a in self.agents.values() if a.state == AgentState.RUNNING),
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate': self.events_processed / (self.events_processed + self.events_failed) 
                           if (self.events_processed + self.events_failed) > 0 else 1.0,
            'queue_size': self.event_queue.qsize() if hasattr(self.event_queue, 'qsize') else 0,
            'task_history_size': len(self.task_history)
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'un agent."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            'id': agent.id,
            'name': agent.name,
            'state': agent.state.name,
            'capabilities': list(agent.capabilities),
            'task_count': agent.task_count,
            'success_count': agent.success_count,
            'success_rate': agent.success_count / agent.task_count if agent.task_count > 0 else 0,
            'current_task': agent.current_task.id if agent.current_task else None,
            'uptime': time.time() - agent.created_at
        }


# Agents prédéfinis

class SamplerAgent:
    """Agent pour génération via diffusion."""
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.agent = Agent(
            id="sampler",
            name="Block Sampler",
            capabilities={"generate", "sample", "diffuse"},
            handler=self.handle
        )
    
    async def handle(self, event: Event) -> Dict[str, Any]:
        """Traite les événements de génération."""
        from sampler.block_diffuser import BlockDiffuser, DiffusionConfig
        
        config = DiffusionConfig(
            num_steps=event.data.get('steps', 16),
            block_size=event.data.get('block_size', 64)
        )
        
        diffuser = BlockDiffuser(config)
        blocks = diffuser.generate_sequence(num_blocks=event.data.get('num_blocks', 4))
        
        return {
            'blocks_generated': len(blocks),
            'config': {
                'steps': config.num_steps,
                'block_size': config.block_size
            }
        }


class MemoryAgent:
    """Agent pour gestion de la mémoire."""
    
    def __init__(self, orchestrator: Orchestrator, memory_path: str = "aria_memory.db"):
        self.orchestrator = orchestrator
        self.memory_path = memory_path
        self.agent = Agent(
            id="memory",
            name="Aria Memory",
            capabilities={"store", "retrieve", "query", "search"},
            handler=self.handle
        )
    
    async def handle(self, event: Event) -> Dict[str, Any]:
        """Traite les événements mémoire."""
        from memory.aria import AriaMemory, MemoryEntry
        
        aria = AriaMemory(self.memory_path)
        
        action = event.data.get('action')
        
        if action == 'store':
            entry = MemoryEntry(
                layer=event.data.get('layer', 2),
                content=event.data.get('content', ''),
                tags=event.data.get('tags', []),
                priority=event.data.get('priority', 0.5)
            )
            entry_id = aria.store(entry)
            return {'stored': True, 'id': entry_id}
        
        elif action == 'query':
            entries = aria.query(
                layer=event.data.get('layer'),
                tags=event.data.get('tags'),
                limit=event.data.get('limit', 10)
            )
            return {'count': len(entries), 'entries': [e.id for e in entries]}
        
        elif action == 'stats':
            return aria.get_stats()
        
        return {'error': 'Unknown action'}


def demo():
    """Démonstration de l'orchestrateur."""
    import asyncio
    
    async def run_demo():
        print("=" * 50)
        print("Soe-Orret: Orchestrator Demo")
        print("=" * 50)
        
        orch = Orchestrator(max_workers=2)
        
        # Crée et enregistre les agents
        sampler = SamplerAgent(orch)
        memory = MemoryAgent(orch, "/tmp/aria_demo.db")
        
        orch.register_agent(sampler.agent)
        orch.register_agent(memory.agent)
        
        print(f"\nAgents enregistrés: {len(orch.agents)}")
        for aid, agent in orch.agents.items():
            print(f"  - {agent.name} ({aid}): {agent.capabilities}")
        
        # Démarre
        await orch.start()
        print("\nOrchestrateur démarré")
        
        # Émet des événements
        print("\nÉmission d'événements:")
        
        await orch.emit_simple(
            "generate",
            {'steps': 16, 'num_blocks': 2, 'block_size': 32},
            priority=EventPriority.HIGH
        )
        print("  - Event: generate (HIGH)")
        
        await orch.emit_simple(
            "store",
            {'action': 'store', 'content': 'Test entry', 'layer': 1},
            priority=EventPriority.NORMAL
        )
        print("  - Event: store (NORMAL)")
        
        await orch.emit_simple(
            "query",
            {'action': 'stats'},
            priority=EventPriority.LOW
        )
        print("  - Event: query (LOW)")
        
        # Attend le traitement
        print("\nTraitement...")
        await asyncio.sleep(2)
        
        # Stats
        stats = orch.get_stats()
        print("\nStatistiques:")
        print(f"  Events processed: {stats['events_processed']}")
        print(f"  Events failed: {stats['events_failed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        
        # Arrête
        await orch.stop()
        print("\nOrchestrateur arrêté")
        
        # Cleanup
        import os
        if os.path.exists("/tmp/aria_demo.db"):
            os.unlink("/tmp/aria_demo.db")
        
        print("\n" + "=" * 50)
        print("Demo terminée!")
        print("=" * 50)
    
    asyncio.run(run_demo())


if __name__ == "__main__":
    demo()
