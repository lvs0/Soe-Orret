# SOE Architecture Document

## Overview

SOE (Symbiotic Operating Environment) is a decentralized AI ecosystem designed
for edge deployment. It consists of four core pillars:

1. **SOE-O1** — Multi-phase reasoning engine
2. **SOE-Orret** — Block Diffusion Language Model (dLLM)
3. **Synapse Runtime** — P2P coordination layer
4. **Format Enchante** — Adaptive compression

## System Architecture

```
                    ┌─────────────────────────────┐
                    │        USER INPUT            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    SOE ORCHESTRATOR          │
                    │  (6-Phase Cognitive Cycle)   │
                    └──┬──────┬──────┬──────┬─────┘
                       │      │      │      │
          ┌────────────▼┐ ┌───▼────┐ ┌▼─────┐ ┌▼──────────┐
          │  REASONING   │ │MEMORY  │ │TOOLS │ │ INFERENCE  │
          │  SOE-O1      │ │ ARIA   │ │      │ │ ENGINE     │
          │              │ │        │ │      │ │            │
          │ ┌──────────┐ │ │L1 Work │ │Web   │ │ Ollama     │
          │ │6-Phase   │ │ │L2 Epi  │ │Code  │ │ Groq       │
          │ │CoT/ToT   │ │ │L3 Sem  │ │Calc  │ │ GGUF       │
          │ │Pred.Code │ │ │L4 Proc │ │Files │ │ dLLM       │
          │ │MetaCog   │ │ │L5 World│ │      │ │            │
          │ └──────────┘ │ │        │ │      │ │            │
          └──────────────┘ └────────┘ └──────┘ └────────────┘
                       │
          ┌────────────▼──────────────────────┐
          │        SECURITY (POLYGONE)         │
          │  AES-256-GCM | Shamir SSS | Audit  │
          └───────────────────────────────────┘
```

## Data Flow

1. **Input** arrives at the Orchestrator
2. **Perception**: Extract entities and classify intent (InferenceEngine)
3. **Intuition**: Retrieve relevant memories (ARIA L1-L3)
4. **Analysis**: Deep reasoning via CoT, ToT, or Predictive Coding
5. **Synthesis**: Construct answer, invoke tools if needed
6. **Validation**: Self-critique and constraint checking
7. **Expression**: Format for human readability
8. **Learning**: Store episode, update emotional state, metacognitive reflection

## Key Design Decisions

### Why dLLM over Autoregressive?
- Bidirectional context: each token sees all others
- Parallel block generation
- Revision of low-confidence tokens
- Solves the Reversal Curse

### Why 6-Phase Reasoning?
- Matches human cognitive cycle (Perception -> ... -> Expression)
- Complexity-adaptive (simple queries skip phases)
- Each phase produces a ThoughtPacket with quality/confidence scores

### Why 5-Layer Memory?
- Mimics biological memory consolidation
- Automatic TTL and promotion between layers
- FAISS integration for semantic search at L3+

### Why M-Levels?
- Adaptive capability scaling based on available resources
- Progressive enhancement: M1 runs anywhere, M9 needs massive compute
- Experience-based progression system

## File Structure

```
soe/src/soe/
├── core/types.py           # 180 lines  — all shared types
├── memory/aria.py          # 200 lines  — SQLite 5-layer memory
├── memory/hierarchy.py     # 180 lines  — high-level memory API
├── reasoning/engine.py     # 120 lines  — 6-phase reasoner
├── reasoning/predictive_coding.py  # 180 lines — Free Energy brain
├── reasoning/metacognition.py      # 120 lines — self-awareness
├── inference/engine.py     # 140 lines  — multi-backend inference
├── agents/manager.py       # 70 lines   — agent routing
├── tools/registry.py       # 80 lines   — tool registry
├── orchestration/soe_orchestrator.py # 200 lines — central hub
├── security/polygone.py    # 100 lines  — encryption layer
├── training/dllm.py        # 60 lines   — dLLM config
└── api/server.py           # 120 lines  — FastAPI server
```

Total: ~1,750 lines of clean, modular Python.

## Legacy Code Mapping

| New Module | Legacy Source |
|-----------|--------------|
| `core/types.py` | `core/m_levels.py`, `core/emotional_system.py` |
| `memory/aria.py` | `memory/aria.py`, `memory/hierarchy.py` |
| `reasoning/engine.py` | `core/multi_phase_reasoning.py` |
| `reasoning/predictive_coding.py` | `src/reasoning/predictive_coding.py` |
| `reasoning/metacognition.py` | `src/reasoning/metacognition.py` |
| `inference/engine.py` | `src/inference/engine.py`, `inference/api/server.py` |
| `agents/manager.py` | `src/agents/manager.py`, `agents/orchestrator.py` |
| `orchestration/soe_orchestrator.py` | `src/orchestrator/soe_orchestrator.py` |
| `security/polygone.py` | `core/polygone_integration.py` |
| `training/dllm.py` | `training/dllm_architecture.py` |
