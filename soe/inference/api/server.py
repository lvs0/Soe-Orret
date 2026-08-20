"""
API REST SOE-Orret — Compatible OpenAI.
Lancement : ORRET_MODEL_PATH=./orret-dllm-7b uvicorn server:app --port 8080
Compatible : curl, openai SDK, Continue.dev, OpenWebUI
"""
import json, time, uuid, os, sys, yaml
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path("~/soe").expanduser()))

# Load provider config
CONFIG_PATH = Path(__file__).parents[2] / "config" / "providers.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

def daily_tokens_used():
    cnt_path = Path(__file__).parents[2] / "counters.json"
    try:
        with open(cnt_path) as f:
            data = json.load(f)
        return data.get("fireworks_tokens_today", 0)
    except Exception:
        return 0

def choose_provider(prompt: str) -> str:
    lower = prompt.lower()
    if any(kw in lower for kw in CONFIG["routing"]["keywords_openrouter"]):
        return "openrouter/free_auto"
    if daily_tokens_used() < CONFIG["providers"]["fireworks"]["daily_limit"]:
        return "fireworks/daily"
    return "openrouter/free_auto"


app = FastAPI(title="SOE-Orret API", version="1.0.0",
              description="dLLM Symbiotique — Intelligence Artificielle Libre")
_agent = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "orret-dllm-7b"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.on_event("startup")
async def startup():
    global _agent
    model_path = os.environ.get("ORRET_MODEL_PATH", "")
    
    # Mode Ollama (priorité pour X250)
    if os.environ.get("USE_OLLAMA", "false").lower() == "true":
        print("[API] Mode Ollama activé")
        import requests
        class _OllamaSampler:
            def __init__(self, model="orret"):
                self.model = model
                self.base_url = "http://localhost:11434"
            def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                try:
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict": max_new_tokens,
                                "temperature": temperature
                            }
                        },
                        timeout=30
                    )
                    return response.json().get("response", "")
                except Exception as e:
                    print(f"[API] Erreur Ollama: {e}")
                    return f"Erreur: {str(e)}"
            @property
            def tokenizer(self):
                class _T:
                    def apply_chat_template(self, messages, **kw):
                        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                return _T()
        sampler = _OllamaSampler(os.environ.get("OLLAMA_MODEL", "orret"))
    
    # Mode GGUF direct (llama-cpp-python)
    elif not model_path or not Path(model_path).exists():
        gguf_dir = Path("~/soe/models/gguf").expanduser()
        ggufs    = list(gguf_dir.glob("*.gguf"))
        if ggufs:
            print(f"[API] Mode GGUF via llama-cpp-python: {ggufs[0]}")
            from llama_cpp import Llama
            llm = Llama(model_path=str(ggufs[0]), n_ctx=4096,
                        n_threads=os.cpu_count(), verbose=False)
            class _LlamaSampler:
                def __init__(self, llm):
                    self.llm = llm
                def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                    r = self.llm(prompt, max_tokens=max_new_tokens, temperature=temperature)
                    return r["choices"][0]["text"]
                @property
                def tokenizer(self):
                    class _T:
                        def apply_chat_template(self, messages, **kw):
                            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    return _T()
            sampler = _LlamaSampler(llm)
        else:
            print("[API] Aucun modèle trouvé — mode dégradé avec sampler minimal.")
            class _DegradedSampler:
                """Sampler minimal qui fonctionne même sans modèle. Utilisé en fallback ultime."""
                def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                    return ("[SOE en mode dégradé] — Aucun modèle chargé. "
                            "Installez Ollama ou placez un fichier .gguf dans ~/soe/models/gguf/")
                @property
                def tokenizer(self):
                    class _T:
                        def apply_chat_template(self, messages, **kw):
                            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    return _T()
            sampler = _DegradedSampler()
    
    # Mode dLLM (HF converti) - seulement si fichier existe
    else:
        if Path(model_path).exists():
            try:
                from inference.engine.dllm_sampler import BlockDiffusionSampler
                sampler = BlockDiffusionSampler(model_path)
            except Exception as e:
                print(f"[API] Erreur chargement dLLM, fallback GGUF: {e}")
                # Fallback vers GGUF
                gguf_dir = Path("~/soe/models/gguf").expanduser()
                ggufs = list(gguf_dir.glob("*.gguf"))
                if ggufs:
                    from llama_cpp import Llama
                    llm = Llama(model_path=str(ggufs[0]), n_ctx=4096,
                                n_threads=os.cpu_count(), verbose=False)
                    class _LlamaSampler:
                        def __init__(self, llm):
                            self.llm = llm
                        def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                            r = self.llm(prompt, max_tokens=max_new_tokens, temperature=temperature)
                            return r["choices"][0]["text"]
                        @property
                        def tokenizer(self):
                            class _T:
                                def apply_chat_template(self, messages, **kw):
                                    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                            return _T()
                    sampler = _LlamaSampler(llm)
                else:
                    print("[API] Fallback GGUF échoué — mode dégradé.")
                    # Créer un sampler dégradé minimal plutôt que de retourner None
                    class _DegradedSampler:
                        def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                            return ("[SOE en mode dégradé] — Aucun modèle chargé. "
                                    "Installez Ollama ou placez un fichier .gguf dans ~/soe/models/gguf/")
                        @property
                        def tokenizer(self):
                            class _T:
                                def apply_chat_template(self, messages, **kw):
                                    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                            return _T()
                    sampler = _DegradedSampler()
        else:
            print(f"[API] Modèle introuvable: {model_path}")
            # Fallback ultime : sampler dégradé
            class _DegradedSampler:
                def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kw):
                    return ("[SOE en mode dégradé] — Modèle introuvable. "
                            "Vérifiez la variable ORRET_MODEL_PATH.")
                @property
                def tokenizer(self):
                    class _T:
                        def apply_chat_template(self, messages, **kw):
                            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    return _T()
            sampler = _DegradedSampler()

    # Initialiser systèmes SOE (sans dépendances transformers)
    try:
        from memory.hierarchy       import OrretMemorySystem
        from core.m_levels          import MLevelManager
        from agents.orchestrator    import OrretAgent

        memory    = OrretMemorySystem()
        m_manager = MLevelManager()

        # Systèmes optionnels (peuvent échouer sans transformers)
        emotions = None
        reasoner = None
        plasticity = None

        try:
            from core.emotional_system import EmotionalOrret
            emotions = EmotionalOrret(sampler.generate)
        except Exception as e:
            print(f"[API] Système émotionnel désactivé: {e}")

        try:
            from core.multi_phase_reasoning import MultiPhaseReasoner
            reasoner = MultiPhaseReasoner(sampler.generate)
        except Exception as e:
            print(f"[API] Raisonnement multi-phase désactivé: {e}")

        try:
            from core.plasticity import NeuralPlasticityEngine
            plasticity = NeuralPlasticityEngine()
        except Exception as e:
            print(f"[API] Plasticité neurale désactivée: {e}")

        _agent = OrretAgent(sampler, memory, emotions, reasoner,
                            m_level_manager=m_manager,
                            plasticity_engine=plasticity, verbose=False)
        print("[API] SOE-Orret prêt → http://0.0.0.0:8080")
        print(f"[API] Niveau M actuel: {m_manager.current_level.value}")
    except Exception as e:
        print(f"[API] Erreur initialisation systèmes: {e}")
        print("[API] API en mode dégradé (sampler simple)")
        _agent = None

@app.get("/")
def root():
    return {"name": "SOE-Orret", "version": "1.0.0",
            "description": "dLLM Symbiotique — Intelligence Artificielle Libre",
            "author": "Lévy, 14 ans, France"}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": "orret-dllm-7b", "object": "model", "created": 1700000000,
         "owned_by": "lvs0"},
        {"id": "orret-ollama", "object": "model", "created": 1700000000,
         "owned_by": "lvs0"}
    ]}

@app.get("/soe/health")
def health():
    """Endpoint de santé — indique le mode actif de l'agent."""
    if not _agent:
        return {"status": "degraded", "model": None, "agent": False,
                "message": "Aucun modèle chargé — vérifiez les dépendances"}
    model_name = type(_agent.sampler).__name__
    return {"status": "ok", "model": model_name, "agent": True}

@app.get("/soe/m_level")
def get_m_level():
    """Retourne le niveau M actuel et les stats"""
    if not _agent or not _agent.m_manager:
        raise HTTPException(503, "Agent non initialisé")
    return _agent.m_manager.get_stats()

@app.post("/soe/m_level/upgrade")
def upgrade_m_level(level: str):
    """Upgrade manuel vers un niveau M"""
    if not _agent or not _agent.m_manager:
        raise HTTPException(503, "Agent non initialisé")
    from core.m_levels import MLevel
    try:
        target = MLevel(level.lower())
        success = _agent.m_manager.upgrade_to(target, reason="Upgrade manuel API")
        return {"success": success, "current_level": _agent.m_manager.current_level.value}
    except ValueError:
        raise HTTPException(400, f"Niveau invalide: {level}")

@app.post("/soe/openclaw")
async def openclaw_integration(req: dict):
    """Endpoint pour intégration OpenClaw"""
    if not _agent:
        raise HTTPException(503, "Agent non initialisé")
    
    task = req.get("task", "")
    context = req.get("context", "")
    
    # Exécuter via SOE
    prompt = f"Task OpenClaw: {task}\nContext: {context}"
    response = _agent.run(prompt)
    
    return {
        "task": task,
        "response": response,
        "m_level": _agent.m_manager.current_level.value if _agent.m_manager else "unknown"
    }

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if not _agent:
        raise HTTPException(503, "Modèle non chargé. Vérifiez ORRET_MODEL_PATH.")
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "Aucun message user")
    t0 = time.time()
    provider = choose_provider(user_msgs[-1].content)
    # Simple log
    print(f"[API] Using provider {provider}")
    response = _agent.run(user_msgs[-1].content)
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response},
                     "finish_reason": "stop"}],
        "usage":   {"prompt_tokens":     len(user_msgs[-1].content.split()),
                    "completion_tokens": len(response.split()),
                    "generation_time_s": round(time.time() - t0, 2)},
    }

if __name__ == "__main__":
    print("SOE-Orret API → http://localhost:8080")
    print("Docs         → http://localhost:8080/docs")
    uvicorn.run(app, host="0.0.0.0", port=8080)
