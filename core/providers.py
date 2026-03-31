#!/usr/bin/env python3
"""
SOE Provider System
==================
Manages LLM providers and data providers for SOE.

Providers are hot-swappable modules that provide:
- LLM inference endpoints
- Data source connections
- Tool integrations

Usage:
  python3 providers.py list
  python3 providers.py add openrouter --api-key sk-xxx
  python3 providers.py test --provider minimax
"""

import os, sys, json, argparse, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import urllib.request
import urllib.parse

SOE_ROOT = Path.home() / "soe"
PROVIDERS_DIR = SOE_ROOT / "core" / "providers"
PROVIDERS_DIR.mkdir(exist_ok=True)
PROVIDERS_FILE = PROVIDERS_DIR / "registry.json"


# ─── Provider Registry ──────────────────────────────────────
@dataclass
class Provider:
    name: str
    type: str  # "llm" | "data" | "tool"
    base_url: str
    api_key: Optional[str] = None
    models: List[str] = field(default_factory=list)
    enabled: bool = True
    score: float = 0.0  # Quality score (uptime, speed, cost)
    latency_ms: int = 0
    last_test: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("api_key", None)  # Never expose keys
        return d

    @property
    def is_llm(self) -> bool:
        return self.type == "llm"

    @property
    def is_data(self) -> bool:
        return self.type == "data"


class ProviderRegistry:
    """Hot-swappable provider registry."""

    BUILTIN_PROVIDERS: Dict[str, Provider] = {
        "minimax-portal": Provider(
            name="minimax-portal",
            type="llm",
            base_url="https://api.minimax.io/anthropic",
            models=["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"],
            score=0.95,
            config={"api": "anthropic-messages"}
        ),
        "groq": Provider(
            name="groq",
            type="llm",
            base_url="https://api.groq.com/openai/v1",
            models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            score=0.88,
            config={"reasoning": True}
        ),
        "openrouter": Provider(
            name="openrouter",
            type="llm",
            base_url="https://openrouter.ai/api/v1",
            models=["deepseek/deepseek-r1", "meta-llama/llama-3.3-70b-instruct", "qwen/qwen2.5-72b-instruct"],
            score=0.82
        ),
        "qwen-portal": Provider(
            name="qwen-portal",
            type="llm",
            base_url="https://portal.qwen.ai/v1",
            models=["coder-model", "vision-model"],
            score=0.78
        ),
        "ollama": Provider(
            name="ollama",
            type="llm",
            base_url="http://127.0.0.1:11434",
            models=["deepseek-r1:1.5b", "qwen2.5-coder:3b", "llama3.2:3b"],
            score=0.70,
            config={"api": "ollama"}
        ),
    }

    DATA_PROVIDERS: Dict[str, Provider] = {
        "github": Provider(
            name="github",
            type="data",
            base_url="https://api.github.com",
            models=[],  # Not an LLM
            score=0.90,
            config={"requires_token": True}
        ),
        "arxiv": Provider(
            name="arxiv",
            type="data",
            base_url="http://export.arxiv.org/api",
            models=[],
            score=0.85
        ),
        "huggingface": Provider(
            name="huggingface",
            type="data",
            base_url="https://huggingface.co/api",
            models=[],
            score=0.88
        ),
        "searxng": Provider(
            name="searxng",
            type="data",
            base_url="http://localhost:8080",
            models=[],
            score=0.75,
            config={"search": True}
        ),
    }

    def __init__(self):
        self.providers: Dict[str, Provider] = {}
        self.load()

    def load(self):
        if PROVIDERS_FILE.exists():
            try:
                data = json.loads(PROVIDERS_FILE.read_text())
                for p in data.get("providers", []):
                    self.providers[p["name"]] = Provider(**p)
            except Exception as e:
                print(f"⚠ Registry load error: {e}")
        # Merge builtins
        for name, p in {**self.BUILTIN_PROVIDERS, **self.DATA_PROVIDERS}.items():
            if name not in self.providers:
                self.providers[name] = p

    def save(self):
        PROVIDERS_FILE.write_text(json.dumps({
            "providers": [p.to_dict() for p in self.providers.values()],
            "updated_at": datetime.now().isoformat()
        }, indent=2))

    def add(self, name: str, provider_type: str, base_url: str, api_key: str = None, models: List[str] = None):
        p = Provider(
            name=name, type=provider_type, base_url=base_url,
            api_key=api_key, models=models or []
        )
        self.providers[name] = p
        self.save()
        print(f"✅ Provider '{name}' added ({provider_type})")

    def remove(self, name: str):
        if name in self.providers and name not in self.BUILTIN_PROVIDERS:
            del self.providers[name]
            self.save()
            print(f"✅ Provider '{name}' removed")
        else:
            print(f"⚠ Cannot remove builtin provider '{name}'")

    def test(self, name: str) -> Dict[str, Any]:
        """Test provider connectivity and latency."""
        if name not in self.providers:
            return {"error": f"Unknown provider: {name}"}

        p = self.providers[name]
        results = {
            "name": name,
            "type": p.type,
            "base_url": p.base_url,
            "tests": {}
        }

        if p.type == "llm":
            # Simple connectivity test
            try:
                start = time.time()
                if "ollama" in p.base_url:
                    url = f"{p.base_url}/api/tags"
                    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
                else:
                    url = f"{p.base_url}/models"
                    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {p.api_key or ''}"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    latency = (time.time() - start) * 1000
                    results["tests"]["connectivity"] = {
                        "status": "✅",
                        "latency_ms": round(latency, 1),
                        "http_status": resp.status
                    }
            except Exception as e:
                results["tests"]["connectivity"] = {
                    "status": "❌",
                    "error": str(e)
                }

        elif p.type == "data":
            try:
                start = time.time()
                if "github" in p.base_url:
                    url = f"{p.base_url}/repositories"
                    req = urllib.request.Request(url, headers={"User-Agent": "SOEProviders/2.0"})
                elif "arxiv" in p.base_url:
                    url = f"{p.base_url}?search_query=all:test&start=0&max_results=1"
                    req = urllib.request.Request(url, headers={"User-Agent": "SOEProviders/2.0"})
                elif "huggingface" in p.base_url:
                    url = f"{p.base_url}/datasets?search=test"
                    req = urllib.request.Request(url, headers={"User-Agent": "SOEProviders/2.0"})
                else:
                    url = p.base_url
                    req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    latency = (time.time() - start) * 1000
                    results["tests"]["connectivity"] = {
                        "status": "✅",
                        "latency_ms": round(latency, 1),
                        "http_status": resp.status
                    }
            except Exception as e:
                results["tests"]["connectivity"] = {
                    "status": "❌",
                    "error": str(e)
                }

        return results

    def list(self, filter_type: str = None):
        types = {"llm": "🤖 LLM Providers", "data": "📡 Data Providers"}
        print(f"\n{'='*50}")
        print(f"  SOE Provider Registry ({len(self.providers)} providers)")
        print(f"{'='*50}")

        by_type = {}
        for p in self.providers.values():
            if filter_type and p.type != filter_type:
                continue
            if p.type not in by_type:
                by_type[p.type] = []
            by_type[p.type].append(p)

        for t, providers in by_type.items():
            label = types.get(t, t)
            print(f"\n{label}")
            print("-" * 40)
            for p in sorted(providers, key=lambda x: x.score, reverse=True):
                status = "🟢" if p.score > 0.8 else ("🟡" if p.score > 0.5 else "🔴")
                models_str = ", ".join(p.models[:3]) + ("..." if len(p.models) > 3 else "")
                print(f"  {status} {p.name}")
                print(f"       URL: {p.base_url}")
                if p.models:
                    print(f"       Models: {models_str}")
                print(f"       Score: {p.score:.2f} | Latency: {p.latency_ms}ms")

    def get_best_llm(self, require_reasoning: bool = False) -> Optional[Provider]:
        """Get the best available LLM provider."""
        candidates = [
            p for p in self.providers.values()
            if p.is_llm and p.enabled and p.score > 0.7
        ]
        if require_reasoning:
            candidates = [p for p in candidates if p.config.get("reasoning")]
        return sorted(candidates, key=lambda p: p.score, reverse=True)[0] if candidates else None


# ─── CLI ───────────────────────────────────────────────────
def main():
    registry = ProviderRegistry()

    parser = argparse.ArgumentParser(description="SOE Provider System")
    parser.add_argument("action", choices=["list", "add", "remove", "test", "best"])
    parser.add_argument("name", nargs="?", help="Provider name")
    parser.add_argument("--type", choices=["llm", "data"], help="Provider type for 'add'")
    parser.add_argument("--url", help="Base URL for 'add'")
    parser.add_argument("--key", help="API key for 'add'")
    parser.add_argument("--models", nargs="+", help="Model list for 'add'")
    parser.add_argument("--filter", choices=["llm", "data"], help="Filter for 'list'")
    args = parser.parse_args()

    if args.action == "list":
        registry.list(filter_type=args.filter)

    elif args.action == "add":
        if not all([args.name, args.type, args.url]):
            print("⚠ Usage: providers.py add <name> --type <llm|data> --url <base_url> [--key <api_key>] [--models <models...>]")
            return
        registry.add(args.name, args.type, args.url, args.key, args.models)

    elif args.action == "remove":
        if not args.name:
            print("⚠ Usage: providers.py remove <name>")
            return
        registry.remove(args.name)

    elif args.action == "test":
        if not args.name:
            print("⚠ Usage: providers.py test <name>")
            return
        result = registry.test(args.name)
        print(json.dumps(result, indent=2))

    elif args.action == "best":
        best = registry.get_best_llm(require_reasoning=True)
        if best:
            print(f"🏆 Best LLM: {best.name} (score: {best.score:.2f})")
        else:
            print("⚠ No available LLM providers")


if __name__ == "__main__":
    main()
