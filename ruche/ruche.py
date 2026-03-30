#!/usr/bin/env python3
"""
SOE Ruche — Distributed Data Collection System
Collects high-quality content from multiple sources, formats as .loop files.
Author: CLAW-PRIME / Synapse for L-VS
"""
import os, sys, json, time, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import urllib.request
import urllib.error

# ─── Config ────────────────────────────────────────────────
SOE_ROOT = Path.home() / "soe"
DATASETS_DIR = SOE_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

@dataclass
class SourceConfig:
    name: str
    url_pattern: str
    keywords: List[str]
    min_score: float = 0.7
    max_items: int = 1000
    lang: str = "fr,en"

@dataclass
class RucheConfig:
    categories: List[str] = field(default_factory=list)
    max_per_category: int = 500
    sources: List[SourceConfig] = field(default_factory=list)

# ─── Quality Scorer ────────────────────────────────────────
class QualityScorer:
    """Heuristic quality scoring — no AI required."""

    RARE_WORDS_FR = {
        "paradigme", "ontologie", "épistémologie", "causalité", "heuristique",
        "métacognition", "abstraction", "généricité", "invariant", "optimisation",
        "itératif", "récursif", "convergence", "invariant", "topologie",
        "algorithme", "complexité", "apprentissage", "supervisé", "non-supervisé"
    }
    RARE_WORDS_EN = {
        "paradigm", "ontology", "epistemology", "causality", "heuristic",
        "metacognition", "abstraction", "genericity", "invariant", "optimization",
        "iterative", "recursive", "convergence", "algorithm", "complexity"
    }

    @classmethod
    def score(cls, text: str, url: str = "", lang: str = "fr,en") -> float:
        if not text or len(text) < 50:
            return 0.0
        score = 0.3
        words = text.lower().split()
        rare = cls.RARE_WORDS_FR | cls.RARE_WORDS_EN
        rare_count = sum(1 for w in words if any(r in w for r in rare))
        score += min(rare_count * 0.04, 0.3)
        score += min(len(words) / 500, 0.2)  # length bonus
        # Source quality signals
        if any(tld in url for tld in [".edu", ".gov", ".org", "arxiv", "huggingface"]):
            score += 0.1
        # Freshness — newer content gets bonus (simplified: assume recent)
        score = min(score, 1.0)
        return score

# ─── Content Fetchers ──────────────────────────────────────
class ContentFetcher:
    """Fetch content from various sources."""

    @staticmethod
    def fetch_url(url: str, timeout: int = 10) -> Optional[str]:
        """Fetch raw HTML from URL."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; SOERuche/1.0)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ct = resp.headers.get("Content-Type", "")
                if "html" not in ct and "text" not in ct:
                    return None
                data = resp.read()
                enc = resp.headers.get_content_charset() or "utf-8"
                return data.decode(enc, errors="replace")
        except Exception:
            return None

    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """Simple HTML text extraction — strip tags and scripts."""
        import re
        # Remove scripts, styles, nav
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def fetch_arxiv(arxiv_id: str) -> Optional[Dict]:
        """Fetch arXiv paper abstract + PDF text (first 2KB)."""
        abstract_url = f"https://arxiv.org/abs/{arxiv_id}"
        html = ContentFetcher.fetch_url(abstract_url)
        if not html:
            return None
        import re
        abstract = re.search(r'class="abstract mathjax">(.+?)</blockquote>', html, re.DOTALL)
        if abstract:
            text = ContentFetcher.extract_text_from_html(abstract.group(1))
            return {"prompt": f"Résume l'article arXiv {arxiv_id}:", "response": text[:2000]}
        return None

    @staticmethod
    def fetch_huggingface_dataset(name: str, split: str = "train") -> List[Dict]:
        """Try to fetch dataset metadata from HuggingFace."""
        api_url = f"https://huggingface.co/api/datasets/{name}"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "SOERuche/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return [{
                    "prompt": f"Dataset: {name}",
                    "response": data.get("description", "")[:1000],
                    "source": f"https://huggingface.co/datasets/{name}",
                    "category": "dataset",
                }]
        except Exception:
            return []

# ─── Loop Builder ──────────────────────────────────────────
class LoopBuilder:
    """Build .loop files from collected data."""

    def __init__(self, category: str, output_path: Optional[Path] = None):
        self.category = category
        self.output_path = output_path or DATASETS_DIR / f"{category}.loop"
        self.items: List[Dict[str, Any]] = []
        # Lazy import looplib
        sys.path.insert(0, str(SOE_ROOT / "core" / "looplib"))
        self._loop_writer = None
        self._venv_python = os.environ.get("VIRTUAL_ENV", "") + "/bin/python3"
        if not self._venv_python or not Path(self._venv_python).exists():
            self._venv_python = "python3"

    def add(self, item: Dict[str, Any]):
        """Add a data item with auto-quality scoring."""
        text = item.get("prompt", "") + " " + item.get("response", "")
        lang = item.get("lang", "fr,en")
        score = item.get("quality_score", QualityScorer.score(text, item.get("url",""), lang))
        self.items.append({
            **item,
            "quality_score": score,
            "category": self.category,
            "timestamp": int(time.time()),
        })

    def add_batch(self, items: List[Dict]):
        for item in items:
            self.add(item)

    def build(self, batch_size: int = 200) -> Path:
        """Write items to .loop file using the looplib."""
        if not self.items:
            print(f"⟨ruche⟩ No items for {self.category}")
            return self.output_path

        # Use the looplib writer
        try:
            from looplib import LoopWriter, LoopConfig
        except ImportError:
            print(f"⟨ruche⟩ looplib not found at {SOE_ROOT / 'core/looplib'}")
            # Fall back to JSONL
            jsonl = self.output_path.with_suffix(".jsonl")
            with open(jsonl, "w") as f:
                for item in self.items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"⟨ruche⟩ Wrote JSONL fallback → {jsonl} ({len(self.items)} items)")
            return jsonl

        # Sort by quality
        self.items.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        # Write in batches
        venv = Path.home() / "soe" / ".venv" / "bin" "python3"
        cfg = LoopConfig()
        import tempfile
        tmp_jsonl = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for item in self.items:
            tmp_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp_jsonl.close()

        from looplib import create_loop
        create_loop(tmp_jsonl.name, str(self.output_path), self.category)
        os.unlink(tmp_jsonl.name)

        print(f"⟨ruche⟩ {self.category}: {len(self.items)} items → {self.output_path}")
        return self.output_path

# ─── Ruche CLI ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SOE Ruche — Data Collector")
    parser.add_argument("--category", "-c", default="general", help="Category name")
    parser.add_argument("--sources", "-s", nargs="+", help="Source URLs or IDs")
    parser.add_argument("--arxiv", "-a", nargs="+", help="arXiv paper IDs")
    parser.add_argument("--hf", nargs="+", help="HuggingFace dataset names")
    parser.add_argument("--url", help="Single URL to scrape")
    parser.add_argument("--output", "-o", help="Output .loop path")
    args = parser.parse_args()

    builder = LoopBuilder(args.category, Path(args.output) if args.output else None)

    if args.url:
        html = ContentFetcher.fetch_url(args.url)
        if html:
            text = ContentFetcher.extract_text_from_html(html)
            score = QualityScorer.score(text, args.url)
            # Split into chunks
            chunks = [text[i:i+1000] for i in range(0, min(len(text), 5000), 1000)]
            for i, chunk in enumerate(chunks):
                builder.add({"prompt": f"Contenu de {args.url} (partie {i+1}):", "response": chunk, "url": args.url, "quality_score": score})

    if args.arxiv:
        for aid in args.arxiv:
            paper = ContentFetcher.fetch_arxiv(aid)
            if paper:
                builder.add(paper)
                print(f"⟨ruche⟩ arXiv {aid}: fetched")

    if args.hf:
        for ds_name in args.hf:
            items = ContentFetcher.fetch_huggingface_dataset(ds_name)
            builder.add_batch(items)
            print(f"⟨ruche⟩ HF {ds_name}: {len(items)} items")

    if args.sources:
        for url in args.sources:
            html = ContentFetcher.fetch_url(url)
            if html:
                text = ContentFetcher.extract_text_from_html(html)
                # Split into prompt/response chunks
                score = QualityScorer.score(text, url)
                for i in range(0, len(text)-500, 400):
                    chunk = text[i:i+600]
                    builder.add({"prompt": f"Extrait de {url}:", "response": chunk, "url": url, "quality_score": score})

    if not args.url and not args.arxiv and not args.hf and not args.sources:
        # Demo: collect some hardcoded quality content
        print("⟨ruche⟩ Demo mode — collecting hardcoded content")
        demo_items = [
            {"prompt": "Explique l'attention mechanism.", "response": "L'attention permet au modèle de ponderer l'importance de chaque token d'entrée.", "source": "soe_demo"},
            {"prompt": "Qu'est-ce que LoRA ?", "response": "LoRA = Low-Rank Adaptation. Technique de fine-tuning efficient qui decompose les poids en matrices de faible rang.", "source": "soe_demo"},
            {"prompt": "Explique les transformers.", "response": "Les transformers utilisent l'auto-attention multi-tete pour traiter des sequences en paralelo.", "source": "soe_demo"},
        ]
        builder.add_batch(demo_items)

    builder.build()
    print(f"⟨ruche⟩ Done. {len(builder.items)} items collected for '{args.category}'.")

if __name__ == "__main__":
    main()
