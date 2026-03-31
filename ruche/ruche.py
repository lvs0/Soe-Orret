#!/usr/bin/env python3
"""
SOE Ruche v2 — Advanced Data Collection System
==============================================
Collects high-quality content from multiple sources → .loop files.
No AI required — heuristic quality scoring.

Sources: GitHub, arXiv, HuggingFace, YouTube, Web (SearxNG), RSS

Usage:
  python3 ruche.py collect --category coding --max 1000
  python3 ruche.py status
  python3 ruche.py list-sources
"""

import os, sys, json, time, argparse, hashlib, urllib.request, urllib.parse, ssl
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser

# ─── Paths ────────────────────────────────────────────────
SOE_ROOT = Path.home() / "soe"
DATASETS_DIR = SOE_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
RAW_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# ─── Quality Scorer ────────────────────────────────────────
class QualityScorer:
    """Heuristic quality scoring — no AI needed."""

    TECH_KEYWORDS = {
        "fr": ["algorithme", "architecture", "optimisation", "apprentissage", "machine",
               "réseau", "neurone", "transformer", "attention", "inférence", "entraînement",
               "modèle", "prédiction", "classification", "régression", "gradient",
               "backprop", "token", "embedding", "latent", "causal", "ontologie",
               "épistémologie", "paradigme", "heuristique", "métacognition"],
        "en": ["algorithm", "architecture", "optimization", "machine learning", "neural",
               "transformer", "attention", "inference", "training", "model", "prediction",
               "classification", "regression", "gradient", "backprop", "token", "embedding",
               "latent", "causal", "ontology", "epistemology", "paradigm", "heuristic"]
    }

    HIGH_QUALITY_TLDS = {".edu", ".gov", ".org", ".ai", ".io"}
    MEDIUM_QUALITY = {"github.com", "huggingface.co", "arxiv.org", "stackoverflow.com"}

    @classmethod
    def score(cls, text: str, url: str = "", meta: Dict = None) -> float:
        if not text or len(text) < 80:
            return 0.0

        score = 0.25
        text_lower = text.lower()
        words = text_lower.split()

        # Keyword density
        all_kw = cls.TECH_KEYWORDS["fr"] + cls.TECH_KEYWORDS["en"]
        kw_count = sum(1 for w in words if any(kw in w for kw in all_kw))
        kw_ratio = kw_count / max(len(words), 1)
        score += min(kw_ratio * 3.0, 0.35)

        # Length bonus (substantial content)
        word_count = len(words)
        if word_count > 500:
            score += 0.10
        if word_count > 2000:
            score += 0.05

        # Source quality
        url_lower = url.lower()
        if any(tld in url_lower for tld in cls.HIGH_QUALITY_TLDS):
            score += 0.15
        elif any(src in url_lower for src in cls.MEDIUM_QUALITY):
            score += 0.08

        # Freshness from meta
        if meta:
            age_days = meta.get("age_days", 999)
            if age_days < 30:
                score += 0.08
            elif age_days < 180:
                score += 0.04

        # Code detection bonus
        if any(marker in text for marker in ["```", "def ", "class ", "import ", "fn ", "pub fn", "//"]):
            score += 0.05

        return min(score, 1.0)


# ─── Source: GitHub ───────────────────────────────────────
class GitHubSource:
    """Collect from GitHub repos, READMEs, issues."""

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")

    def _headers(self) -> Dict:
        h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "SOERuche/2.0"}
        if self.token:
            h["Authorization"] = f"token {self.token}"
        return h

    def search_repos(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search GitHub for repos matching query."""
        items = []
        page = 1
        while len(items) < max_results and page <= 5:
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page=100&page={page}"
            try:
                req = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    items.extend(data.get("items", []))
                    if len(data.get("items", [])) < 100:
                        break
                    page += 1
                    time.sleep(0.5)  # Rate limit respect
            except Exception as e:
                print(f"  GitHub search error: {e}")
                break
        return items[:max_results]

    def fetch_readme(self, owner: str, repo: str) -> Optional[str]:
        """Fetch repo README content."""
        for name in ["README.md", "readme.md", "README", "readme"]:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{name}"
            try:
                req = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    if "content" in data:
                        import base64
                        return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            except:
                continue
        return None

    def collect(self, keywords: List[str], max_per_kw: int = 20) -> List[Dict]:
        """Main collection entry point."""
        results = []
        for kw in keywords:
            print(f"  🔍 GitHub: searching '{kw}'...")
            repos = self.search_repos(kw, max_results=max_per_kw)
            for repo in repos:
                owner = repo.get("owner", {}).get("login", "")
                name = repo.get("name", "")
                readme = self.fetch_readme(owner, name) if owner else None
                if readme:
                    qscore = QualityScorer.score(readme, f"github.com/{owner}/{name}")
                    results.append({
                        "content": readme[:10000],  # Limit size
                        "url": f"github.com/{owner}/{name}",
                        "source": "github",
                        "category": kw,
                        "title": repo.get("full_name", name),
                        "description": repo.get("description", ""),
                        "stars": repo.get("stargazers_count", 0),
                        "quality_score": qscore,
                        "collected_at": datetime.now().isoformat()
                    })
            time.sleep(1)
        return results


# ─── Source: arXiv ────────────────────────────────────────
class ArxivSource:
    """Collect from arXiv papers."""

    ARXIV_API = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 30) -> List[Dict]:
        results = []
        url = f"{self.ARXIV_API}?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SOERuche/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                import xml.etree.ElementTree as ET
                root = ET.parse(resp).getroot()
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    published = entry.find("atom:published", ns)
                    link = entry.find("atom:id", ns)

                    title_text = title.text.replace("\n", " ").strip() if title is not None else ""
                    summary_text = summary.text.replace("\n", " ").strip() if summary is not None else ""
                    pub_date = published.text[:10] if published is not None else ""

                    try:
                        age_days = (datetime.now() - datetime.fromisoformat(pub_date)).days
                    except:
                        age_days = 999

                    content = f"# {title_text}\n\n{summary_text}"
                    qscore = QualityScorer.score(content, "arxiv.org", {"age_days": age_days})

                    results.append({
                        "content": content,
                        "url": link.text if link is not None else "",
                        "source": "arxiv",
                        "category": query,
                        "title": title_text,
                        "quality_score": qscore,
                        "published": pub_date,
                        "age_days": age_days,
                        "collected_at": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"  arXiv error: {e}")
        return results


# ─── Source: HuggingFace ───────────────────────────────────
class HuggingFaceSource:
    """Collect dataset descriptions and metadata from HuggingFace."""

    HF_API = "https://huggingface.co/api"

    def search_datasets(self, query: str, max_results: int = 30) -> List[Dict]:
        results = []
        url = f"{self.HF_API}/datasets?search={urllib.parse.quote(query)}&sort=downloads&direction=-1&limit={max_results}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SOERuche/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for ds in data[:max_results]:
                    title = ds.get("id", "")
                    desc = ds.get("description", "") or ""
                    downloads = ds.get("downloads", 0)
                    tags = ds.get("tags", [])

                    content = f"# {title}\n\n{desc}\n\nTags: {', '.join(tags[:20])}"
                    qscore = QualityScorer.score(content, "huggingface.co")

                    # Fetch a sample of the dataset README
                    try:
                        readme_url = f"https://huggingface.co/datasets/{title}/raw/main/README.md"
                        req2 = urllib.request.Request(readme_url, headers={"User-Agent": "SOERuche/2.0"})
                        with urllib.request.urlopen(req2, timeout=5) as r2:
                            readme = r2.read().decode("utf-8", errors="ignore")[:5000]
                            content = f"# {title}\n\n{readme}"
                            qscore = QualityScorer.score(content, f"huggingface.co/datasets/{title}")
                    except:
                        pass

                    results.append({
                        "content": content[:10000],
                        "url": f"https://huggingface.co/datasets/{title}",
                        "source": "huggingface",
                        "category": query,
                        "title": title,
                        "description": desc[:500],
                        "downloads": downloads,
                        "quality_score": qscore,
                        "collected_at": datetime.now().isoformat()
                    })
                    time.sleep(0.2)
        except Exception as e:
            print(f"  HuggingFace error: {e}")
        return results


# ─── Source: YouTube ───────────────────────────────────────
class YouTubeSource:
    """Collect from YouTube via yt-dlp (subtitles + metadata)."""

    def __init__(self):
        self.yt_dlp = None
        try:
            import yt_dlp
            self.yt_dlp = yt_dlp
        except ImportError:
            print("  ⚠ yt-dlp not installed, YouTube collection disabled")
            print("  Install: pip install yt-dlp --break-system-packages")

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        if not self.yt_dlp:
            return []
        results = []
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
            "max_results": max_results,
            "extra_info_to_return": ["title", "description", "view_count", "like_count", "upload_date"],
        }
        try:
            with self.yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch:{query}", download=False)
                if not info or "entries" not in info:
                    return []
                for video in info["entries"]:
                    if not video:
                        continue
                    try:
                        age_days = 0
                        upload_date = video.get("upload_date", "")
                        if upload_date:
                            try:
                                age_days = (datetime.now() - datetime.strptime(upload_date, "%Y%m%d")).days
                            except:
                                age_days = 0

                        # Get subtitles
                        subtitle_text = ""
                        subs = video.get("automatic_chapters") or []
                        if not subs:
                            chapters = video.get("chapters") or []
                            subtitle_text = " ".join(c.get("title", "") for c in chapters[:20])
                        else:
                            subtitle_text = " ".join(c.get("title", "") for c in subs[:20])

                        description = video.get("description", "") or ""
                        content = f"# {video.get('title', '')}\n\n{description[:2000]}\n\nChapters: {subtitle_text}"
                        qscore = QualityScorer.score(content, "youtube.com", {"age_days": age_days})

                        results.append({
                            "content": content[:10000],
                            "url": f"https://youtube.com/watch?v={video.get('id', '')}",
                            "source": "youtube",
                            "category": query,
                            "title": video.get("title", ""),
                            "views": video.get("view_count", 0),
                            "likes": video.get("like_count", 0),
                            "duration_sec": video.get("duration", 0),
                            "quality_score": qscore,
                            "age_days": age_days,
                            "collected_at": datetime.now().isoformat()
                        })
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"  YouTube error: {e}")
        return results


# ─── Source: RSS Feeds ─────────────────────────────────────
class RSSSource:
    """Collect from RSS feeds."""

    FEEDS = {
        "tech_fr": [
            "https://www.lemondeinformatique.fr/rss/flux_rss.xml",
            "https://www.01net.com/rss/news/",  # may need different
        ],
        "ai_en": [
            "https://arxiv.org/rss/cs.AI",
            "https://arxiv.org/rss/cs.LG",
            "https://blog.google/technology/ai/rss/",
            "https://www.deepmind.com/blog/rss",
        ],
        "dev": [
            "https://dev.to/feed",
            "https://stackoverflow.com/feeds/tag/python",
        ]
    }

    def fetch_feed(self, url: str) -> List[Dict]:
        results = []
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "SOERuche/2.0"})
            for entry in feed.entries[:30]:
                content = ""
                if hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "content"):
                    content = entry.content[0].value
                elif hasattr(entry, "description"):
                    content = entry.description

                # Clean HTML
                import re
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()

                pub_date = ""
                if hasattr(entry, "published"):
                    pub_date = entry.published

                title = entry.get("title", "") or ""
                link = entry.get("link", "") or ""

                age_days = 999
                try:
                    from email.utils import parsedate
                    import time
                    if pub_date:
                        t = parsedate(pub_date)
                        if t:
                            age_days = (datetime.now() - datetime.fromtimestamp(time.mktime(t))).days
                except:
                    pass

                qscore = QualityScorer.score(content, link, {"age_days": age_days})
                results.append({
                    "content": f"# {title}\n\n{content[:8000]}",
                    "url": link,
                    "source": "rss",
                    "category": "general",
                    "title": title,
                    "quality_score": qscore,
                    "age_days": age_days,
                    "collected_at": datetime.now().isoformat()
                })
        except Exception as e:
            print(f"  RSS error ({url}): {e}")
        return results

    def collect(self, categories: List[str] = None) -> List[Dict]:
        all_results = []
        feeds_to_fetch = []
        if categories:
            for cat in categories:
                if cat in self.FEEDS:
                    feeds_to_fetch.extend(self.FEEDS[cat])
        else:
            for feeds in self.FEEDS.values():
                feeds_to_fetch.extend(feeds)

        feeds_to_fetch = list(set(feeds_to_fetch))  # Dedup
        for feed_url in feeds_to_fetch:
            print(f"  📡 RSS: fetching {feed_url[:60]}...")
            all_results.extend(self.fetch_feed(feed_url))
            time.sleep(0.5)
        return all_results


# ─── Source: Web Search ────────────────────────────────────
class WebSource:
    """Simple web search via SearxNG or direct fetch."""

    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        results = []
        try:
            # Try SearxNG first
            params = {"q": query, "format": "json", "engines": "google,bing", "limit": max_results}
            url = f"{self.searx_url}/search?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "SOERuche/2.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                for r in data.get("results", [])[:max_results]:
                    content = ""
                    try:
                        req2 = urllib.request.Request(r["url"], headers={"User-Agent": "SOERuche/2.0"})
                        with urllib.request.urlopen(req2, timeout=5) as r2:
                            ct = r2.headers.get("Content-Type", "")
                            if "text/html" in ct:
                                from html.parser import HTMLParser
                                class TextExtractor(HTMLParser):
                                    def __init__(self):
                                        super().__init__()
                                        self.text = []
                                        self.skip = False
                                    def handle_starttag(self, tag, attrs):
                                        if tag in ["script", "style", "nav", "header", "footer"]:
                                            self.skip = True
                                    def handle_endtag(self, tag):
                                        if tag in ["script", "style", "nav", "header", "footer"]:
                                            self.skip = False
                                    def handle_data(self, data):
                                        if not self.skip:
                                            self.text.append(data)
                                p = TextExtractor()
                                p.feed(r2.read().decode("utf-8", errors="ignore")[:50000])
                                content = " ".join(p.text)[:5000]
                            else:
                                content = r2.read().decode("utf-8", errors="ignore")[:5000]
                    except:
                        content = r.get("content", "") or ""

                    qscore = QualityScorer.score(content, r.get("url", ""))
                    results.append({
                        "content": f"# {r.get('title', '')}\n\n{content[:8000]}",
                        "url": r.get("url", ""),
                        "source": "web",
                        "category": query,
                        "title": r.get("title", ""),
                        "quality_score": qscore,
                        "engine": r.get("engine", ""),
                        "collected_at": datetime.now().isoformat()
                    })
                    time.sleep(0.3)
        except Exception as e:
            print(f"  Web search error: {e}")
        return results


# ─── Loop Writer ──────────────────────────────────────────
def write_loop(items: List[Dict], output_path: Path, category: str):
    """Write collected items to .loop file."""
    sys.path.insert(0, str(SOE_ROOT / "core" / "looplib"))
    try:
        from looplib import LoopWriter
    except ImportError:
        print("  ⚠ looplib not available, writing JSON instead")
        output_path = output_path.with_suffix(".jsonl")
        with open(output_path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✅ Wrote {len(items)} items to {output_path}")
        return

    # Write as JSONL first (looplib may need schema setup)
    jsonl_path = output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  ✅ Collected {len(items)} items → {output_path} (+ {jsonl_path})")


# ─── Main Collector ────────────────────────────────────────
class RucheCollector:
    """Main orchestrator for data collection."""

    CATEGORIES = {
        "ai_ml": {
            "keywords": ["machine learning", "deep learning", "neural network", "transformer", "LLM"],
            "github": True, "arxiv": True, "huggingface": True, "youtube": True, "rss": ["ai_en"]
        },
        "coding": {
            "keywords": ["python async", "rust programming", "golang tutorial", "typescript advanced", "algorithms"],
            "github": True, "arxiv": False, "huggingface": True, "youtube": True, "rss": ["dev"]
        },
        "math": {
            "keywords": ["linear algebra machine learning", "statistics bayesian", "calculus optimization"],
            "github": True, "arxiv": True, "huggingface": False, "youtube": True, "rss": ["ai_en"]
        },
        "science": {
            "keywords": ["physics simulation", "computational biology", "quantum computing"],
            "github": True, "arxiv": True, "huggingface": False, "youtube": True, "rss": ["ai_en"]
        },
        "philosophy": {
            "keywords": ["consciousness AI", "epistemology", "causality philosophy", "ethics artificial intelligence"],
            "github": True, "arxiv": True, "huggingface": False, "youtube": True, "rss": ["tech_fr", "ai_en"]
        }
    }

    def __init__(self, max_per_category: int = 500):
        self.max_per_category = max_per_category
        self.github = GitHubSource()
        self.arxiv = ArxivSource()
        self.huggingface = HuggingFaceSource()
        self.youtube = YouTubeSource()
        self.rss = RSSSource()
        self.web = WebSource()

    def collect_category(self, cat_name: str, config: Dict) -> List[Dict]:
        print(f"\n📦 Collecting category: {cat_name}")
        results = []
        keywords = config.get("keywords", [])

        # GitHub
        if config.get("github") and keywords:
            items_per_kw = max(5, self.max_per_category // len(keywords) // 3)
            for kw in keywords:
                print(f"  🔍 GitHub: {kw}")
                try:
                    results.extend(self.github.collect([kw], max_per_kw=items_per_kw))
                except Exception as e:
                    print(f"  ⚠ GitHub error: {e}")

        # arXiv
        if config.get("arxiv") and keywords:
            for kw in keywords[:3]:  # Limit to avoid rate limiting
                print(f"  📄 arXiv: {kw}")
                try:
                    results.extend(self.arxiv.search(kw, max_results=20))
                except Exception as e:
                    print(f"  ⚠ arXiv error: {e}")
                time.sleep(2)

        # HuggingFace
        if config.get("huggingface") and keywords:
            for kw in keywords[:3]:
                print(f"  🤗 HuggingFace: {kw}")
                try:
                    results.extend(self.huggingface.search_datasets(kw, max_results=20))
                except Exception as e:
                    print(f"  ⚠ HF error: {e}")
                time.sleep(1)

        # YouTube
        if config.get("youtube") and keywords:
            for kw in keywords[:2]:
                print(f"  🎬 YouTube: {kw}")
                try:
                    results.extend(self.youtube.search(kw, max_results=10))
                except Exception as e:
                    print(f"  ⚠ YouTube error: {e}")

        # RSS
        rss_cats = config.get("rss", [])
        if rss_cats:
            for cat in rss_cats:
                try:
                    results.extend(self.rss.collect([cat]))
                except Exception as e:
                    print(f"  ⚠ RSS error: {e}")

        # Filter by quality
        filtered = [r for r in results if r.get("quality_score", 0) >= 0.4]
        print(f"  📊 {len(results)} collected → {len(filtered)} above quality threshold")
        return filtered

    def collect_all(self, categories: List[str] = None) -> Dict[str, List[Dict]]:
        if categories is None:
            categories = list(self.CATEGORIES.keys())

        all_data = {}
        for cat_name in categories:
            if cat_name not in self.CATEGORIES:
                print(f"⚠ Unknown category: {cat_name}")
                continue
            items = self.collect_category(cat_name, self.CATEGORIES[cat_name])
            if items:
                all_data[cat_name] = items
                # Write immediately
                out_path = RAW_DIR / f"{cat_name}_{datetime.now().strftime('%Y%m%d')}.loop"
                write_loop(items, out_path, cat_name)
        return all_data


# ─── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SOE Ruche v2 — Data Collection")
    parser.add_argument("action", choices=["collect", "status", "list-sources"], help="Action")
    parser.add_argument("--category", "-c", nargs="+", help="Categories to collect")
    parser.add_argument("--max", "-m", type=int, default=500, help="Max items per category")
    parser.add_argument("--all", "-a", action="store_true", help="Collect all categories")
    args = parser.parse_args()

    if args.action == "list-sources":
        print("📡 Available sources:")
        print("  • GitHub     — repos, READMEs, issues")
        print("  • arXiv      — research papers (cs.AI, cs.LG)")
        print("  • HuggingFace — datasets, model cards")
        print("  • YouTube    — videos, subtitles, chapters")
        print("  • RSS feeds  — tech news, dev blogs")
        print("  • Web search — via SearxNG (optional)")
        print("\n📂 Categories:", ", ".join(RucheCollector.CATEGORIES.keys()))
        return

    if args.action == "status":
        print("📊 Ruche Status")
        print(f"  Raw dir: {RAW_DIR}")
        print(f"  Processed dir: {PROCESSED_DIR}")
        for f in sorted(RAW_DIR.glob("*")):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  • {f.name}: {size_mb:.1f} MB")
        return

    if args.action == "collect":
        collector = RucheCollector(max_per_category=args.max)
        cats = args.category or (list(RucheCollector.CATEGORIES.keys()) if args.all else ["ai_ml", "coding"])
        print(f"🚀 Starting collection: {cats}")
        start = time.time()
        result = collector.collect_all(cats)
        elapsed = time.time() - start
        total = sum(len(v) for v in result.values())
        print(f"\n✅ Done in {elapsed:.0f}s — {total} items collected across {len(result)} categories")
        return


if __name__ == "__main__":
    main()
