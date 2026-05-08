"""
Per-collection ingest metadata (last-ingest timestamp, source mode,
pages added) stored at ./rag_chroma_db/.ingest_meta.json. Lives next
to the Chroma DB so its scoped per-machine and never committed to git.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Optional

META_PATH = Path("rag_chroma_db") / ".ingest_meta.json"


def _load() -> Dict[str, dict]:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return {}


def _save(data: Dict[str, dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def record_ingest(collection: str, *, mode: str, pages: int = 0,
                  chunks_added: int = 0) -> None:
    data = _load()
    data[collection] = {
        "ts": time.time(),
        "mode": mode,
        "pages": pages,
        "chunks_added": chunks_added,
    }
    _save(data)


def get_meta(collection: str) -> Optional[dict]:
    return _load().get(collection)


def all_meta() -> Dict[str, dict]:
    return _load()


def forget(collection: str) -> None:
    data = _load()
    if collection in data:
        del data[collection]
        _save(data)


def humanize_ago(ts: float) -> str:
    if not ts:
        return "no ingest yet"
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} hr ago"
    return f"{int(delta // 86400)} day ago"
