import json
import hashlib
import threading
import time
from pathlib import Path

_CACHE_LOCK = threading.Lock()


def _cache_file_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "llm_cache.json"


def _load_cache() -> dict:
    path = _cache_file_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(cache: dict) -> None:
    path = _cache_file_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _make_key(query: str, context: str, user_groups: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n")
    h.update(user_groups.encode("utf-8"))
    h.update(b"\n")
    h.update(context.encode("utf-8"))
    return h.hexdigest()


def get_cached_response(query: str, context: str, user_groups: str):
    key = _make_key(query, context, user_groups)
    with _CACHE_LOCK:
        cache = _load_cache()
        entry = cache.get(key)
        if entry:
            return entry.get("response")
    return None


def set_cached_response(query: str, context: str, response: str, user_groups: str) -> None:
    key = _make_key(query, context, user_groups)
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[key] = {"response": response, "ts": time.time()}
        _write_cache(cache)
