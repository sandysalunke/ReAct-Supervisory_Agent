import json
import hashlib
import threading
import time
from pathlib import Path
from helpers.embeddings import AzureEmbeddingWrapper
import math
from typing import Optional
from utils.tracing import tracer
from opentelemetry import trace

_CACHE_LOCK = threading.Lock()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _cache_file_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/RAG-chache"
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


def _make_key(query: str, user_groups: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n")
    h.update(user_groups.encode("utf-8"))
    return h.hexdigest()


def _security_scope(user_groups: str) -> str:
    return hashlib.sha256(
        user_groups.encode("utf-8")
    ).hexdigest()


def get_cached_response(query: str, user_groups: str):
    with tracer.start_as_current_span("cache.get_exact", attributes={"cache.type": "exact"}):
        key = _make_key(query, user_groups)
        with _CACHE_LOCK:
            cache = _load_cache()
            entry = cache.get(key)
            if entry:
                    trace.get_current_span().set_attribute("cache.hit", True)
                    return entry.get("response")
    
        trace.get_current_span().set_attribute("cache.hit", False)
        return None


def get_semantic_cached_response(
    query: str,
    query_embedding: list[float],
    user_groups: str,
    threshold: float = 0.90
) -> Optional[str]:  # Embed the new question
    with tracer.start_as_current_span("cache.get_semantic", attributes={"threshold": threshold, "cache.type": "semantic"}):
        # Security boundary
        current_scope = _security_scope(user_groups)

        with _CACHE_LOCK:
            cache = _load_cache()

        best_score = -1.0
        best_entry = None

        for entry in cache.values():

            # Do not compare against another security scope
            if entry.get("security_scope") != current_scope:
                continue

            cached_embedding = entry.get("query_embedding")

            if not cached_embedding:
                continue

            score = _cosine_similarity(
                query_embedding,
                cached_embedding
            )

            if score > best_score:
                best_score = score
                best_entry = entry

        trace.get_current_span().set_attribute("cache.score", float(best_score))
        
        if best_entry is not None and best_score >= threshold:
            trace.get_current_span().set_attribute("cache.hit", True)
            return best_entry.get("response")

        trace.get_current_span().set_attribute("cache.hit", False)
        return None


def set_cached_response(query: str, query_embedding: list[float], response: str, user_groups: str) -> None:
    key = _make_key(query, user_groups)
    security_scope = _security_scope(user_groups)
    with _CACHE_LOCK:
        cache = _load_cache()
        cache[key] = {
            "query": query, 
            "query_embedding": query_embedding, 
            "security_scope": security_scope, 
            "response": response, 
            "ts": time.time()}
        
        _write_cache(cache)

        with tracer.start_as_current_span("cache.set", attributes={"user_groups": user_groups}):
            trace.get_current_span().set_attribute("cache.key", key)
            trace.get_current_span().set_attribute("cache.stored", True)

