"""Embedding engine for archival memory vector search.

Supports multiple backends:
- Local: sentence-transformers (all-MiniLM-L6-v2, bge-small-zh, etc.)
- Local: Ollama embedding models (nomic-embed-text, etc.)
- Remote: OpenAI (text-embedding-3-small, text-embedding-3-large)
- Remote: Any OpenAI-compatible API (Azure, Cohere, 智谱, vLLM, etc.)

Configuration via config.yaml memory.embedding section.
Graceful degradation: if embedding unavailable, returns None.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default request timeout (seconds) for remote backends.
_DEFAULT_TIMEOUT = 30.0


class _EmbeddingCache:
    """Bounded LRU cache for embedding vectors.

    Memory budget: 5000 entries * 384 dims * 4 bytes/float ~= 7.5MB.
    Adjustable via config.

    Thread-safety: relies on CPython GIL for atomic OrderedDict ops.
    Sufficient for the Dream Engine's background thread + main loop.
    """

    def __init__(self, max_entries: int = 5000):
        self._store: "OrderedDict[str, List[float]]" = OrderedDict()
        self._max = max(1, int(max_entries))
        self._hits = 0
        self._misses = 0

    def get(self, text: str):
        """Look up cached vector. Returns list or None."""
        key = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, text: str, vector):
        """Store vector, evicting LRU if needed."""
        if vector is None:
            return
        key = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        self._store[key] = vector
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def clear(self):
        """Clear all cached embeddings."""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_entries": self._max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


class EmbeddingBackend(ABC):
    """Abstract base for embedding backends."""

    @abstractmethod
    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text. Returns None on failure."""

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts. Default: iterate embed()."""
        return [self.embed(t) for t in texts]

    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions. Returns 0 if unknown/unavailable."""

    def is_available(self) -> bool:
        """Whether the backend is ready for use. Subclasses may override."""
        return True


# ---------------------------------------------------------------------------
# Local backends
# ---------------------------------------------------------------------------


class SentenceTransformerBackend(EmbeddingBackend):
    """Local embedding via sentence-transformers library.

    Lazy-loads the library and model on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache: bool = True):
        self._model_name = model_name
        self._model = None
        self._cache = cache
        self._dimensions: Optional[int] = None
        self._loaded = False
        self._unavailable = False

    def _load_model(self) -> None:
        """Lazy load. If sentence-transformers is not installed, mark unavailable."""
        if self._loaded or self._unavailable:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
            try:
                self._dimensions = int(self._model.get_sentence_embedding_dimension())
            except Exception:  # pragma: no cover - defensive
                self._dimensions = None
            self._loaded = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Run: uv pip install sentence-transformers"
            )
            self._model = None
            self._unavailable = True
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model '%s': %s", self._model_name, exc
            )
            self._model = None
            self._unavailable = True

    def embed(self, text: str) -> Optional[List[float]]:
        self._load_model()
        if self._model is None:
            return None
        try:
            vec = self._model.encode(text, convert_to_numpy=False, show_progress_bar=False)
            # Normalize to plain python list of floats.
            try:
                return [float(x) for x in vec]
            except TypeError:
                # Some versions may return tensor-like objects.
                return [float(x) for x in list(vec)]
        except Exception as exc:
            logger.warning("SentenceTransformer embed failed: %s", exc)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        self._load_model()
        if self._model is None:
            return [None for _ in texts]
        try:
            vecs = self._model.encode(
                list(texts), convert_to_numpy=False, show_progress_bar=False
            )
            results: List[Optional[List[float]]] = []
            for v in vecs:
                try:
                    results.append([float(x) for x in v])
                except Exception:
                    results.append(None)
            return results
        except Exception as exc:
            logger.warning("SentenceTransformer batch embed failed: %s", exc)
            return [None for _ in texts]

    def dimensions(self) -> int:
        if self._dimensions is None and not self._loaded and not self._unavailable:
            self._load_model()
        return int(self._dimensions or 0)

    def is_available(self) -> bool:
        if not self._loaded and not self._unavailable:
            self._load_model()
        return self._model is not None


class OllamaBackend(EmbeddingBackend):
    """Local embedding via Ollama API (http://localhost:11434)."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dimensions: Optional[int] = None
        self._unavailable = False

    def _post_embedding(self, text: str) -> Optional[List[float]]:
        try:
            import httpx  # type: ignore
        except ImportError:
            logger.warning("httpx is required for OllamaBackend but is not installed")
            self._unavailable = True
            return None

        url = f"{self._base_url}/api/embeddings"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    url, json={"model": self._model_name, "prompt": text}
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Ollama embedding request failed: %s", exc)
            return None

        embedding = data.get("embedding") if isinstance(data, dict) else None
        if not embedding or not isinstance(embedding, list):
            logger.warning("Ollama returned no embedding for model '%s'", self._model_name)
            return None
        try:
            vec = [float(x) for x in embedding]
        except (TypeError, ValueError) as exc:
            logger.warning("Ollama embedding payload was not numeric: %s", exc)
            return None
        if self._dimensions is None:
            self._dimensions = len(vec)
        return vec

    def embed(self, text: str) -> Optional[List[float]]:
        if self._unavailable:
            return None
        return self._post_embedding(text)

    def dimensions(self) -> int:
        return int(self._dimensions or 0)

    def is_available(self) -> bool:
        return not self._unavailable


# ---------------------------------------------------------------------------
# Remote backends (OpenAI / OpenAI-compatible)
# ---------------------------------------------------------------------------


class _OpenAIClientMixin:
    """Shared helpers for OpenAI-style embedding backends."""

    _client: Any = None
    _client_loaded: bool = False
    _unavailable: bool = False

    def _build_client(
        self,
        api_key: Optional[str],
        base_url: Optional[str] = None,
    ) -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            logger.warning(
                "openai package not installed; remote embedding backend unavailable"
            )
            self._unavailable = True
            return None

        kwargs: Dict[str, Any] = {"timeout": _DEFAULT_TIMEOUT}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return OpenAI(**kwargs)
        except Exception as exc:
            logger.warning("Failed to create OpenAI client: %s", exc)
            self._unavailable = True
            return None

    def _embed_via_client(
        self,
        client: Any,
        model_name: str,
        texts: List[str],
        explicit_dimensions: Optional[int],
    ) -> Optional[List[List[float]]]:
        if client is None:
            return None
        # Data Shield: redact sensitive content before sending to remote API
        try:
            from agent.data_shield import get_data_shield
            _ds = get_data_shield()
            if _ds and _ds.enabled and _ds.shield_embedding:
                shielded_texts = []
                for t in texts:
                    shielded, _ = _ds.shield_text(t)
                    shielded_texts.append(shielded)
                texts = shielded_texts
        except Exception:
            pass  # Silently fall through with original texts
        kwargs: Dict[str, Any] = {"model": model_name, "input": texts}
        if explicit_dimensions:
            kwargs["dimensions"] = int(explicit_dimensions)
        try:
            resp = client.embeddings.create(**kwargs)
        except TypeError:
            # Some compatible servers don't accept `dimensions` kwarg.
            kwargs.pop("dimensions", None)
            try:
                resp = client.embeddings.create(**kwargs)
            except Exception as exc:
                logger.warning("Embedding API call failed: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Embedding API call failed: %s", exc)
            return None

        try:
            data = resp.data
            return [list(item.embedding) for item in data]
        except Exception as exc:
            logger.warning("Failed to parse embedding response: %s", exc)
            return None


class OpenAIBackend(_OpenAIClientMixin, EmbeddingBackend):
    """Remote embedding via OpenAI API.

    Uses the openai SDK already in core dependencies.
    """

    # Known dimension defaults for the well-known OpenAI models.
    _DEFAULT_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._model_name = model_name
        self._explicit_dims = dimensions
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("EMBEDDING_API_KEY")
        )
        self._client = None
        self._client_loaded = False
        self._unavailable = False
        self._dimensions: Optional[int] = (
            dimensions if dimensions else self._DEFAULT_DIMS.get(model_name)
        )

    def _ensure_client(self) -> Any:
        if self._client_loaded:
            return self._client
        if not self._api_key:
            logger.warning(
                "OpenAIBackend has no API key (set OPENAI_API_KEY or EMBEDDING_API_KEY)"
            )
            self._unavailable = True
            self._client_loaded = True
            return None
        self._client = self._build_client(self._api_key)
        self._client_loaded = True
        return self._client

    def embed(self, text: str) -> Optional[List[float]]:
        result = self.embed_batch([text])
        if not result or result[0] is None:
            return None
        return result[0]

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        client = self._ensure_client()
        if client is None:
            return [None for _ in texts]
        vectors = self._embed_via_client(
            client, self._model_name, list(texts), self._explicit_dims
        )
        if vectors is None:
            return [None for _ in texts]
        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
        return [list(v) for v in vectors]

    def dimensions(self) -> int:
        return int(self._dimensions or 0)

    def is_available(self) -> bool:
        if self._unavailable:
            return False
        return bool(self._api_key)


class OpenAICompatibleBackend(_OpenAIClientMixin, EmbeddingBackend):
    """Remote embedding via any OpenAI-compatible API endpoint.

    Works with Azure OpenAI, Cohere, 智谱API, 通义千问, self-hosted vLLM, etc.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._model_name = model_name
        self._base_url = base_url
        self._explicit_dims = dimensions
        self._api_key = (
            api_key
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"  # vLLM and similar self-hosted services often accept any key.
        )
        self._client = None
        self._client_loaded = False
        self._unavailable = False
        self._dimensions: Optional[int] = dimensions

    def _ensure_client(self) -> Any:
        if self._client_loaded:
            return self._client
        if not self._base_url:
            logger.warning("OpenAICompatibleBackend requires a base_url")
            self._unavailable = True
            self._client_loaded = True
            return None
        self._client = self._build_client(self._api_key, self._base_url)
        self._client_loaded = True
        return self._client

    def embed(self, text: str) -> Optional[List[float]]:
        result = self.embed_batch([text])
        if not result or result[0] is None:
            return None
        return result[0]

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        client = self._ensure_client()
        if client is None:
            return [None for _ in texts]
        vectors = self._embed_via_client(
            client, self._model_name, list(texts), self._explicit_dims
        )
        if vectors is None:
            return [None for _ in texts]
        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
        return [list(v) for v in vectors]

    def dimensions(self) -> int:
        return int(self._dimensions or 0)

    def is_available(self) -> bool:
        return not self._unavailable and bool(self._base_url)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


_BACKEND_ALIASES = {
    "sentence_transformers": "sentence_transformers",
    "sentence-transformers": "sentence_transformers",
    "st": "sentence_transformers",
    "local": "sentence_transformers",
    "ollama": "ollama",
    "openai": "openai",
    "openai_compatible": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "compatible": "openai_compatible",
}


class EmbeddingEngine:
    """Main embedding engine - selects and manages the active backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize from config dict (memory.embedding section).

        config keys:
          backend: str - "sentence_transformers" | "openai" | "openai_compatible" | "ollama"
          model: str - model name
          dimensions: int - vector dimensions (optional, auto-detected for some backends)
          cache_model: bool - cache loaded model in process (default True)
          api_key: str - API key for remote backends (falls back to EMBEDDING_API_KEY env var)
          base_url: str - custom endpoint for openai_compatible backend
        """
        self._config: Dict[str, Any] = dict(config or {})
        self._backend: Optional[EmbeddingBackend] = None
        self._unavailable = False
        self._dimensions_override: Optional[int] = self._config.get("dimensions")
        # LRU cache for repeated text embeddings.
        _cache_enabled = bool(self._config.get("cache_enabled", True))
        _cache_max = int(self._config.get("cache_max_entries", 5000) or 5000)
        self._cache: Optional[_EmbeddingCache] = (
            _EmbeddingCache(_cache_max) if _cache_enabled else None
        )
        self._init_backend()

        # Track the most-recently-created engine so the gateway memory
        # monitor can reach the underlying cache during emergency cleanup.
        global _active_engine
        _active_engine = self

    # ------------------------------------------------------------------ init

    def _init_backend(self) -> None:
        cfg = self._config
        raw_backend = str(cfg.get("backend") or "sentence_transformers").strip().lower()
        backend_kind = _BACKEND_ALIASES.get(raw_backend)
        if backend_kind is None:
            logger.warning("Unknown embedding backend '%s'; embedding disabled", raw_backend)
            self._unavailable = True
            return

        try:
            if backend_kind == "sentence_transformers":
                model = cfg.get("model") or "all-MiniLM-L6-v2"
                cache = bool(cfg.get("cache_model", True))
                self._backend = SentenceTransformerBackend(model_name=model, cache=cache)
            elif backend_kind == "ollama":
                model = cfg.get("model") or "nomic-embed-text"
                base_url = cfg.get("base_url") or "http://localhost:11434"
                self._backend = OllamaBackend(model_name=model, base_url=base_url)
            elif backend_kind == "openai":
                model = cfg.get("model") or "text-embedding-3-small"
                self._backend = OpenAIBackend(
                    model_name=model,
                    api_key=cfg.get("api_key"),
                    dimensions=cfg.get("dimensions"),
                )
            elif backend_kind == "openai_compatible":
                model = cfg.get("model")
                base_url = cfg.get("base_url")
                if not model or not base_url:
                    logger.warning(
                        "openai_compatible backend requires both 'model' and 'base_url'"
                    )
                    self._unavailable = True
                    return
                self._backend = OpenAICompatibleBackend(
                    model_name=model,
                    base_url=base_url,
                    api_key=cfg.get("api_key"),
                    dimensions=cfg.get("dimensions"),
                )
        except Exception as exc:
            logger.warning("Failed to initialize embedding backend '%s': %s", backend_kind, exc)
            self._backend = None
            self._unavailable = True

    # ---------------------------------------------------------------- public

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed text. Returns None if backend unavailable."""
        if not isinstance(text, str) or not text:
            return None
        # Cache hit: skip availability check + backend call entirely.
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return cached
        if not self.is_available():
            return None
        try:
            result = self._backend.embed(text)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None
        if self._cache is not None and result is not None:
            self._cache.put(text, result)
        return result

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts. Cached entries skip the backend round-trip."""
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if self._cache is not None:
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text:
                    # Leave as None; do not query backend for empty input.
                    continue
                cached = self._cache.get(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            for i, text in enumerate(texts):
                if isinstance(text, str) and text:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if not uncached_texts:
            return results
        if not self.is_available():
            return results
        try:
            computed = self._backend.embed_batch(list(uncached_texts))  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Batch embedding failed: %s", exc)
            return results
        if not computed:
            return results
        for idx, vec in zip(uncached_indices, computed):
            results[idx] = vec
            if self._cache is not None and vec is not None:
                self._cache.put(texts[idx], vec)
        return results

    def clear_cache(self) -> None:
        """Clear embedding cache (used by memory monitor emergency cleanup)."""
        if self._cache is not None:
            self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        if self._cache is not None:
            return self._cache.stats()
        return {"enabled": False}

    def is_available(self) -> bool:
        """Check if embedding is available (backend loaded successfully)."""
        if self._unavailable or self._backend is None:
            return False
        try:
            return bool(self._backend.is_available())
        except Exception:
            return False

    @property
    def dimensions(self) -> int:
        """Return vector dimensions."""
        if self._dimensions_override:
            return int(self._dimensions_override)
        if self._backend is None:
            return 0
        try:
            return int(self._backend.dimensions())
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Pure Python implementation. If numpy is available, uses it for performance.
    """
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    try:
        import numpy as np  # type: ignore

        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 0 else 0.0
    except ImportError:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        denom = norm_a * norm_b
        return dot / denom if denom > 0 else 0.0


def create_embedding_engine(config: Optional[Dict[str, Any]] = None) -> EmbeddingEngine:
    """Create an EmbeddingEngine from the memory.embedding config section.

    If config is None, tries to read from hermes_cli.config.cfg_get("memory.embedding").
    Returns an engine instance (may be unavailable if backend fails to load).
    """
    if config is None:
        config = _load_config_from_cli()
    return EmbeddingEngine(config or {})


# ---------------------------------------------------------------------------
# Module-level access for memory monitor
# ---------------------------------------------------------------------------

# Tracks the most recently created EmbeddingEngine so the gateway memory
# monitor can reach the underlying cache during emergency cleanup.
_active_engine: Optional["EmbeddingEngine"] = None


def get_embedding_cache() -> Optional[Any]:
    """Return the active embedding cache object, if any.

    The memory monitor calls this during emergency cleanup and invokes
    ``cache.clear()`` if the returned object exposes that method. Returns
    ``None`` when no engine has been created or no cache is available.
    """
    if _active_engine is None:
        return None
    cache = getattr(_active_engine, "_cache", None)
    if cache is None or isinstance(cache, bool):
        return None
    return cache


def _load_config_from_cli() -> Dict[str, Any]:
    """Best-effort lookup of memory.embedding from hermes_cli.config.

    Returns an empty dict if the module is not importable or the key is missing.
    """
    try:
        from hermes_cli.config import cfg_get  # type: ignore
    except ImportError:
        logger.debug("hermes_cli.config not importable; using empty embedding config")
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to import hermes_cli.config: %s", exc)
        return {}

    try:
        value = cfg_get("memory.embedding")
    except Exception as exc:
        logger.debug("cfg_get('memory.embedding') failed: %s", exc)
        return {}
    if isinstance(value, dict):
        return value
    return {}


__all__ = [
    "EmbeddingBackend",
    "EmbeddingEngine",
    "SentenceTransformerBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "cosine_similarity",
    "create_embedding_engine",
    "get_embedding_cache",
]
