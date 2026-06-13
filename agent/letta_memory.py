"""Three-layer memory core logic (Letta/MemGPT-style).

This module orchestrates the three memory tiers on top of
:class:`agent.memory_db.MemoryDB` and (optionally)
:class:`agent.embedding_engine.EmbeddingEngine`:

* :class:`CoreMemory` — small, always-in-context blocks (persona / human).
  Replaces the legacy ``MEMORY.md`` / ``USER.md`` files.  The system prompt
  always sees a *frozen snapshot* captured at session start — live edits
  via tools persist to the DB but do not mutate the prompt mid-conversation
  (preserves prompt caching).
* :class:`RecallMemory` — every chat message; FTS5 search by default,
  augmented with semantic search when an embedding engine is present.
* :class:`ArchivalMemory` — long-term knowledge base with hybrid
  (FTS5 + vector) retrieval.  Falls back to FTS5-only when embeddings
  are unavailable.

:class:`LettaMemorySystem` is the single entry point: construct it once
during agent startup, call :meth:`LettaMemorySystem.initialize`, and pass
``.core``, ``.recall``, ``.archival`` to whichever subsystem needs them.

Design notes
------------
* **Graceful degradation** — every vector-search code path tolerates a
  ``None`` / unavailable embedding engine and silently falls back to FTS5.
* **Thread safety** — all DB I/O goes through ``MemoryDB``, which manages
  per-thread connections plus a write lock.
* **Embedding storage** — vectors are packed as little-endian float32
  via :func:`struct.pack` and stored in the ``archival_entries.embedding``
  BLOB column.
"""

from __future__ import annotations

import json
import logging
import re as _re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.embedding_engine import EmbeddingEngine, cosine_similarity
from agent.memory_db import MemoryDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity extraction patterns (lightweight regex-based NER).
#
# Compiled once at module load for performance. Each pattern targets a
# different entity class; matches are deduplicated and capped per type by
# :meth:`ArchivalMemory._extract_entities`.
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS = {
    "person": _re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'),
    "tech": _re.compile(
        r'\b(?:Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|React|Vue|Angular|'
        r'Django|Flask|FastAPI|Docker|K8s|Kubernetes|Redis|MySQL|PostgreSQL|SQLite|'
        r'MongoDB|AWS|Azure|GCP|Linux|Windows|macOS|Git|Node\.js|npm|pip|conda|'
        r'Hermes|PyTorch|TensorFlow|OpenAI|Anthropic|LangChain)\b',
        _re.IGNORECASE,
    ),
    "url": _re.compile(r'https?://[^\s<>"\']+'),
    "path": _re.compile(r'(?:/[\w.-]+){2,}|(?:[A-Z]:\\[\w\\.-]+)'),
    "date": _re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    "email": _re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "persona": {
        "description": "Agent's personal notes, observations, and learned facts",
        "char_limit": 2200,
    },
    "human": {
        "description": "What the agent knows about the user",
        "char_limit": 1375,
    },
}

# Reciprocal Rank Fusion constant (Cormack et al., 2009).  60 is the
# canonical value and works well across heterogeneous score scales.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Embedding (de)serialisation
# ---------------------------------------------------------------------------


def _serialize_embedding(vec: List[float]) -> bytes:
    """Pack a float vector as little-endian float32 for SQLite BLOB storage."""
    if not vec:
        return b""
    return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))


def _deserialize_embedding(data: Optional[bytes]) -> List[float]:
    """Unpack a float32 BLOB produced by :func:`_serialize_embedding`."""
    if not data:
        return []
    n = len(data) // 4  # float32 = 4 bytes
    if n == 0:
        return []
    try:
        return list(struct.unpack(f"<{n}f", data[: n * 4]))
    except struct.error:
        logger.debug("letta_memory: failed to deserialize embedding blob", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryBlock:
    """A single core memory block."""

    label: str
    description: str
    value: str
    char_limit: int = 2200
    updated_at: float = 0.0


@dataclass
class RecallEntry:
    """A recall memory search result."""

    id: int
    session_id: str
    role: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0  # populated for ranked results


@dataclass
class ArchivalEntry:
    """An archival memory entry."""

    id: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    relevance_score: float = 0.0  # populated for ranked results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_recall(row: Dict[str, Any]) -> RecallEntry:
    return RecallEntry(
        id=int(row.get("id") or 0),
        session_id=str(row.get("session_id") or ""),
        role=str(row.get("role") or ""),
        content=str(row.get("content") or ""),
        timestamp=float(row.get("timestamp") or 0.0),
        metadata=dict(row.get("metadata") or {}),
        relevance_score=float(row.get("relevance_score") or 0.0),
    )


def _row_to_archival(row: Dict[str, Any]) -> ArchivalEntry:
    return ArchivalEntry(
        id=int(row.get("id") or 0),
        content=str(row.get("content") or ""),
        metadata=dict(row.get("metadata") or {}),
        created_at=float(row.get("created_at") or 0.0),
        relevance_score=float(row.get("relevance_score") or 0.0),
    )


# ---------------------------------------------------------------------------
# Core Memory
# ---------------------------------------------------------------------------


class CoreMemory:
    """In-context memory blocks rendered into every system prompt.

    Default blocks: ``persona`` (agent self-notes) and ``human`` (user
    profile).  Additional blocks can be configured via the constructor.

    The system prompt always uses a *frozen snapshot* captured by
    :meth:`load_snapshot`.  Tool-driven edits update the live DB state
    immediately but leave the snapshot untouched until the next session
    boot — this is intentional, so prompt-prefix caching keeps working.
    """

    def __init__(
        self,
        db: MemoryDB,
        block_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._db = db
        self._block_configs: Dict[str, Dict[str, Any]] = (
            dict(block_configs) if block_configs else dict(DEFAULT_BLOCK_CONFIGS)
        )
        self._snapshot: Dict[str, str] = {}
        self._snapshot_descriptions: Dict[str, str] = {}
        self._ensure_default_blocks()

    # ------------------------------------------------------------------
    # Block lifecycle
    # ------------------------------------------------------------------

    def _ensure_default_blocks(self) -> None:
        """Create configured blocks in the DB if they don't already exist."""
        for label, cfg in self._block_configs.items():
            existing = self._db.get_block(label)
            if existing is None:
                description = str(cfg.get("description") or "")
                char_limit = int(cfg.get("char_limit") or 2200)
                self._db.upsert_block(
                    label=label,
                    value="",
                    description=description,
                    char_limit=char_limit,
                )

    def load_snapshot(self) -> None:
        """Capture the current block state for the system prompt.

        Call once at session start.  Subsequent edits via :meth:`update_block`
        / :meth:`replace_in_block` do *not* refresh the snapshot.
        """
        self._snapshot.clear()
        self._snapshot_descriptions.clear()
        for row in self._db.list_blocks():
            label = str(row.get("label") or "")
            if not label:
                continue
            self._snapshot[label] = str(row.get("value") or "")
            self._snapshot_descriptions[label] = str(row.get("description") or "")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_block(self, label: str) -> Optional[MemoryBlock]:
        """Return the live DB state of ``label`` (not the snapshot)."""
        row = self._db.get_block(label)
        if row is None:
            return None
        return MemoryBlock(
            label=str(row.get("label") or label),
            description=str(row.get("description") or ""),
            value=str(row.get("value") or ""),
            char_limit=int(row.get("char_limit") or 2200),
            updated_at=float(row.get("updated_at") or 0.0),
        )

    def list_blocks(self) -> List[MemoryBlock]:
        """Return all blocks (live state)."""
        out: List[MemoryBlock] = []
        for row in self._db.list_blocks():
            out.append(
                MemoryBlock(
                    label=str(row.get("label") or ""),
                    description=str(row.get("description") or ""),
                    value=str(row.get("value") or ""),
                    char_limit=int(row.get("char_limit") or 2200),
                    updated_at=float(row.get("updated_at") or 0.0),
                )
            )
        return out

    def get_block_age(self, label: str) -> Optional[float]:
        """Return the number of hours since *label* was last updated.

        Returns ``None`` if the block does not exist or has never been
        written to (no ``updated_at`` recorded).
        """
        row = self._db.get_block(label)
        if row is None:
            return None
        try:
            updated_at = float(row.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            return None
        if updated_at <= 0.0:
            return None
        return max(0.0, (time.time() - updated_at) / 3600.0)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update_block(self, label: str, new_value: str) -> Tuple[bool, str]:
        """Replace a block's value.  Enforces ``char_limit``.

        Returns ``(success, message)``.  The system-prompt snapshot is
        deliberately *not* refreshed.
        """
        if new_value is None:
            new_value = ""
        block = self.get_block(label)
        if block is None:
            return False, f"Block '{label}' does not exist."

        if len(new_value) > block.char_limit:
            return (
                False,
                (
                    f"New value exceeds char_limit for block '{label}' "
                    f"({len(new_value)} > {block.char_limit})."
                ),
            )

        try:
            self._db.upsert_block(
                label=label,
                value=new_value,
                description=block.description,
                char_limit=block.char_limit,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("CoreMemory.update_block failed for %s", label)
            return False, f"Failed to update block '{label}': {exc}"

        return True, f"Block '{label}' updated ({len(new_value)}/{block.char_limit} chars)."

    def replace_in_block(
        self, label: str, old_str: str, new_str: str
    ) -> Tuple[bool, str]:
        """Surgically replace ``old_str`` with ``new_str`` inside a block."""
        block = self.get_block(label)
        if block is None:
            return False, f"Block '{label}' does not exist."
        if not old_str:
            return False, "old_str must be non-empty."
        if old_str not in block.value:
            return False, f"Substring not found in block '{label}'."

        # Replace only the first occurrence to keep edits surgical.
        updated = block.value.replace(old_str, new_str, 1)
        return self.update_block(label, updated)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_blocks(
        labels_in_order: List[str],
        values: Dict[str, str],
        descriptions: Dict[str, str],
    ) -> str:
        if not labels_in_order:
            return "<core_memory>\n</core_memory>"
        lines: List[str] = ["<core_memory>"]
        for label in labels_in_order:
            value = values.get(label, "")
            desc = descriptions.get(label, "")
            # Escape only the double-quote in the description attribute;
            # the prompt format intentionally mirrors XML-ish tagging.
            attr = desc.replace('"', "&quot;")
            lines.append(f'<{label} description="{attr}">')
            lines.append(value)
            lines.append(f"</{label}>")
        lines.append("</core_memory>")
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Render the *frozen* snapshot for system prompt injection."""
        labels = sorted(self._snapshot.keys())
        return self._format_blocks(labels, self._snapshot, self._snapshot_descriptions)

    def format_live_state(self) -> str:
        """Render the current DB state (used by tool responses)."""
        blocks = self.list_blocks()
        labels = sorted(b.label for b in blocks)
        values = {b.label: b.value for b in blocks}
        descriptions = {b.label: b.description for b in blocks}
        return self._format_blocks(labels, values, descriptions)


# ---------------------------------------------------------------------------
# Recall Memory
# ---------------------------------------------------------------------------


class RecallMemory:
    """Searchable conversation history with optional semantic ranking."""

    def __init__(
        self,
        db: MemoryDB,
        embedding_engine: Optional[EmbeddingEngine] = None,
        time_decay_rate: float = 0.1,
    ) -> None:
        self._db = db
        self._embedding = embedding_engine
        self._time_decay_rate = time_decay_rate

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a message.  Returns the new row id."""
        if not session_id:
            raise ValueError("session_id is required")
        if role is None:
            role = ""
        if content is None:
            content = ""
        return int(
            self._db.add_recall_message(
                session_id=session_id,
                role=role,
                content=content,
                timestamp=timestamp,
                metadata=metadata or {},
            )
        )

    def search(
        self,
        query: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> List[RecallEntry]:
        """FTS5 + (optional) semantic search, merged via RRF."""
        if limit <= 0:
            return []
        query = (query or "").strip()
        if not query:
            return []

        fts_rows = self._db.search_recall(query, limit=limit * 2, session_id=session_id)
        fts_results: List[RecallEntry] = [_row_to_recall(r) for r in fts_rows]

        if not self._embedding_available():
            for idx, entry in enumerate(fts_results):
                entry.relevance_score = 1.0 / (_RRF_K + idx + 1)
            results = fts_results[:limit]
            return self._apply_time_decay(results, self._time_decay_rate)

        # Semantic pass: rank a wider candidate pool by cosine similarity.
        query_vec: Optional[List[float]] = None
        try:
            query_vec = self._embedding.embed(query) if self._embedding else None  # type: ignore[union-attr]
        except Exception:
            logger.debug("RecallMemory: embedding query failed", exc_info=True)
            query_vec = None

        semantic_results: List[RecallEntry] = []
        if query_vec:
            # Score against the most-recent N session messages.  Recall
            # has no embeddings stored — we embed candidates on the fly,
            # which is acceptable for small/medium history sizes.
            candidate_pool = limit * 5
            recent = self._collect_recall_candidates(session_id, candidate_pool)
            scored: List[Tuple[float, RecallEntry]] = []
            texts = [c.content for c in recent]
            try:
                vecs = self._embedding.embed_batch(texts) if self._embedding else []  # type: ignore[union-attr]
            except Exception:
                logger.debug("RecallMemory: batch embed failed", exc_info=True)
                vecs = []
            for entry, vec in zip(recent, vecs or []):
                if not vec:
                    continue
                score = cosine_similarity(query_vec, list(vec))
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda t: t[0], reverse=True)
            for score, entry in scored[: limit * 2]:
                entry.relevance_score = float(score)
                semantic_results.append(entry)

        return self._apply_time_decay(
            self._merge_rrf_recall(fts_results, semantic_results, limit),
            self._time_decay_rate,
        )

    @staticmethod
    def _apply_time_decay(entries: List[RecallEntry], decay_rate: float = 0.1) -> List[RecallEntry]:
        """Apply time decay to search results — recent results score higher.

        The decay formula is: ``score *= 1 / (1 + age_hours * decay_rate)``.
        With the default rate of 0.1, a result’s score halves after ~10 hours.
        """
        if not entries or decay_rate <= 0:
            return entries
        now = time.time()
        for entry in entries:
            age_hours = max(0.0, (now - entry.timestamp) / 3600.0)
            time_decay = 1.0 / (1.0 + age_hours * decay_rate)
            entry.relevance_score = entry.relevance_score * time_decay
        # Re-sort by decayed score
        entries.sort(key=lambda e: e.relevance_score, reverse=True)
        return entries

    def get_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RecallEntry]:
        """Return messages for ``session_id`` (chronological order)."""
        rows = self._db.get_recall_messages(session_id, limit=limit, offset=offset)
        return [_row_to_recall(r) for r in rows]

    def get_message_count(self, session_id: Optional[str] = None) -> int:
        """Count messages, optionally limited to one session."""
        conn = self._db.connect()
        if session_id is None:
            row = conn.execute("SELECT COUNT(*) AS n FROM recall_messages").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM recall_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["n"])
        except (TypeError, KeyError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embedding_available(self) -> bool:
        if self._embedding is None:
            return False
        try:
            return bool(self._embedding.is_available())
        except Exception:
            return False

    def _collect_recall_candidates(
        self, session_id: Optional[str], limit: int
    ) -> List[RecallEntry]:
        """Pull recent messages to score semantically."""
        conn = self._db.connect()
        if session_id is None:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, timestamp, metadata_json
                FROM recall_messages
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, timestamp, metadata_json
                FROM recall_messages
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()
        out: List[RecallEntry] = []
        for row in rows:
            data = {key: row[key] for key in row.keys()}
            raw_meta = data.pop("metadata_json", "") or "{}"
            try:
                import json as _json

                meta = _json.loads(raw_meta)
                if not isinstance(meta, dict):
                    meta = {}
            except (ValueError, TypeError):
                meta = {}
            data["metadata"] = meta
            out.append(_row_to_recall(data))
        return out

    @staticmethod
    def _merge_rrf_recall(
        fts: List[RecallEntry],
        semantic: List[RecallEntry],
        limit: int,
    ) -> List[RecallEntry]:
        scores: Dict[int, float] = {}
        keep: Dict[int, RecallEntry] = {}
        for rank, entry in enumerate(fts):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            keep.setdefault(entry.id, entry)
        for rank, entry in enumerate(semantic):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            keep.setdefault(entry.id, entry)
        ordered_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        out: List[RecallEntry] = []
        for entry_id in ordered_ids[:limit]:
            entry = keep[entry_id]
            entry.relevance_score = float(scores[entry_id])
            out.append(entry)
        return out


# ---------------------------------------------------------------------------
# Archival Memory
# ---------------------------------------------------------------------------


class ArchivalMemory:
    """Hybrid (FTS5 + vector) long-term knowledge base."""

    def __init__(
        self,
        db: MemoryDB,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ) -> None:
        self._db = db
        self._embedding = embedding_engine

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities(content: str) -> Dict[str, List[str]]:
        """Extract entities from ``content`` via lightweight regex NER.

        Returns a mapping ``entity_type -> list of unique matched values``,
        capped at five entries per type. An empty dict is returned for empty
        / falsy input.
        """
        if not content:
            return {}
        entities: Dict[str, List[str]] = {}
        for entity_type, pattern in _ENTITY_PATTERNS.items():
            matches = list(set(pattern.findall(content)))
            if matches:
                entities[entity_type] = matches[:5]
        return entities

    def insert(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a new entry and return its id."""
        if content is None:
            content = ""
        metadata = dict(metadata) if metadata else {}
        # Auto-extract entities if not already provided by the caller so that
        # downstream entity-prefix search can match them.
        if "entities" not in metadata:
            entities = self._extract_entities(content)
            if entities:
                metadata["entities"] = entities
        embedding_blob: Optional[bytes] = None
        if self._embedding_available() and content.strip():
            try:
                vec = self._embedding.embed(content) if self._embedding else None  # type: ignore[union-attr]
            except Exception:
                logger.debug("ArchivalMemory: embedding insert failed", exc_info=True)
                vec = None
            if vec:
                embedding_blob = _serialize_embedding(list(vec))
        return int(
            self._db.add_archival_entry(
                content=content,
                embedding=embedding_blob,
                metadata=metadata,
            )
        )

    def get_entry(self, entry_id: int) -> Optional[ArchivalEntry]:
        conn = self._db.connect()
        row = conn.execute(
            """
            SELECT id, content, metadata_json, created_at
            FROM archival_entries
            WHERE id = ?
            """,
            (int(entry_id),),
        ).fetchone()
        if row is None:
            return None
        import json as _json

        raw_meta = row["metadata_json"] or "{}"
        try:
            meta = _json.loads(raw_meta)
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            meta = {}
        return ArchivalEntry(
            id=int(row["id"]),
            content=str(row["content"] or ""),
            metadata=meta,
            created_at=float(row["created_at"] or 0.0),
        )

    def get_all_entries(self, limit: int = 100, offset: int = 0) -> List[ArchivalEntry]:
        conn = self._db.connect()
        rows = conn.execute(
            """
            SELECT id, content, metadata_json, created_at
            FROM archival_entries
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (int(limit), int(offset)),
        ).fetchall()
        import json as _json

        out: List[ArchivalEntry] = []
        for row in rows:
            try:
                meta = _json.loads(row["metadata_json"] or "{}")
                if not isinstance(meta, dict):
                    meta = {}
            except (ValueError, TypeError):
                meta = {}
            out.append(
                ArchivalEntry(
                    id=int(row["id"]),
                    content=str(row["content"] or ""),
                    metadata=meta,
                    created_at=float(row["created_at"] or 0.0),
                )
            )
        return out

    def get_entry_count(self) -> int:
        conn = self._db.connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM archival_entries").fetchone()
        if row is None:
            return 0
        try:
            return int(row["n"])
        except (TypeError, KeyError, IndexError):
            return 0

    def delete(self, entry_id: int) -> bool:
        try:
            return bool(self._db.delete_archival_entry(int(entry_id)))
        except Exception:
            logger.exception("ArchivalMemory.delete failed for %s", entry_id)
            return False

    def update(self, entry_id: int, content: str) -> bool:
        if content is None:
            content = ""
        embedding_blob: Optional[bytes] = None
        if self._embedding_available() and content.strip():
            try:
                vec = self._embedding.embed(content) if self._embedding else None  # type: ignore[union-attr]
            except Exception:
                logger.debug("ArchivalMemory: embedding update failed", exc_info=True)
                vec = None
            if vec:
                embedding_blob = _serialize_embedding(list(vec))
        try:
            return bool(
                self._db.update_archival_entry(
                    entry_id=int(entry_id),
                    content=content,
                    embedding=embedding_blob,
                )
            )
        except Exception:
            logger.exception("ArchivalMemory.update failed for %s", entry_id)
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[ArchivalEntry]:
        """Hybrid search: FTS5 + vector similarity, merged by RRF.

        Final scores are re-weighted by per-entry importance (tracked in
        the ``archival_scoring`` table); each access also bumps the entry's
        access counter and ``last_accessed_at``.
        """
        if top_k <= 0:
            return []
        query = (query or "").strip()
        if not query:
            return []

        # Entity-based search: "entity:tech:Python" or "entity:person:Alice".
        # Bypasses FTS/vector hybrid in favour of metadata LIKE matching.
        if query.startswith("entity:"):
            _entity_parts = query[7:].split(":", 1)
            if len(_entity_parts) == 2:
                _etype = _entity_parts[0].strip()
                _evalue = _entity_parts[1].strip()
                if _etype and _evalue:
                    rows = self._db.search_archival_by_entity(
                        _etype, _evalue, limit=top_k
                    )
                    results: List[ArchivalEntry] = []
                    for r in rows:
                        raw_meta = r.get("metadata")
                        if isinstance(raw_meta, str):
                            try:
                                meta = json.loads(raw_meta)
                                if not isinstance(meta, dict):
                                    meta = {}
                            except (ValueError, TypeError):
                                meta = {}
                        else:
                            meta = dict(raw_meta) if raw_meta else {}
                        entry = ArchivalEntry(
                            id=int(r.get("id") or 0),
                            content=str(r.get("content") or ""),
                            metadata=meta,
                            created_at=float(r.get("created_at") or 0.0),
                            relevance_score=1.0,
                        )
                        results.append(entry)
                    return self._apply_importance_weighting(results)
            # Fall through to normal search if entity: parsing fails.

        fts_rows = self._db.search_archival_fts(query, limit=top_k * 2)
        fts_results: List[ArchivalEntry] = [_row_to_archival(r) for r in fts_rows]

        if not self._embedding_available():
            for idx, entry in enumerate(fts_results):
                entry.relevance_score = 1.0 / (_RRF_K + idx + 1)
            return self._apply_importance_weighting(fts_results[:top_k])

        # Vector pass.
        try:
            query_vec = self._embedding.embed(query) if self._embedding else None  # type: ignore[union-attr]
        except Exception:
            logger.debug("ArchivalMemory: embedding query failed", exc_info=True)
            query_vec = None

        vector_results: List[ArchivalEntry] = []
        if query_vec:
            all_entries = self._db.get_archival_entries_for_vector_search()
            scored: List[Tuple[float, ArchivalEntry]] = []
            for raw in all_entries:
                vec = _deserialize_embedding(raw.get("embedding"))
                if not vec:
                    continue
                score = cosine_similarity(query_vec, vec)
                if score <= 0:
                    continue
                entry = _row_to_archival(raw)
                entry.relevance_score = float(score)
                scored.append((score, entry))
            scored.sort(key=lambda t: t[0], reverse=True)
            vector_results = [e for _, e in scored[: top_k * 2]]

        merged = self._merge_rrf_archival(fts_results, vector_results, top_k)
        return self._apply_importance_weighting(merged)

    def _apply_importance_weighting(
        self, results: List[ArchivalEntry]
    ) -> List[ArchivalEntry]:
        """Re-weight ``results`` by stored importance and bump access stats.

        Multiplier range is ``[0.5, 1.0]`` (importance contributes -50%..0%
        to the relevance score), then results are re-sorted descending.
        Failures are swallowed so search degrades gracefully.
        """
        if not results or self._db is None:
            return results
        for entry in results:
            entry_id = getattr(entry, "id", None)
            if not entry_id:
                continue
            try:
                imp = self._db.get_importance(int(entry_id))
            except Exception:
                logger.debug(
                    "ArchivalMemory: get_importance failed for %s",
                    entry_id,
                    exc_info=True,
                )
                imp = 0.5
            try:
                self._db.increment_access(int(entry_id))
            except Exception:
                logger.debug(
                    "ArchivalMemory: increment_access failed for %s",
                    entry_id,
                    exc_info=True,
                )
            entry.relevance_score = float(entry.relevance_score) * (
                0.5 + float(imp) * 0.5
            )
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Consolidation helpers (used by the Dream Engine)
    # ------------------------------------------------------------------

    def find_similar(
        self,
        threshold: float = 0.9,
        limit: int = 10,
    ) -> List[Tuple[int, int, str, str]]:
        """Find pairs of archival entries with high embedding similarity.

        Returns a list of ``(id_a, id_b, content_a, content_b)`` tuples.
        Requires the embedding engine to be available; returns ``[]``
        otherwise.
        """
        if not self._embedding_available():
            return []
        if limit <= 0:
            return []

        # Pull a generous candidate pool so similar pairs aren't missed.
        candidate_pool = max(limit * 5, 20)
        rows = self._db.get_archival_entries_for_vector_search(limit=candidate_pool)
        if len(rows) < 2:
            return []

        entry_data: List[Tuple[int, str, List[float]]] = []
        for row in rows:
            vec = _deserialize_embedding(row.get("embedding"))
            if not vec:
                continue
            entry_data.append(
                (int(row.get("id") or 0), str(row.get("content") or ""), vec)
            )
        if len(entry_data) < 2:
            return []

        pairs: List[Tuple[int, int, str, str]] = []
        seen: set = set()
        for i in range(len(entry_data)):
            id_a, content_a, emb_a = entry_data[i]
            for j in range(i + 1, len(entry_data)):
                id_b, content_b, emb_b = entry_data[j]
                try:
                    sim = cosine_similarity(emb_a, emb_b)
                except Exception:
                    continue
                if sim < threshold:
                    continue
                pair_key = (min(id_a, id_b), max(id_a, id_b))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pairs.append((id_a, id_b, content_a, content_b))
                if len(pairs) >= limit:
                    return pairs
        return pairs

    def merge_entries(
        self,
        ids: List[int],
        merged_content: str,
    ) -> Optional[int]:
        """Merge multiple archival entries into one.

        Deletes the originals and inserts a new entry containing
        ``merged_content``.  Returns the new entry id, or ``None`` on
        failure / empty input.
        """
        if not ids or not merged_content:
            return None
        for entry_id in ids:
            try:
                self._db.delete_archival_entry(int(entry_id))
            except Exception:
                logger.debug(
                    "ArchivalMemory.merge_entries: delete failed for %s",
                    entry_id,
                    exc_info=True,
                )
        try:
            return int(self.insert(merged_content))
        except Exception:
            logger.debug(
                "ArchivalMemory.merge_entries: insert failed", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embedding_available(self) -> bool:
        if self._embedding is None:
            return False
        try:
            return bool(self._embedding.is_available())
        except Exception:
            return False

    @staticmethod
    def _merge_rrf_archival(
        fts: List[ArchivalEntry],
        vector: List[ArchivalEntry],
        top_k: int,
    ) -> List[ArchivalEntry]:
        scores: Dict[int, float] = {}
        keep: Dict[int, ArchivalEntry] = {}
        for rank, entry in enumerate(fts):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            keep.setdefault(entry.id, entry)
        for rank, entry in enumerate(vector):
            scores[entry.id] = scores.get(entry.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            # Prefer the vector copy when it carries a meaningful score.
            existing = keep.get(entry.id)
            if existing is None or existing.relevance_score == 0.0:
                keep[entry.id] = entry
        ordered_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        out: List[ArchivalEntry] = []
        for entry_id in ordered_ids[:top_k]:
            entry = keep[entry_id]
            entry.relevance_score = float(scores[entry_id])
            out.append(entry)
        return out


# ---------------------------------------------------------------------------
# Top-level system
# ---------------------------------------------------------------------------


class LettaMemorySystem:
    """Top-level coordinator for the three memory tiers."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self._config: Dict[str, Any] = dict(config or {})
        self._db = MemoryDB(db_path)
        self._embedding: Optional[EmbeddingEngine] = self._init_embedding()
        self.core = CoreMemory(self._db, self._get_block_configs())
        self.recall = RecallMemory(
            self._db, self._embedding, self._get_time_decay_rate()
        )
        self.archival = ArchivalMemory(self._db, self._embedding)
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Capture core-memory snapshot for the system prompt."""
        if self._initialized:
            return
        self.core.load_snapshot()
        self._initialized = True

    def shutdown(self) -> None:
        """Close all DB connections."""
        try:
            self._db.close()
        except Exception:
            logger.debug("LettaMemorySystem.shutdown: DB close failed", exc_info=True)

    # ------------------------------------------------------------------
    # Mood detection
    # ------------------------------------------------------------------

    def _update_mood(self, user_message: str) -> None:
        """Detect emotion via auxiliary LLM with 3-turn decay."""
        try:
            from agent.emotion_detector import detect_emotion_prompt, parse_emotion_response

            # Call auxiliary model for emotion detection
            prompt = detect_emotion_prompt(user_message)
            from agent.auxiliary_client import call_llm
            response = call_llm(
                task="emotion_detection",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            # Extract text from response object
            raw_text = ""
            if hasattr(response, 'choices') and response.choices:
                msg = response.choices[0].message
                raw_text = getattr(msg, 'content', '') or ''
            elif hasattr(response, 'content'):
                raw_text = response.content or ''
            else:
                raw_text = str(response)

            emotion, confidence = parse_emotion_response(raw_text)

            # Decay logic: 3 consecutive neutral turns resets mood
            if emotion == "neutral":
                self._mood_neutral_count = getattr(self, '_mood_neutral_count', 0) + 1
                if self._mood_neutral_count >= 3:
                    current = self.core.get_block("mood")
                    if current and getattr(current, 'value', '') not in ("", "neutral"):
                        self.core.update_block("mood", "neutral")
            else:
                self._mood_neutral_count = 0
                if confidence >= 0.5:
                    self.core.update_block("mood", f"{emotion}")
        except Exception:
            pass  # Never crash message flow

    def __enter__(self) -> "LettaMemorySystem":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def db(self) -> MemoryDB:
        return self._db

    @property
    def embedding(self) -> Optional[EmbeddingEngine]:
        return self._embedding

    @property
    def embedding_available(self) -> bool:
        if self._embedding is None:
            return False
        try:
            return bool(self._embedding.is_available())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _init_embedding(self) -> Optional[EmbeddingEngine]:
        """Build the embedding engine, returning ``None`` on any failure."""
        emb_cfg = self._config.get("embedding") if isinstance(self._config, dict) else None
        if emb_cfg is None:
            return None
        if isinstance(emb_cfg, dict) and emb_cfg.get("enabled") is False:
            return None
        try:
            engine = EmbeddingEngine(emb_cfg if isinstance(emb_cfg, dict) else {})
        except Exception as exc:
            logger.warning("Failed to construct EmbeddingEngine: %s", exc)
            return None
        return engine

    def _get_block_configs(self) -> Dict[str, Dict[str, Any]]:
        """Pull ``memory.core_memory.blocks`` from config (with fallback)."""
        cfg = self._config.get("core_memory") if isinstance(self._config, dict) else None
        if not isinstance(cfg, dict):
            return dict(DEFAULT_BLOCK_CONFIGS)
        blocks = cfg.get("blocks")
        if not isinstance(blocks, dict) or not blocks:
            return dict(DEFAULT_BLOCK_CONFIGS)

        normalised: Dict[str, Dict[str, Any]] = {}
        for label, raw in blocks.items():
            if not isinstance(raw, dict):
                continue
            description = str(raw.get("description") or "")
            try:
                char_limit = int(raw.get("char_limit") or 2200)
            except (TypeError, ValueError):
                char_limit = 2200
            normalised[str(label)] = {
                "description": description,
                "char_limit": char_limit,
            }
        return normalised or dict(DEFAULT_BLOCK_CONFIGS)

    def _get_time_decay_rate(self) -> float:
        """Pull ``memory.recall.time_decay_rate`` from config (default 0.1)."""
        recall_cfg = self._config.get("recall") if isinstance(self._config, dict) else None
        if not isinstance(recall_cfg, dict):
            return 0.1
        try:
            rate = float(recall_cfg.get("time_decay_rate", 0.1))
        except (TypeError, ValueError):
            return 0.1
        return max(0.0, rate)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "ArchivalEntry",
    "ArchivalMemory",
    "CoreMemory",
    "DEFAULT_BLOCK_CONFIGS",
    "LettaMemorySystem",
    "MemoryBlock",
    "RecallEntry",
    "RecallMemory",
]
