"""SQLite-backed storage for the three-layer memory system.

This module provides :class:`MemoryDB`, a unified persistence layer for:

* **Core Memory** – small, always-in-context blocks (persona, human, …).
* **Recall Memory** – conversational history, searched via FTS5.
* **Archival Memory** – vector-indexed long-term knowledge entries with
  optional embeddings and FTS5 text search as a fallback.

Design notes
------------
* **WAL mode with fallback** — Writes prefer WAL for concurrent reads, but
  fall back to ``DELETE`` on filesystems where WAL is unsupported
  (NFS / SMB / some FUSE mounts).  Mirrors the resilience pattern in
  :mod:`hermes_state`.
* **Thread-safe** — A per-thread :class:`sqlite3.Connection` is cached via
  :class:`threading.local`, plus a write lock to serialize cross-thread
  writes safely.
* **No heavy dependencies** — Standard library only plus
  :mod:`hermes_constants` for profile-aware path resolution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# Markers SQLite raises on filesystems that don't support WAL — copied from
# the same pattern used by hermes_state.py.
_WAL_INCOMPAT_MARKERS = (
    "locking protocol",   # SQLITE_PROTOCOL on NFS/SMB
    "not authorized",     # Some FUSE mounts block WAL pragma outright
    "disk i/o error",     # Flaky network FS during WAL setup
)

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # Core Memory blocks ------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS memory_blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT UNIQUE NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        value TEXT NOT NULL DEFAULT '',
        char_limit INTEGER NOT NULL DEFAULT 2200,
        updated_at REAL NOT NULL
    )
    """,
    # Recall Memory ----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS recall_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        timestamp REAL NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_recall_session ON recall_messages(session_id, timestamp)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS recall_messages_fts USING fts5(
        content,
        content=recall_messages,
        content_rowid=id
    )
    """,
    # Triggers keep recall_messages_fts in sync with recall_messages.
    """
    CREATE TRIGGER IF NOT EXISTS recall_ai AFTER INSERT ON recall_messages BEGIN
        INSERT INTO recall_messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS recall_ad AFTER DELETE ON recall_messages BEGIN
        INSERT INTO recall_messages_fts(recall_messages_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS recall_au AFTER UPDATE ON recall_messages BEGIN
        INSERT INTO recall_messages_fts(recall_messages_fts, rowid, content)
        VALUES('delete', old.id, old.content);
        INSERT INTO recall_messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    # Archival Memory --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS archival_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        embedding BLOB,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS archival_entries_fts USING fts5(
        content,
        content=archival_entries,
        content_rowid=id
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS archival_ai AFTER INSERT ON archival_entries BEGIN
        INSERT INTO archival_entries_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS archival_ad AFTER DELETE ON archival_entries BEGIN
        INSERT INTO archival_entries_fts(archival_entries_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS archival_au AFTER UPDATE ON archival_entries BEGIN
        INSERT INTO archival_entries_fts(archival_entries_fts, rowid, content)
        VALUES('delete', old.id, old.content);
        INSERT INTO archival_entries_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    # Schema metadata --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS memory_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # Dream Engine state -----------------------------------------------------
    # Cross-session persistence for Dream Engine counters (turn_count,
    # total_turns, last_dream_time).  Required so that Gateway-mode requests,
    # which spin up a fresh AIAgent per turn, can still trigger consolidation.
    """
    CREATE TABLE IF NOT EXISTS dream_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    # Archival importance scoring -------------------------------------------
    # Tracks per-entry importance, access frequency, and recency for
    # importance-weighted re-ranking of archival search results.
    """
    CREATE TABLE IF NOT EXISTS archival_scoring (
        archival_id INTEGER PRIMARY KEY,
        importance REAL NOT NULL DEFAULT 0.5,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed_at REAL,
        FOREIGN KEY(archival_id) REFERENCES archival_entries(id) ON DELETE CASCADE
    )
    """,
    # Dream Engine distill results -----------------------------------------
    # Tracks each successfully applied distill fact so that pattern
    # learning can analyse recurring keys across multiple dream cycles.
    """
    CREATE TABLE IF NOT EXISTS dream_distill_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dream_cycle INTEGER NOT NULL,
        block_label TEXT NOT NULL,
        fact_key TEXT NOT NULL,
        fact_value TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'applied',
        created_at REAL NOT NULL
    )
    """,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_journal_mode(conn: sqlite3.Connection) -> str:
    """Try to enable WAL; fall back to DELETE on incompatible filesystems."""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return "wal"
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            logger.warning(
                "memory.db: WAL not supported here (%s) — falling back to DELETE",
                exc,
            )
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
                return "delete"
            except sqlite3.OperationalError:
                logger.exception("memory.db: failed to set journal_mode=DELETE")
                return "unknown"
        raise


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _decode_metadata(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# MemoryDB
# ---------------------------------------------------------------------------


class MemoryDB:
    """SQLite-backed storage for the three-layer memory system."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = Path(db_path) if db_path else (get_hermes_home() / "memory.db")
        self._local = threading.local()
        # Single global write lock to make cross-thread writes safe even
        # though each thread holds its own connection.
        self._write_lock = threading.RLock()
        # Track connections so close() can release them all.
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._schema_ready = False
        self._schema_lock = threading.Lock()

        # Make sure the parent directory exists.
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("memory.db: failed to create parent dir %s", self.db_path.parent)

        # Eagerly initialise the schema on the creating thread so that
        # subsequent reads from any thread find a fully-formed database.
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return a thread-local connection with WAL mode enabled."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # autocommit; we manage transactions manually
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            _set_journal_mode(conn)
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            logger.exception("memory.db: failed to apply pragmas on %s", self.db_path)

        self._local.conn = conn
        with self._all_conns_lock:
            self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        """Close every cached connection across all threads."""
        with self._all_conns_lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                logger.debug("memory.db: error closing connection", exc_info=True)
        # Drop the thread-local handle so the next call to connect() opens
        # a fresh connection.
        if hasattr(self._local, "conn"):
            try:
                delattr(self._local, "conn")
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables, triggers, and FTS indexes if they don't exist."""
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            conn = self.connect()
            with self._write_lock:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    for stmt in _SCHEMA_STATEMENTS:
                        conn.execute(stmt)
                    conn.execute("COMMIT")
                except sqlite3.Error:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise

                current = self._get_schema_version()
                if current == 0:
                    self._set_schema_version(SCHEMA_VERSION)
                elif current < SCHEMA_VERSION:
                    # Future migrations land here.  For v1 there's nothing
                    # to upgrade — just bump the recorded version.
                    self._set_schema_version(SCHEMA_VERSION)
            self._schema_ready = True

    def _get_schema_version(self) -> int:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT value FROM memory_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def _set_schema_version(self, version: int) -> None:
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT INTO memory_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(int(version)),),
            )

    # ------------------------------------------------------------------
    # Core Memory CRUD
    # ------------------------------------------------------------------

    def get_block(self, label: str) -> Optional[Dict[str, Any]]:
        """Return the block keyed by *label* or ``None`` if missing."""
        conn = self.connect()
        row = conn.execute(
            "SELECT id, label, description, value, char_limit, updated_at "
            "FROM memory_blocks WHERE label = ?",
            (label,),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def upsert_block(
        self,
        label: str,
        value: str,
        description: str = "",
        char_limit: int = 2200,
    ) -> None:
        """Insert a block, or update its value/description/limit if it exists."""
        now = time.time()
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                """
                INSERT INTO memory_blocks(label, description, value, char_limit, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET
                    description = excluded.description,
                    value       = excluded.value,
                    char_limit  = excluded.char_limit,
                    updated_at  = excluded.updated_at
                """,
                (label, description, value, int(char_limit), now),
            )

    def list_blocks(self) -> List[Dict[str, Any]]:
        """Return every memory block, ordered by label."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT id, label, description, value, char_limit, updated_at "
            "FROM memory_blocks ORDER BY label"
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def delete_block(self, label: str) -> bool:
        """Delete *label*. Returns ``True`` if a row was removed."""
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM memory_blocks WHERE label = ?", (label,)
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Recall Memory CRUD
    # ------------------------------------------------------------------

    def add_recall_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a message to recall memory.  Returns the new row id."""
        ts = float(timestamp) if timestamp is not None else time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO recall_messages(session_id, role, content, timestamp, metadata_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (session_id, role, content, ts, meta_json),
            )
            return int(cursor.lastrowid or 0)

    def search_recall(
        self,
        query: str,
        limit: int = 20,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search against recall messages via FTS5."""
        if not query or not query.strip():
            return []
        conn = self.connect()
        sql = (
            "SELECT r.id, r.session_id, r.role, r.content, r.timestamp, r.metadata_json "
            "FROM recall_messages_fts f "
            "JOIN recall_messages r ON r.id = f.rowid "
            "WHERE recall_messages_fts MATCH ? "
        )
        params: list[Any] = [query]
        if session_id is not None:
            sql += "AND r.session_id = ? "
            params.append(session_id)
        sql += "ORDER BY r.timestamp DESC LIMIT ?"
        params.append(int(limit))

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS5 raises on malformed queries (e.g. unbalanced quotes).
            logger.debug("memory.db: FTS5 query rejected: %r", query, exc_info=True)
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _decode_metadata(data.pop("metadata_json", "") or "")
            results.append(data)
        return results

    def get_recall_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return the most recent messages for *session_id* in ascending order."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, timestamp, metadata_json
            FROM recall_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            (session_id, int(limit), int(offset)),
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _decode_metadata(data.pop("metadata_json", "") or "")
            results.append(data)
        return results

    # ------------------------------------------------------------------
    # Archival Memory CRUD
    # ------------------------------------------------------------------

    def add_archival_entry(
        self,
        content: str,
        embedding: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a new archival entry and return its id."""
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = time.time()
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(
                """
                INSERT INTO archival_entries(content, embedding, metadata_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (content, embedding, meta_json, now),
            )
            return int(cursor.lastrowid or 0)

    def search_archival_by_entity(
        self,
        entity_type: str,
        entity_value: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search archival entries by entity value stored in metadata_json.

        Uses a broad ``LIKE`` match against the metadata JSON string so the
        query works on SQLite builds without the JSON1 extension. Expected
        metadata shape::

            {"entities": {"tech": ["Python", "Docker"], ...}}

        ``entity_type`` is currently unused in the WHERE clause (the
        substring search keys off ``entity_value``); it is accepted for
        API symmetry and future refinement.
        """
        if not entity_value:
            return []
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT id, content, metadata_json, created_at
            FROM archival_entries
            WHERE metadata_json LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f'%"{entity_value}"%', int(limit)),
        ).fetchall()
        return [
            {
                "id": r[0],
                "content": r[1],
                "metadata": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def search_archival_by_metadata_key(
        self, key: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Find archival entries that have a specific key in metadata_json.

        Uses a substring ``LIKE`` match against the JSON text so the query
        works on SQLite builds without the JSON1 extension.  Used by the
        interest graph aggregator to pull every entry tagged with a given
        metadata key (e.g. ``"entities"``).
        """
        if not key:
            return []
        conn = self.connect()
        rows = conn.execute(
            "SELECT id, content, metadata_json FROM archival_entries "
            "WHERE metadata_json LIKE ? LIMIT ?",
            (f'%"{key}"%', int(limit)),
        ).fetchall()
        return [
            {"id": r[0], "content": r[1], "metadata_json": r[2]}
            for r in rows
        ]

    def search_archival_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """FTS5 text search across archival entries."""
        if not query or not query.strip():
            return []
        conn = self.connect()
        try:
            rows = conn.execute(
                """
                SELECT a.id, a.content, a.embedding, a.metadata_json, a.created_at
                FROM archival_entries_fts f
                JOIN archival_entries a ON a.id = f.rowid
                WHERE archival_entries_fts MATCH ?
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (query, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.debug("memory.db: archival FTS5 query rejected: %r", query, exc_info=True)
            return []
        results: List[Dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _decode_metadata(data.pop("metadata_json", "") or "")
            results.append(data)
        return results

    def get_archival_entries_for_vector_search(
        self,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return entries that have an embedding, ready for vector scoring."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT id, content, embedding, metadata_json, created_at
            FROM archival_entries
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            data = _row_to_dict(row)
            data["metadata"] = _decode_metadata(data.pop("metadata_json", "") or "")
            results.append(data)
        return results

    def update_archival_embedding(self, entry_id: int, embedding: bytes) -> None:
        """Replace the embedding bytes for *entry_id*."""
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "UPDATE archival_entries SET embedding = ? WHERE id = ?",
                (embedding, int(entry_id)),
            )

    def delete_archival_entry(self, entry_id: int) -> bool:
        """Remove an archival entry. Returns ``True`` on success."""
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM archival_entries WHERE id = ?", (int(entry_id),)
            )
            return cursor.rowcount > 0

    def update_archival_entry(
        self,
        entry_id: int,
        content: str,
        embedding: Optional[bytes] = None,
    ) -> bool:
        """Update an archival entry's content and (optionally) embedding."""
        conn = self.connect()
        with self._write_lock:
            if embedding is None:
                cursor = conn.execute(
                    "UPDATE archival_entries SET content = ? WHERE id = ?",
                    (content, int(entry_id)),
                )
            else:
                cursor = conn.execute(
                    "UPDATE archival_entries SET content = ?, embedding = ? WHERE id = ?",
                    (content, embedding, int(entry_id)),
                )
            return cursor.rowcount > 0

    def get_random_archival_entries(
        self, count: int = 10, exclude_ids: Optional[list] = None
    ) -> list:
        """Get random archival entries for idle consolidation.

        Returns list of dicts with id, content, embedding, metadata, created_at.
        """
        conn = self.connect()
        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            rows = conn.execute(
                f"""
                SELECT id, content, embedding, metadata_json, created_at
                FROM archival_entries
                WHERE id NOT IN ({placeholders})
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (*exclude_ids, count),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, content, embedding, metadata_json, created_at
                FROM archival_entries
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (count,),
            ).fetchall()
        return [
            {"id": r[0], "content": r[1], "embedding": r[2], "metadata": r[3], "created_at": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Archival importance scoring CRUD
    # ------------------------------------------------------------------

    def get_importance(self, archival_id: int) -> float:
        """Get importance score for an archival entry (default 0.5)."""
        conn = self.connect()
        row = conn.execute(
            "SELECT importance FROM archival_scoring WHERE archival_id = ?",
            (int(archival_id),),
        ).fetchone()
        return float(row[0]) if row else 0.5

    def increment_access(self, archival_id: int) -> None:
        """Increment access count and update last_accessed_at."""
        now = time.time()
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                """
                INSERT INTO archival_scoring(archival_id, importance, access_count, last_accessed_at)
                VALUES(?, 0.5, 1, ?)
                ON CONFLICT(archival_id) DO UPDATE SET
                    access_count = access_count + 1,
                    last_accessed_at = ?
                """,
                (int(archival_id), now, now),
            )

    def set_importance(self, archival_id: int, score: float) -> None:
        """Set importance score (0.0-1.0) for an archival entry."""
        score = max(0.0, min(1.0, float(score)))
        now = time.time()
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                """
                INSERT INTO archival_scoring(archival_id, importance, access_count, last_accessed_at)
                VALUES(?, ?, 0, ?)
                ON CONFLICT(archival_id) DO UPDATE SET importance = ?
                """,
                (int(archival_id), score, now, score),
            )

    def get_entries_by_importance(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top archival entries sorted by importance score."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT a.id, a.content, s.importance, s.access_count
            FROM archival_entries a
            LEFT JOIN archival_scoring s ON a.id = s.archival_id
            ORDER BY COALESCE(s.importance, 0.5) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [
            {
                "id": int(r[0]),
                "content": str(r[1] or ""),
                "importance": float(r[2]) if r[2] is not None else 0.5,
                "access_count": int(r[3]) if r[3] is not None else 0,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Dream Engine state CRUD
    # ------------------------------------------------------------------

    def get_dream_state(self, key: str) -> Optional[str]:
        """Return the persisted Dream Engine value for *key*, or ``None``."""
        conn = self.connect()
        row = conn.execute(
            "SELECT value FROM dream_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_dream_state(self, key: str, value: str) -> None:
        """Upsert a Dream Engine state value keyed by *key*."""
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO dream_state(key, value, updated_at) "
                "VALUES(?, ?, ?)",
                (key, value, time.time()),
            )

    # ------------------------------------------------------------------
    # Dream distill results CRUD (pattern learning)
    # ------------------------------------------------------------------

    def add_distill_result(
        self,
        cycle: int,
        block: str,
        key: str,
        value: str,
        status: str = "applied",
    ) -> None:
        """Record a distill result for pattern learning."""
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                """
                INSERT INTO dream_distill_results(
                    dream_cycle, block_label, fact_key, fact_value, status, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (int(cycle), block, key, value, status, time.time()),
            )

    def get_distill_results(
        self, since_cycle: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get distill results since a given dream cycle."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT id, dream_cycle, block_label, fact_key, fact_value, status, created_at
            FROM dream_distill_results
            WHERE dream_cycle >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(since_cycle), int(limit)),
        ).fetchall()
        return [
            {
                "id": r[0],
                "cycle": r[1],
                "block": r[2],
                "key": r[3],
                "value": r[4],
                "status": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def get_recurring_keys(self, min_count: int = 3) -> List[Dict[str, Any]]:
        """Get fact keys that appear ``>= min_count`` times for pattern learning.

        Returns a list of dicts ``[{"key", "value", "count", "block"}]``
        where ``value``/``block`` come from one of the matching rows (an
        arbitrary representative — most recent ordering is not required).
        """
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT fact_key, fact_value, COUNT(*) as cnt, block_label
            FROM dream_distill_results
            WHERE status = 'applied'
            GROUP BY fact_key
            HAVING cnt >= ?
            ORDER BY cnt DESC
            LIMIT 20
            """,
            (int(min_count),),
        ).fetchall()
        return [
            {"key": r[0], "value": r[1], "count": r[2], "block": r[3]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemoryDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["MemoryDB", "SCHEMA_VERSION"]
