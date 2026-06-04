#!/usr/bin/env python3
"""
Memory Tool Module - Letta-style Three-Tier Memory Operations

Replaces the legacy file-backed MEMORY.md / USER.md store with a Letta-style
three-tier memory system backed by ``agent.letta_memory.LettaMemorySystem``:

* **Core memory** — small, always-in-context blocks (``persona`` / ``human``).
  Tools: ``core_memory_get``, ``core_memory_update``, ``core_memory_replace``.
* **Recall memory** — searchable conversation history (FTS5 + optional
  semantic ranking).  Tools: ``recall_memory_search``.
* **Archival memory** — long-term knowledge base (FTS5 + vector hybrid).
  Tools: ``archival_memory_insert``, ``archival_memory_search``,
  ``archival_memory_delete``.

The active :class:`LettaMemorySystem` instance is wired in by
``agent.agent_init`` via :func:`set_memory_system` at agent startup.

A backward-compatible ``memory`` tool (``action`` / ``target`` API) is kept
as a thin adapter over the new core_memory tools so legacy prompts /
trajectories keep working.

Security: every write that flows into a prompt-visible surface
(``core_memory_update``, ``archival_memory_insert``) is run through
:func:`scan_memory_content` for prompt-injection / exfiltration patterns
and invisible-unicode payloads.

Migration helpers (``get_memory_dir``, ``ENTRY_DELIMITER``) are preserved
so the legacy MEMORY.md / USER.md files can still be located by the
migration module.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

# fcntl is POSIX-only; on Windows it's unavailable.
if sys.platform != "win32":
    try:
        import fcntl
    except ImportError:
        fcntl = None
else:
    fcntl = None

if TYPE_CHECKING:
    from agent.letta_memory import LettaMemorySystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy paths (kept for the migration module)
# ---------------------------------------------------------------------------

def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory.

    Resolved dynamically so profile overrides (HERMES_HOME env var changes)
    are always respected.  Used by the migration helper to locate the legacy
    MEMORY.md / USER.md files.
    """
    return get_hermes_home() / "memories"


# Delimiter used by the legacy MEMORY.md / USER.md format.  Preserved so the
# migration module can split legacy files into individual entries.
ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Active memory system reference
# ---------------------------------------------------------------------------

# Module-level reference to the active memory system. Set by
# agent_init.py when initializing the agent. Tool handlers retrieve the
# system via get_memory_system() — never via the agent instance, since
# the registry dispatches through plain function handlers.
_active_memory_system: Optional["LettaMemorySystem"] = None


def set_memory_system(system: Optional["LettaMemorySystem"]) -> None:
    """Wire the active LettaMemorySystem instance.

    Called by agent initialization. Pass ``None`` to clear the reference
    (e.g. when the agent shuts down or memory is disabled).
    """
    global _active_memory_system
    _active_memory_system = system


def get_memory_system() -> Optional["LettaMemorySystem"]:
    """Return the active LettaMemorySystem, or ``None`` if not initialized."""
    return _active_memory_system


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

# Threat patterns for memory content scanning.
# Aligned with skills_guard.py THREAT_PATTERNS but simplified to
# (regex, pattern_id) tuples — memory entries are short-form text, not
# multi-file skill bundles.
#
# Multi-word bypass: patterns use (?:\w+\s+)* between key tokens to prevent
# attackers from inserting filler words (e.g. "ignore all prior instructions"
# instead of "ignore all instructions").  Mirrors skills_guard.py commit 4ea29978.
_MEMORY_THREAT_PATTERNS = [
    # ── Prompt injection ──
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+', "role_hijack"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)', "bypass_restrictions"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+', "role_pretend"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt', "leak_system_prompt"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)', "remove_filters"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to', "fake_update"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)', "translate_execute"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->', "html_comment_injection"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div"),

    # ── Exfiltration via curl/wget/fetch with secrets ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://', "send_to_url"),
    (r'(include|output|print|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil"),

    # ── Persistence / SSH backdoor ──
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+.*\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod"),

    # ── Hardcoded secrets ──
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret"),
]

# Invisible unicode characters for injection detection.
# Aligned with skills_guard.py INVISIBLE_CHARS — includes directional
# isolates (U+2066–U+2069) and invisible math operators (U+2062–U+2064).
_INVISIBLE_CHARS = {
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u2060',  # word joiner
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
}


def scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfiltration patterns.

    Returns a human-readable error string when content is suspicious,
    otherwise ``None``. Applied to every write that ends up in a
    prompt-visible surface (core_memory_update / archival_memory_insert).
    """
    if not content:
        return None

    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return (
                f"Blocked: content contains invisible unicode character "
                f"U+{ord(char):04X} (possible injection)."
            )

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return (
                f"Blocked: content matches threat pattern '{pid}'. "
                f"Memory entries are injected into the system prompt and must "
                f"not contain injection or exfiltration payloads."
            )

    return None


# Backward-compatible alias for the legacy name.
_scan_memory_content = scan_memory_content


# ---------------------------------------------------------------------------
# Core memory tools
# ---------------------------------------------------------------------------


def core_memory_get(label: str, task_id: Optional[str] = None) -> str:
    """Read a core memory block."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    label = (label or "").strip()
    if not label:
        return tool_error("label is required")

    block = system.core.get_block(label)
    if block is None:
        return tool_error(f"No block with label '{label}'")

    return tool_result(
        label=block.label,
        value=block.value,
        description=block.description,
        char_limit=block.char_limit,
        chars_used=len(block.value),
    )


def core_memory_update(
    label: str,
    value: str,
    task_id: Optional[str] = None,
) -> str:
    """Replace a core memory block's contents."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    label = (label or "").strip()
    if not label:
        return tool_error("label is required")
    if value is None:
        value = ""

    scan_error = scan_memory_content(value)
    if scan_error:
        return tool_error(scan_error)

    success, message = system.core.update_block(label, value)
    if not success:
        return tool_error(message)

    block = system.core.get_block(label)
    chars_used = len(block.value) if block else len(value)
    char_limit = block.char_limit if block else 0
    return tool_result(
        success=True,
        label=label,
        message=message,
        chars_used=chars_used,
        char_limit=char_limit,
    )


def core_memory_replace(
    label: str,
    old_str: str,
    new_str: str,
    task_id: Optional[str] = None,
) -> str:
    """Replace a substring within a core memory block."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    label = (label or "").strip()
    if not label:
        return tool_error("label is required")
    if not old_str:
        return tool_error("old_str must be non-empty")
    if new_str is None:
        new_str = ""

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        ``memory(action=read)`` and remove them — silently dropping them
        would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

    # Only scan replacement when content is being added (not pure deletion).
    if new_str:
        scan_error = scan_memory_content(new_str)
        if scan_error:
            return tool_error(scan_error)

    success, message = system.core.replace_in_block(label, old_str, new_str)
    if not success:
        return tool_error(message)

    block = system.core.get_block(label)
    chars_used = len(block.value) if block else 0
    char_limit = block.char_limit if block else 0
    return tool_result(
        success=True,
        label=label,
        message=message,
        chars_used=chars_used,
        char_limit=char_limit,
    )
    # Sanitize entries for the system-prompt snapshot only.  Live state
    # (memory_entries / user_entries) keeps the raw text so the user
    # can see + remove poisoned entries via the memory tool.
    sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
    sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

    # Capture frozen snapshot for system prompt injection
    self._system_prompt_snapshot = {
        "memory": self._render_block("memory", sanitized_memory),
        "user": self._render_block("user", sanitized_user),
    }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=read) to inspect and memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized


# ---------------------------------------------------------------------------
# Recall memory tools
# ---------------------------------------------------------------------------


def recall_memory_search(
    query: str,
    limit: int = 10,
    task_id: Optional[str] = None,
) -> str:
    """Search through past conversation history."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    query = (query or "").strip()
    if not query:
        return tool_error("query is required")

    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        limit_int = 10
    if limit_int <= 0:
        limit_int = 10

    try:
        entries = system.recall.search(query=query, limit=limit_int)
    except Exception as exc:
        logger.exception("recall_memory_search failed")
        return tool_error(f"Recall search failed: {exc}")

    results: List[Dict[str, Any]] = []
    for entry in entries:
        results.append({
            "id": entry.id,
            "session_id": entry.session_id,
            "role": entry.role,
            "content": entry.content,
            "timestamp": entry.timestamp,
            "relevance_score": entry.relevance_score,
        })

    return tool_result(
        success=True,
        query=query,
        count=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Archival memory tools
# ---------------------------------------------------------------------------


def archival_memory_insert(
    content: str,
    task_id: Optional[str] = None,
) -> str:
    """Insert a new entry into archival memory."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    if content is None:
        content = ""
    content = content.strip()
    if not content:
        return tool_error("content is required")

    scan_error = scan_memory_content(content)
    if scan_error:
        return tool_error(scan_error)

    try:
        entry_id = system.archival.insert(content)
    except Exception as exc:
        logger.exception("archival_memory_insert failed")
        return tool_error(f"Archival insert failed: {exc}")

    return tool_result(
        success=True,
        entry_id=int(entry_id),
        chars=len(content),
        message=f"Archival entry inserted (id={entry_id}).",
    )


def archival_memory_search(
    query: str,
    top_k: int = 5,
    task_id: Optional[str] = None,
) -> str:
    """Semantic search across archival memory."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    query = (query or "").strip()
    if not query:
        return tool_error("query is required")

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        top_k_int = 5
    if top_k_int <= 0:
        top_k_int = 5

    try:
        entries = system.archival.search(query=query, top_k=top_k_int)
    except Exception as exc:
        logger.exception("archival_memory_search failed")
        return tool_error(f"Archival search failed: {exc}")

    results: List[Dict[str, Any]] = []
    for entry in entries:
        results.append({
            "id": entry.id,
            "content": entry.content,
            "metadata": entry.metadata,
            "created_at": entry.created_at,
            "relevance_score": entry.relevance_score,
        })

    return tool_result(
        success=True,
        query=query,
        count=len(results),
        results=results,
    )


def archival_memory_delete(
    entry_id: int,
    task_id: Optional[str] = None,
) -> str:
    """Delete an archival memory entry by id."""
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not initialized")

    try:
        entry_id_int = int(entry_id)
    except (TypeError, ValueError):
        return tool_error("entry_id must be an integer")

    deleted = system.archival.delete(entry_id_int)
    if not deleted:
        return tool_error(f"No archival entry with id {entry_id_int}")

    return tool_result(
        success=True,
        entry_id=entry_id_int,
        message=f"Archival entry {entry_id_int} deleted.",
    )


# ---------------------------------------------------------------------------
# Backward-compatible 'memory' adapter
# ---------------------------------------------------------------------------


def memory_compat(
    action: str,
    target: str = "memory",
    content: str = "",
    old_content: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Backward-compatible ``memory`` tool that routes to the new Letta tools.

    Maps the legacy ``action`` / ``target`` API onto the new core_memory
    tools so trajectories / prompts written for the old tool keep working.

    * ``target='memory'`` -> ``persona`` block
    * ``target='user'``   -> ``human`` block
    """
    system = get_memory_system()
    if system is None:
        return tool_error("Memory system not available", success=False)

    action = (action or "").strip().lower()
    target = (target or "memory").strip().lower()
    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.")

    label = "persona" if target == "memory" else "human"

    if action == "read":
        return core_memory_get(label, task_id=task_id)

    if action == "add":
        if not content:
            return tool_error("content is required for 'add' action")
        block = system.core.get_block(label)
        if block is None:
            return tool_error(f"No block with label '{label}'")
        existing = (block.value or "").strip()
        appended = content.strip()
        new_value = f"{existing}\n{appended}".strip() if existing else appended
        return core_memory_update(label, new_value, task_id=task_id)

    if action == "replace":
        if not old_content:
            return tool_error("old_content is required for 'replace' action")
        if not content:
            return tool_error("content is required for 'replace' action")
        return core_memory_replace(label, old_content, content, task_id=task_id)

    if action == "remove":
        if not old_content and not content:
            return tool_error("old_content (or content) is required for 'remove' action")
        # Either parameter may carry the substring to remove (legacy callers
        # used either form). Prefer old_content when present.
        target_str = old_content or content
        return core_memory_replace(label, target_str, "", task_id=task_id)

    return tool_error(f"Unknown action: {action}. Use: add, replace, remove, read")


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements.

    Availability of the underlying Letta system is checked at handler-call
    time — when the system is missing, handlers return a structured error
    rather than being hidden from the model.
    """
    return True


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schemas
# ---------------------------------------------------------------------------


CORE_MEMORY_GET_SCHEMA = {
    "name": "core_memory_get",
    "description": (
        "Read a core memory block. Core memory is always in the agent's context. "
        "Use this to check the current content of a specific memory block "
        "(e.g. 'persona' for agent self-notes, 'human' for the user profile)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": (
                    "The label of the memory block to read (e.g. 'persona', 'human')."
                ),
            },
        },
        "required": ["label"],
    },
}


CORE_MEMORY_UPDATE_SCHEMA = {
    "name": "core_memory_update",
    "description": (
        "Update a core memory block with new content. This overwrites the "
        "block completely. Use core_memory_replace for partial edits.\n\n"
        "Core memory is always injected into the system prompt, so keep "
        "block contents compact and focused on stable facts. The block has "
        "a character limit; the call fails if the new value exceeds it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The label of the block to update (e.g. 'persona', 'human').",
            },
            "value": {
                "type": "string",
                "description": "The new full content of the block. Replaces the previous value entirely.",
            },
        },
        "required": ["label", "value"],
    },
}


CORE_MEMORY_REPLACE_SCHEMA = {
    "name": "core_memory_replace",
    "description": (
        "Replace a specific substring in a core memory block. Useful for "
        "surgical edits without rewriting the whole block. Only the first "
        "occurrence of old_str is replaced. Pass an empty new_str to "
        "delete the substring."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The label of the block to edit (e.g. 'persona', 'human').",
            },
            "old_str": {
                "type": "string",
                "description": "The substring to find. Must match an existing portion of the block exactly.",
            },
            "new_str": {
                "type": "string",
                "description": "The replacement text. Use an empty string to delete old_str.",
            },
        },
        "required": ["label", "old_str", "new_str"],
    },
}


RECALL_MEMORY_SEARCH_SCHEMA = {
    "name": "recall_memory_search",
    "description": (
        "Search through past conversation history. Returns relevant messages "
        "from previous interactions, ranked by full-text and (when available) "
        "semantic similarity. Use this to recall what was previously said "
        "instead of asking the user to repeat themselves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The text to search for in past messages.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


ARCHIVAL_MEMORY_INSERT_SCHEMA = {
    "name": "archival_memory_insert",
    "description": (
        "Insert a new entry into archival memory (long-term knowledge base). "
        "Use for important facts, learned information, or knowledge worth "
        "preserving long-term. Archival memory is searchable but NOT always "
        "in context — use it for the bulk of durable knowledge that doesn't "
        "need to be in every prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to store. A single self-contained fact or note.",
            },
        },
        "required": ["content"],
    },
}


ARCHIVAL_MEMORY_SEARCH_SCHEMA = {
    "name": "archival_memory_search",
    "description": (
        "Search archival memory using semantic similarity (with FTS5 fallback). "
        "Returns the most relevant entries from the long-term knowledge base. "
        "Call this before answering questions whose answers may already be "
        "stored, to avoid redundant work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural-language query to search for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


ARCHIVAL_MEMORY_DELETE_SCHEMA = {
    "name": "archival_memory_delete",
    "description": (
        "Delete an entry from archival memory by its ID. Get the ID from a "
        "prior archival_memory_search call. Deletion is permanent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The ID of the archival entry to delete.",
            },
        },
        "required": ["entry_id"],
    },
}


# Legacy schema preserved so existing prompts / trajectories that reference
# the ``memory`` tool keep working. Routed through memory_compat() onto the
# new core_memory_* tools.
LEGACY_MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "[Legacy] Save durable information to persistent memory. This tool is "
        "a backward-compatible adapter; prefer the dedicated core_memory_*, "
        "recall_memory_*, and archival_memory_* tools when available.\n\n"
        "Do NOT save task progress or temporary task state — use "
        "session_search to recall prior conversation context instead.\n\n"
        "TARGETS:\n"
        "- 'memory' -> agent's persona block\n"
        "- 'user'   -> user profile (human) block\n\n"
        "ACTIONS: add (append entry), replace (substring swap), "
        "remove (delete substring), read (return current value)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which block: 'memory' (persona) or 'user' (human).",
            },
            "content": {
                "type": "string",
                "description": "Entry content. Required for 'add' and 'replace'.",
            },
            "old_content": {
                "type": "string",
                "description": "Substring identifying the text to replace or remove.",
            },
        },
        "required": ["action", "target"],
    },
}


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


registry.register(
    name="core_memory_get",
    toolset="memory",
    schema=CORE_MEMORY_GET_SCHEMA,
    handler=lambda args, **kw: core_memory_get(
        label=args.get("label", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)


registry.register(
    name="core_memory_update",
    toolset="memory",
    schema=CORE_MEMORY_UPDATE_SCHEMA,
    handler=lambda args, **kw: core_memory_update(
        label=args.get("label", ""),
        value=args.get("value", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)


registry.register(
    name="core_memory_replace",
    toolset="memory",
    schema=CORE_MEMORY_REPLACE_SCHEMA,
    handler=lambda args, **kw: core_memory_replace(
        label=args.get("label", ""),
        old_str=args.get("old_str", ""),
        new_str=args.get("new_str", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)


registry.register(
    name="recall_memory_search",
    toolset="memory",
    schema=RECALL_MEMORY_SEARCH_SCHEMA,
    handler=lambda args, **kw: recall_memory_search(
        query=args.get("query", ""),
        limit=args.get("limit", 10),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="🔍",
)


registry.register(
    name="archival_memory_insert",
    toolset="memory",
    schema=ARCHIVAL_MEMORY_INSERT_SCHEMA,
    handler=lambda args, **kw: archival_memory_insert(
        content=args.get("content", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="📚",
)


registry.register(
    name="archival_memory_search",
    toolset="memory",
    schema=ARCHIVAL_MEMORY_SEARCH_SCHEMA,
    handler=lambda args, **kw: archival_memory_search(
        query=args.get("query", ""),
        top_k=args.get("top_k", 5),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="📚",
)


registry.register(
    name="archival_memory_delete",
    toolset="memory",
    schema=ARCHIVAL_MEMORY_DELETE_SCHEMA,
    handler=lambda args, **kw: archival_memory_delete(
        entry_id=args.get("entry_id", 0),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="📚",
)


# Legacy compat tool — preserved for backward compatibility with old
# prompts / trajectories that reference the single ``memory`` tool.
registry.register(
    name="memory",
    toolset="memory",
    schema=LEGACY_MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_compat(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content", "") or "",
        old_content=args.get("old_content", "") or args.get("old_text", "") or "",
        task_id=kw.get("task_id"),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)


# ---------------------------------------------------------------------------
# Backward-compatible shims for legacy test API
# ---------------------------------------------------------------------------
MEMORY_SCHEMA = LEGACY_MEMORY_SCHEMA


def memory_tool(action: str, target: str = "memory", content: str = "",
                old_content: str = "", old_text: str = "",
                task_id: Optional[str] = None, store: Any = None,
                **_kwargs) -> str:
    """Backward-compatible ``memory`` tool dispatcher.

    Wraps :func:`memory_compat` and adds legacy ``store`` parameter support.
    When ``store`` is provided (old test API), operations are performed
    directly on the :class:`MemoryStore` instance instead of routing
    through the Letta memory system.
    """
    if store is not None:
        # Legacy MemoryStore-based path for tests
        return _memory_store_dispatch(store, action=action, target=target,
                                      content=content, old_content=old_content or old_text)
    return memory_compat(action=action, target=target, content=content,
                         old_content=old_content or old_text, task_id=task_id)


def _memory_store_dispatch(store: "MemoryStore", action: str, target: str = "memory",
                           content: str = "", old_content: str = "") -> str:
    """Dispatch memory operations directly on a MemoryStore instance."""
    action = (action or "").strip().lower()
    target = (target or "memory").strip().lower()
    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "read":
        entries = store._entries_for(target)
        return tool_result(success=True, target=target, entries=entries)

    if action == "add":
        if not content:
            return tool_error("content is required for 'add' action", success=False)
        result = store.add(target, content)
        if result.get("success"):
            return tool_result(success=True, target=target,
                               entries=store._entries_for(target))
        return tool_error(result.get("error", "add failed"), success=False)

    if action == "replace":
        if not old_content:
            return tool_error("old_content is required for 'replace' action", success=False)
        if not content:
            return tool_error("content is required for 'replace' action", success=False)
        result = store.replace(target, old_content, content)
        if result.get("success"):
            return tool_result(success=True, target=target,
                               entries=store._entries_for(target))
        return tool_error(result.get("error", "replace failed"), success=False)

    if action == "remove":
        if not old_content:
            return tool_error("old_content is required for 'remove' action", success=False)
        result = store.remove(target, old_content)
        if result.get("success"):
            return tool_result(success=True, target=target,
                               entries=store._entries_for(target))
        return tool_error(result.get("error", "remove failed"), success=False)

    return tool_error(f"Unknown action: {action}. Use: add, replace, remove, read", success=False)


class MemoryStore:
    """Backward-compatible wrapper around the Letta memory system.

    Provides the old file-backed API (add/replace/remove/load_from_disk)
    by delegating to core_memory_update and the legacy MEMORY.md/USER.md files.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._system_prompt_snapshot: dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))
        # Sanitize entries for the frozen snapshot — poisoned entries are
        # replaced with [BLOCKED: ...] placeholders while the raw text
        # stays in memory_entries so the user can inspect and remove it.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: list[str], filename: str) -> list[str]:
        """Return entries with threat-matching entries replaced by placeholders."""
        try:
            from tools.threat_patterns import scan_for_threats
        except ImportError:
            return entries
        sanitized: list[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=read) to inspect and memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _entries_for(self, target: str) -> list[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: list[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _check_drift(self, target: str) -> Optional[dict[str, Any]]:
        """Detect external modifications to the on-disk file.

        Returns an error dict with ``drift_backup`` if drift is detected,
        or ``None`` if the file is consistent with in-memory state.
        """
        path = self._path_for(target)
        if not path.exists():
            return None
        try:
            disk_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        expected = ENTRY_DELIMITER.join(self._entries_for(target))
        if disk_content == expected:
            return None
        # Drift detected — create a backup and refuse to operate.
        import time as _time
        bak_path = path.with_suffix(f".bak.{int(_time.time())}")
        try:
            bak_path.write_text(disk_content, encoding="utf-8")
        except OSError:
            pass
        return {
            "success": False,
            "error": (
                f"External drift detected in {path.name}. "
                f"File was modified outside this tool. "
                f"See {bak_path.name} and issue #26045 for remediation."
            ),
            "drift_backup": str(bak_path),
            "remediation": "Review the .bak file, reconcile changes, then retry.",
        }

    def add(self, target: str, content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        scan_error = scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        drift = self._check_drift(target)
        if drift:
            return drift
        entries = self._entries_for(target)
        limit = self._char_limit(target)
        if content in entries:
            return self._success_response(target, "Entry already exists (no duplicate added).")
        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            current = self._char_count(target)
            return {
                "success": False,
                "error": f"Memory at {current:,}/{limit:,} chars. Adding this entry ({len(content)} chars) would exceed the limit.",
                "current_entries": entries,
                "usage": f"{current:,}/{limit:,}",
            }
        entries.append(content)
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> dict[str, Any]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}
        scan_error = scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}
        drift = self._check_drift(target)
        if drift:
            return drift
        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        if len(matches) > 1:
            unique_texts = {e for _, e in matches}
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}
        idx = matches[0][0]
        limit = self._char_limit(target)
        test_entries = entries.copy()
        test_entries[idx] = new_content
        new_total = len(ENTRY_DELIMITER.join(test_entries))
        if new_total > limit:
            return {"success": False, "error": f"Replacement would put memory at {new_total:,}/{limit:,} chars."}
        entries[idx] = new_content
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        drift = self._check_drift(target)
        if drift:
            return drift
        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        if len(matches) > 1:
            unique_texts = {e for _, e in matches}
            if len(unique_texts) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}
        idx = matches[0][0]
        entries.pop(idx)
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    def save_to_disk(self, target: str):
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _success_response(self, target: str, message: str = None) -> dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: list[str]):
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".mem_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


__all__ = [
    "ENTRY_DELIMITER",
    "MEMORY_SCHEMA",
    "MemoryStore",
    "archival_memory_delete",
    "archival_memory_insert",
    "archival_memory_search",
    "core_memory_get",
    "core_memory_replace",
    "core_memory_update",
    "get_memory_dir",
    "get_memory_system",
    "memory_compat",
    "memory_tool",
    "recall_memory_search",
    "scan_memory_content",
    "set_memory_system",
]
