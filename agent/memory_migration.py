"""Memory migration — MEMORY.md/USER.md → SQLite.

Automatically migrates legacy file-based memory to the new SQLite-backed
Letta memory system on first run. Safe to call multiple times (idempotent).

Called during agent initialization if legacy memory files are detected.

Migration strategy
------------------
* ``MEMORY.md`` → ``persona`` core block; overflow spills into archival.
* ``USER.md``   → ``human``   core block; overflow spills into archival.
* Each successfully processed file is renamed to ``<file>.migrated`` so
  subsequent runs skip the work.
* All exceptions are caught and recorded in the returned stats dict so a
  partial failure never blocks agent startup.

The legacy file format uses ``§`` (section sign) on its own line as the
entry delimiter, i.e. entries are joined by ``"\n§\n"``.  Each entry can
itself span multiple lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Entry delimiter used by the legacy file-based memory system.
ENTRY_DELIMITER = "\n§\n"

# Default char limits — must match agent.letta_memory.DEFAULT_BLOCK_CONFIGS.
# Duplicated here as a safety net for the (rare) case where the live block
# has not been created yet when migration runs.
_DEFAULT_PERSONA_LIMIT = 2200
_DEFAULT_HUMAN_LIMIT = 1375


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def needs_migration() -> bool:
    """Return ``True`` if at least one legacy file is present and unmigrated.

    Cheap — only stats up to four paths.  Safe to call on every startup.
    """
    memory_dir = get_hermes_home() / "memories"
    memory_file = memory_dir / "MEMORY.md"
    user_file = memory_dir / "USER.md"

    if memory_file.exists() and not (memory_dir / "MEMORY.md.migrated").exists():
        return True
    if user_file.exists() and not (memory_dir / "USER.md.migrated").exists():
        return True
    return False


def migrate_to_letta(memory_system) -> Dict[str, Any]:
    """Migrate legacy ``MEMORY.md`` / ``USER.md`` into ``memory_system``.

    Args:
        memory_system: A ``LettaMemorySystem``-shaped object (duck-typed to
            avoid circular imports).  Must expose ``.core.get_block``,
            ``.core.update_block``, and ``.archival.insert``.

    Returns:
        A dict with the migration stats::

            {
                "migrated": bool,         # True if anything was actually moved
                "persona_entries": int,   # entries merged into the persona block
                "human_entries":   int,   # entries merged into the human block
                "archival_entries": int,  # overflow entries pushed to archival
                "errors":          list[str],
            }
    """
    memory_dir = get_hermes_home() / "memories"
    memory_file = memory_dir / "MEMORY.md"
    user_file = memory_dir / "USER.md"

    stats: Dict[str, Any] = {
        "migrated": False,
        "persona_entries": 0,
        "human_entries": 0,
        "archival_entries": 0,
        "errors": [],
    }

    # --- MEMORY.md → persona block (+ archival overflow) --------------------
    if memory_file.exists() and not (memory_dir / "MEMORY.md.migrated").exists():
        _migrate_file(
            memory_system=memory_system,
            file_path=memory_file,
            block_label="persona",
            default_char_limit=_DEFAULT_PERSONA_LIMIT,
            stats=stats,
            core_count_key="persona_entries",
        )
        try:
            memory_file.rename(memory_dir / "MEMORY.md.migrated")
        except Exception as exc:  # pragma: no cover - filesystem-dependent
            stats["errors"].append(f"Failed to rename MEMORY.md: {exc}")

    # --- USER.md → human block (+ archival overflow) ------------------------
    if user_file.exists() and not (memory_dir / "USER.md.migrated").exists():
        _migrate_file(
            memory_system=memory_system,
            file_path=user_file,
            block_label="human",
            default_char_limit=_DEFAULT_HUMAN_LIMIT,
            stats=stats,
            core_count_key="human_entries",
        )
        try:
            user_file.rename(memory_dir / "USER.md.migrated")
        except Exception as exc:  # pragma: no cover - filesystem-dependent
            stats["errors"].append(f"Failed to rename USER.md: {exc}")

    stats["migrated"] = (
        stats["persona_entries"]
        + stats["human_entries"]
        + stats["archival_entries"]
    ) > 0

    if stats["migrated"]:
        logger.info(
            "Memory migration complete: %d persona entries, %d human entries, "
            "%d archival entries",
            stats["persona_entries"],
            stats["human_entries"],
            stats["archival_entries"],
        )
        if stats["errors"]:
            logger.warning(
                "Memory migration encountered %d errors: %s",
                len(stats["errors"]),
                stats["errors"],
            )

    return stats


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_entries(file_path: Path) -> List[str]:
    """Parse a legacy memory file into deduplicated, trimmed entries.

    Handles empty files, files with no delimiters, and trailing whitespace.
    Exact duplicates are dropped while preserving the first occurrence.
    """
    if not file_path.exists():
        return []
    try:
        content = file_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Failed to read legacy memory file %s: %s", file_path, exc)
        return []

    if not content:
        return []

    entries = [e.strip() for e in content.split(ENTRY_DELIMITER)]

    seen = set()
    unique: List[str] = []
    for entry in entries:
        if entry and entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique


def _resolve_char_limit(memory_system, label: str, fallback: int) -> int:
    """Look up the configured char limit for a block, with a safe fallback."""
    try:
        block = memory_system.core.get_block(label)
    except Exception as exc:
        logger.debug("memory_migration: get_block(%s) failed: %s", label, exc)
        return fallback
    if block is None:
        return fallback
    try:
        return int(block.char_limit)
    except (AttributeError, TypeError, ValueError):
        return fallback


def _split_for_block(entries: List[str], char_limit: int) -> tuple[List[str], List[str]]:
    """Greedy split: pack entries into the core block until ``char_limit``.

    Each entry consumes ``len(entry) + 1`` chars (newline separator).
    Entries that can't fit go into the archival overflow list, preserving
    the original order.
    """
    core: List[str] = []
    overflow: List[str] = []
    running = 0
    for entry in entries:
        cost = len(entry) + (1 if core else 0)  # newline only after first entry
        if running + cost <= char_limit:
            core.append(entry)
            running += cost
        else:
            overflow.append(entry)
    return core, overflow


def _migrate_file(
    memory_system,
    file_path: Path,
    block_label: str,
    default_char_limit: int,
    stats: Dict[str, Any],
    core_count_key: str,
) -> None:
    """Migrate one legacy file into the named core block + archival overflow.

    Mutates ``stats`` in place.  Never raises — every error is captured.
    """
    try:
        entries = _parse_entries(file_path)
    except Exception as exc:  # defensive — _parse_entries already catches I/O
        stats["errors"].append(f"Failed to parse {file_path.name}: {exc}")
        return

    if not entries:
        return

    char_limit = _resolve_char_limit(memory_system, block_label, default_char_limit)
    combined = "\n".join(entries)

    if len(combined) <= char_limit:
        # Common case: everything fits in the core block.
        try:
            success, msg = memory_system.core.update_block(block_label, combined)
            if success:
                stats[core_count_key] = len(entries)
            else:
                stats["errors"].append(
                    f"Failed to update {block_label} block: {msg}"
                )
        except Exception as exc:
            stats["errors"].append(
                f"Exception updating {block_label} block: {exc}"
            )
        return

    # Overflow path: greedily pack core, archive the rest.
    core_entries, archival_entries = _split_for_block(entries, char_limit)

    if core_entries:
        try:
            success, msg = memory_system.core.update_block(
                block_label, "\n".join(core_entries)
            )
            if success:
                stats[core_count_key] = len(core_entries)
            else:
                stats["errors"].append(
                    f"Failed to update {block_label} block: {msg}"
                )
        except Exception as exc:
            stats["errors"].append(
                f"Exception updating {block_label} block: {exc}"
            )

    for entry in archival_entries:
        try:
            memory_system.archival.insert(
                content=entry,
                metadata={"source": "migration", "original_file": file_path.name},
            )
            stats["archival_entries"] += 1
        except Exception as exc:
            stats["errors"].append(f"Archival insert failed: {exc}")


__all__ = ["ENTRY_DELIMITER", "migrate_to_letta", "needs_migration"]
