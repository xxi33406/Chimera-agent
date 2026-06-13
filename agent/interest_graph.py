"""Interest graph built from entity tags in archival memory."""

import json
from collections import Counter
from typing import Optional


def build_interest_profile(db, top_n: int = 15) -> dict[str, list[str]]:
    """Aggregate entity tags from archival entries into interest categories.

    Returns: {"tech": ["Python", "Docker", ...], "person": ["Alice", ...]}
    """
    rows = db.search_archival_by_metadata_key("entities")

    counters: dict[str, Counter] = {}
    for row in rows:
        meta = row.get("metadata_json") or "{}"
        try:
            metadata = json.loads(meta) if isinstance(meta, str) else meta
        except (json.JSONDecodeError, TypeError):
            continue
        entities = metadata.get("entities", {})
        for etype, values in entities.items():
            if etype not in counters:
                counters[etype] = Counter()
            for v in values:
                counters[etype][v] += 1

    # Build top interests per category
    profile = {}
    for etype, counter in counters.items():
        top = [item for item, _ in counter.most_common(top_n)]
        if top:
            profile[etype] = top

    return profile


def format_interests_block(profile: dict[str, list[str]], max_chars: int = 800) -> str:
    """Format interest profile for Core Memory block."""
    lines = []
    for category, items in profile.items():
        lines.append(f"{category}: {', '.join(items[:8])}")
    text = "\n".join(lines)
    return text[:max_chars]


def parse_interests_text(text: str) -> dict[str, list[str]]:
    """Parse interests block text back into dict format.

    Input format: "tech: Python, Docker, Linux\nperson: Alice, Bob"
    Returns: {"tech": ["Python", "Docker", "Linux"], "person": ["Alice", "Bob"]}
    """
    if not text or not text.strip():
        return {}
    result = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        category, values_str = line.split(":", 1)
        category = category.strip()
        values = [v.strip() for v in values_str.split(",") if v.strip()]
        if category and values:
            result[category] = values
    return result


def refresh_interest_graph(memory_system) -> None:
    """Rebuild interest graph and update Core Memory interests block.

    Called during Dream Engine Phase 6. Zero LLM cost — pure SQL aggregation.
    """
    try:
        db = memory_system._db
        profile = build_interest_profile(db)
        if profile:
            text = format_interests_block(profile)
            memory_system._core.update_block("interests", text)
    except Exception:
        pass  # Never crash dream flow for interest graph
