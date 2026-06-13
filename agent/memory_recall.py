"""Topic-driven memory recall — triggers when user mentions known entities."""

import time
from typing import List, Optional

# Cooldown: minimum 10 minutes between recalls
_RECALL_COOLDOWN_SECONDS = 600
_last_recall_time: float = 0.0


def check_topic_recall(user_message: str, memory_system, db) -> str:
    """Check if user message matches archival entities; return hint or empty string.

    Trigger condition:
    - User message contains entities that exist in archival memory
    - Cooldown since last recall has elapsed (10 min)
    - At least one archival entry is found for the matched entity
    """
    global _last_recall_time

    now = time.time()
    if now - _last_recall_time < _RECALL_COOLDOWN_SECONDS:
        return ""

    # Extract entities from current message (reuse ArchivalMemory's static method)
    from agent.letta_memory import ArchivalMemory
    entities = ArchivalMemory._extract_entities(user_message)
    if not entities:
        return ""

    # Search archival for matching entities
    for etype, values in entities.items():
        for value in values[:3]:  # Limit to top 3 entities per type
            try:
                rows = db.search_archival_by_entity(etype, value, limit=3)
                if rows:
                    # Found related memories — format hint
                    _last_recall_time = now
                    return _format_topic_hint(value, rows[:1])
            except Exception:
                continue

    return ""


def _format_topic_hint(topic: str, memories: List[dict]) -> str:
    """Format topic-matched memories as system prompt hint."""
    if not memories:
        return ""
    snippets = []
    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue
        if len(content) > 120:
            content = content[:117] + "..."
        snippets.append(content)

    if not snippets:
        return ""

    recall_text = "; ".join(snippets)
    return (
        f"\n[共同记忆提示: 用户提到了「{topic}」，"
        f"你们之前有过相关经历——{recall_text}。"
        f"如果自然地相关，可以简短提及，但不要刻意。不相关则忽略。]"
    )
