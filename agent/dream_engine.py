"""Dream Engine — background memory consolidation after N user turns.

Runs in a daemon thread after the configured number of user turns complete.
Performs four operations: Distill, Archive, Merge, Decay.
All operations are best-effort — failures are logged but never bubble up
to the main conversation loop.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DreamEngine:
    """Background memory consolidation — runs after N turns during idle.

    Lifecycle:
        1. Created once per AIAgent session (if enabled in config).
        2. on_turn_complete() called after each assistant response.
        3. After trigger_turns reached, spawns a daemon thread to consolidate.
        4. Thread runs _dream() pipeline, then resets counter.
    """

    def __init__(
        self,
        memory_system: Any,
        config: Optional[Dict[str, Any]] = None,
        auxiliary_fn: Optional[Callable[..., Optional[str]]] = None,
    ) -> None:
        """
        Args:
            memory_system: LettaMemorySystem instance.
            config: ``dream_engine`` section from config.yaml.
            auxiliary_fn: Optional callable ``(prompt, task="dream") -> str``
                          for LLM-assisted operations.  If ``None``, only
                          rule-based ops run.
        """
        self.memory = memory_system
        self.config: Dict[str, Any] = dict(config or {})
        self._auxiliary_fn = auxiliary_fn

        self._trigger_turns: int = int(self.config.get("trigger_turns", 5))
        self._deep_trigger_turns: int = int(self.config.get("deep_trigger_turns", 20))
        self._use_llm: bool = bool(self.config.get("use_llm", True))
        self._max_archival_merge: int = int(self.config.get("max_archival_merge", 10))
        self._distill_to_core: bool = bool(self.config.get("distill_to_core", True))
        self._log_dreams: bool = bool(self.config.get("log_dreams", True))

        # Idle consolidation config
        self._idle_enabled: bool = bool(self.config.get("idle_consolidation", True))
        self._idle_threshold_seconds: int = int(self.config.get("idle_threshold_minutes", 30)) * 60
        self._idle_sample_size: int = int(self.config.get("idle_sample_size", 10))
        self._idle_merge_threshold: float = float(self.config.get("idle_merge_threshold", 0.85))

        # Pattern learning config (used during deep dreams).
        self._pattern_learning_interval: int = int(
            self.config.get("pattern_learning_interval", 5)
        )
        self._pattern_min_frequency: int = int(
            self.config.get("pattern_min_frequency", 3)
        )

        self._turn_count: int = 0
        self._total_turns: int = 0  # for deep dream tracking
        self._running: bool = False
        self._lock = threading.Lock()
        self._last_dream_time: float = 0.0

        # Collected messages since last dream
        self._recent_messages: List[Dict[str, Any]] = []

        # Restore persisted state (cross-session)
        self._restore_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_turn_complete(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Called after each turn completes.  Tracks messages and maybe
        triggers a dream.

        Args:
            session_id: Current session ID.
            messages: Recent messages from this turn (last ~10 messages).
        """
        self._turn_count += 1
        self._total_turns += 1
        self._persist_state()

        # Accumulate recent messages for dream processing.
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") in ("user", "assistant"):
                self._recent_messages.append(msg)
        # Keep only the last 30 messages to bound memory.
        if len(self._recent_messages) > 30:
            self._recent_messages = self._recent_messages[-30:]

        # Idle consolidation check: if enough time since last dream AND not currently running
        if (self._idle_enabled
            and not self._running
            and self._last_dream_time > 0
            and time.time() - self._last_dream_time > self._idle_threshold_seconds):

            # Check last idle consolidation time to avoid spamming
            _idle_db = self._get_db()
            _last_idle_str = _idle_db.get_dream_state("last_idle_consolidation") if _idle_db else None
            _should_idle = True
            if _last_idle_str:
                try:
                    _should_idle = time.time() - float(_last_idle_str) > self._idle_threshold_seconds
                except (ValueError, TypeError):
                    pass

            if _should_idle:
                _idle_thread = threading.Thread(
                    target=self._idle_consolidation,
                    args=(session_id,),
                    daemon=True,
                    name="idle-consolidation",
                )
                _idle_thread.start()
                return  # Don't trigger dream in same turn as idle consolidation

        # Check whether we should dream.
        is_deep = (
            self._total_turns > 0
            and self._total_turns % self._deep_trigger_turns == 0
        )

        if self._turn_count >= self._trigger_turns:
            self._schedule_dream(session_id, is_deep=is_deep)

    @property
    def is_running(self) -> bool:
        """Whether a dream is currently in progress."""
        return self._running

    @property
    def turns_until_dream(self) -> int:
        """How many more turns until the next dream triggers."""
        return max(0, self._trigger_turns - self._turn_count)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_dream(self, session_id: str, is_deep: bool = False) -> None:
        """Spawn background thread for consolidation (debounced)."""
        with self._lock:
            if self._running:
                return
            self._running = True

        dream_messages = list(self._recent_messages)
        thread = threading.Thread(
            target=self._dream,
            args=(session_id, dream_messages, is_deep),
            daemon=True,
            name="dream-engine",
        )
        thread.start()

    def _dream(
        self,
        session_id: str,
        recent_messages: List[Dict[str, Any]],
        is_deep: bool,
    ) -> None:
        """Execute the consolidation pipeline (background thread).

        Light dream (every ``trigger_turns``): distill + archive.
        Deep dream (every ``deep_trigger_turns``): full pipeline including
        merge + decay.
        """
        dream_start = time.time()
        dream_log_parts: List[str] = []

        try:
            logger.info(
                "Dream Engine: starting %s dream (turns=%d, messages=%d)",
                "deep" if is_deep else "light",
                self._turn_count,
                len(recent_messages),
            )

            # Phase 1: Distill — extract key facts → core memory.
            if self._distill_to_core and recent_messages:
                distill_result = self._distill(session_id, recent_messages)
                if distill_result:
                    dream_log_parts.append(f"Distilled: {distill_result}")

            # Phase 2: Archive — identify valuable knowledge → archival.
            if recent_messages:
                archive_result = self._archive(session_id, recent_messages)
                if archive_result:
                    dream_log_parts.append(f"Archived: {archive_result}")

            # Deep dream only: merge + decay.
            if is_deep:
                merge_result = self._merge_similar(session_id)
                if merge_result:
                    dream_log_parts.append(f"Merged: {merge_result}")

                decay_result = self._decay_outdated(session_id)
                if decay_result:
                    dream_log_parts.append(f"Decayed: {decay_result}")

                # Phase 5: Pattern learning (periodic during deep dreams).
                _pattern_interval = max(1, int(self._pattern_learning_interval))
                _deep_step = max(1, int(self._deep_trigger_turns))
                if (
                    self._total_turns > 0
                    and (self._total_turns // _deep_step) % _pattern_interval == 0
                ):
                    try:
                        pattern_result = self._learn_patterns()
                        if pattern_result:
                            dream_log_parts.append(f"Patterns: {pattern_result}")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Pattern learning failed: %s", exc)

                # Phase 6: Refresh interest graph (no LLM, pure aggregation)
                try:
                    from agent.interest_graph import refresh_interest_graph
                    refresh_interest_graph(self._memory)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Interest graph refresh failed: %s", e)

            elapsed = time.time() - dream_start
            logger.info(
                "Dream Engine: completed in %.1fs — %s",
                elapsed,
                "; ".join(dream_log_parts) or "no changes",
            )

            if self._log_dreams and dream_log_parts:
                self._write_dream_log(session_id, dream_log_parts, is_deep, elapsed)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dream Engine: error during consolidation: %s", exc, exc_info=True
            )
        finally:
            with self._lock:
                self._running = False
            self._turn_count = 0
            self._last_dream_time = time.time()
            self._persist_state()  # Save reset state

    # ------------------------------------------------------------------
    # Phase 1 — Distill
    # ------------------------------------------------------------------

    def _distill(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract key facts from recent messages → update core memory blocks."""
        if not messages:
            return None

        if self._use_llm and self._auxiliary_fn is not None:
            conv_text = self._format_messages_for_prompt(messages)
            return self._distill_with_llm(session_id, conv_text)
        return self._distill_rule_based(session_id, messages)

    def _distill_with_llm(
        self, session_id: str, conv_text: str
    ) -> Optional[str]:
        """Use LLM to extract facts and update core memory."""
        prompt = (
            "Review the following recent conversation and extract KEY FACTS "
            "about the user (preferences, personal info, stated goals, "
            "corrections to previous assumptions). Also identify any facts "
            "about the assistant's persona that should be remembered.\n\n"
            "Return ONLY valid JSON:\n"
            '{"facts": [{"block": "persona|human", "key": "short_label", '
            '"value": "concise fact"}]}\n\n'
            'If no new facts worth remembering, return: {"facts": []}\n\n'
            f"Recent conversation:\n{conv_text}"
        )

        try:
            response = self._call_auxiliary(prompt)
            if not response:
                return None

            data = self._parse_json_response(response)
            if not data or not data.get("facts"):
                return None

            facts = data["facts"]
            if not isinstance(facts, list):
                return None

            updated = 0
            for fact in facts[:10]:  # cap per dream
                if not isinstance(fact, dict):
                    continue
                block_label = str(fact.get("block") or "human")
                key = str(fact.get("key") or "").strip()
                value = str(fact.get("value") or "").strip()
                if not key or not value:
                    continue
                if self._append_fact_to_block(block_label, key, value):
                    updated += 1
                    # Record distill result for pattern learning.
                    try:
                        _pdb = self._get_db()
                        if _pdb:
                            _pdb.add_distill_result(
                                cycle=self._total_turns,
                                block=block_label,
                                key=key,
                                value=value,
                                status="applied",
                            )
                    except Exception:
                        pass

            return f"{updated} facts to core memory" if updated > 0 else None

        except Exception as exc:  # noqa: BLE001
            logger.debug("Dream distill LLM error: %s", exc)
            return None

    def _distill_rule_based(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Rule-based fact extraction (no LLM required).

        Looks for explicit self-declarations like ``I am...``,
        ``My name is...``, ``I prefer...``, ``I work at...`` etc.
        """
        patterns = [
            (r"(?:my name is|i'?m called|call me)\s+([A-Za-z][\w\-]*)", "name"),
            (r"i (?:work|am working) (?:at|for|in)\s+(.+?)(?:\.|,|$)", "work"),
            (r"i (?:prefer|like|love|enjoy)\s+(.+?)(?:\.|,|$)", "preference"),
            (r"i (?:use|am using)\s+(.+?)(?:\.|,|$)", "tools"),
            (r"i (?:live|am) in\s+(.+?)(?:\.|,|$)", "location"),
        ]

        extracted: List[tuple] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue
            for pattern, label in patterns:
                for match in re.findall(pattern, content, re.IGNORECASE):
                    val = match.strip() if isinstance(match, str) else ""
                    if val:
                        extracted.append((label, val))

        if not extracted:
            return None

        updated = 0
        for label, value in extracted[:5]:
            if self._append_fact_to_block("human", label, value):
                updated += 1

        return f"{updated} rule-based facts" if updated > 0 else None

    def _append_fact_to_block(
        self, block_label: str, key: str, value: str
    ) -> bool:
        """Append ``- key: value`` into the named core block if not present.

        Returns True on a successful update.  Silently skips blocks that
        don't exist or that exceed their char_limit.
        """
        core = getattr(self.memory, "core", None)
        if core is None:
            return False
        try:
            block = core.get_block(block_label)
        except Exception:
            return False
        if block is None:
            return False
        current = getattr(block, "value", "") or ""
        # Cheap dedupe: skip if the key already appears in the block.
        if key.lower() in current.lower():
            return False
        fact_line = f"- {key}: {value}"
        new_value = (current.rstrip() + "\n" + fact_line).lstrip("\n")
        try:
            ok, _msg = core.update_block(block_label, new_value)
            return bool(ok)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Pattern learning (deep dreams only)
    # ------------------------------------------------------------------

    def _learn_patterns(self) -> Optional[str]:
        """Synthesize recurring distill facts into the ``patterns`` core block.

        Called periodically during deep dreams. Identifies fact keys that
        appear frequently across multiple dream cycles and either asks the
        auxiliary LLM to synthesize behavioural patterns from them, or
        falls back to writing the most frequent facts directly. The result
        is appended to the ``patterns`` block, trimming oldest lines if
        the configured char_limit would be exceeded.
        """
        db = self._get_db()
        if not db or not hasattr(db, "get_recurring_keys"):
            return None

        min_freq = max(1, int(self._pattern_min_frequency))
        try:
            recurring = db.get_recurring_keys(min_count=min_freq)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pattern learning: get_recurring_keys failed: %s", exc)
            return None
        if not recurring:
            return None

        core = getattr(self.memory, "core", None)
        if core is None:
            return None

        # The patterns block may not exist yet; handle gracefully.
        patterns_block = None
        try:
            patterns_block = core.get_block("patterns")
        except Exception:
            patterns_block = None

        existing = ""
        if patterns_block is not None:
            existing = getattr(patterns_block, "value", "") or ""

        char_limit = 1500
        if patterns_block is not None:
            char_limit = int(getattr(patterns_block, "char_limit", 1500) or 1500)

        def _trim(text: str) -> str:
            if len(text) <= char_limit:
                return text
            lines = text.split("\n")
            while len("\n".join(lines)) > char_limit and len(lines) > 3:
                lines.pop(0)
            return "\n".join(lines)

        # LLM-assisted synthesis.
        if self._use_llm and self._auxiliary_fn is not None:
            prompt = (
                "Based on these frequently observed facts about the user, "
                "synthesize 3-5 high-level behavioral patterns or "
                "preferences. Each pattern should be a concise, actionable "
                "insight.\n"
                'Return ONLY valid JSON: {"patterns": ["pattern1", "pattern2", ...]}\n\n'
                "Recurring facts:\n"
                + "\n".join(
                    f"- {r['key']}: {r['value']} (seen {r['count']}x)"
                    for r in recurring[:20]
                )
            )
            try:
                response = self._call_auxiliary(prompt)
                data = self._parse_json_response(response) if response else None
                if data and isinstance(data.get("patterns"), list):
                    patterns_text = "\n".join(
                        f"- {p}"
                        for p in data["patterns"][:5]
                        if isinstance(p, str) and p.strip()
                    )
                    if patterns_text:
                        new_value = _trim(
                            (existing.rstrip() + "\n" + patterns_text).strip()
                        )
                        try:
                            ok, _msg = core.update_block("patterns", new_value)
                            if ok:
                                return f"{len(data['patterns'])} rules learned"
                        except Exception:
                            return None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Pattern learning LLM failed: %s", exc)
            return None

        # Rule-based fallback: write the most recurring facts directly.
        facts_text = "\n".join(
            f"- {r['key']}: {r['value']}" for r in recurring[:5]
        )
        if not facts_text:
            return None
        new_value = _trim((existing.rstrip() + "\n" + facts_text).strip())
        try:
            ok, _msg = core.update_block("patterns", new_value)
            if ok:
                return f"{len(recurring[:5])} patterns from frequency"
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Phase 2 — Archive
    # ------------------------------------------------------------------

    def _archive(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Identify valuable knowledge → archival memory."""
        if not messages:
            return None

        if self._use_llm and self._auxiliary_fn is not None:
            conv_text = self._format_messages_for_prompt(messages)
            return self._archive_with_llm(session_id, conv_text)
        return self._archive_rule_based(session_id, messages)

    def _archive_with_llm(
        self, session_id: str, conv_text: str
    ) -> Optional[str]:
        """Use LLM to identify archival-worthy knowledge."""
        prompt = (
            "Review the following conversation and identify KNOWLEDGE ENTRIES "
            "worth archiving for long-term reference. Focus on:\n"
            "- Technical decisions or solutions\n"
            "- Project-specific information\n"
            "- User-stated requirements or constraints\n"
            "- Important context that might be needed later\n\n"
            "Return ONLY valid JSON:\n"
            '{"entries": [{"content": "concise knowledge entry", '
            '"tags": "comma,separated,tags"}]}\n\n'
            'If nothing worth archiving, return: {"entries": []}\n\n'
            f"Conversation:\n{conv_text}"
        )

        try:
            response = self._call_auxiliary(prompt)
            if not response:
                return None

            data = self._parse_json_response(response)
            if not data or not data.get("entries"):
                return None

            entries = data["entries"]
            if not isinstance(entries, list):
                return None

            archival = getattr(self.memory, "archival", None)
            if archival is None:
                return None

            inserted = 0
            for entry in entries[:5]:  # cap per dream
                if not isinstance(entry, dict):
                    continue
                content = str(entry.get("content") or "").strip()
                if not content or len(content) < 10:
                    continue
                # Cheap dedupe: substring overlap with top-3 hits.
                try:
                    existing = archival.search(content[:50], top_k=3)
                except Exception:
                    existing = []
                duplicate = False
                lower = content.lower()
                for e in existing or []:
                    e_text = (getattr(e, "content", "") or "").lower()
                    if not e_text:
                        continue
                    if lower in e_text or e_text in lower:
                        duplicate = True
                        break
                if duplicate:
                    continue
                try:
                    tags_str = str(entry.get("tags") or "")
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    metadata = (
                        {"source": "llm_archive", "tags": tags}
                        if tags
                        else {"source": "llm_archive"}
                    )
                    new_id = archival.insert(content, metadata=metadata)
                    inserted += 1
                    # LLM-flagged entries get higher initial importance.
                    self._set_archival_importance(new_id, 0.8)
                except Exception:
                    pass

            return f"{inserted} new entries" if inserted > 0 else None

        except Exception as exc:  # noqa: BLE001
            logger.debug("Dream archive LLM error: %s", exc)
            return None

    def _archive_rule_based(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Rule-based archival: long assistant responses with code/technical
        content."""
        archival = getattr(self.memory, "archival", None)
        if archival is None:
            return None

        archived = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if len(content) > 200 and (
                "```" in content or "def " in content or "class " in content
            ):
                snippet = content[:500] + ("..." if len(content) > 500 else "")
                try:
                    new_id = archival.insert(f"[Technical note] {snippet}")
                    archived += 1
                    # Rule-based archives get the default importance.
                    self._set_archival_importance(new_id, 0.5)
                except Exception:
                    continue
                if archived >= 3:
                    break

        return f"{archived} technical entries" if archived > 0 else None

    # ------------------------------------------------------------------
    # Phase 3 — Merge similar archival entries (deep dreams only)
    # ------------------------------------------------------------------

    def _merge_similar(self, session_id: str) -> Optional[str]:
        """Find and merge semantically similar archival entries."""
        archival = getattr(self.memory, "archival", None)
        if archival is None:
            return None
        if not hasattr(archival, "find_similar") or not hasattr(
            archival, "merge_entries"
        ):
            return None

        try:
            pairs = archival.find_similar(
                threshold=0.9, limit=self._max_archival_merge
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dream merge: find_similar error: %s", exc)
            return None

        if not pairs:
            return None

        merged_count = 0
        for pair in pairs:
            try:
                id_a, id_b, content_a, content_b = pair
            except Exception:
                continue

            # Capture source importance so the merged entry inherits the max.
            src_importance = self._get_max_importance([id_a, id_b])

            if self._use_llm and self._auxiliary_fn is not None:
                merge_prompt = (
                    "Merge these two similar memory entries into one concise "
                    "entry:\n"
                    f"Entry 1: {content_a}\n"
                    f"Entry 2: {content_b}\n\n"
                    "Return ONLY the merged text (no JSON, no explanation):"
                )
                try:
                    merged_text = self._call_auxiliary(merge_prompt)
                except Exception:
                    merged_text = None
                merged_text = (merged_text or "").strip()
                if len(merged_text) > 10:
                    try:
                        new_id = archival.merge_entries([id_a, id_b], merged_text)
                        merged_count += 1
                        self._set_archival_importance(new_id, src_importance)
                        continue
                    except Exception:
                        pass
                # fall through to rule-based on LLM failure

            # Rule-based: keep the longer entry, drop the shorter.
            keeper = content_a if len(content_a) >= len(content_b) else content_b
            try:
                new_id = archival.merge_entries([id_a, id_b], keeper)
                merged_count += 1
                self._set_archival_importance(new_id, src_importance)
            except Exception:
                continue

        return f"{merged_count} pairs" if merged_count > 0 else None

    # ------------------------------------------------------------------
    # Idle consolidation (lightweight, no LLM)
    # ------------------------------------------------------------------

    def _idle_consolidation(self, session_id: str) -> None:
        """Run lightweight consolidation during idle time (no LLM needed).

        Randomly samples archival entries, finds similar pairs via embedding,
        merges highly similar ones, and updates access counts.
        """
        archival = getattr(self.memory, "archival", None)
        if not archival or not hasattr(archival, "_db"):
            return

        db = archival._db
        sample_size = self._idle_sample_size
        merged_ids: set = set()

        try:
            # 1. Random sample
            entries = db.get_random_archival_entries(count=sample_size)
            if len(entries) < 2:
                return

            # 2. Find similar pairs (using embedding if available)
            merged_count = 0
            if hasattr(archival, '_embedding_available') and archival._embedding_available():
                try:
                    from agent.letta_memory import _deserialize_embedding, cosine_similarity
                except ImportError:
                    logger.debug("Idle consolidation: cannot import embedding utils")
                    return

                pairs_checked = 0
                for i in range(len(entries)):
                    if entries[i]["id"] in merged_ids:
                        continue
                    vec_i = _deserialize_embedding(entries[i].get("embedding"))
                    if not vec_i:
                        continue
                    for j in range(i + 1, len(entries)):
                        if entries[j]["id"] in merged_ids:
                            continue
                        vec_j = _deserialize_embedding(entries[j].get("embedding"))
                        if not vec_j:
                            continue
                        sim = cosine_similarity(vec_i, vec_j)
                        if sim > self._idle_merge_threshold:
                            try:
                                archival.merge_entries(entries[i]["id"], entries[j]["id"])
                                merged_ids.add(entries[i]["id"])
                                merged_ids.add(entries[j]["id"])
                                merged_count += 1
                            except Exception:
                                pass
                        pairs_checked += 1
                        if pairs_checked > 20:
                            break
                    if pairs_checked > 20:
                        break

            # 3. Update importance for accessed entries (skip merged ones)
            for entry in entries:
                if entry["id"] not in merged_ids:
                    try:
                        db.increment_access(entry["id"])
                    except Exception:
                        pass

            # 4. Log result
            if merged_count > 0:
                logger.info(
                    "Idle consolidation: merged %d pairs from %d samples",
                    merged_count, len(entries)
                )

            # 5. Persist last idle time
            self_db = self._get_db()
            if self_db:
                self_db.set_dream_state("last_idle_consolidation", str(time.time()))

        except Exception as e:
            logger.debug("Idle consolidation error: %s", e)

    # ------------------------------------------------------------------
    # Phase 4 — Decay outdated entries (deep dreams only)
    # ------------------------------------------------------------------

    def _decay_outdated(self, session_id: str) -> Optional[str]:
        """Decay outdated archival importance + drop contradicted core lines.

        Archival decay is purely time-based and runs whenever a deep dream
        fires.  Core-memory contradiction detection still requires an LLM.
        """
        decay_parts: List[str] = []

        archival_decayed = self._decay_archival_importance()
        if archival_decayed > 0:
            decay_parts.append(f"{archival_decayed} archival imp")

        if not (self._use_llm and self._auxiliary_fn is not None):
            return ", ".join(decay_parts) if decay_parts else None

        core = getattr(self.memory, "core", None)
        if core is None:
            return ", ".join(decay_parts) if decay_parts else None

        try:
            blocks = core.list_blocks()
        except Exception:
            return ", ".join(decay_parts) if decay_parts else None
        if not blocks:
            return ", ".join(decay_parts) if decay_parts else None

        decayed = 0
        for block in blocks:
            label = getattr(block, "label", None)
            if not label:
                continue
            content = getattr(block, "value", "") or ""
            if not content or len(content) < 50:
                continue

            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip().startswith("- ")
            ]
            if len(lines) < 3:
                continue

            check_prompt = (
                "Review these memory entries for CONTRADICTIONS or OUTDATED "
                "info. Return ONLY JSON:\n"
                '{"outdated": ["exact line to remove", ...], '
                '"reason": "brief explanation"}\n'
                'If none found: {"outdated": [], "reason": "all consistent"}\n\n'
                "Entries:\n" + "\n".join(lines)
            )

            try:
                response = self._call_auxiliary(check_prompt)
            except Exception:
                continue
            if not response:
                continue

            data = self._parse_json_response(response)
            if not data or not data.get("outdated"):
                continue

            outdated_lines = data["outdated"]
            if not isinstance(outdated_lines, list):
                continue

            new_content = content
            removed_here = 0
            for outdated_line in outdated_lines[:3]:
                if not isinstance(outdated_line, str):
                    continue
                if outdated_line in new_content:
                    new_content = new_content.replace(outdated_line, "")
                    removed_here += 1

            if removed_here == 0:
                continue

            # Tidy up double blank lines that may result from removals.
            new_content = re.sub(r"\n{3,}", "\n\n", new_content).strip()
            try:
                ok, _msg = core.update_block(label, new_content)
                if ok:
                    decayed += removed_here
            except Exception:
                continue

        return f"{decayed} outdated entries" if decayed > 0 else None

    def _decay_archival_importance(
        self, decay_threshold_hours: float = 168.0, decay_factor: float = 0.9
    ) -> int:
        """Reduce importance for archival entries that haven't been accessed
        recently.

        Returns the number of rows touched.  Defaults: decay anything not
        accessed in the last week, multiplying importance by 0.9, with a
        floor of 0.1.
        """
        archival = getattr(self.memory, "archival", None)
        if archival is None:
            return 0
        db = getattr(archival, "_db", None) or self._get_db()
        if db is None:
            return 0
        try:
            conn = db.connect()
            cutoff = time.time() - max(0.0, float(decay_threshold_hours)) * 3600.0
            rows = conn.execute(
                """
                SELECT archival_id, importance FROM archival_scoring
                WHERE last_accessed_at IS NOT NULL
                  AND last_accessed_at < ?
                  AND importance > 0.1
                """,
                (cutoff,),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dream decay: archival scan failed: %s", exc)
            return 0

        decayed = 0
        for row in rows:
            try:
                archival_id = int(row[0])
                current = float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            new_imp = max(0.1, current * float(decay_factor))
            try:
                db.set_importance(archival_id, new_imp)
                decayed += 1
            except Exception:
                continue
        return decayed

    # ------------------------------------------------------------------
    # Archival importance helpers
    # ------------------------------------------------------------------

    def _archival_db(self):
        """Return the MemoryDB used by archival memory, or ``None``."""
        archival = getattr(self.memory, "archival", None)
        if archival is None:
            return None
        return getattr(archival, "_db", None) or self._get_db()

    def _set_archival_importance(
        self, entry_id: Optional[int], score: float
    ) -> None:
        """Best-effort wrapper around ``MemoryDB.set_importance``."""
        if not entry_id:
            return
        db = self._archival_db()
        if db is None or not hasattr(db, "set_importance"):
            return
        try:
            db.set_importance(int(entry_id), float(score))
        except Exception:
            logger.debug(
                "Dream Engine: set_importance failed for %s", entry_id, exc_info=True
            )

    def _get_max_importance(self, entry_ids: List[int]) -> float:
        """Return the highest stored importance among ``entry_ids`` (default 0.5)."""
        db = self._archival_db()
        if db is None or not hasattr(db, "get_importance"):
            return 0.5
        best = 0.5
        for entry_id in entry_ids:
            if not entry_id:
                continue
            try:
                imp = float(db.get_importance(int(entry_id)))
            except Exception:
                continue
            if imp > best:
                best = imp
        return best

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_dream_log(
        self,
        session_id: str,
        parts: List[str],
        is_deep: bool,
        elapsed: float,
    ) -> None:
        """Write a dream summary entry to archival memory."""
        archival = getattr(self.memory, "archival", None)
        if archival is None:
            return

        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        dream_type = "Deep Dream" if is_deep else "Light Dream"
        log_entry = (
            f"[{dream_type} {dt_str}] {'; '.join(parts)} "
            f"(took {elapsed:.1f}s)"
        )
        try:
            archival.insert(log_entry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cross-session state persistence
    # ------------------------------------------------------------------

    def _get_db(self):
        """Locate the underlying :class:`MemoryDB` for state persistence."""
        if hasattr(self.memory, "_db"):
            return self.memory._db
        # Try core -> _db or archival -> _db
        if hasattr(self.memory, "core") and hasattr(self.memory.core, "_db"):
            return self.memory.core._db
        if hasattr(self.memory, "archival") and hasattr(
            self.memory.archival, "_db"
        ):
            return self.memory.archival._db
        return None

    def _persist_state(self) -> None:
        """Save dream state to SQLite for cross-session persistence."""
        try:
            db = self._get_db()
            if db:
                db.set_dream_state("turn_count", str(self._turn_count))
                db.set_dream_state("total_turns", str(self._total_turns))
                db.set_dream_state(
                    "last_dream_time", str(self._last_dream_time)
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Dream Engine: failed to persist state: %s", e)

    def _restore_state(self) -> None:
        """Restore dream state from SQLite."""
        try:
            db = self._get_db()
            if db:
                tc = db.get_dream_state("turn_count")
                if tc is not None:
                    self._turn_count = int(tc)
                tt = db.get_dream_state("total_turns")
                if tt is not None:
                    self._total_turns = int(tt)
                ldt = db.get_dream_state("last_dream_time")
                if ldt is not None:
                    self._last_dream_time = float(ldt)
        except Exception as e:  # noqa: BLE001
            logger.debug("Dream Engine: failed to restore state: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_auxiliary(self, prompt: str) -> Optional[str]:
        """Invoke the configured auxiliary LLM, swallowing any error."""
        if self._auxiliary_fn is None:
            return None
        try:
            try:
                return self._auxiliary_fn(prompt, task="dream")
            except TypeError:
                # Allow callables that don't accept the ``task`` kwarg.
                return self._auxiliary_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dream auxiliary call failed: %s", exc)
            return None

    @staticmethod
    def _format_messages_for_prompt(
        messages: List[Dict[str, Any]],
        max_chars: int = 3000,
    ) -> str:
        """Format messages for an LLM prompt, truncating to ``max_chars``."""
        lines: List[str] = []
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            line = f"{role}: {content[:500]}"
            if total + len(line) > max_chars:
                lines.append("... (truncated)")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from an LLM response, handling markdown code blocks."""
        if not response:
            return None
        text = response.strip()

        # Strip a fenced code block if present.
        if text.startswith("```"):
            split_lines = text.split("\n")
            if len(split_lines) >= 3:
                if split_lines[-1].strip() == "```":
                    split_lines = split_lines[1:-1]
                else:
                    split_lines = split_lines[1:]
                text = "\n".join(split_lines).strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract a JSON object from mixed text.
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if match:
            try:
                data = json.loads(match.group())
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None


__all__ = ["DreamEngine"]
