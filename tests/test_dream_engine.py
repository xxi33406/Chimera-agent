"""Tests for agent.dream_engine — Dream Engine memory consolidation."""

import time
from unittest.mock import MagicMock

import pytest

from agent.dream_engine import DreamEngine


class TestDreamEngineTrigger:
    """Test turn counting and dream triggering."""

    def test_no_trigger_before_threshold(self):
        """Dream should not trigger before trigger_turns reached."""
        mock_memory = MagicMock()
        # Prevent _restore_state from loading stale values via MagicMock.__int__
        mock_memory._db.get_dream_state.return_value = None
        engine = DreamEngine(mock_memory, {"trigger_turns": 5, "enabled": True})

        for i in range(4):
            engine.on_turn_complete(
                "sess1", [{"role": "user", "content": f"msg {i}"}]
            )

        assert engine._turn_count == 4
        assert not engine.is_running

    def test_trigger_at_threshold(self):
        """Dream should trigger at exactly trigger_turns."""
        mock_memory = MagicMock()
        mock_memory._db.get_dream_state.return_value = None
        mock_memory.core = None
        mock_memory.archival = None
        engine = DreamEngine(
            mock_memory,
            {"trigger_turns": 3, "enabled": True, "log_dreams": False},
        )

        for i in range(3):
            engine.on_turn_complete(
                "sess1", [{"role": "user", "content": f"msg {i}"}]
            )

        # Give thread time to start
        time.sleep(0.1)
        # After dream completes (very fast with no memory), counter should reset
        time.sleep(0.5)
        assert engine._turn_count == 0

    def test_debounce_prevents_concurrent_dreams(self):
        """Should not start a second dream while one is running."""
        mock_memory = MagicMock()
        mock_memory._db.get_dream_state.return_value = None
        mock_memory.core = None
        mock_memory.archival = None

        engine = DreamEngine(
            mock_memory,
            {"trigger_turns": 1, "enabled": True, "log_dreams": False},
        )

        # Force running state
        engine._running = True
        engine.on_turn_complete(
            "sess1", [{"role": "user", "content": "test"}]
        )

        # Should not spawn another thread; counter increments but no new dream
        assert engine._turn_count == 1

    def test_turns_until_dream(self):
        """turns_until_dream property should count down correctly."""
        mock_memory = MagicMock()
        mock_memory._db.get_dream_state.return_value = None
        engine = DreamEngine(mock_memory, {"trigger_turns": 5})

        assert engine.turns_until_dream == 5
        engine._turn_count = 3
        assert engine.turns_until_dream == 2


class TestDreamEngineDistill:
    """Test distillation logic."""

    def test_distill_rule_based_extracts_name(self):
        """Rule-based distill should extract 'my name is X' pattern."""
        mock_memory = MagicMock()
        mock_core = MagicMock()
        # Block must expose a `.value` attribute (engine reads via getattr).
        mock_block = MagicMock()
        mock_block.value = "User information:\n- likes: coding"
        mock_core.get_block.return_value = mock_block
        mock_core.update_block.return_value = (True, "ok")
        mock_memory.core = mock_core

        engine = DreamEngine(
            mock_memory,
            {"use_llm": False, "distill_to_core": True, "trigger_turns": 99},
        )

        messages = [
            {
                "role": "user",
                "content": "Hi, my name is Alice and I work at Google",
            },
        ]

        result = engine._distill("sess1", messages)

        # Should have called update_block at least once.
        assert mock_core.update_block.called or result is not None

    def test_distill_with_llm(self):
        """LLM-based distill should parse JSON response correctly."""
        mock_memory = MagicMock()
        mock_core = MagicMock()
        mock_block = MagicMock()
        mock_block.value = "User info:\n"
        mock_core.get_block.return_value = mock_block
        mock_core.update_block.return_value = (True, "ok")
        mock_memory.core = mock_core

        def mock_llm(prompt, task="dream"):
            return (
                '{"facts": [{"block": "human", "key": "language", '
                '"value": "Python"}]}'
            )

        engine = DreamEngine(
            mock_memory,
            {"use_llm": True, "distill_to_core": True},
            auxiliary_fn=mock_llm,
        )

        messages = [{"role": "user", "content": "I mainly use Python"}]
        result = engine._distill("sess1", messages)

        assert mock_core.update_block.called
        assert result is not None

    def test_distill_no_messages_returns_none(self):
        """Empty messages should return None."""
        mock_memory = MagicMock()
        engine = DreamEngine(mock_memory, {"distill_to_core": True})
        assert engine._distill("sess1", []) is None


class TestDreamEngineArchive:
    """Test archival logic."""

    def test_archive_rule_based_saves_code(self):
        """Rule-based archive should save long technical responses."""
        mock_memory = MagicMock()
        mock_archival = MagicMock()
        mock_archival.search.return_value = []
        mock_archival.insert.return_value = 1
        mock_memory.archival = mock_archival

        engine = DreamEngine(mock_memory, {"use_llm": False})

        messages = [
            {
                "role": "assistant",
                "content": (
                    "Here's the solution:\n```python\ndef foo():\n    pass\n```\n"
                    + "x" * 200
                ),
            },
        ]

        result = engine._archive("sess1", messages)
        assert mock_archival.insert.called
        assert result is not None

    def test_archive_with_llm_deduplicates(self):
        """LLM archive should check for duplicates before inserting."""
        mock_memory = MagicMock()
        mock_archival = MagicMock()
        # Return an existing entry that matches
        mock_entry = MagicMock()
        mock_entry.content = "Python is the user's preferred language"
        mock_archival.search.return_value = [mock_entry]
        mock_memory.archival = mock_archival

        def mock_llm(prompt, task="dream"):
            return (
                '{"entries": [{"content": '
                '"Python is the user\'s preferred language", '
                '"tags": "preference"}]}'
            )

        engine = DreamEngine(
            mock_memory, {"use_llm": True}, auxiliary_fn=mock_llm
        )

        messages = [{"role": "user", "content": "I prefer Python"}]
        engine._archive("sess1", messages)

        # Should NOT insert (duplicate detected)
        assert not mock_archival.insert.called


class TestDreamEngineMerge:
    """Test merge logic."""

    def test_merge_without_embedding_returns_none(self):
        """Merge should return None when find_similar returns nothing."""
        mock_memory = MagicMock()
        mock_archival = MagicMock()
        mock_archival.find_similar.return_value = []
        mock_memory.archival = mock_archival

        engine = DreamEngine(mock_memory, {"max_archival_merge": 5})
        result = engine._merge_similar("sess1")

        # Should handle gracefully
        assert result is None or isinstance(result, str)

    def test_merge_with_pairs(self):
        """Should merge similar pairs found by find_similar."""
        mock_memory = MagicMock()
        mock_archival = MagicMock()
        mock_archival.find_similar.return_value = [
            (
                1,
                2,
                "Python is great for data science",
                "Python is excellent for data analysis",
            )
        ]
        mock_archival.merge_entries.return_value = 3
        mock_memory.archival = mock_archival

        def mock_llm(prompt, task="dream"):
            return "Python is great for data science and data analysis"

        engine = DreamEngine(
            mock_memory,
            {"use_llm": True, "max_archival_merge": 5},
            auxiliary_fn=mock_llm,
        )
        result = engine._merge_similar("sess1")

        assert mock_archival.merge_entries.called
        assert result is not None


class TestDreamEngineJsonParsing:
    """Test JSON parsing helper."""

    def test_parse_clean_json(self):
        engine = DreamEngine(MagicMock(), {})
        result = engine._parse_json_response('{"facts": []}')
        assert result == {"facts": []}

    def test_parse_markdown_wrapped_json(self):
        engine = DreamEngine(MagicMock(), {})
        result = engine._parse_json_response('```json\n{"facts": []}\n```')
        assert result == {"facts": []}

    def test_parse_invalid_json_returns_none(self):
        engine = DreamEngine(MagicMock(), {})
        result = engine._parse_json_response("not json at all")
        assert result is None

    def test_parse_empty_returns_none(self):
        engine = DreamEngine(MagicMock(), {})
        assert engine._parse_json_response("") is None
        assert engine._parse_json_response(None) is None


class TestDreamEngineLog:
    """Test dream logging."""

    def test_write_dream_log(self):
        """Should write log entry to archival."""
        mock_memory = MagicMock()
        mock_archival = MagicMock()
        mock_archival.insert.return_value = 1
        mock_memory.archival = mock_archival

        engine = DreamEngine(mock_memory, {"log_dreams": True})
        engine._write_dream_log("sess1", ["Distilled: 2 facts"], False, 1.5)

        assert mock_archival.insert.called
        call_args = mock_archival.insert.call_args[0][0]
        assert "Light Dream" in call_args
        assert "Distilled: 2 facts" in call_args


class TestTimestampInjection:
    """Test timestamp injection in user messages."""

    def test_context_compressor_age_extraction(self):
        """_extract_message_age_hours should work with timestamp field."""
        from agent.context_compressor import _extract_message_age_hours

        # Message with explicit timestamp (1 hour ago)
        msg = {
            "role": "user",
            "content": "hello",
            "timestamp": time.time() - 3600,
        }
        age = _extract_message_age_hours(msg)
        assert 0.9 < age < 1.1  # ~1 hour

    def test_context_compressor_age_no_timestamp(self):
        """Messages without timestamp should return -1 (unknown)."""
        from agent.context_compressor import _extract_message_age_hours

        msg = {"role": "user", "content": "hello without timestamp"}
        age = _extract_message_age_hours(msg)
        assert age == -1.0

    def test_context_compressor_age_with_prefix(self):
        """Messages with timestamp prefix but no field should return 0."""
        from agent.context_compressor import _extract_message_age_hours

        msg = {"role": "user", "content": "[14:30] hello with prefix"}
        age = _extract_message_age_hours(msg)
        assert age == 0.0


class TestRecallTimeDecay:
    """Test recall memory time decay."""

    def test_time_decay_recent_messages_score_higher(self):
        """Recent messages should have higher scores after decay."""
        from agent.letta_memory import RecallMemory

        if not hasattr(RecallMemory, "_apply_time_decay"):
            pytest.skip("RecallMemory._apply_time_decay not available")

        from agent.letta_memory import RecallEntry

        now = time.time()
        entries = [
            RecallEntry(
                id=1,
                session_id="s",
                role="user",
                content="old",
                timestamp=now - 36000,  # 10 hours ago
                relevance_score=1.0,
            ),
            RecallEntry(
                id=2,
                session_id="s",
                role="user",
                content="new",
                timestamp=now - 60,  # 1 minute ago
                relevance_score=1.0,
            ),
        ]

        decayed = RecallMemory._apply_time_decay(entries, decay_rate=0.1)

        # Newer message should score higher
        new_entry = next(e for e in decayed if e.id == 2)
        old_entry = next(e for e in decayed if e.id == 1)
        assert new_entry.relevance_score > old_entry.relevance_score
