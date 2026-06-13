"""Tests for Companion Intelligence Layer: emotion detection, interest graph, memory recall, adaptive tone."""
import os
import random
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEmotionDetector:
    """Tests for LLM-based emotion detection."""

    def test_detect_emotion_prompt_builds_correctly(self):
        from agent.emotion_detector import detect_emotion_prompt
        prompt = detect_emotion_prompt("我今天好开心啊")
        assert "我今天好开心啊" in prompt
        assert "happy" in prompt  # labels in prompt
        assert "neutral" in prompt

    def test_detect_emotion_prompt_truncates_long_message(self):
        from agent.emotion_detector import detect_emotion_prompt
        long_msg = "a" * 500
        prompt = detect_emotion_prompt(long_msg)
        # Message should be truncated to 200 chars
        assert "a" * 200 in prompt
        assert "a" * 201 not in prompt

    def test_parse_emotion_response_valid_labels(self):
        from agent.emotion_detector import parse_emotion_response
        assert parse_emotion_response("happy")[0] == "happy"
        assert parse_emotion_response("sad")[0] == "sad"
        assert parse_emotion_response("anxious")[0] == "anxious"
        assert parse_emotion_response("angry")[0] == "angry"
        assert parse_emotion_response("tired")[0] == "tired"
        assert parse_emotion_response("neutral")[0] == "neutral"

    def test_parse_emotion_response_confidence(self):
        from agent.emotion_detector import parse_emotion_response
        _, conf = parse_emotion_response("happy")
        assert conf == 0.8
        _, conf = parse_emotion_response("neutral")
        assert conf == 0.0

    def test_parse_emotion_response_with_punctuation(self):
        from agent.emotion_detector import parse_emotion_response
        assert parse_emotion_response("happy.")[0] == "happy"
        assert parse_emotion_response("sad。")[0] == "sad"
        assert parse_emotion_response("tired!")[0] == "tired"

    def test_parse_emotion_response_empty(self):
        from agent.emotion_detector import parse_emotion_response
        assert parse_emotion_response("")[0] == "neutral"
        assert parse_emotion_response(None)[0] == "neutral"

    def test_parse_emotion_response_fuzzy_match(self):
        from agent.emotion_detector import parse_emotion_response
        # "happy_state" contains "happy" as a substring
        label, conf = parse_emotion_response("happy_state")
        assert label == "happy"
        assert conf == 0.6

    def test_parse_emotion_response_invalid(self):
        from agent.emotion_detector import parse_emotion_response
        assert parse_emotion_response("xyz_unknown")[0] == "neutral"

    def test_tone_guidance_keys(self):
        from agent.emotion_detector import TONE_GUIDANCE, VALID_EMOTIONS
        for emo in VALID_EMOTIONS:
            assert emo in TONE_GUIDANCE


class TestToneGuidance:
    """Test TONE_GUIDANCE mapping."""

    def test_all_emotions_have_guidance(self):
        from agent.emotion_detector import TONE_GUIDANCE
        assert "happy" in TONE_GUIDANCE
        assert "sad" in TONE_GUIDANCE
        assert "anxious" in TONE_GUIDANCE
        assert "angry" in TONE_GUIDANCE
        assert "tired" in TONE_GUIDANCE
        assert "neutral" in TONE_GUIDANCE

    def test_neutral_is_empty(self):
        from agent.emotion_detector import TONE_GUIDANCE
        assert TONE_GUIDANCE["neutral"] == ""

    def test_non_neutral_have_content(self):
        from agent.emotion_detector import TONE_GUIDANCE
        for key in ["happy", "sad", "anxious", "angry", "tired"]:
            assert len(TONE_GUIDANCE[key]) > 5


class TestInterestGraph:
    """Test interest graph building and formatting."""

    def test_build_profile_from_entities(self):
        from agent.interest_graph import build_interest_profile
        mock_db = MagicMock()
        mock_db.search_archival_by_metadata_key.return_value = [
            {"id": "1", "content": "test", "metadata_json": '{"entities": {"tech": ["Python", "Docker"], "person": ["Alice"]}}'},
            {"id": "2", "content": "test2", "metadata_json": '{"entities": {"tech": ["Python", "Linux"]}}'},
        ]
        profile = build_interest_profile(mock_db)
        assert "tech" in profile
        assert "Python" in profile["tech"]
        assert profile["tech"][0] == "Python"  # Most frequent first

    def test_build_profile_empty_db(self):
        from agent.interest_graph import build_interest_profile
        mock_db = MagicMock()
        mock_db.search_archival_by_metadata_key.return_value = []
        profile = build_interest_profile(mock_db)
        assert profile == {}

    def test_build_profile_invalid_json(self):
        from agent.interest_graph import build_interest_profile
        mock_db = MagicMock()
        mock_db.search_archival_by_metadata_key.return_value = [
            {"id": "1", "content": "x", "metadata_json": "not json"},
        ]
        profile = build_interest_profile(mock_db)
        assert profile == {}

    def test_format_interests_block(self):
        from agent.interest_graph import format_interests_block
        profile = {"tech": ["Python", "Docker", "Linux"], "person": ["Alice", "Bob"]}
        text = format_interests_block(profile)
        assert "tech:" in text
        assert "Python" in text
        assert "person:" in text

    def test_format_interests_block_max_chars(self):
        from agent.interest_graph import format_interests_block
        profile = {"tech": [f"item_{i}" for i in range(50)]}
        text = format_interests_block(profile, max_chars=100)
        assert len(text) <= 100

    def test_parse_interests_text(self):
        from agent.interest_graph import parse_interests_text
        text = "tech: Python, Docker, Linux\nperson: Alice, Bob"
        result = parse_interests_text(text)
        assert result["tech"] == ["Python", "Docker", "Linux"]
        assert result["person"] == ["Alice", "Bob"]

    def test_parse_interests_text_empty(self):
        from agent.interest_graph import parse_interests_text
        assert parse_interests_text("") == {}
        assert parse_interests_text(None) == {}

    def test_refresh_interest_graph_exception_safe(self):
        from agent.interest_graph import refresh_interest_graph
        mock_memory = MagicMock()
        mock_memory._db.search_archival_by_metadata_key.side_effect = Exception("DB error")
        # Should not raise
        refresh_interest_graph(mock_memory)


class TestMemoryRecall:
    """Tests for topic-driven memory recall."""

    def test_check_topic_recall_cooldown(self):
        """Should return empty when cooldown not elapsed."""
        import agent.memory_recall as mr
        # Force last recall to now
        mr._last_recall_time = time.time()
        result = mr.check_topic_recall("test message", None, None)
        assert result == ""

    def test_check_topic_recall_no_entities(self):
        """Should return empty when no entities found in message."""
        import agent.memory_recall as mr
        mr._last_recall_time = 0.0  # Reset cooldown
        # Very short message unlikely to have entities
        result = mr.check_topic_recall("hi", None, None)
        assert result == ""

    def test_format_topic_hint_empty(self):
        from agent.memory_recall import _format_topic_hint
        assert _format_topic_hint("test", []) == ""
        assert _format_topic_hint("test", [{"content": ""}]) == ""

    def test_format_topic_hint_with_content(self):
        from agent.memory_recall import _format_topic_hint
        memories = [{"content": "我们讨论了Python项目"}]
        hint = _format_topic_hint("Python", memories)
        assert "Python" in hint
        assert "我们讨论了Python项目" in hint
        assert "共同记忆提示" in hint

    def test_format_topic_hint_truncates_long_content(self):
        from agent.memory_recall import _format_topic_hint
        long_content = "x" * 200
        memories = [{"content": long_content}]
        hint = _format_topic_hint("topic", memories)
        assert "..." in hint
        assert len(hint) < 200  # Truncated

    def test_check_topic_recall_with_mock_db(self):
        """Full flow with mocked dependencies."""
        import agent.memory_recall as mr
        mr._last_recall_time = 0.0  # Reset cooldown

        class MockDB:
            def search_archival_by_entity(self, etype, value, limit=3):
                if value == "Python":
                    return [{"content": "一起学了Python基础"}]
                return []

        # Mock _extract_entities on the real ArchivalMemory class so the
        # local `from agent.letta_memory import ArchivalMemory` inside
        # check_topic_recall picks up the patched staticmethod.
        import unittest.mock as mock
        with mock.patch(
            "agent.letta_memory.ArchivalMemory._extract_entities",
            return_value={"topic": ["Python"]},
        ):
            result = mr.check_topic_recall("我在学Python", None, MockDB())
            assert "Python" in result
            assert "共同记忆提示" in result


# Path to the active-message skill — sys.path tweak required because the
# directory contains a hyphen and is not a regular Python package.
_ACTIVE_MESSAGE_DIR = (
    Path(__file__).resolve().parent.parent / "optional-skills" / "active-message"
)


class TestActiveMessageEnhancement:
    """Test active-message build_context helpers."""

    def _import_build_context(self):
        sys.path.insert(0, str(_ACTIVE_MESSAGE_DIR))
        # Drop stale module cache so a fresh import picks up the patched env.
        for mod_name in ("build_context", "active_message_lib"):
            sys.modules.pop(mod_name, None)
        try:
            import build_context  # type: ignore
            return build_context
        finally:
            try:
                sys.path.remove(str(_ACTIVE_MESSAGE_DIR))
            except ValueError:
                pass

    def test_load_interests_from_core_no_db(self, tmp_path, monkeypatch):
        """Should return 'none' gracefully when DB doesn't exist."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "no-such-home"))
        try:
            build_context = self._import_build_context()
        except ImportError:
            pytest.skip("build_context not importable from test context")
            return
        assert build_context._load_interests_from_core() == "none"

    def test_load_mood_from_core_no_db(self, tmp_path, monkeypatch):
        """Should return 'neutral' gracefully when DB doesn't exist."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "no-such-home"))
        try:
            build_context = self._import_build_context()
        except ImportError:
            pytest.skip("build_context not importable from test context")
            return
        assert build_context._load_mood_from_core() == "neutral"
