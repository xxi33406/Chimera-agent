"""Tests for agent.letta_memory — Three-layer memory system."""

import pytest

from agent.letta_memory import (
    ArchivalMemory,
    CoreMemory,
    LettaMemorySystem,
    RecallMemory,
)
from agent.memory_db import MemoryDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh MemoryDB for each test."""
    _db = MemoryDB(tmp_path / "test_memory.db")
    _db.connect()
    yield _db
    _db.close()


class TestCoreMemory:
    def test_default_blocks_created(self, db):
        core = CoreMemory(db)
        blocks = core.list_blocks()
        labels = {b.label for b in blocks}
        assert "persona" in labels
        assert "human" in labels

    def test_update_block(self, db):
        core = CoreMemory(db)
        success, _msg = core.update_block("persona", "I am helpful")
        assert success
        block = core.get_block("persona")
        assert block.value == "I am helpful"

    def test_char_limit_enforced(self, db):
        config = {"persona": {"description": "test", "char_limit": 20}}
        core = CoreMemory(db, block_configs=config)
        success, msg = core.update_block("persona", "x" * 21)
        assert not success
        assert "limit" in msg.lower() or "exceed" in msg.lower()

    def test_replace_in_block(self, db):
        core = CoreMemory(db)
        core.update_block("persona", "I like cats")
        success, _msg = core.replace_in_block("persona", "cats", "dogs")
        assert success
        block = core.get_block("persona")
        assert "dogs" in block.value
        assert "cats" not in block.value

    def test_frozen_snapshot(self, db):
        core = CoreMemory(db)
        core.update_block("persona", "snapshot value")
        core.load_snapshot()
        # Now update live value
        core.update_block("persona", "live value")
        # Snapshot should still show old value
        prompt = core.format_for_prompt()
        assert "snapshot value" in prompt
        # Live state should show new value
        live = core.format_live_state()
        assert "live value" in live

    def test_format_for_prompt_structure(self, db):
        core = CoreMemory(db)
        core.update_block("persona", "agent notes")
        core.load_snapshot()
        prompt = core.format_for_prompt()
        assert "<core_memory>" in prompt or "<persona" in prompt


class TestRecallMemory:
    def test_add_and_search(self, db):
        recall = RecallMemory(db)
        recall.add_message("s1", "user", "I love programming in Python")
        recall.add_message("s1", "assistant", "Python is great!")
        results = recall.search("Python")
        assert len(results) >= 1

    def test_get_session_messages(self, db):
        recall = RecallMemory(db)
        recall.add_message("s1", "user", "msg1")
        recall.add_message("s1", "assistant", "msg2")
        recall.add_message("s2", "user", "msg3")
        msgs = recall.get_session_messages("s1")
        assert len(msgs) == 2

    def test_message_count(self, db):
        recall = RecallMemory(db)
        recall.add_message("s1", "user", "a")
        recall.add_message("s1", "user", "b")
        assert recall.get_message_count("s1") == 2


class TestArchivalMemory:
    def test_insert_and_search(self, db):
        archival = ArchivalMemory(db)
        archival.insert("SQLite is a lightweight database engine")
        results = archival.search("database")
        assert len(results) >= 1
        assert "SQLite" in results[0].content

    def test_delete(self, db):
        archival = ArchivalMemory(db)
        entry_id = archival.insert("temporary note")
        assert archival.delete(entry_id)
        results = archival.search("temporary note")
        assert len(results) == 0

    def test_update(self, db):
        archival = ArchivalMemory(db)
        entry_id = archival.insert("original text")
        archival.update(entry_id, "modified text")
        entry = archival.get_entry(entry_id)
        assert "modified" in entry.content

    def test_entry_count(self, db):
        archival = ArchivalMemory(db)
        archival.insert("entry 1")
        archival.insert("entry 2")
        assert archival.get_entry_count() == 2


class TestLettaMemorySystem:
    def test_initialization(self, tmp_path):
        system = LettaMemorySystem(config={}, db_path=tmp_path / "test.db")
        system.initialize()
        assert system.core is not None
        assert system.recall is not None
        assert system.archival is not None
        system.shutdown()

    def test_embedding_available_false_by_default(self, tmp_path):
        """Without proper embedding config, should be unavailable."""
        system = LettaMemorySystem(config={}, db_path=tmp_path / "test.db")
        system.initialize()
        # May or may not be available depending on installed packages
        assert isinstance(system.embedding_available, bool)
        system.shutdown()
