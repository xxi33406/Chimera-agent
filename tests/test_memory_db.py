"""Tests for agent.memory_db — SQLite memory database layer."""

from agent.memory_db import MemoryDB


class TestMemoryDBSchema:
    """Test database schema creation and migrations."""

    def test_creates_db_file(self, tmp_path):
        """DB file should be created on first connect."""
        db_path = tmp_path / "test_memory.db"
        db = MemoryDB(db_path)
        db.connect()
        assert db_path.exists()
        db.close()

    def test_schema_version_set(self, tmp_path):
        """Schema version should be set after init."""
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        version = db._get_schema_version()
        assert version >= 1
        db.close()

    def test_tables_created(self, tmp_path):
        """All required tables should exist."""
        db = MemoryDB(tmp_path / "test.db")
        conn = db.connect()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "memory_blocks" in tables
        assert "recall_messages" in tables
        assert "archival_entries" in tables
        assert "memory_meta" in tables
        db.close()


class TestMemoryDBBlocks:
    """Test core memory block CRUD."""

    def test_upsert_and_get_block(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.upsert_block("persona", "test value", description="test desc", char_limit=2200)
        block = db.get_block("persona")
        assert block is not None
        assert block["label"] == "persona"
        assert block["value"] == "test value"
        assert block["description"] == "test desc"
        db.close()

    def test_upsert_updates_existing(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.upsert_block("persona", "v1")
        db.upsert_block("persona", "v2")
        block = db.get_block("persona")
        assert block["value"] == "v2"
        db.close()

    def test_list_blocks(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.upsert_block("persona", "p")
        db.upsert_block("human", "h")
        blocks = db.list_blocks()
        labels = {b["label"] for b in blocks}
        assert "persona" in labels
        assert "human" in labels
        db.close()

    def test_delete_block(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.upsert_block("temp", "val")
        assert db.delete_block("temp") is True
        assert db.get_block("temp") is None
        db.close()

    def test_get_nonexistent_block_returns_none(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        assert db.get_block("nonexistent") is None
        db.close()


class TestMemoryDBRecall:
    """Test recall memory operations."""

    def test_add_and_get_messages(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.add_recall_message("session1", "user", "hello")
        db.add_recall_message("session1", "assistant", "hi there")
        msgs = db.get_recall_messages("session1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        db.close()

    def test_search_recall_fts(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.add_recall_message("s1", "user", "I love Python programming")
        db.add_recall_message("s1", "user", "The weather is nice today")
        results = db.search_recall("Python")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]
        db.close()

    def test_search_recall_by_session(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        db.add_recall_message("s1", "user", "Python code")
        db.add_recall_message("s2", "user", "Python data")
        results = db.search_recall("Python", session_id="s1")
        assert all(r["session_id"] == "s1" for r in results)
        db.close()


class TestMemoryDBArchival:
    """Test archival memory operations."""

    def test_add_and_search_archival(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        entry_id = db.add_archival_entry(
            "SQLite is a great database", metadata={"source": "test"}
        )
        assert entry_id > 0
        results = db.search_archival_fts("SQLite database")
        assert len(results) >= 1
        db.close()

    def test_delete_archival_entry(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        entry_id = db.add_archival_entry("temp entry")
        assert db.delete_archival_entry(entry_id) is True
        results = db.search_archival_fts("temp entry")
        assert len(results) == 0
        db.close()

    def test_update_archival_entry(self, tmp_path):
        db = MemoryDB(tmp_path / "test.db")
        db.connect()
        entry_id = db.add_archival_entry("original content")
        db.update_archival_entry(entry_id, "updated content")
        results = db.search_archival_fts("updated content")
        assert len(results) >= 1
        db.close()
