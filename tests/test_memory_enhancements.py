"""Tests for memory system enhancements: entity tagging, pattern learning, idle consolidation."""
import time
import json
import tempfile
import os
import pytest


# --- 1. Entity extraction ---
class TestEntityExtraction:
    def test_extract_tech_entities(self):
        """Extracts technology names from content."""
        from agent.letta_memory import ArchivalMemory
        entities = ArchivalMemory._extract_entities("I use Python and Docker for deployment")
        assert "tech" in entities
        assert "Python" in entities["tech"] or "python" in [e.lower() for e in entities["tech"]]

    def test_extract_person_entities(self):
        """Extracts capitalized person names."""
        from agent.letta_memory import ArchivalMemory
        entities = ArchivalMemory._extract_entities("John Smith met Alice Johnson yesterday")
        assert "person" in entities
        assert any("John Smith" in p for p in entities["person"])

    def test_extract_url_entities(self):
        """Extracts URLs from content."""
        from agent.letta_memory import ArchivalMemory
        entities = ArchivalMemory._extract_entities("Visit https://example.com/page for info")
        assert "url" in entities
        assert any("example.com" in u for u in entities["url"])

    def test_extract_date_entities(self):
        """Extracts date patterns."""
        from agent.letta_memory import ArchivalMemory
        entities = ArchivalMemory._extract_entities("The deadline is 2026-05-27")
        assert "date" in entities
        assert "2026-05-27" in entities["date"]

    def test_extract_empty_content(self):
        """Empty content returns empty dict."""
        from agent.letta_memory import ArchivalMemory
        assert ArchivalMemory._extract_entities("") == {}
        assert ArchivalMemory._extract_entities("no entities here at all") == {}

    def test_entities_capped_at_5(self):
        """Each entity type is capped at 5 matches."""
        from agent.letta_memory import ArchivalMemory
        text = "Python Java JavaScript TypeScript Go Rust React Vue Angular Django Flask FastAPI"
        entities = ArchivalMemory._extract_entities(text)
        assert "tech" in entities
        assert len(entities["tech"]) <= 5


# --- 2. Entity search in DB ---
class TestEntitySearch:
    def test_search_by_entity(self):
        """search_archival_by_entity finds entries with matching entity."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_entity.db")
        db = MemoryDB(db_path)

        # Insert entries with metadata containing entities
        meta1 = json.dumps({"entities": {"tech": ["Python", "Docker"]}})
        meta2 = json.dumps({"entities": {"tech": ["Java", "Spring"]}})

        conn = db.connect()
        conn.execute(
            "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, ?, ?)",
            ("Python deployment guide", meta1, time.time())
        )
        conn.execute(
            "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, ?, ?)",
            ("Java microservices", meta2, time.time())
        )
        conn.commit()

        results = db.search_archival_by_entity("tech", "Python")
        assert len(results) == 1
        assert "Python" in results[0]["content"]

    def test_search_no_match(self):
        """Returns empty list when no entries match."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_entity2.db")
        db = MemoryDB(db_path)

        results = db.search_archival_by_entity("tech", "Haskell")
        assert results == []


# --- 3. Distill results table ---
class TestDistillResults:
    def test_table_created(self):
        """dream_distill_results table exists after init."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_distill.db")
        db = MemoryDB(db_path)
        conn = db.connect()

        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dream_distill_results'"
        ).fetchone()
        assert row is not None

    def test_add_and_get_results(self):
        """Can add and retrieve distill results."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_distill2.db")
        db = MemoryDB(db_path)

        db.add_distill_result(cycle=1, block="human", key="name", value="Alice", status="applied")
        db.add_distill_result(cycle=2, block="human", key="hobby", value="coding", status="applied")

        results = db.get_distill_results(since_cycle=0)
        assert len(results) == 2
        assert results[0]["key"] in ("name", "hobby")

    def test_recurring_keys(self):
        """get_recurring_keys returns keys appearing >= min_count times."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_distill3.db")
        db = MemoryDB(db_path)

        # Add "name" 3 times, "hobby" 1 time
        for i in range(3):
            db.add_distill_result(cycle=i, block="human", key="name", value="Alice", status="applied")
        db.add_distill_result(cycle=4, block="human", key="hobby", value="coding", status="applied")

        recurring = db.get_recurring_keys(min_count=3)
        assert len(recurring) == 1
        assert recurring[0]["key"] == "name"
        assert recurring[0]["count"] == 3

    def test_recurring_keys_empty_below_threshold(self):
        """Returns empty when no keys meet min_count."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_distill4.db")
        db = MemoryDB(db_path)

        db.add_distill_result(cycle=1, block="human", key="name", value="Bob", status="applied")

        recurring = db.get_recurring_keys(min_count=3)
        assert recurring == []


# --- 4. Random archival sampling ---
class TestRandomSampling:
    def test_random_sample(self):
        """get_random_archival_entries returns requested count."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_random.db")
        db = MemoryDB(db_path)

        conn = db.connect()
        for i in range(20):
            conn.execute(
                "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, '{}', ?)",
                (f"entry {i}", time.time())
            )
        conn.commit()

        results = db.get_random_archival_entries(count=5)
        assert len(results) == 5
        # All should have id and content
        for r in results:
            assert "id" in r
            assert "content" in r

    def test_random_exclude_ids(self):
        """Excluded IDs are not returned."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_random2.db")
        db = MemoryDB(db_path)

        conn = db.connect()
        for i in range(10):
            conn.execute(
                "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, '{}', ?)",
                (f"entry {i}", time.time())
            )
        conn.commit()

        results = db.get_random_archival_entries(count=10, exclude_ids=[1, 2, 3])
        returned_ids = [r["id"] for r in results]
        assert 1 not in returned_ids
        assert 2 not in returned_ids
        assert 3 not in returned_ids

    def test_random_empty_table(self):
        """Returns empty list from empty table."""
        from agent.memory_db import MemoryDB
        db_path = os.path.join(tempfile.mkdtemp(), "test_random3.db")
        db = MemoryDB(db_path)

        results = db.get_random_archival_entries(count=5)
        assert results == []


# --- 5. DreamEngine idle config ---
class TestIdleConfig:
    def test_idle_config_defaults(self):
        """DreamEngine reads idle config with defaults."""
        from unittest.mock import MagicMock
        from agent.dream_engine import DreamEngine

        mock_memory = MagicMock()
        mock_memory._db = MagicMock()
        mock_memory._db.get_dream_state = MagicMock(return_value=None)

        engine = DreamEngine(
            memory_system=mock_memory,
            config={},
            auxiliary_fn=None,
        )

        assert engine._idle_enabled == True
        assert engine._idle_threshold_seconds == 30 * 60  # 30 min default
        assert engine._idle_sample_size == 10
        assert engine._idle_merge_threshold == 0.85

    def test_idle_config_custom(self):
        """DreamEngine respects custom idle config."""
        from unittest.mock import MagicMock
        from agent.dream_engine import DreamEngine

        mock_memory = MagicMock()
        mock_memory._db = MagicMock()
        mock_memory._db.get_dream_state = MagicMock(return_value=None)

        engine = DreamEngine(
            memory_system=mock_memory,
            config={
                "idle_consolidation": False,
                "idle_threshold_minutes": 60,
                "idle_sample_size": 20,
                "idle_merge_threshold": 0.90,
            },
            auxiliary_fn=None,
        )

        assert engine._idle_enabled == False
        assert engine._idle_threshold_seconds == 60 * 60
        assert engine._idle_sample_size == 20
        assert engine._idle_merge_threshold == 0.90


# --- 6. Pattern learning config ---
class TestPatternConfig:
    def test_pattern_config_defaults(self):
        """DreamEngine reads pattern learning config with defaults."""
        from unittest.mock import MagicMock
        from agent.dream_engine import DreamEngine

        mock_memory = MagicMock()
        mock_memory._db = MagicMock()
        mock_memory._db.get_dream_state = MagicMock(return_value=None)

        engine = DreamEngine(
            memory_system=mock_memory,
            config={},
            auxiliary_fn=None,
        )

        assert engine._pattern_learning_interval == 5
        assert engine._pattern_min_frequency == 3
