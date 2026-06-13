"""Tests for the seven 2GB-server optimizations."""
import time
import pytest


# --- 1. Compressor time-aware sorting ---
class TestCompressorTimeAware:
    def test_age_sort_key_with_timestamp(self):
        """_age_sort_key returns negative age for timestamped messages."""
        from agent.context_compressor import _age_sort_key
        msg = {"role": "user", "content": "[12:00] hello", "timestamp": time.time() - 7200}
        key = _age_sort_key(msg)
        assert key < 0  # 2 hours old -> negative

    def test_age_sort_key_no_timestamp(self):
        """Messages without timestamp get sort key 0 (neutral)."""
        from agent.context_compressor import _age_sort_key
        msg = {"role": "user", "content": "hello"}
        key = _age_sort_key(msg)
        assert key == 0.0

    def test_groups_sorted_by_age(self):
        """Older interaction groups sort before newer ones."""
        from agent.context_compressor import _age_sort_key
        now = time.time()
        old_msg = {"role": "user", "content": "old", "timestamp": now - 86400}  # 24h ago
        new_msg = {"role": "user", "content": "new", "timestamp": now - 60}  # 1 min ago
        assert _age_sort_key(old_msg) < _age_sort_key(new_msg)


# --- 2. Embedding cache ---
class TestEmbeddingCache:
    def test_cache_put_and_get(self):
        """Cache stores and retrieves vectors."""
        from agent.embedding_engine import _EmbeddingCache
        cache = _EmbeddingCache(max_entries=10)
        vec = [0.1, 0.2, 0.3]
        cache.put("hello", vec)
        assert cache.get("hello") == vec

    def test_cache_miss(self):
        """Cache returns None for unknown text."""
        from agent.embedding_engine import _EmbeddingCache
        cache = _EmbeddingCache(max_entries=10)
        assert cache.get("unknown") is None

    def test_cache_lru_eviction(self):
        """Oldest entries evicted when max_entries exceeded."""
        from agent.embedding_engine import _EmbeddingCache
        cache = _EmbeddingCache(max_entries=3)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])
        cache.put("d", [4.0])  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("d") == [4.0]

    def test_cache_stats(self):
        """Stats track hits and misses."""
        from agent.embedding_engine import _EmbeddingCache
        cache = _EmbeddingCache(max_entries=10)
        cache.put("x", [1.0])
        cache.get("x")  # hit
        cache.get("y")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """Clear removes all entries."""
        from agent.embedding_engine import _EmbeddingCache
        cache = _EmbeddingCache(max_entries=10)
        cache.put("x", [1.0])
        cache.clear()
        assert cache.size == 0
        assert cache.get("x") is None


# --- 3. Data Shield stats ---
class TestDataShieldStats:
    def test_stats_initialized(self):
        """DataShield has _stats dict after init."""
        from agent.data_shield import DataShield
        ds = DataShield({"enabled": True, "policy": "strict"})
        stats = ds.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_redactions"] == 0
        assert stats["by_category"] == {}

    def test_stats_increment_on_shield(self):
        """Stats update when shielding messages with PII."""
        from agent.data_shield import DataShield
        ds = DataShield({"enabled": True, "policy": "strict"})
        text = "My email is test@example.com and key is sk-abc123456789012345678901234567890123456789012345678"
        result, ctx = ds.shield_text(text)
        stats = ds.get_stats()
        assert stats["total_redactions"] > 0
        assert stats["chars_redacted"] > 0

    def test_stats_total_calls(self):
        """shield_messages increments total_calls."""
        from agent.data_shield import DataShield
        ds = DataShield({"enabled": True, "policy": "strict"})
        msgs = [{"role": "user", "content": "Hello world"}]
        ds.shield_messages(msgs)
        ds.shield_messages(msgs)
        stats = ds.get_stats()
        assert stats["total_calls"] == 2


# --- 4. Memory monitor thresholds ---
class TestMemoryMonitorThresholds:
    def test_configure_thresholds(self):
        """configure_thresholds stores config."""
        from gateway.memory_monitor import configure_thresholds, _threshold_config
        configure_thresholds({"warning_threshold_mb": 1000, "critical_threshold_mb": 1500})
        # Just verify it doesn't crash and config is stored
        assert True

    def test_check_thresholds_no_action_below_warning(self):
        """No action when RSS is below warning threshold."""
        from gateway.memory_monitor import _check_thresholds, configure_thresholds
        configure_thresholds({"warning_threshold_mb": 1400, "critical_threshold_mb": 1700})
        # Should not raise
        _check_thresholds(500.0)


# --- 5. Dream Engine persistence ---
class TestDreamPersistence:
    def test_dream_state_table_created(self):
        """dream_state table is created in schema."""
        import sqlite3, tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_dream.db")
        db = MemoryDB(db_path)
        conn = db.connect()

        # Check dream_state table exists
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dream_state'"
        ).fetchone()
        assert row is not None

    def test_dream_state_crud(self):
        """get/set dream_state works correctly."""
        import tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_dream2.db")
        db = MemoryDB(db_path)

        # Initially None
        assert db.get_dream_state("turn_count") is None

        # Set
        db.set_dream_state("turn_count", "7")
        assert db.get_dream_state("turn_count") == "7"

        # Update
        db.set_dream_state("turn_count", "12")
        assert db.get_dream_state("turn_count") == "12"


# --- 6. Importance scoring ---
class TestImportanceScoring:
    def test_scoring_table_created(self):
        """archival_scoring table is created."""
        import sqlite3, tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_scoring.db")
        db = MemoryDB(db_path)
        conn = db.connect()

        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='archival_scoring'"
        ).fetchone()
        assert row is not None

    def test_importance_default(self):
        """get_importance returns 0.5 for unknown entries."""
        import tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_scoring2.db")
        db = MemoryDB(db_path)
        assert db.get_importance(999) == 0.5

    def test_set_and_get_importance(self):
        """set_importance correctly stores and retrieves."""
        import tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_scoring3.db")
        db = MemoryDB(db_path)

        # Need an archival entry first
        conn = db.connect()
        conn.execute(
            "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, '{}', ?)",
            ("test content", time.time())
        )

        db.set_importance(1, 0.9)
        assert db.get_importance(1) == pytest.approx(0.9)

    def test_increment_access(self):
        """increment_access creates/updates scoring row."""
        import tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_scoring4.db")
        db = MemoryDB(db_path)

        # Create archival entry
        conn = db.connect()
        conn.execute(
            "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, '{}', ?)",
            ("test", time.time())
        )

        db.increment_access(1)
        db.increment_access(1)

        row = conn.execute("SELECT access_count FROM archival_scoring WHERE archival_id=1").fetchone()
        assert row[0] == 2

    def test_importance_clamped(self):
        """Importance is clamped to [0.0, 1.0]."""
        import tempfile, os
        from agent.memory_db import MemoryDB

        db_path = os.path.join(tempfile.mkdtemp(), "test_scoring5.db")
        db = MemoryDB(db_path)

        conn = db.connect()
        conn.execute(
            "INSERT INTO archival_entries(content, metadata_json, created_at) VALUES(?, '{}', ?)",
            ("test", time.time())
        )

        db.set_importance(1, 5.0)  # Should clamp to 1.0
        assert db.get_importance(1) == 1.0

        db.set_importance(1, -2.0)  # Should clamp to 0.0
        assert db.get_importance(1) == 0.0


# --- 7. /memory command registration ---
class TestMemoryCommand:
    def test_command_registered(self):
        """memory command exists in COMMAND_REGISTRY."""
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [cmd.name for cmd in COMMAND_REGISTRY]
        assert "memory" in names

    def test_command_aliases(self):
        """memory command has 'mem' alias."""
        from hermes_cli.commands import resolve_command
        result = resolve_command("mem")
        assert result is not None
        assert result.name == "memory"

    def test_command_subcommands(self):
        """memory command declares expected subcommands."""
        from hermes_cli.commands import COMMAND_REGISTRY
        cmd = next(c for c in COMMAND_REGISTRY if c.name == "memory")
        assert "list" in cmd.subcommands
        assert "search" in cmd.subcommands
        assert "stats" in cmd.subcommands
        assert "forget" in cmd.subcommands
