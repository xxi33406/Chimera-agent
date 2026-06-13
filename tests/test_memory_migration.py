"""Tests for agent.memory_migration — Legacy file migration."""

from unittest.mock import patch

from agent.letta_memory import LettaMemorySystem
from agent.memory_migration import (
    ENTRY_DELIMITER,
    _parse_entries,
    migrate_to_letta,
    needs_migration,
)


class TestNeedsMigration:
    def test_no_files_no_migration(self, tmp_path):
        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            assert needs_migration() is False

    def test_memory_file_exists_needs_migration(self, tmp_path):
        memories_dir = tmp_path / "memories"
        memories_dir.mkdir()
        (memories_dir / "MEMORY.md").write_text("entry1", encoding="utf-8")
        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            assert needs_migration() is True

    def test_already_migrated_no_migration(self, tmp_path):
        memories_dir = tmp_path / "memories"
        memories_dir.mkdir()
        (memories_dir / "MEMORY.md").write_text("entry1", encoding="utf-8")
        (memories_dir / "MEMORY.md.migrated").write_text("", encoding="utf-8")
        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            assert needs_migration() is False


class TestParseEntries:
    def test_single_entry(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("single entry", encoding="utf-8")
        entries = _parse_entries(f)
        assert entries == ["single entry"]

    def test_multiple_entries(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text(
            f"entry1{ENTRY_DELIMITER}entry2{ENTRY_DELIMITER}entry3",
            encoding="utf-8",
        )
        entries = _parse_entries(f)
        assert entries == ["entry1", "entry2", "entry3"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("", encoding="utf-8")
        entries = _parse_entries(f)
        assert entries == []

    def test_deduplication(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text(
            f"dup{ENTRY_DELIMITER}dup{ENTRY_DELIMITER}unique", encoding="utf-8"
        )
        entries = _parse_entries(f)
        assert entries == ["dup", "unique"]

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "missing.md"
        entries = _parse_entries(f)
        assert entries == []


class TestMigrateToLetta:
    def test_migration_basic(self, tmp_path):
        # Setup legacy files
        memories_dir = tmp_path / "memories"
        memories_dir.mkdir()
        (memories_dir / "MEMORY.md").write_text(
            "agent note 1\n§\nagent note 2", encoding="utf-8"
        )
        (memories_dir / "USER.md").write_text("user likes Python", encoding="utf-8")

        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            system = LettaMemorySystem(config={}, db_path=tmp_path / "memory.db")
            system.initialize()

            stats = migrate_to_letta(system)

            assert stats["migrated"] is True
            assert stats["persona_entries"] > 0
            assert stats["human_entries"] > 0

            # Verify data in new system
            persona = system.core.get_block("persona")
            assert "agent note" in persona.value

            human = system.core.get_block("human")
            assert "Python" in human.value

            # Verify files renamed
            assert (memories_dir / "MEMORY.md.migrated").exists()
            assert (memories_dir / "USER.md.migrated").exists()

            system.shutdown()

    def test_migration_overflow_to_archival(self, tmp_path):
        """Entries exceeding char_limit should go to archival memory."""
        memories_dir = tmp_path / "memories"
        memories_dir.mkdir()
        # Create content that exceeds default 2200 char limit
        long_entries = "\n§\n".join(
            [f"Long entry number {i} " + "x" * 200 for i in range(20)]
        )
        (memories_dir / "MEMORY.md").write_text(long_entries, encoding="utf-8")

        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            system = LettaMemorySystem(config={}, db_path=tmp_path / "memory.db")
            system.initialize()

            stats = migrate_to_letta(system)

            assert stats["migrated"] is True
            assert stats["archival_entries"] > 0  # Some overflow went to archival

            system.shutdown()

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice should not duplicate data."""
        memories_dir = tmp_path / "memories"
        memories_dir.mkdir()
        (memories_dir / "MEMORY.md").write_text("test entry", encoding="utf-8")

        with patch("agent.memory_migration.get_hermes_home", return_value=tmp_path):
            system = LettaMemorySystem(config={}, db_path=tmp_path / "memory.db")
            system.initialize()

            stats1 = migrate_to_letta(system)
            assert stats1["migrated"] is True

            # Second call should not migrate again
            stats2 = migrate_to_letta(system)
            assert stats2["migrated"] is False

            system.shutdown()
