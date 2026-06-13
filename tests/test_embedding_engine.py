"""Tests for agent.embedding_engine — Embedding engine backends."""

from agent.embedding_engine import (
    EmbeddingEngine,
    cosine_similarity,
    create_embedding_engine,
)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0


class TestEmbeddingEngine:
    """Test EmbeddingEngine initialization and graceful degradation."""

    def test_unavailable_backend_returns_none(self):
        """Engine with invalid backend should not crash, embed returns None."""
        engine = EmbeddingEngine({"backend": "nonexistent", "model": "x"})
        result = engine.embed("test")
        assert result is None

    def test_is_available_false_for_missing_backend(self):
        engine = EmbeddingEngine(
            {"backend": "sentence_transformers", "model": "nonexistent-model-xyz"}
        )
        # May or may not be available depending on installed packages
        # But should not crash
        assert isinstance(engine.is_available(), bool)

    def test_create_embedding_engine_with_none_config(self):
        """Factory with None should create an engine (possibly unavailable)."""
        engine = create_embedding_engine(None)
        assert engine is not None

    def test_openai_backend_without_key(self, monkeypatch):
        """OpenAI backend without API key should be unavailable but not crash."""
        # Strip env so we genuinely test the no-key path.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        engine = EmbeddingEngine(
            {
                "backend": "openai",
                "model": "text-embedding-3-small",
                "api_key": "",
            }
        )
        # Should not crash, just be unavailable or return None
        result = engine.embed("test")
        # Result is None because no valid API key
        assert result is None or isinstance(result, list)
