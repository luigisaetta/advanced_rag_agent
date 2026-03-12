"""
Tests for BM25 cache persistence helpers.
"""

from pathlib import Path

import core.bm25_cache as cache_module


class _FakeEngine:
    def __init__(self, table_name="COLL01", text_column="TEXT", batch_size=40):
        """Initialize the instance."""
        self.table_name = table_name
        self.text_column = text_column
        self.batch_size = batch_size
        self.docs = [
            {"page_content": "x", "metadata": {"source": "s", "page_label": "1"}}
        ]
        self.texts = ["x"]
        self.tokenized_texts = [["x"]]
        self.bm25 = object()

    def to_serialized_payload(self):
        """To serialized payload."""
        return {
            "table_name": self.table_name,
            "text_column": self.text_column,
            "batch_size": self.batch_size,
            "docs": self.docs,
            "texts": self.texts,
            "tokenized_texts": self.tokenized_texts,
        }

    @staticmethod
    def from_serialized_payload(payload):
        """Build fake engine from serialized payload."""
        return _FakeEngine(
            payload["table_name"], payload["text_column"], payload["batch_size"]
        )


def test_save_and_load_cache_file(monkeypatch, tmp_path: Path):
    """Test test save and load cache file."""
    cache = cache_module.BM25Cache()
    monkeypatch.setattr(cache_module, "BM25OracleSearch", _FakeEngine)
    cache.get_or_create(table_name="COLL01", text_column="TEXT", batch_size=40)

    out_file = tmp_path / "bm25.pkl"
    cache.save_to_file(out_file)
    assert out_file.exists()

    loaded_cache = cache_module.BM25Cache()
    loaded = loaded_cache.load_from_file(out_file)

    assert loaded == 1
    assert loaded_cache.stats()["size"] == 1


def test_ensure_registered_collections_cached_creates_file(monkeypatch, tmp_path: Path):
    """Test test ensure registered collections cached creates file."""
    cache = cache_module.BM25Cache()
    cache_file = tmp_path / "bm25_cache.pkl"

    monkeypatch.setattr(cache, "_cache_file_path", lambda: cache_file)
    monkeypatch.setattr(cache_module, "BM25OracleSearch", _FakeEngine)

    entries_count, loaded_from_file, path = cache.ensure_registered_collections_cached(
        collections=["COLL01", "CONTRATTI"],
        text_column="TEXT",
        batch_size=40,
    )

    assert path == cache_file
    assert loaded_from_file is False
    assert entries_count == 2
    assert cache_file.exists()
