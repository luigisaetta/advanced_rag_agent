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


def test_save_and_load_cache_file(monkeypatch, tmp_path: Path):
    """Test test save and load cache file."""
    cache = cache_module.BM25Cache()
    key = cache._key("COLL01", "TEXT", 40)  # pylint: disable=protected-access
    cache._entries[key] = (
        cache_module._BM25CacheEntry(  # pylint: disable=protected-access
            engine=_FakeEngine(),
            created_at=123.0,
            ttl_seconds=0,
        )
    )

    out_file = tmp_path / "bm25.pkl"
    cache.save_to_file(out_file)
    assert out_file.exists()

    monkeypatch.setattr(
        cache_module.BM25OracleSearch,
        "from_serialized_payload",
        staticmethod(
            lambda payload: _FakeEngine(
                payload["table_name"], payload["text_column"], payload["batch_size"]
            )
        ),
    )

    loaded_cache = cache_module.BM25Cache()
    loaded = loaded_cache.load_from_file(out_file)

    assert loaded == 1
    assert loaded_cache.stats()["size"] == 1


def test_ensure_registered_collections_cached_creates_file(monkeypatch, tmp_path: Path):
    """Test test ensure registered collections cached creates file."""
    cache = cache_module.BM25Cache()
    cache_file = tmp_path / "bm25_cache.pkl"

    monkeypatch.setattr(cache, "_cache_file_path", lambda: cache_file)

    def _fake_get_or_create(
        table_name, text_column, batch_size=40, ttl_seconds=None, force_refresh=False
    ):
        """Helper for fake get or create."""
        key = cache._key(
            table_name, text_column, batch_size
        )  # pylint: disable=protected-access
        cache._entries[key] = (
            cache_module._BM25CacheEntry(  # pylint: disable=protected-access
                engine=_FakeEngine(
                    table_name=table_name,
                    text_column=text_column,
                    batch_size=batch_size,
                ),
                created_at=1.0,
                ttl_seconds=0 if ttl_seconds is None else ttl_seconds,
            )
        )
        return cache._entries[key].engine  # pylint: disable=protected-access

    monkeypatch.setattr(cache, "get_or_create", _fake_get_or_create)

    entries_count, loaded_from_file, path = cache.ensure_registered_collections_cached(
        collections=["COLL01", "CONTRATTI"],
        text_column="TEXT",
        batch_size=40,
    )

    assert path == cache_file
    assert loaded_from_file is False
    assert entries_count == 2
    assert cache_file.exists()
