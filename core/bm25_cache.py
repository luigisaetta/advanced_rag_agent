"""
File name: core/bm25_cache.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11

License: MIT

Description:
    In-process cache manager for BM25 indexes to support stateless request handling.
"""

import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from core.bm25_search import BM25OracleSearch
from core.utils import get_console_logger
from config import COLLECTION_LIST, BM25_CACHE_DIR

logger = get_console_logger()

# Default cache TTL disabled (can be overridden with env var).
# Use 0 to keep cache entries until explicit invalidate/force_refresh.
DEFAULT_BM25_CACHE_TTL_SECONDS = int(os.getenv("BM25_CACHE_TTL_SECONDS", "0"))
BM25_CACHE_FILENAME = "bm25_cache.pkl"


@dataclass
class _BM25CacheEntry:
    engine: BM25OracleSearch
    created_at: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Is expired."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


class BM25Cache:
    """
    Thread-safe in-memory cache for BM25 engines.
    """

    def __init__(self, default_ttl_seconds: int = DEFAULT_BM25_CACHE_TTL_SECONDS):
        """Initialize the instance."""
        self.default_ttl_seconds = default_ttl_seconds
        self._entries: dict[tuple[str, str, int], _BM25CacheEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _cache_file_path() -> Path:
        """Helper for cache file path."""
        return Path(BM25_CACHE_DIR).expanduser().resolve() / BM25_CACHE_FILENAME

    @staticmethod
    def _key(
        table_name: str, text_column: str, batch_size: int
    ) -> tuple[str, str, int]:
        """Helper for key."""
        return (table_name.upper(), text_column.upper(), int(batch_size))

    def get_or_create(
        self,
        table_name: str,
        text_column: str,
        batch_size: int = 40,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> BM25OracleSearch:
        """
        Return a cached BM25 engine, creating/rebuilding it when needed.
        """
        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        key = self._key(table_name, text_column, batch_size)

        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and not force_refresh
                and not entry.is_expired()
                and entry.engine.bm25 is not None
            ):
                return entry.engine

            logger.info(
                "Building BM25 cache entry for table=%s, column=%s, batch_size=%d",
                table_name,
                text_column,
                batch_size,
            )
            engine = BM25OracleSearch(
                table_name=table_name,
                text_column=text_column,
                batch_size=batch_size,
            )

            self._entries[key] = _BM25CacheEntry(
                engine=engine, created_at=time.time(), ttl_seconds=ttl
            )
            return engine

    def search(
        self,
        query: str,
        table_name: str,
        text_column: str,
        top_n: int = 5,
        batch_size: int = 40,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Run BM25 search using a cached engine.
        """
        engine = self.get_or_create(
            table_name=table_name,
            text_column=text_column,
            batch_size=batch_size,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )
        return engine.search(query=query, top_n=top_n)

    def search_docs(
        self,
        query: str,
        table_name: str,
        text_column: str,
        top_n: int = 5,
        batch_size: int = 40,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Run BM25 search using a cached engine and return docs with metadata.
        """
        engine = self.get_or_create(
            table_name=table_name,
            text_column=text_column,
            batch_size=batch_size,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )
        return engine.search_docs(query=query, top_n=top_n)

    def invalidate(
        self, table_name: str, text_column: str, batch_size: int = 40
    ) -> bool:
        """
        Remove a single cache entry.
        """
        key = self._key(table_name, text_column, batch_size)
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        with self._lock:
            self._entries.clear()

    def save_to_file(self, file_path: Path | None = None) -> Path:
        """
        Persist current cache entries to a pickle file.
        """
        path = file_path or self._cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "entries": [
                    {
                        "table_name": table_name,
                        "text_column": text_column,
                        "batch_size": batch_size,
                        "created_at": entry.created_at,
                        "ttl_seconds": entry.ttl_seconds,
                        "engine": entry.engine.to_serialized_payload(),
                    }
                    for (
                        table_name,
                        text_column,
                        batch_size,
                    ), entry in self._entries.items()
                    if entry.engine is not None and entry.engine.bm25 is not None
                ],
            }

        with path.open("wb") as handle:
            pickle.dump(payload, handle)

        logger.info(
            "Saved BM25 cache file: %s (entries=%d)", path, len(payload["entries"])
        )
        return path

    def load_from_file(self, file_path: Path | None = None) -> int:
        """
        Load cache entries from a pickle file. Returns loaded entries count.
        """
        path = file_path or self._cache_file_path()
        if not path.exists():
            return 0

        with path.open("rb") as handle:
            payload = pickle.load(handle)

        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        loaded = 0

        with self._lock:
            self._entries.clear()
            for item in entries:
                try:
                    table_name = str(item["table_name"]).upper()
                    text_column = str(item["text_column"]).upper()
                    batch_size = int(item["batch_size"])
                    created_at = float(item.get("created_at", time.time()))
                    ttl_seconds = int(item.get("ttl_seconds", self.default_ttl_seconds))
                    engine_payload = item["engine"]
                    engine = BM25OracleSearch.from_serialized_payload(engine_payload)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "Skipping invalid BM25 cache entry from file: %s", exc
                    )
                    continue

                key = self._key(table_name, text_column, batch_size)
                self._entries[key] = _BM25CacheEntry(
                    engine=engine,
                    created_at=created_at,
                    ttl_seconds=ttl_seconds,
                )
                loaded += 1

        logger.info("Loaded BM25 cache file: %s (entries=%d)", path, loaded)
        return loaded

    def ensure_registered_collections_cached(
        self,
        collections: list[str] | None = None,
        text_column: str = "TEXT",
        batch_size: int = 40,
    ) -> tuple[int, bool, Path]:
        """
        Ensure BM25 indexes are available for all configured collections.
        If cache file exists, load it first. Missing collections are then built.
        If file does not exist, build all and save.
        Returns: (entries_count, loaded_from_file, cache_path)
        """
        path = self._cache_file_path()
        target_collections = [
            c.strip().upper() for c in (collections or COLLECTION_LIST) if c
        ]
        loaded_from_file = False

        if path.exists():
            try:
                loaded = self.load_from_file(path)
                loaded_from_file = loaded > 0
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to load BM25 cache from file %s: %s", path, exc)

        # Build any missing entries (or all entries when file is missing/empty).
        for table_name in target_collections:
            key = self._key(table_name, text_column, batch_size)
            with self._lock:
                exists = (
                    key in self._entries and self._entries[key].engine.bm25 is not None
                )
            if exists:
                continue
            try:
                self.get_or_create(
                    table_name=table_name,
                    text_column=text_column,
                    batch_size=batch_size,
                    force_refresh=True,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to build BM25 cache for %s: %s", table_name, exc)

        # Persist current state so next startup can load directly.
        try:
            self.save_to_file(path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to save BM25 cache file %s: %s", path, exc)

        with self._lock:
            entries_count = len(self._entries)
        return entries_count, loaded_from_file, path

    def stats(self) -> dict:
        """
        Return a lightweight snapshot of cache metadata.
        """
        with self._lock:
            items = []
            now = time.time()
            for (table_name, text_column, batch_size), entry in self._entries.items():
                age_seconds = int(now - entry.created_at)
                items.append(
                    {
                        "table_name": table_name,
                        "text_column": text_column,
                        "batch_size": batch_size,
                        "age_seconds": age_seconds,
                        "ttl_seconds": entry.ttl_seconds,
                        "expired": entry.is_expired(),
                        "indexed_docs": len(entry.engine.texts),
                        "is_initialized": entry.engine.bm25 is not None,
                    }
                )
            return {
                "default_ttl_seconds": self.default_ttl_seconds,
                "size": len(items),
                "entries": items,
            }


_bm25_cache_singleton: BM25Cache | None = None
_singleton_lock = threading.Lock()


def get_bm25_cache() -> BM25Cache:
    """
    Return process-wide BM25 cache singleton.
    """
    global _bm25_cache_singleton
    with _singleton_lock:
        if _bm25_cache_singleton is None:
            _bm25_cache_singleton = BM25Cache()
            entries_count, loaded_from_file, path = (
                _bm25_cache_singleton.ensure_registered_collections_cached()
            )
            logger.info(
                "BM25 cache ready. loaded_from_file=%s entries=%d path=%s",
                loaded_from_file,
                entries_count,
                path,
            )
        return _bm25_cache_singleton
