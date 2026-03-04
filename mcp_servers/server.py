"""
File name: mcp_servers/server.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: FastMCP server exposing BM25 cache tools with optional startup prewarm.
"""

import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Ensure project root is importable even when running from `mcp_servers/` directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import COLLECTION_LIST, TOP_K, ENABLE_HYBRID_SEARCH
from core.bm25_cache import get_bm25_cache
from core.utils import get_console_logger

logger = get_console_logger("bm25_mcp_server")

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8010"))
BM25_TEXT_COLUMN = os.getenv("BM25_TEXT_COLUMN", "TEXT")
BM25_BATCH_SIZE = int(os.getenv("BM25_BATCH_SIZE", "40"))
BM25_PREWARM_ENABLED = os.getenv("BM25_PREWARM_ENABLED", "true").lower() == "true"
BM25_PREWARM_COLLECTIONS = os.getenv("BM25_PREWARM_COLLECTIONS", "")

mcp = FastMCP("bm25-cache-server")


def _collections_to_prewarm() -> list[str]:
    """Helper for collections to prewarm."""
    if BM25_PREWARM_COLLECTIONS.strip():
        return [
            item.strip() for item in BM25_PREWARM_COLLECTIONS.split(",") if item.strip()
        ]
    return list(COLLECTION_LIST)


def prewarm_cache() -> dict[str, Any]:
    """
    Pre-build BM25 indexes at startup to avoid first-request latency.
    """
    cache = get_bm25_cache()
    warmed = []
    errors = []

    if not ENABLE_HYBRID_SEARCH:
        logger.info("ENABLE_HYBRID_SEARCH is false. Skipping BM25 prewarm.")
        return {
            "prewarm_enabled": False,
            "hybrid_enabled": False,
            "warmed": warmed,
            "errors": errors,
        }

    if not BM25_PREWARM_ENABLED:
        return {
            "prewarm_enabled": False,
            "hybrid_enabled": True,
            "warmed": warmed,
            "errors": errors,
        }

    for collection_name in _collections_to_prewarm():
        try:
            cache.get_or_create(
                table_name=collection_name,
                text_column=BM25_TEXT_COLUMN,
                batch_size=BM25_BATCH_SIZE,
            )
            warmed.append(collection_name)
            logger.info("Prewarmed BM25 cache for collection=%s", collection_name)
        except Exception as exc:
            errors.append({"collection_name": collection_name, "error": str(exc)})
            logger.error(
                "Failed to prewarm BM25 cache for collection=%s: %s",
                collection_name,
                exc,
            )

    return {
        "prewarm_enabled": True,
        "hybrid_enabled": True,
        "warmed": warmed,
        "errors": errors,
    }


@mcp.tool
def ping() -> dict[str, Any]:
    """
    Health/ping tool.
    """
    return {"status": "ok", "service": "bm25-cache-server"}


@mcp.tool
def bm25_search(
    query: str,
    collection_name: str,
    top_n: int = TOP_K,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Run BM25 search on the given collection.
    """
    if not ENABLE_HYBRID_SEARCH:
        return {
            "collection_name": collection_name,
            "top_n": top_n,
            "results": [],
            "error": "ENABLE_HYBRID_SEARCH is false",
        }

    cache = get_bm25_cache()
    results = cache.search(
        query=query,
        table_name=collection_name,
        text_column=BM25_TEXT_COLUMN,
        top_n=top_n,
        batch_size=BM25_BATCH_SIZE,
        force_refresh=force_refresh,
    )
    return {
        "collection_name": collection_name,
        "top_n": top_n,
        "results": [{"text": text, "score": float(score)} for text, score in results],
    }


@mcp.tool
def bm25_refresh(collection_name: str) -> dict[str, Any]:
    """
    Force cache rebuild for one collection.
    """
    if not ENABLE_HYBRID_SEARCH:
        return {"collection_name": collection_name, "refreshed": False}

    cache = get_bm25_cache()
    cache.get_or_create(
        table_name=collection_name,
        text_column=BM25_TEXT_COLUMN,
        batch_size=BM25_BATCH_SIZE,
        force_refresh=True,
    )
    return {"collection_name": collection_name, "refreshed": True}


@mcp.tool
def bm25_cache_stats() -> dict[str, Any]:
    """
    Return in-memory cache stats.
    """
    return get_bm25_cache().stats()


def run_server() -> None:
    """
    Start FastMCP server over HTTP transport.
    """
    startup = prewarm_cache()
    logger.info("BM25 MCP startup: %s", startup)
    mcp.run(transport="http", host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    run_server()
