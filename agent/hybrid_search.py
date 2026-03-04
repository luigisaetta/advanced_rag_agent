"""
File name: hybrid_search.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11

Description:
    This module defines the hybrid retrieval step in the graph.
    It merges semantic retrieval results with BM25 results.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.hybrid_search import HybridSearch

License:
    This code is released under the MIT License.
"""

import config as app_config
from langchain_core.runnables import Runnable

from agent.agent_state import State
from core.bm25_cache import get_bm25_cache
from core.utils import get_console_logger

logger = get_console_logger()


class HybridSearch(Runnable):
    """
    Hybrid retrieval node that enriches semantic results with BM25 candidates.
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Helper for normalize text."""
        return " ".join((text or "").split()).strip().lower()

    def _merge_docs(self, semantic_docs: list, bm25_docs: list) -> list:
        """
        Merge semantic and BM25 docs, deduplicating by normalized content.
        If the same chunk comes from both retrievals, mark it as hybrid provenance.
        """
        merged = []
        index_by_content = {}

        # Add semantic docs first.
        for doc in semantic_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content:
                continue

            metadata = doc.get("metadata") or {}
            metadata.setdefault("retrieval_type", "semantic")
            doc["metadata"] = metadata

            index_by_content[content] = len(merged)
            merged.append(doc)

        # Merge BM25 docs and upgrade provenance when same chunk is already present.
        for doc in bm25_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content:
                continue

            if content in index_by_content:
                existing = merged[index_by_content[content]]
                existing_meta = existing.get("metadata") or {}
                existing_type = existing_meta.get("retrieval_type", "semantic")
                if existing_type != "semantic+bm25":
                    existing_meta["retrieval_type"] = "semantic+bm25"
                    existing["metadata"] = existing_meta
                continue

            metadata = doc.get("metadata") or {}
            metadata.setdefault("retrieval_type", "bm25")
            doc["metadata"] = metadata
            index_by_content[content] = len(merged)
            merged.append(doc)

        return merged

    def _bm25_docs(self, query: str, collection_name: str) -> list:
        """
        Retrieve BM25 candidates and convert them to the same doc shape used by the agent.
        """
        cache = get_bm25_cache()
        results = cache.search_docs(
            query=query,
            table_name=collection_name,
            text_column="TEXT",
            top_n=app_config.HYBRID_TOP_K,
        )
        _docs = []
        for doc in results:
            metadata = doc.get("metadata") or {}
            metadata["retrieval_type"] = "bm25"
            _docs.append({"page_content": doc["page_content"], "metadata": metadata})
        return _docs

    def invoke(self, input: State, config=None, **kwargs):
        """
        Merge retrieval candidates.
        KB-only enrichment:
        - semantic + optional bm25
        """
        retriever_docs = input.get("retriever_docs", [])
        error = input.get("error")

        standalone_question = input.get("standalone_question", "")
        kb_query = input.get("kb_query") or standalone_question
        if not standalone_question:
            return {"retriever_docs": retriever_docs, "error": error}

        merged_docs = retriever_docs
        collection_name = "UNKNOWN"
        if config and config.get("configurable"):
            collection_name = config["configurable"].get("collection_name", "UNKNOWN")

        # 1) optional BM25 branch over DB retrieval
        bm25_docs = []
        if app_config.ENABLE_HYBRID_SEARCH:
            try:
                bm25_docs = self._bm25_docs(kb_query, collection_name)
                merged_docs = self._merge_docs(retriever_docs, bm25_docs)
            except Exception as exc:
                logger.warning(
                    "Hybrid BM25 retrieval failed, fallback to semantic only: %s", exc
                )
                merged_docs = retriever_docs

        logger.info(
            "HybridSearch merged KB docs. semantic=%d bm25=%d merged=%d bm25_top_k=%d",
            len(retriever_docs),
            len(bm25_docs),
            len(merged_docs),
            app_config.HYBRID_TOP_K,
        )

        return {"retriever_docs": merged_docs, "error": error}
