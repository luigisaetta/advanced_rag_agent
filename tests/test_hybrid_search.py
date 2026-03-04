"""
File name: tests/test_hybrid_search.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Unit tests for hybrid retrieval merge and fallback behavior.
"""

import agent.hybrid_search as hybrid_module


def test_hybrid_search_passthrough_when_disabled(monkeypatch):
    """Test test hybrid search passthrough when disabled."""
    monkeypatch.setattr(hybrid_module.app_config, "ENABLE_HYBRID_SEARCH", False)
    node = hybrid_module.HybridSearch()

    state = {
        "retriever_docs": [
            {"page_content": "A", "metadata": {"source": "s", "page_label": "1"}}
        ],
        "error": None,
        "standalone_question": "q",
    }

    out = node.invoke(state, config={"configurable": {"collection_name": "COLL01"}})
    assert out["retriever_docs"] == state["retriever_docs"]
    assert out["error"] is None


def test_hybrid_search_merges_and_deduplicates(monkeypatch):
    """Test test hybrid search merges and deduplicates."""
    monkeypatch.setattr(hybrid_module.app_config, "ENABLE_HYBRID_SEARCH", True)
    node = hybrid_module.HybridSearch()

    semantic_docs = [
        {"page_content": "Chunk A", "metadata": {"source": "s1", "page_label": "1"}},
        {"page_content": "Chunk B", "metadata": {"source": "s2", "page_label": "2"}},
    ]
    bm25_docs = [
        {"page_content": "Chunk A", "metadata": {"source": "bm25", "page_label": ""}},
        {"page_content": "Chunk C", "metadata": {"source": "bm25", "page_label": ""}},
    ]

    monkeypatch.setattr(node, "_bm25_docs", lambda query, collection: bm25_docs)

    state = {"retriever_docs": semantic_docs, "error": None, "standalone_question": "q"}
    out = node.invoke(state, config={"configurable": {"collection_name": "COLL01"}})

    contents = [doc["page_content"] for doc in out["retriever_docs"]]
    assert contents == ["Chunk A", "Chunk B", "Chunk C"]


def test_hybrid_search_fallback_on_bm25_error(monkeypatch):
    """Test test hybrid search fallback on bm25 error."""
    monkeypatch.setattr(hybrid_module.app_config, "ENABLE_HYBRID_SEARCH", True)
    node = hybrid_module.HybridSearch()

    semantic_docs = [
        {"page_content": "Chunk A", "metadata": {"source": "s1", "page_label": "1"}},
    ]

    def _boom(query, collection):
        """Helper for boom."""
        raise RuntimeError("bm25 unavailable")

    monkeypatch.setattr(node, "_bm25_docs", _boom)

    state = {"retriever_docs": semantic_docs, "error": None, "standalone_question": "q"}
    out = node.invoke(state, config={"configurable": {"collection_name": "COLL01"}})
    assert out["retriever_docs"] == semantic_docs
