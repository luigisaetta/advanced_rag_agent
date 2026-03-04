"""
Tests for HYBRID session source floor in reranker.
"""

import agent.reranker as reranker_module


def _doc(text, retrieval_type, source="s.pdf", page="1"):
    """Helper for doc."""
    return {
        "page_content": text,
        "metadata": {
            "source": source,
            "page_label": page,
            "retrieval_type": retrieval_type,
        },
    }


def test_hybrid_floor_adds_session_docs_when_missing(monkeypatch):
    """Test test hybrid floor adds session docs when missing."""
    monkeypatch.setattr(reranker_module, "HYBRID_MIN_SESSION_DOCS", 2)
    node = reranker_module.Reranker()

    retriever_docs = [
        _doc("kb a", "semantic", source="kb.pdf", page="2"),
        _doc("kb b", "bm25", source="kb.pdf", page="8"),
        _doc("sess a", "session_pdf", source="upload.pdf", page="3"),
        _doc("sess b", "session_pdf", source="upload.pdf", page="5"),
    ]

    monkeypatch.setattr(reranker_module, "get_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        node,
        "get_reranked_docs",
        lambda llm, query, docs: [_doc("kb a", "semantic", source="kb.pdf", page="2")],
    )

    out = node.invoke(
        {
            "standalone_question": "q",
            "search_intent": "HYBRID",
            "retriever_docs": retriever_docs,
            "error": None,
        },
        config={"configurable": {"enable_reranker": True}},
    )

    final_docs = out["reranker_docs"]
    session_count = sum(
        1
        for d in final_docs
        if (d.get("metadata") or {}).get("retrieval_type", "").startswith("session_pdf")
    )
    assert session_count >= 2


def test_hybrid_floor_not_applied_to_global_kb(monkeypatch):
    """Test test hybrid floor not applied to global kb."""
    monkeypatch.setattr(reranker_module, "HYBRID_MIN_SESSION_DOCS", 2)
    node = reranker_module.Reranker()

    retriever_docs = [
        _doc("kb a", "semantic", source="kb.pdf", page="2"),
        _doc("sess a", "session_pdf", source="upload.pdf", page="3"),
    ]

    monkeypatch.setattr(reranker_module, "get_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        node,
        "get_reranked_docs",
        lambda llm, query, docs: [_doc("kb a", "semantic", source="kb.pdf", page="2")],
    )

    out = node.invoke(
        {
            "standalone_question": "q",
            "search_intent": "GLOBAL_KB",
            "retriever_docs": retriever_docs,
            "error": None,
        },
        config={"configurable": {"enable_reranker": True}},
    )

    assert len(out["reranker_docs"]) == 1
    assert out["reranker_docs"][0]["metadata"]["retrieval_type"] == "semantic"
