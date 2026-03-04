"""
File name: test_workflow_routing.py
Author: Luigi Saetta
Last modified: 01-03-2026
Python Version: 3.11
License: MIT
Description: End-to-end workflow routing tests with mocked nodes.
"""

import pytest
from langchain_core.runnables import Runnable


rag_module = pytest.importorskip("agent.rag_agent")


class _FakeModerator(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {"error": input_state.get("error")}


class _FakeQueryRewriter(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {
            "standalone_question": input_state.get("user_request", ""),
            "error": input_state.get("error"),
        }


class _FakeIntentClassifier(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        configurable = (config or {}).get("configurable", {})
        session_vs = configurable.get("session_pdf_vector_store")
        chunks_count = configurable.get("session_pdf_chunks_count", 0)
        advanced_enabled = bool(configurable.get("enable_advanced_analysis", False))
        if session_vs is None or chunks_count <= 0:
            return {
                "search_intent": "GLOBAL_KB",
                "has_session_pdf": False,
                "advanced_analysis_enabled": advanced_enabled,
                "error": input_state.get("error"),
            }

        forced_intent = configurable.get("forced_intent", "GLOBAL_KB")
        return {
            "search_intent": forced_intent,
            "has_session_pdf": True,
            "advanced_analysis_enabled": advanced_enabled,
            "error": input_state.get("error"),
        }


class _FakeSearch(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        docs = [
            {
                "page_content": "semantic chunk",
                "metadata": {
                    "source": "kb.pdf",
                    "page_label": "1",
                    "retrieval_type": "semantic",
                },
            }
        ]
        return {"retriever_docs": docs, "error": input_state.get("error")}


class _FakeHybridQueryBuilder(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {
            "kb_query": f"kb::{input_state.get('standalone_question', '')}",
            "error": input_state.get("error"),
        }


class _FakeSessionSearch(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        docs = [
            {
                "page_content": "session only chunk",
                "metadata": {
                    "source": "uploaded.pdf",
                    "page_label": "3",
                    "retrieval_type": "session_pdf",
                },
            }
        ]
        return {"retriever_docs": docs, "error": input_state.get("error")}


class _FakeHybridSearch(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        docs = list(input_state.get("retriever_docs", []))
        docs.append(
            {
                "page_content": "bm25 chunk",
                "metadata": {
                    "source": "kb.pdf",
                    "page_label": "2",
                    "retrieval_type": "bm25",
                },
            }
        )
        return {"retriever_docs": docs, "error": input_state.get("error")}


class _FakeHybridSessionSearch(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        docs = []
        if input_state.get("search_intent") == "HYBRID":
            docs.append(
                {
                    "page_content": "session hybrid chunk",
                    "metadata": {
                        "source": "uploaded.pdf",
                        "page_label": "4",
                        "retrieval_type": "session_pdf",
                    },
                }
            )
        return {"session_retriever_docs": docs, "error": input_state.get("error")}


class _FakeHybridDocsMerge(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        kb_docs = list(input_state.get("retriever_docs", []))
        session_docs = list(input_state.get("session_retriever_docs", []))
        return {
            "retriever_docs": kb_docs + session_docs,
            "error": input_state.get("error"),
        }


class _FakeAdvancedPlanner(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {"advanced_plan": [], "error": input_state.get("error")}


class _FakeAdvancedRunner(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {
            "advanced_step_outputs": ["### Step 1 - mock\nmock step output"],
            "citations": [],
            "error": input_state.get("error"),
        }


class _FakeAdvancedFinalSynthesis(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {
            "final_answer": "advanced placeholder with synthesis",
            "citations": input_state.get("citations", []),
            "error": input_state.get("error"),
        }


class _FakeRerank(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        docs = input_state.get("retriever_docs", [])
        citations = [
            {
                "source": (doc.get("metadata") or {}).get("source", "unknown"),
                "page": (doc.get("metadata") or {}).get("page_label", ""),
                "retrieval_type": (doc.get("metadata") or {}).get(
                    "retrieval_type", "semantic"
                ),
            }
            for doc in docs
        ]
        return {
            "reranker_docs": docs,
            "citations": citations,
            "error": input_state.get("error"),
        }


class _FakeAnswer(Runnable):
    def invoke(self, input_state, config=None, **kwargs):
        """Invoke."""
        return {"final_answer": "ok", "error": input_state.get("error")}


def _build_workflow_with_mocks(monkeypatch):
    """Helper for build workflow with mocks."""
    monkeypatch.setattr(rag_module, "ContentModerator", _FakeModerator)
    monkeypatch.setattr(rag_module, "QueryRewriter", _FakeQueryRewriter)
    monkeypatch.setattr(rag_module, "IntentClassifier", _FakeIntentClassifier)
    monkeypatch.setattr(rag_module, "HybridQueryBuilder", _FakeHybridQueryBuilder)
    monkeypatch.setattr(rag_module, "SemanticSearch", _FakeSearch)
    monkeypatch.setattr(rag_module, "SessionVectorSearch", _FakeSessionSearch)
    monkeypatch.setattr(rag_module, "HybridSearch", _FakeHybridSearch)
    monkeypatch.setattr(rag_module, "HybridSessionSearch", _FakeHybridSessionSearch)
    monkeypatch.setattr(rag_module, "HybridDocsMerge", _FakeHybridDocsMerge)
    monkeypatch.setattr(rag_module, "AdvancedPlanner", _FakeAdvancedPlanner)
    monkeypatch.setattr(rag_module, "AdvancedAnalysisRunner", _FakeAdvancedRunner)
    monkeypatch.setattr(
        rag_module, "AdvancedFinalSynthesis", _FakeAdvancedFinalSynthesis
    )
    monkeypatch.setattr(rag_module, "Reranker", _FakeRerank)
    monkeypatch.setattr(rag_module, "AnswerGenerator", _FakeAnswer)
    return rag_module.create_workflow()


def _run_workflow(app, config):
    """Helper for run workflow."""
    state = {"user_request": "test question", "chat_history": [], "error": None}
    events = list(app.stream(state, config=config))
    step_names = [next(iter(event.keys())) for event in events]
    by_step = {}
    for event in events:
        for key, value in event.items():
            by_step[key] = value
    return step_names, by_step


def test_global_kb_route_without_session_pdf(monkeypatch):
    """Test test global kb route without session pdf."""
    app = _build_workflow_with_mocks(monkeypatch)
    steps, by_step = _run_workflow(
        app,
        {
            "configurable": {
                "forced_intent": "HYBRID",
                "session_pdf_vector_store": None,
                "session_pdf_chunks_count": 0,
            }
        },
    )

    assert "Search" in steps
    assert "HybridSearch" in steps
    assert "HybridFlow" not in steps
    assert "SessionSearch" not in steps
    assert by_step["IntentClassifier"]["search_intent"] == "GLOBAL_KB"


def test_session_doc_route_uses_only_session_search(monkeypatch):
    """Test test session doc route uses only session search."""
    app = _build_workflow_with_mocks(monkeypatch)
    steps, by_step = _run_workflow(
        app,
        {
            "configurable": {
                "forced_intent": "SESSION_DOC",
                "session_pdf_vector_store": object(),
                "session_pdf_chunks_count": 5,
            }
        },
    )

    assert "SessionSearch" in steps
    assert "Search" not in steps
    assert "HybridFlow" not in steps
    assert "HybridSearch" not in steps
    retrieval_types = {c["retrieval_type"] for c in by_step["Rerank"]["citations"]}
    assert retrieval_types == {"session_pdf"}


def test_hybrid_route_merges_db_and_session_provenance(monkeypatch):
    """Test test hybrid route merges db and session provenance."""
    app = _build_workflow_with_mocks(monkeypatch)
    steps, by_step = _run_workflow(
        app,
        {
            "configurable": {
                "forced_intent": "HYBRID",
                "session_pdf_vector_store": object(),
                "session_pdf_chunks_count": 5,
            }
        },
    )

    assert "HybridFlow" in steps
    assert "SessionSearch" not in steps
    retrieval_types = {c["retrieval_type"] for c in by_step["Rerank"]["citations"]}
    assert retrieval_types == {"semantic", "bm25", "session_pdf"}


def test_global_route_contains_db_provenance(monkeypatch):
    """Test test global route contains db provenance."""
    app = _build_workflow_with_mocks(monkeypatch)
    _steps, by_step = _run_workflow(
        app,
        {
            "configurable": {
                "forced_intent": "GLOBAL_KB",
                "session_pdf_vector_store": object(),
                "session_pdf_chunks_count": 5,
            }
        },
    )

    retrieval_types = {c["retrieval_type"] for c in by_step["Rerank"]["citations"]}
    assert retrieval_types == {"semantic", "bm25"}


def test_hybrid_route_uses_advanced_subgraph_when_enabled(monkeypatch):
    """Test test hybrid route uses advanced subgraph when enabled."""
    app = _build_workflow_with_mocks(monkeypatch)
    steps, by_step = _run_workflow(
        app,
        {
            "configurable": {
                "forced_intent": "HYBRID",
                "session_pdf_vector_store": object(),
                "session_pdf_chunks_count": 5,
                "enable_advanced_analysis": True,
            }
        },
    )

    assert "AdvancedAnalysisFlow" in steps
    assert "HybridFlow" not in steps
    assert "Rerank" not in steps
    assert by_step["AdvancedAnalysisFlow"]["final_answer"]
