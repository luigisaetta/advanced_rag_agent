"""
Unit tests for HYBRID KB query expansion node.
"""

import agent.hybrid_query_builder as node_module


class _Doc:
    def __init__(self, text, metadata=None):
        """Initialize the instance."""
        self.page_content = text
        self.metadata = metadata or {}


class _SessionVS:
    def __init__(self, docs):
        """Initialize the instance."""
        self.docs = docs

    def similarity_search(self, query, k):
        """Similarity search."""
        return self.docs[:k]


class _LLM:
    def __init__(self, text):
        """Initialize the instance."""
        self._text = text

    class _Resp:
        def __init__(self, content):
            """Initialize the instance."""
            self.content = content

    def invoke(self, _messages):
        """Invoke."""
        return self._Resp(self._text)


def test_hybrid_query_builder_passthrough_for_non_hybrid():
    """Test test hybrid query builder passthrough for non hybrid."""
    node = node_module.HybridQueryBuilder()
    state = {
        "search_intent": "GLOBAL_KB",
        "standalone_question": "what is x",
        "error": None,
    }
    out = node.invoke(state, config={"configurable": {"model_id": "m"}})
    assert out["kb_query"] == "what is x"


def test_hybrid_query_builder_builds_kb_query(monkeypatch):
    """Test test hybrid query builder builds kb query."""
    monkeypatch.setattr(
        node_module, "get_llm", lambda model_id, temperature=0.0: _LLM("expanded query")
    )

    node = node_module.HybridQueryBuilder()
    state = {
        "search_intent": "HYBRID",
        "user_request": "evaluate this document with KB",
        "standalone_question": "evaluate this document with KB",
        "error": None,
    }
    cfg = {
        "configurable": {
            "model_id": "openai.gpt-oss-120b",
            "session_pdf_vector_store": _SessionVS(
                [
                    _Doc(
                        "supplier is Acme, annual volume is 12000",
                        {"source": "u.pdf", "page_label": "2"},
                    ),
                    _Doc(
                        "contract duration 24 months",
                        {"source": "u.pdf", "page_label": "3"},
                    ),
                ]
            ),
        }
    }
    out = node.invoke(state, config=cfg)
    assert out["kb_query"] == "expanded query"


def test_hybrid_query_builder_fallback_on_llm_error(monkeypatch):
    """Test test hybrid query builder fallback on llm error."""

    def _raise(*args, **kwargs):
        """Helper for raise."""
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(node_module, "get_llm", _raise)

    node = node_module.HybridQueryBuilder()
    state = {
        "search_intent": "HYBRID",
        "standalone_question": "baseline q",
        "error": None,
    }
    cfg = {
        "configurable": {
            "model_id": "openai.gpt-oss-120b",
            "session_pdf_vector_store": _SessionVS(
                [_Doc("foo", {"source": "u.pdf", "page_label": "1"})]
            ),
        }
    }
    out = node.invoke(state, config=cfg)
    assert out["kb_query"] == "baseline q"
