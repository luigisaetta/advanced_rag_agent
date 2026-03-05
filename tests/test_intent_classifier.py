"""
File name: test_intent_classifier.py
Author: Luigi Saetta
Last modified: 01-03-2026
Python Version: 3.11
License: MIT
Description: Unit tests for intent classification node routing behavior.
"""

from types import SimpleNamespace

from agent.intent_classifier import IntentClassifier


def test_intent_classifier_forces_global_when_no_session_pdf():
    """Test test intent classifier forces global when no session pdf."""
    node = IntentClassifier()
    out = node.invoke(
        {"user_request": "What is this about?", "error": None},
        config={
            "configurable": {
                "session_pdf_vector_store": None,
                "session_pdf_chunks_count": 0,
            }
        },
    )
    assert out["search_intent"] == "GLOBAL_KB"
    assert out["has_session_pdf"] is False
    assert out["advanced_analysis_session_only"] is False


def test_intent_classifier_calls_llm_when_session_pdf_exists(monkeypatch):
    """Test test intent classifier calls llm when session pdf exists."""
    node = IntentClassifier()

    class _FakeLlm:
        def invoke(self, _messages):
            """Invoke."""
            return SimpleNamespace(content='{"intent": "SESSION_DOC"}')

    monkeypatch.setattr("agent.intent_classifier.get_llm", lambda **kwargs: _FakeLlm())

    out = node.invoke(
        {"user_request": "Use the uploaded file", "error": None},
        config={
            "configurable": {
                "session_pdf_vector_store": object(),
                "session_pdf_chunks_count": 3,
            }
        },
    )

    assert out["search_intent"] == "SESSION_DOC"
    assert out["has_session_pdf"] is True
    assert out["advanced_analysis_session_only"] is True


def test_intent_classifier_rejects_invalid_hybrid_typo():
    """Test test intent classifier rejects invalid hybrid typo."""
    node = IntentClassifier()
    assert node._normalize_intent("HYBRYD") == "GLOBAL_KB"
