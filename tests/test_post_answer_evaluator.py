"""
Unit tests for post-answer evaluator behavior.
"""

from types import SimpleNamespace

import agent.post_answer_evaluator as pae_module
from agent.post_answer_evaluator import PostAnswerEvaluator


class _FakeLlm:
    def __init__(self, content: str):
        """Store deterministic model response content."""
        self._content = content

    def invoke(self, _messages):
        """Return a simple LLM-like response object."""
        return SimpleNamespace(content=self._content)


class _FakeFeedback:
    def __init__(self, calls: list):
        """Store list where insert calls will be collected."""
        self._calls = calls

    def insert_feedback(self, **kwargs):
        """Capture feedback payload."""
        self._calls.append(kwargs)


def _base_state(final_answer="Final answer text."):
    """Return a minimal valid evaluator input state."""
    return {
        "user_request": "What are the key obligations?",
        "standalone_question": "What are the key obligations?",
        "retriever_docs": [
            {
                "page_content": "Obligation A and B are listed.",
                "metadata": {
                    "source": "contract.pdf",
                    "page_label": "2",
                    "retrieval_type": "semantic",
                },
            }
        ],
        "reranker_docs": [
            {
                "page_content": "Obligation A and B are listed.",
                "metadata": {
                    "source": "contract.pdf",
                    "page_label": "2",
                    "retrieval_type": "rerank",
                },
            }
        ],
        "final_answer": final_answer,
        "citations": [{"source": "contract.pdf", "page": "2"}],
        "error": None,
    }


def test_post_answer_evaluator_skips_when_answer_is_empty():
    """Empty final answer should bypass evaluation and persistence."""
    node = PostAnswerEvaluator()
    out = node.invoke(_base_state(final_answer=""))
    assert out["final_answer"] == ""
    assert out["post_answer_root_cause"] == ""
    assert out["post_answer_reason"] == ""
    assert out["post_answer_confidence"] == 0.0


def test_post_answer_evaluator_skips_when_disabled():
    """Disabled evaluator should return passthrough output."""
    node = PostAnswerEvaluator()
    out = node.invoke(
        _base_state(),
        config={"configurable": {"post_answer_evaluation_enabled": False}},
    )
    assert out["final_answer"] == "Final answer text."
    assert out["post_answer_root_cause"] == ""
    assert out["post_answer_confidence"] == 0.0


def test_post_answer_evaluator_success_and_persists_feedback(monkeypatch):
    """Evaluator parses model output and writes feedback."""
    node = PostAnswerEvaluator()
    feedback_calls = []

    monkeypatch.setattr(
        pae_module, "get_llm", lambda **_kwargs: _FakeLlm("irrelevant raw payload")
    )
    monkeypatch.setattr(
        pae_module,
        "extract_json_from_text",
        lambda _raw: {
            "root_cause": "retrieval",
            "reason": "Missing crucial evidence in retrieval set.",
            "confidence": 0.82,
        },
    )
    monkeypatch.setattr(
        pae_module, "PostAnswerFeedback", lambda: _FakeFeedback(feedback_calls)
    )

    out = node.invoke(
        _base_state(),
        config={
            "configurable": {
                "model_id": "answer-model",
                "post_answer_evaluation_model_id": "eval-model",
                "embed_model_id": "embed-model",
                "reranker_model_id": "reranker-model",
                "top_k": 12,
            }
        },
    )

    assert out["post_answer_root_cause"] == "RETRIEVAL"
    assert out["post_answer_reason"] == "Missing crucial evidence in retrieval set."
    assert out["post_answer_confidence"] == 0.82
    assert len(feedback_calls) == 1
    assert feedback_calls[0]["question"] == "What are the key obligations?"
    assert feedback_calls[0]["root_cause"] == "RETRIEVAL"
    assert feedback_calls[0]["config_data"]["top_k"] == 12


def test_post_answer_evaluator_normalizes_invalid_fields(monkeypatch):
    """Unknown cause and out-of-range confidence are normalized."""
    node = PostAnswerEvaluator()
    feedback_calls = []

    monkeypatch.setattr(pae_module, "get_llm", lambda **_kwargs: _FakeLlm("unused raw"))
    monkeypatch.setattr(
        pae_module,
        "extract_json_from_text",
        lambda _raw: {
            "root_cause": "something_else",
            "reason": "N/A",
            "confidence": 7,
        },
    )
    monkeypatch.setattr(
        pae_module, "PostAnswerFeedback", lambda: _FakeFeedback(feedback_calls)
    )

    out = node.invoke(_base_state(), config={"configurable": {}})
    assert out["post_answer_root_cause"] == "NO_ISSUE"
    assert out["post_answer_confidence"] == 1.0
    assert len(feedback_calls) == 1


def test_post_answer_evaluator_handles_exception_and_falls_back(monkeypatch):
    """Any evaluator failure should not break downstream answer."""
    node = PostAnswerEvaluator()

    def _boom(**_kwargs):
        raise RuntimeError("x")

    monkeypatch.setattr(pae_module, "get_llm", _boom)

    out = node.invoke(_base_state(), config={"configurable": {}})
    assert out["final_answer"] == "Final answer text."
    assert out["post_answer_root_cause"] == ""
    assert out["post_answer_reason"] == ""
    assert out["post_answer_confidence"] == 0.0
