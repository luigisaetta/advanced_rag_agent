"""
Unit tests for post-answer evaluation LangGraph subagent wrapper.
"""

from langchain_core.runnables import Runnable

from agent.post_answer_evaluation_agent import (
    create_post_answer_evaluation_agent,
    create_post_answer_evaluation_workflow,
)


class _FakePostAnswerEvaluator(Runnable):
    """Simple node stub for deterministic workflow tests."""

    def invoke(self, input_state, config=None, **kwargs):  # noqa: A002
        _ = (config, kwargs)
        return {
            "final_answer": input_state.get("final_answer", ""),
            "citations": input_state.get("citations", []),
            "post_answer_root_cause": "RETRIEVAL",
            "post_answer_reason": "missing doc",
            "post_answer_confidence": 0.8,
            "post_answer_quality_score": 4,
            "error": input_state.get("error"),
        }


def test_create_post_answer_evaluation_workflow_invokes_node():
    """Workflow should execute evaluator node and expose node outputs."""
    workflow = create_post_answer_evaluation_workflow(
        post_answer_evaluator=_FakePostAnswerEvaluator()
    )
    out = workflow.invoke(
        {
            "user_request": "q",
            "standalone_question": "q",
            "retriever_docs": [],
            "reranker_docs": [],
            "final_answer": "a",
            "citations": [],
            "error": None,
        }
    )
    assert out["post_answer_root_cause"] == "RETRIEVAL"
    assert out["post_answer_confidence"] == 0.8
    assert out["post_answer_quality_score"] == 4


def test_create_post_answer_evaluation_agent_normalizes_output_fields():
    """Agent wrapper should return stable output keys."""
    agent = create_post_answer_evaluation_agent(
        post_answer_evaluator=_FakePostAnswerEvaluator()
    )
    out = agent.invoke({"final_answer": "a", "citations": []})
    assert out["final_answer"] == "a"
    assert out["post_answer_root_cause"] == "RETRIEVAL"
    assert out["post_answer_reason"] == "missing doc"
    assert out["post_answer_confidence"] == 0.8
    assert out["post_answer_quality_score"] == 4
    assert out["error"] is None
