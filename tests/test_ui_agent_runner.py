"""
Unit tests for ui.agent_runner helpers.
"""

# pylint: disable=protected-access

from types import SimpleNamespace

import ui.agent_runner as runner_module


class _FakeLogger:
    """Collect warnings emitted by helper functions under test."""

    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        """Store warning log invocations for assertions."""
        self.warnings.append((msg, args))

    def info(self, msg, *args):
        """No-op info logger for API compatibility."""
        _ = (msg, args)


def test_build_agent_config_reads_session_and_runtime_flags(monkeypatch):
    """Config builder should map streamlit session values into configurable payload."""
    fake_st = SimpleNamespace(
        session_state=SimpleNamespace(
            model_id="model-x",
            enable_reranker=True,
            enable_advanced_analysis=True,
            enable_tracing=False,
            prompt_profile="default",
            collection_name="COLL01",
            thread_id="thread-1",
            session_pdf_vector_store="vs",
            session_pdf_chunks_count=9,
            session_pdf_docs=[{"page_content": "x"}],
            enable_risk_validation=True,
            enable_post_answer_evaluation=False,
        )
    )
    monkeypatch.setattr(runner_module, "st", fake_st)

    def callback(*_args):
        """No-op callback used to verify reference passthrough."""

    cfg = runner_module._build_agent_config(callback)["configurable"]

    assert cfg["model_id"] == "model-x"
    assert cfg["collection_name"] == "COLL01"
    assert cfg["thread_id"] == "thread-1"
    assert cfg["enable_tracing"] is False
    assert cfg["enable_reranker"] is True
    assert cfg["enable_advanced_analysis"] is True
    assert cfg["advanced_analysis_enable_risk_validation"] is True
    assert cfg["post_answer_evaluation_enabled"] is False
    assert cfg["progress_callback"] is callback


def test_has_hybrid_db_signal_detects_bm25_and_semantic_bm25():
    """Hybrid signal is true when retrieval_type contains bm25 provenance."""
    assert (
        runner_module._has_hybrid_db_signal(
            [{"metadata": {"retrieval_type": "semantic"}}]
        )
        is False
    )
    assert (
        runner_module._has_hybrid_db_signal([{"metadata": {"retrieval_type": "bm25"}}])
        is True
    )
    assert (
        runner_module._has_hybrid_db_signal(
            [{"metadata": {"retrieval_type": "semantic+bm25"}}]
        )
        is True
    )


def test_run_post_answer_evaluation_submits_when_hybrid_db_context_present(monkeypatch):
    """Evaluator should run asynchronously only for hybrid DB-backed answers."""
    logger = _FakeLogger()
    calls = {"submit": 0, "invoke": 0}

    class _FakeEvaluator:
        def invoke(self, input_state, config=None):  # noqa: A002
            """Validate post-eval payload and count invocations."""
            calls["invoke"] += 1
            assert input_state["standalone_question"] == "rewritten question"
            assert input_state["final_answer"] == "final answer"
            assert config["configurable"]["post_answer_evaluation_enabled"] is True

        def name(self):
            """Return fake evaluator name for diagnostics."""
            return "fake-evaluator"

    class _FakeExecutor:
        def submit(self, fn):
            """Execute submitted function inline for deterministic tests."""
            calls["submit"] += 1
            fn()

        def shutdown(self):
            """No-op shutdown hook for API compatibility."""

    monkeypatch.setattr(
        runner_module, "_POST_ANSWER_EVALUATION_AGENT", _FakeEvaluator()
    )
    monkeypatch.setattr(runner_module, "_POST_EVAL_EXECUTOR", _FakeExecutor())

    by_step = {
        "IntentClassifier": {"has_session_pdf": False},
        "QueryRewrite": {"standalone_question": "rewritten question"},
        "HybridSearch": {
            "retriever_docs": [
                {
                    "page_content": "doc",
                    "metadata": {"retrieval_type": "bm25", "source": "s"},
                }
            ]
        },
        "Rerank": {"reranker_docs": [{"page_content": "reranked"}]},
    }
    agent_config = {"configurable": {"post_answer_evaluation_enabled": True}}

    runner_module._run_post_answer_evaluation_if_needed(
        by_step=by_step,
        question="original question",
        final_answer="final answer",
        agent_config=agent_config,
        logger=logger,
    )

    assert calls["submit"] == 1
    assert calls["invoke"] == 1


def test_run_post_answer_evaluation_skips_for_session_pdf(monkeypatch):
    """No post-answer evaluation should run for session-PDF intent."""
    calls = {"submit": 0}

    class _FakeExecutor:
        def submit(self, _fn):
            """Track submit attempts without executing function."""
            calls["submit"] += 1

        def shutdown(self):
            """No-op shutdown hook for API compatibility."""

    monkeypatch.setattr(runner_module, "_POST_EVAL_EXECUTOR", _FakeExecutor())

    runner_module._run_post_answer_evaluation_if_needed(
        by_step={"IntentClassifier": {"has_session_pdf": True}},
        question="q",
        final_answer="a",
        agent_config={"configurable": {"post_answer_evaluation_enabled": True}},
        logger=_FakeLogger(),
    )

    assert calls["submit"] == 0
