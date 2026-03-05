"""
Regression tests for Advanced Analysis session-only behavior.
"""

from types import SimpleNamespace

from agent.advanced_analysis import (
    AdvancedAnalysisRunner,
    AdvancedFinalSynthesis,
    AdvancedPlanner,
)


class _FakeLlm:
    def __init__(self, content: str):
        """Initialize the fake LLM response payload."""
        self._content = content

    def invoke(self, _messages):
        """Return a deterministic fake response."""
        return SimpleNamespace(content=self._content)


def _session_doc(text: str, page: str = "1") -> dict:
    """Build one serialized session document item."""
    return {
        "page_content": text,
        "metadata": {
            "source": "uploaded.pdf",
            "page_label": page,
            "retrieval_type": "session_pdf",
        },
    }


def test_planner_normalize_plan_forces_no_kb_in_session_only():
    """Ensure normalized plan disables KB fields in session-only mode."""
    raw_plan = [
        {
            "step": 1,
            "section": "Overview",
            "chunk_numbers": [1],
            "objective": "Summarize",
            "kb_search_needed": True,
            "kb_query": "external query",
        }
    ]
    normalized = AdvancedPlanner._normalize_plan(
        raw_plan, max_actions=3, session_only=True
    )
    assert normalized[0]["kb_search_needed"] is False
    assert normalized[0]["kb_query"] == ""


def test_runner_session_only_never_calls_kb_search(monkeypatch):
    """Ensure KB retrieval is skipped even when plan requests it."""
    runner = AdvancedAnalysisRunner()
    monkeypatch.setattr(
        "agent.advanced_analysis.get_llm", lambda **_kwargs: _FakeLlm("ok")
    )

    calls = {"kb": 0}

    def _kb_search(*_args, **_kwargs):
        calls["kb"] += 1
        return [{"page_content": "kb chunk", "metadata": {"source": "kb.pdf"}}]

    monkeypatch.setattr(runner, "_kb_search_docs", _kb_search)

    input_state = {
        "user_request": "Analyze this document.",
        "advanced_plan": [
            {
                "step": 1,
                "section": "Overview",
                "chunk_numbers": [1],
                "objective": "Summarize",
                "kb_search_needed": True,
                "kb_query": "should be ignored",
            }
        ],
        "error": None,
    }
    config = {
        "configurable": {
            "model_id": "fake-model",
            "main_language": "same as the question",
            "advanced_analysis_session_only": True,
            "session_pdf_docs": [_session_doc("Italiano testo documento.", "1")],
            "collection_name": "COLL01",
            "advanced_analysis_kb_top_k": 3,
            "advanced_analysis_step_max_words": 120,
            "session_pdf_vector_store": None,
        }
    }

    out = runner.invoke(input_state, config=config)
    assert calls["kb"] == 0
    assert out["error"] is None


def test_runner_session_only_uses_language_from_document_chunks(monkeypatch):
    """Ensure step titles follow document language in session-only mode."""
    runner = AdvancedAnalysisRunner()
    monkeypatch.setattr(
        "agent.advanced_analysis.get_llm",
        lambda **_kwargs: _FakeLlm("contenuto sintetico"),
    )

    input_state = {
        "user_request": "Please analyze the document.",
        "advanced_plan": [
            {
                "step": 1,
                "section": "Panoramica",
                "chunk_numbers": [1],
                "objective": "Sintesi",
                "kb_search_needed": False,
                "kb_query": "",
            }
        ],
        "error": None,
    }
    config = {
        "configurable": {
            "model_id": "fake-model",
            "main_language": "same as the question",
            "advanced_analysis_session_only": True,
            "session_pdf_docs": [
                _session_doc(
                    "Questo documento descrive le clausole della fornitura e i termini finali.",
                    "1",
                )
            ],
            "collection_name": "COLL01",
            "advanced_analysis_kb_top_k": 3,
            "advanced_analysis_step_max_words": 120,
            "session_pdf_vector_store": None,
        }
    }

    out = runner.invoke(input_state, config=config)
    assert out["advanced_step_outputs"][0].startswith("### Passo 1 -")


def test_runner_non_session_only_uses_request_language(monkeypatch):
    """Ensure non-session-only mode follows request language policy."""
    runner = AdvancedAnalysisRunner()
    monkeypatch.setattr(
        "agent.advanced_analysis.get_llm", lambda **_kwargs: _FakeLlm("summary text")
    )

    input_state = {
        "user_request": "Please analyze this document.",
        "advanced_plan": [
            {
                "step": 1,
                "section": "Overview",
                "chunk_numbers": [1],
                "objective": "Summary",
                "kb_search_needed": False,
                "kb_query": "",
            }
        ],
        "error": None,
    }
    config = {
        "configurable": {
            "model_id": "fake-model",
            "main_language": "same as the question",
            "advanced_analysis_session_only": False,
            "session_pdf_docs": [
                _session_doc(
                    "Questo documento descrive le clausole della fornitura e i termini finali.",
                    "1",
                )
            ],
            "collection_name": "COLL01",
            "advanced_analysis_kb_top_k": 3,
            "advanced_analysis_step_max_words": 120,
            "session_pdf_vector_store": None,
        }
    }

    out = runner.invoke(input_state, config=config)
    assert out["advanced_step_outputs"][0].startswith("### Step 1 -")


def test_final_synthesis_uses_document_language_labels_in_session_only(monkeypatch):
    """Ensure final synthesis section title is localized from document language."""
    node = AdvancedFinalSynthesis()
    monkeypatch.setattr(
        "agent.advanced_analysis.get_llm",
        lambda **_kwargs: _FakeLlm("sintesi conclusiva"),
    )

    input_state = {
        "user_request": "Please analyze this document.",
        "advanced_step_outputs": ["### Passo 1 - Panoramica\ncontenuto"],
        "citations": [],
        "error": None,
    }
    config = {
        "configurable": {
            "model_id": "fake-model",
            "main_language": "same as the question",
            "advanced_analysis_session_only": True,
            "session_pdf_docs": [
                _session_doc(
                    "Questo documento descrive le clausole della fornitura e i termini finali.",
                    "1",
                )
            ],
            "advanced_analysis_step_max_words": 200,
        }
    }

    out = node.invoke(input_state, config=config)
    assert "## Sintesi Finale" in out["final_answer"]
    assert "## Final Synthesis" not in out["final_answer"]
