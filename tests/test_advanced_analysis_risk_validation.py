"""
Unit tests for the minimal Advanced Analysis risk validation step.
"""

from types import SimpleNamespace

from agent.advanced_analysis import RiskValidator


class _FakeLlm:
    def __init__(self, responses: list):
        """Initialize deterministic sequential responses."""
        self._responses = list(responses)

    def invoke(self, _messages):
        """Return the next response."""
        if self._responses:
            return SimpleNamespace(content=self._responses.pop(0))
        return SimpleNamespace(content="")


def _base_input_state():
    """Build a minimal input state for risk validation."""
    return {
        "user_request": "Analyze the document using the available KB.",
        "advanced_step_outputs": ["### Step 1 - Overview\ntext"],
        "final_answer": "Some negative findings are present.",
        "citations": [],
        "error": None,
    }


def test_risk_validation_disabled_passthrough(monkeypatch):
    """Ensure disabled risk validation is a passthrough."""
    node = RiskValidator()
    out = node.invoke(
        _base_input_state(),
        config={"configurable": {"advanced_analysis_enable_risk_validation": False}},
    )
    assert "Risk Validation" not in out["final_answer"]


def test_risk_validation_enabled_adds_section_without_critical_findings(monkeypatch):
    """Ensure enabled validation appends section even with no critical findings."""
    node = RiskValidator()
    fake_llm = _FakeLlm(
        ['{"critical_negative_findings": false, "claims_to_validate": []}']
    )
    monkeypatch.setattr("agent.advanced_analysis.get_llm", lambda **_kwargs: fake_llm)
    out = node.invoke(
        _base_input_state(),
        config={
            "configurable": {
                "model_id": "fake-model",
                "main_language": "en",
                "advanced_analysis_enable_risk_validation": True,
            }
        },
    )
    assert "## Risk Validation" in out["final_answer"]


def test_risk_validation_critical_findings_calls_kb_when_not_session_only(monkeypatch):
    """Ensure critical findings trigger KB validation in non session-only mode."""
    node = RiskValidator()
    fake_llm = _FakeLlm(
        [
            '{"critical_negative_findings": true, "claims_to_validate": ["major penalty risk"]}',
            "Validation result from KB evidence.",
        ]
    )
    monkeypatch.setattr("agent.advanced_analysis.get_llm", lambda **_kwargs: fake_llm)

    calls = {"kb": 0}

    def _kb_search(self, query, collection_name, top_k):  # noqa: ARG001
        calls["kb"] += 1
        return [
            {
                "page_content": "KB evidence",
                "metadata": {
                    "source": "kb.pdf",
                    "page_label": "2",
                    "retrieval_type": "semantic",
                },
            }
        ]

    monkeypatch.setattr("agent.advanced_analysis.AdvancedAnalysisRunner._kb_search_docs", _kb_search)

    out = node.invoke(
        _base_input_state(),
        config={
            "configurable": {
                "model_id": "fake-model",
                "main_language": "en",
                "collection_name": "COLL01",
                "advanced_analysis_enable_risk_validation": True,
                "advanced_analysis_session_only": False,
            }
        },
    )
    assert calls["kb"] == 1
    assert "## Risk Validation" in out["final_answer"]
    assert "Validation result from KB evidence." in out["final_answer"]

