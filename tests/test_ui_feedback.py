"""
Unit tests for ui.feedback callbacks.
"""

from types import SimpleNamespace

import ui.feedback as feedback_module


class _SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    def __init__(self):
        self.session_state = _SessionState()


class _FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, *args):
        self.lines.append(args)


def test_register_feedback_persists_last_qa_and_disables_prompt(monkeypatch):
    """Register feedback stores stars and latest chat turn pair."""
    fake_st = _FakeStreamlit()
    fake_st.session_state.feedback = 3
    fake_st.session_state.get_feedback = True
    fake_st.session_state.chat_history = [
        SimpleNamespace(content="older"),
        SimpleNamespace(content="older answer"),
        SimpleNamespace(content="latest question"),
        SimpleNamespace(content="latest answer"),
    ]

    inserted = {}

    class _FakeRagFeedback:
        def insert_feedback(self, question, answer, feedback):
            inserted["question"] = question
            inserted["answer"] = answer
            inserted["feedback"] = feedback

    monkeypatch.setattr(feedback_module, "st", fake_st)
    monkeypatch.setattr(feedback_module, "RagFeedback", _FakeRagFeedback)

    logger = _FakeLogger()
    feedback_module.register_feedback(logger)

    assert inserted == {
        "question": "latest question",
        "answer": "latest answer",
        "feedback": 4,
    }
    assert fake_st.session_state.get_feedback is False
