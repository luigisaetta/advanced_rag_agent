"""
Unit tests for ui.sidebar.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import ui.sidebar as sidebar_module


class _SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStatusSlot:
    def __init__(self):
        self.info_calls = []
        self.success_calls = []
        self.error_calls = []

    def info(self, text):
        self.info_calls.append(text)

    def success(self, text):
        self.success_calls.append(text)

    def error(self, text):
        self.error_calls.append(text)


class _FakeProgress:
    def __init__(self):
        self.values = []

    def progress(self, value):
        self.values.append(value)


class _FakeSidebar:
    def __init__(self, button_values=None, checkbox_values=None, uploader=None):
        self.button_values = button_values or {}
        self.checkbox_values = checkbox_values or {}
        self.uploader = uploader
        self.warnings = []
        self.errors = []
        self.captions = []
        self.headers = []
        self.status_slots = []
        self.progress_bars = []

    def button(self, label):
        return bool(self.button_values.get(label, False))

    def checkbox(self, label, value=False, disabled=False, help=None):  # noqa: ARG002
        return self.checkbox_values.get(label, value)

    def expander(self, *_args, **_kwargs):
        return nullcontext()

    def file_uploader(self, *_args, **_kwargs):
        return self.uploader

    def header(self, text):
        self.headers.append(text)

    def caption(self, text):
        self.captions.append(text)

    def warning(self, text):
        self.warnings.append(text)

    def error(self, text):
        self.errors.append(text)

    def empty(self):
        slot = _FakeStatusSlot()
        self.status_slots.append(slot)
        return slot

    def progress(self, _initial):
        bar = _FakeProgress()
        self.progress_bars.append(bar)
        return bar


class _FakeStreamlit:
    def __init__(self, sidebar, selectbox_values=None, checkbox_values=None):
        self.sidebar = sidebar
        self.session_state = _SessionState()
        self._selectbox_values = selectbox_values or {}
        self._checkbox_values = checkbox_values or {}

    def text_input(self, *args, **kwargs):  # noqa: ARG002
        return None

    def selectbox(self, label, options, index=0, **kwargs):  # noqa: ARG002
        return self._selectbox_values.get(label, options[index])

    def checkbox(self, label, value=False, disabled=False):  # noqa: ARG002
        return self._checkbox_values.get(label, value)


class _FakeUpload:
    def __init__(self, name, data: bytes):
        self.name = name
        self._data = data

    def read(self):
        return self._data


class _FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, msg, *args):
        self.errors.append((msg, args))


def test_render_sidebar_returns_upload_and_scan_actions(monkeypatch):
    """Sidebar render should wire controls and return PDF actions."""
    upload = _FakeUpload("doc.pdf", b"%PDF-1.4")
    sidebar = _FakeSidebar(
        button_values={"Clear Chat History": True, "Scan PDF in memory": True},
        checkbox_values={"Advanced Analysis": True, "Risk Validation": True},
        uploader=upload,
    )
    fake_st = _FakeStreamlit(
        sidebar=sidebar,
        selectbox_values={"LLM model": "m1", "Collection name": "COLL01"},
        checkbox_values={
            "Enable Reranker": False,
            "Enable tracing": True,
            "Enable Post Answer Evaluation": True,
        },
    )
    fake_st.session_state.model_id = "m1"
    fake_st.session_state.prompt_profile = list(sidebar_module.PROMPT_PROFILES.keys())[0]
    fake_st.session_state.enable_tracing = False
    fake_st.session_state.enable_post_answer_evaluation = False
    fake_st.session_state.enable_risk_validation = False
    fake_st.session_state.session_pdf_name = "existing.pdf"
    fake_st.session_state.session_pdf_chunks_count = 3

    reset_calls = {"n": 0}
    monkeypatch.setattr(sidebar_module, "st", fake_st)

    session_pdf, scan_requested = sidebar_module.render_sidebar(
        lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1)
    )

    assert reset_calls["n"] == 1
    assert session_pdf is upload
    assert scan_requested is True
    assert fake_st.session_state.enable_reranker is False
    assert fake_st.session_state.enable_advanced_analysis is True
    assert fake_st.session_state.enable_risk_validation is True
    assert len(sidebar.captions) == 1


def test_scan_pdf_and_store_in_session_warns_without_upload(monkeypatch):
    """Missing upload should show warning and return."""
    sidebar = _FakeSidebar()
    fake_st = _FakeStreamlit(sidebar=sidebar)
    monkeypatch.setattr(sidebar_module, "st", fake_st)

    sidebar_module.scan_pdf_and_store_in_session(None, logger=_FakeLogger())
    assert sidebar.warnings == ["Please upload a PDF first."]


def test_scan_pdf_and_store_in_session_success(monkeypatch):
    """Successful scan stores vector store and serialized docs in session."""
    upload = _FakeUpload("policy.pdf", b"pdf-bytes")
    sidebar = _FakeSidebar()
    fake_st = _FakeStreamlit(sidebar=sidebar)
    monkeypatch.setattr(sidebar_module, "st", fake_st)

    docs = [
        SimpleNamespace(
            page_content="content A",
            metadata={"source": "policy.pdf", "page_label": "1"},
        )
    ]

    def _fake_scan(**kwargs):
        kwargs["on_progress"](1, 2)
        return docs, 2

    class _FakeVS:
        def __init__(self, embedding):
            self.embedding = embedding
            self.added = []

        def add_documents(self, _docs):
            self.added.extend(_docs)

    monkeypatch.setattr(sidebar_module, "scan_pdf_to_docs_with_vlm", _fake_scan)
    monkeypatch.setattr(sidebar_module, "get_embedding_model", lambda: "embed-model")
    monkeypatch.setattr(sidebar_module, "InMemoryVectorStore", _FakeVS)
    monkeypatch.setattr(
        sidebar_module,
        "docs_serializable",
        lambda input_docs: [
            {"page_content": d.page_content, "metadata": d.metadata} for d in input_docs
        ],
    )

    sidebar_module.scan_pdf_and_store_in_session(upload, logger=_FakeLogger())

    assert fake_st.session_state.session_pdf_name == "policy.pdf"
    assert fake_st.session_state.session_pdf_chunks_count == 1
    assert fake_st.session_state.session_pdf_docs[0]["page_content"] == "content A"
    assert fake_st.session_state.session_pdf_vector_store.embedding == "embed-model"
    assert sidebar.status_slots[0].success_calls
    assert sidebar.progress_bars[0].values[-1] == 100


def test_scan_pdf_and_store_in_session_handles_scan_error(monkeypatch):
    """Scan exceptions should be logged and surfaced in sidebar/status."""
    upload = _FakeUpload("broken.pdf", b"pdf-bytes")
    sidebar = _FakeSidebar()
    fake_st = _FakeStreamlit(sidebar=sidebar)
    logger = _FakeLogger()
    monkeypatch.setattr(sidebar_module, "st", fake_st)

    def _boom(**_kwargs):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(sidebar_module, "scan_pdf_to_docs_with_vlm", _boom)

    sidebar_module.scan_pdf_and_store_in_session(upload, logger=logger)

    assert logger.errors
    assert sidebar.status_slots[0].error_calls == ["PDF scan failed."]
    assert sidebar.errors == ["scan failed"]
