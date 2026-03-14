"""
Unit tests for regression dataset generation helpers.
"""

# pylint: disable=protected-access

from scripts.eval import generate_regression_from_pdf as gen


def test_clean_page_text_removes_doc_title_header():
    """Synthetic document header is removed from extracted page text."""
    raw = "# Doc. title: Contract X\nBody content on this page."
    assert gen._clean_page_text(raw) == "Body content on this page."


def test_is_not_too_empty_enforces_minimum_content():
    """Short/noisy text should be rejected."""
    assert gen._is_not_too_empty("short", min_chars=20) is False
    assert gen._is_not_too_empty("This page has enough readable text.", min_chars=10)


def test_parse_questions_json_extracts_payload_from_wrapped_text():
    """Parser should tolerate pre/post text around JSON."""
    raw = 'prefix {"questions":[{"question":"Q1","chunk_number":"1"}]} suffix'
    out = gen._parse_questions_json(raw)
    assert out == [{"question": "Q1", "chunk_number": "1"}]


def test_question_is_valid_rejects_forbidden_patterns():
    """Question filter blocks references to sections and numeric ranges."""
    assert gen._question_is_valid("What are the payment terms?", "contract_file")
    assert not gen._question_is_valid("What is in section 1.2?", "contract_file")
    assert not gen._question_is_valid("What is on page 3-4?", "contract_file")
    assert not gen._question_is_valid("What says contract file?", "contract_file")


def test_to_jsonl_rows_maps_fields_and_ids():
    """Tuple rows are converted to expected eval JSONL schema."""
    rows = gen._to_jsonl_rows(
        qa_rows=[("What is the duration?", "7")],
        source_name="doc.pdf",
        id_prefix="kb_auto",
        start_id=4,
    )
    assert rows == [
        {
            "id": "kb_auto_004",
            "question": "What is the duration?",
            "expected_intent": "GLOBAL_KB",
            "expected_sources": ["kb"],
            "expected_citations": [{"source": "doc.pdf", "page": "7"}],
        }
    ]


def test_generate_questions_filters_invalid_pages_and_duplicates(monkeypatch):
    """Generation keeps only unique, valid questions on allowed pages."""

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

        def text(self):
            """Return response content for test assertions/debugging."""
            return self.content

    class _FakeLLM:
        def invoke(self, _messages):
            """Return a deterministic fake LLM response payload."""
            return _FakeResponse(
                '{"questions":['
                '{"question":"What is the payment term?","chunk_number":"1"},'
                '{"question":"What is the payment term?","chunk_number":"1"},'
                '{"question":"What is in section 1.2?","chunk_number":"1"},'
                '{"question":"What is the termination clause?","chunk_number":"2"},'
                '{"question":"What is in scope?","chunk_number":"9"}'
                "]}"
            )

        def model_name(self):
            """Expose a model identifier like a real client might."""
            return "fake-model"

    monkeypatch.setattr(gen, "get_llm", lambda model_id, temperature: _FakeLLM())
    monkeypatch.setattr(gen, "run_with_retry", lambda func, **kwargs: func())

    out = gen._generate_questions(
        selected_pages=[
            {"page": "1", "text": "A"},
            {"page": "2", "text": "B"},
        ],
        questions_count=2,
        model_id="fake-model",
        file_stem="contract_file",
        max_attempts=1,
    )

    assert out == [
        ("What is the payment term?", "1"),
        ("What is the termination clause?", "2"),
    ]
