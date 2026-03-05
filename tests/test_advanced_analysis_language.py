"""
Language-resolution regression tests for advanced analysis.
"""

from agent import advanced_analysis as aa


def test_detect_question_language_defaults_to_english_for_english_query():
    """Test detect question language defaults to english for english query."""
    q = "Analyze the document using the available knowledge base and summarize key findings."
    assert aa._detect_question_language(q) == "en"


def test_detect_question_language_for_italian_query():
    """Test detect question language for italian query."""
    q = "Analizza il documento e indica le clausole della sezione nel capitolo finale."
    assert aa._detect_question_language(q) == "it"


def test_detect_language_from_session_docs_for_italian_document():
    """Test detect language from session docs for italian document."""
    docs = [
        {
            "page_content": (
                "Questo documento descrive le condizioni della fornitura e le clausole "
                "finali. Il cliente e il fornitore concordano i termini nel contratto."
            ),
            "metadata": {"source": "doc_it.pdf", "page_label": "1"},
        }
    ]
    assert aa._detect_language_from_session_docs(docs) == "it"
