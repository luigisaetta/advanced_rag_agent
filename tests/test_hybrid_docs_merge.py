"""Unit tests for hybrid docs merge behavior."""

from agent.hybrid_docs_merge import HybridDocsMerge


def test_hybrid_docs_merge_appends_session_docs():
    """Test test hybrid docs merge appends session docs."""
    node = HybridDocsMerge()
    out = node.invoke(
        {
            "retriever_docs": [
                {"page_content": "KB A", "metadata": {"retrieval_type": "semantic"}}
            ],
            "session_retriever_docs": [
                {
                    "page_content": "Session B",
                    "metadata": {"retrieval_type": "session_pdf"},
                }
            ],
            "error": None,
        }
    )
    contents = [d["page_content"] for d in out["retriever_docs"]]
    assert contents == ["KB A", "Session B"]


def test_hybrid_docs_merge_deduplicates_by_content():
    """Test test hybrid docs merge deduplicates by content."""
    node = HybridDocsMerge()
    out = node.invoke(
        {
            "retriever_docs": [
                {
                    "page_content": "Same chunk",
                    "metadata": {"retrieval_type": "semantic"},
                }
            ],
            "session_retriever_docs": [
                {
                    "page_content": "  same   chunk  ",
                    "metadata": {"retrieval_type": "session_pdf"},
                }
            ],
            "error": None,
        }
    )
    assert len(out["retriever_docs"]) == 1
