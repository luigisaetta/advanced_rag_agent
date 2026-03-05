"""
File name: tests/test_utils.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Unit tests for utility helpers related to JSON extraction, path cleanup, and document serialization.
"""

from types import SimpleNamespace

import pytest

from core import utils


def test_extract_json_from_text_success():
    """Test test extract json from text success."""
    text = 'prefix {"a": 1, "b": "x"} suffix'
    assert utils.extract_json_from_text(text) == {"a": 1, "b": "x"}


def test_extract_json_from_text_raises_when_missing():
    """Test test extract json from text raises when missing."""
    with pytest.raises(ValueError, match="No JSON content found"):
        utils.extract_json_from_text("nothing to parse")


def test_remove_path_from_ref_returns_basename():
    """Test test remove path from ref returns basename."""
    assert utils.remove_path_from_ref("/tmp/docs/file.pdf") == "file.pdf"


def test_docs_serializable_maps_documents():
    """Test test docs serializable maps documents."""
    docs = [
        SimpleNamespace(page_content="hello", metadata={"source": "a.pdf"}),
        SimpleNamespace(page_content="world", metadata=None),
    ]

    serialized = utils.docs_serializable(docs)

    assert serialized == [
        {"page_content": "hello", "metadata": {"source": "a.pdf"}},
        {"page_content": "world", "metadata": {}},
    ]
