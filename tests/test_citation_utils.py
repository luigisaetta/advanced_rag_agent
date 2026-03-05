"""
File name: tests/test_citation_utils.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Unit tests for citation URL and page number parsing helpers.
"""

import config
from core import citation_utils


def test_parse_page_number_from_int():
    """Test test parse page number from int."""
    assert citation_utils.parse_page_number(7) == 7


def test_parse_page_number_from_string():
    """Test test parse page number from string."""
    assert citation_utils.parse_page_number("page 12") == 12


def test_parse_page_number_invalid():
    """Test test parse page number invalid."""
    assert citation_utils.parse_page_number("no page") is None


def test_build_citation_url_uses_base_url_and_padding(monkeypatch):
    """Test test build citation url uses base url and padding."""
    monkeypatch.setattr(config, "CITATION_BASE_URL", "http://127.0.0.1:8008")

    url = citation_utils.build_citation_url("mydoc.pdf", 3)

    assert url == "http://127.0.0.1:8008/mydoc/page0003.png"


def test_build_citation_url_strips_pdf_case_insensitive(monkeypatch):
    """Test test build citation url strips pdf case insensitive."""
    monkeypatch.setattr(config, "CITATION_BASE_URL", "http://localhost:9999/")

    url = citation_utils.build_citation_url("MyBook.PDF", 42)

    assert url == "http://localhost:9999/MyBook/page0042.png"


def test_build_citation_url_encodes_document_name(monkeypatch):
    """Test test build citation url encodes document name."""
    monkeypatch.setattr(config, "CITATION_BASE_URL", "http://localhost:8008/")

    url = citation_utils.build_citation_url("doc name v1.pdf", 1)

    assert url == "http://localhost:8008/doc%20name%20v1/page0001.png"
