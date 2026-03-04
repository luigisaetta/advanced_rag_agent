"""
File name: core/citation_utils.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Helpers to parse page numbers and build consistent citation image URLs.
"""

import re
from urllib.parse import quote

import config


def _citation_base_url() -> str:
    """Helper for citation base url."""
    base = config.CITATION_BASE_URL.strip()
    if not base.endswith("/"):
        base = f"{base}/"
    return base


def parse_page_number(raw) -> int | None:
    """
    Parse page values like 4, "4", "page 4", "p.4".
    """
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, float):
        val = int(raw)
        return val if val >= 0 else None
    if isinstance(raw, str):
        m = re.search(r"\d+", raw)
        if not m:
            return None
        return int(m.group(0))
    return None


def build_citation_url(document_name: str, page_number: int) -> str:
    """
    Build URL in the form:
    <CITATION_BASE_URL>/<ENCODED_DOC_NAME_NO_PDF>/pageNNNN.png
    """
    if not document_name:
        return ""
    doc_no_pdf = re.sub(r"\.pdf$", "", document_name, flags=re.IGNORECASE)
    encoded_doc = quote(doc_no_pdf, safe="")
    return f"{_citation_base_url()}{encoded_doc}/page{page_number:04d}.png"
