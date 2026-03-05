"""
File name: rendering.py
Author: Luigi Saetta
Last modified: 03-03-2026
Python Version: 3.11

Description:
    This module defines rendering helpers for Streamlit chat and sidebar UI elements.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from ui.rendering import render_answer

License:
    This code is released under the MIT License.
"""

import re
from typing import List, Union

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from core.citation_utils import build_citation_url, parse_page_number

USER = "user"
ASSISTANT = "assistant"


def display_msg_on_rerun(chat_hist: List[Union[HumanMessage, AIMessage]]) -> None:
    """Display all messages on rerun."""
    for msg in chat_hist:
        role = USER if isinstance(msg, HumanMessage) else ASSISTANT
        with st.chat_message(role):
            if role == ASSISTANT:
                st.markdown(
                    _normalize_markdown_text(msg.content), unsafe_allow_html=True
                )
            else:
                st.markdown(msg.content)


def _normalize_markdown_text(text: str) -> str:
    """
    Convert common HTML line-break tags to markdown-friendly new lines,
    preserving markdown table rows where <br> is often intentional.
    """
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    normalized_lines = []
    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            # Keep inline breaks inside markdown table cells.
            normalized_lines.append(re.sub(r"(?i)<br\s*/?>", "<br/>", line))
        else:
            normalized_lines.append(re.sub(r"(?i)<br\s*/?>", "\n", line))
    return "".join(normalized_lines)


def render_references(citations: list) -> None:
    """Render references in sidebar including citation links when available."""
    if not citations:
        st.sidebar.write("No references available.")
        return

    def _render_reference_line(ref: dict) -> None:
        """Helper for render reference line."""
        source = ref.get("source", "unknown")
        page = ref.get("page", "")
        retrieval_type = ref.get("retrieval_type", "semantic")
        is_session_pdf_ref = str(retrieval_type).startswith("session_pdf")
        page_number = parse_page_number(page)

        if is_session_pdf_ref:
            st.markdown(
                f'{{"source": "{source}", "page": "{page}", "retrieval_type": "{retrieval_type}"}}'
            )
        elif page_number is not None:
            url = build_citation_url(source, page_number)
            st.markdown(
                f'{{"source": "{source}", "page": "{page}", "retrieval_type": "{retrieval_type}", "link": [{url}]({url})}}'
            )
        else:
            st.markdown(
                f'{{"source": "{source}", "page": "{page}", "retrieval_type": "{retrieval_type}", "link": ""}}'
            )

    with st.sidebar.expander("Show references", expanded=False):
        citations_with_step = [c for c in citations if c.get("step") is not None]
        if citations_with_step and len(citations_with_step) == len(citations):
            ordered_steps = sorted(
                {int(c.get("step")) for c in citations if str(c.get("step")).isdigit()}
            )
            for step_no in ordered_steps:
                st.markdown(f"**Step {step_no}**")
                for ref in citations:
                    ref_step = ref.get("step")
                    if str(ref_step).isdigit() and int(ref_step) == step_no:
                        _render_reference_line(ref)
        else:
            for ref in citations:
                _render_reference_line(ref)


def render_advanced_plan(plan: list) -> None:
    """Render the advanced-analysis plan in a collapsible sidebar section."""
    with st.sidebar.expander("Show advanced plan", expanded=False):
        if not plan:
            st.write("No plan available.")
            return
        for step in plan:
            step_no = step.get("step", "?")
            section = step.get("section", "unknown section")
            objective = step.get("objective", "")
            chunk_numbers = step.get("chunk_numbers", [])
            kb_needed = bool(step.get("kb_search_needed", False))
            kb_query = step.get("kb_query", "")

            st.markdown(f"**Step {step_no} - {section}**")
            if objective:
                st.write(f"Objective: {objective}")
            st.write(f"Chunk numbers: {chunk_numbers}")
            st.write(f"KB search needed: {kb_needed}")
            if kb_query:
                st.write(f"KB query: {kb_query}")
            st.divider()


def render_answer(answer_payload) -> str:
    """Render streamed/final answer payload and return complete text."""
    full_response = ""
    with st.chat_message(ASSISTANT):
        response_container = st.empty()
        if isinstance(answer_payload, str):
            full_response = answer_payload
            response_container.markdown(
                _normalize_markdown_text(full_response),
                unsafe_allow_html=True,
            )
        else:
            for chunk in answer_payload:
                full_response += chunk.content
                response_container.markdown(
                    _normalize_markdown_text(full_response) + "▌",
                    unsafe_allow_html=True,
                )
            response_container.markdown(
                _normalize_markdown_text(full_response),
                unsafe_allow_html=True,
            )
    return _normalize_markdown_text(full_response)
