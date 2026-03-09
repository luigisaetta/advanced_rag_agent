"""
File name: sidebar.py
Author: Luigi Saetta
Last modified: 03-03-2026
Python Version: 3.11

Description:
    This module defines sidebar controls and in-session PDF processing for the UI.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from ui.sidebar import render_sidebar

License:
    This code is released under the MIT License.
"""

import os
import tempfile

import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore

import config
from prompt_profiles import PROMPT_PROFILES
from core.oci_models import get_embedding_model
from core.utils import docs_serializable
from core.session_pdf_vlm import scan_pdf_to_docs_with_vlm


def render_sidebar(reset_callback):
    """Render sidebar controls and return PDF actions."""
    if st.sidebar.button("Clear Chat History"):
        reset_callback()

    with st.sidebar.expander("Options", expanded=False):
        st.text_input(label="LLM Region", value=config.LLM_REGION, disabled=True)
        st.text_input(label="Embed Region", value=config.EMBED_REGION, disabled=True)

        st.session_state.model_id = st.selectbox(
            "LLM model",
            config.MODEL_LIST,
            index=(
                config.MODEL_LIST.index(st.session_state.model_id)
                if st.session_state.model_id in config.MODEL_LIST
                else 0
            ),
        )

        st.text_input(label="Embed Model", value=config.EMBED_MODEL_ID, disabled=True)
        st.text_input(label="Session PDF VLM", value=config.VLM_MODEL_ID, disabled=True)
        st.session_state.collection_name = st.selectbox(
            "Collection name",
            config.COLLECTION_LIST,
        )
        profile_keys = list(PROMPT_PROFILES.keys())
        st.session_state.prompt_profile = st.selectbox(
            "Prompt profile",
            profile_keys,
            index=(
                profile_keys.index(st.session_state.prompt_profile)
                if st.session_state.prompt_profile in profile_keys
                else 0
            ),
            format_func=lambda key: (f"{PROMPT_PROFILES[key]['name']} ({key})"),
            help=(
                "Select the domain framing used across prompts "
                "(classification, retrieval guidance, answer generation)."
            ),
        )

        st.session_state.enable_reranker = st.checkbox(
            "Enable Reranker", value=True, disabled=False
        )
        config.ENABLE_TRACING = st.checkbox(
            "Enable tracing", value=False, disabled=False
        )
        st.session_state.enable_post_answer_evaluation = st.checkbox(
            "Enable Post Answer Evaluation",
            value=st.session_state.enable_post_answer_evaluation,
            disabled=False,
        )

    st.session_state.enable_advanced_analysis = st.sidebar.checkbox(
        "Advanced Analysis", value=False, disabled=False
    )
    st.session_state.enable_risk_validation = st.sidebar.checkbox(
        "Risk Validation",
        value=st.session_state.enable_risk_validation,
        disabled=not st.session_state.enable_advanced_analysis,
        help=(
            "Run an additional post-synthesis validation step. "
            "If critical negative findings are detected, it may query KB (unless session-only mode is active)."
        ),
    )

    st.sidebar.header("Session PDF (in-memory)")
    session_pdf = st.sidebar.file_uploader(
        "Upload a PDF for this session",
        type=["pdf"],
        key="session_pdf_upload",
    )

    scan_requested = st.sidebar.button("Scan PDF in memory")

    if st.session_state.session_pdf_name:
        st.sidebar.caption(
            "In-memory PDF: "
            f"{st.session_state.session_pdf_name} "
            f"({st.session_state.session_pdf_chunks_count} chunks)"
        )

    return session_pdf, scan_requested


def scan_pdf_and_store_in_session(session_pdf, logger) -> None:
    """Scan uploaded PDF and store chunks + vector store in session state."""
    if session_pdf is None:
        st.sidebar.warning("Please upload a PDF first.")
        return

    status_slot = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    tmp_path = ""

    try:
        status_slot.info("Saving uploaded PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(session_pdf.read())
            tmp_path = tmp_file.name
        progress_bar.progress(20)

        status_slot.info("Scanning PDF pages with VLM...")

        def _on_page_progress(current_page: int, total_pages: int):
            """Helper for on page progress."""
            if total_pages > 0:
                pct = 25 + int((current_page / total_pages) * 70)
                progress_bar.progress(min(pct, 95))

        docs, page_count = scan_pdf_to_docs_with_vlm(
            pdf_path=tmp_path,
            vlm_model_id=config.VLM_MODEL_ID,
            max_pages=config.SESSION_PDF_MAX_PAGES,
            source_name=session_pdf.name,
            on_progress=_on_page_progress,
        )

        status_slot.info("Building in-memory vector index...")
        progress_bar.progress(97)
        embed_model = get_embedding_model()
        session_vs = InMemoryVectorStore(embedding=embed_model)
        if docs:
            session_vs.add_documents(docs)

        st.session_state.session_pdf_vector_store = session_vs
        st.session_state.session_pdf_name = session_pdf.name
        st.session_state.session_pdf_chunks_count = len(docs)
        st.session_state.session_pdf_docs = docs_serializable(docs)

        progress_bar.progress(100)
        status_slot.success(
            f"Loaded '{session_pdf.name}' in memory ({page_count} pages, {len(docs)} chunks)."
        )
    except Exception as exc:
        logger.error("Error in session PDF scan: %s", exc)
        status_slot.error("PDF scan failed.")
        st.sidebar.error(str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
