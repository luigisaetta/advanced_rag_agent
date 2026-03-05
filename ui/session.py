"""
File name: session.py
Author: Luigi Saetta
Last modified: 03-03-2026
Python Version: 3.11

Description:
    This module defines session-state helpers for the Streamlit assistant.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from ui.session import init_session_state

License:
    This code is released under the MIT License.
"""

import uuid

import streamlit as st

import config
from agent.rag_agent import create_workflow


def init_session_state() -> None:
    """Initialize Streamlit session keys used by the app."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_workflow()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "model_id" not in st.session_state:
        st.session_state.model_id = (
            config.LLM_MODEL_ID
            if config.LLM_MODEL_ID in config.MODEL_LIST
            else config.MODEL_LIST[0]
        )
    if "enable_reranker" not in st.session_state:
        st.session_state.enable_reranker = True
    if "enable_advanced_analysis" not in st.session_state:
        st.session_state.enable_advanced_analysis = False
    if "enable_risk_validation" not in st.session_state:
        st.session_state.enable_risk_validation = (
            config.ADVANCED_ANALYSIS_ENABLE_RISK_VALIDATION
        )
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = config.COLLECTION_LIST[0]
    if "prompt_profile" not in st.session_state:
        st.session_state.prompt_profile = (
            config.PROMPT_PROFILE
            if config.PROMPT_PROFILE in config.PROMPT_PROFILES
            else "general"
        )
    if "get_feedback" not in st.session_state:
        st.session_state.get_feedback = False
    if "session_pdf_name" not in st.session_state:
        st.session_state.session_pdf_name = ""
    if "session_pdf_chunks_count" not in st.session_state:
        st.session_state.session_pdf_chunks_count = 0
    if "session_pdf_vector_store" not in st.session_state:
        st.session_state.session_pdf_vector_store = None
    if "session_pdf_docs" not in st.session_state:
        st.session_state.session_pdf_docs = []


def reset_conversation() -> None:
    """Reset conversation and in-memory PDF state."""
    st.session_state.chat_history = []
    st.session_state.session_pdf_name = ""
    st.session_state.session_pdf_chunks_count = 0
    st.session_state.session_pdf_vector_store = None
    st.session_state.session_pdf_docs = []
    st.session_state.thread_id = str(uuid.uuid4())


def add_to_chat_history(msg) -> None:
    """Append a message to chat history."""
    st.session_state.chat_history.append(msg)


def get_chat_history():
    """Return chat history (optionally trimmed)."""
    if config.MAX_MSGS_IN_HISTORY > 0:
        return st.session_state.chat_history[-config.MAX_MSGS_IN_HISTORY :]
    return st.session_state.chat_history
