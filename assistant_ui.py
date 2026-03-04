"""
File name: assistant_ui.py
Author: Luigi Saetta
Date created: 2024-12-04
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module provides the UI for the RAG demo

Usage:
    streamlit run assistant_ui.py

License:
    This code is released under the MIT License.

Notes:
    This is part of a  demo for a RAG solution implemented
    using LangGraph

Warnings:
    This module is in development, may change in future versions.
"""

import streamlit as st

from core.utils import get_console_logger
from ui.agent_runner import handle_question
from ui.feedback import register_feedback
from ui.rendering import display_msg_on_rerun
from ui.session import get_chat_history, init_session_state, reset_conversation
from ui.sidebar import render_sidebar, scan_pdf_and_store_in_session


st.set_page_config(page_title="assistant ui", layout="wide")
st.markdown(
    """
    <style>
    .stApp .block-container {
        max-width: 98%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logger = get_console_logger()


init_session_state()

st.title("OCI Enterprise Knowledge Assistant")

session_pdf, scan_requested = render_sidebar(reset_callback=reset_conversation)
if scan_requested:
    scan_pdf_and_store_in_session(session_pdf, logger)

# Display chat messages from history on app rerun
display_msg_on_rerun(get_chat_history())

if question := st.chat_input("Hello, how can I help you?"):
    try:
        handle_question(question, logger)

        if st.session_state.get_feedback:
            st.feedback(
                "stars",
                key="feedback",
                on_change=lambda: register_feedback(logger),
            )
    except Exception as exc:
        err_msg = "An error occurred: " + str(exc)
        logger.error(err_msg)
        st.error(err_msg)
