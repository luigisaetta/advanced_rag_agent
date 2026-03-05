"""
File name: agent_runner.py
Author: Luigi Saetta
Last modified: 03-03-2026
Python Version: 3.11

Description:
    This module manages agent execution and response streaming for the Streamlit UI.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from ui.agent_runner import handle_question

License:
    This code is released under the MIT License.
"""

import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from py_zipkin import Encoding
from py_zipkin.zipkin import zipkin_span

import config
from agent.rag_agent import State
from core.transport import http_transport
from core.utils import redact_agent_config_for_log
from ui.rendering import render_advanced_plan, render_answer, render_references
from ui.session import add_to_chat_history, get_chat_history


def _build_agent_config(progress_callback):
    """Helper for build agent config."""
    return {
        "configurable": {
            "model_id": st.session_state.model_id,
            "enable_reranker": st.session_state.enable_reranker,
            "enable_advanced_analysis": st.session_state.enable_advanced_analysis,
            # Default is false. SESSION_DOC sets this at runtime via classifier output.
            # This key allows explicit script/config control when needed.
            "advanced_analysis_session_only": False,
            "enable_tracing": config.ENABLE_TRACING,
            "main_language": config.MAIN_LANGUAGE,
            "prompt_profile": st.session_state.prompt_profile,
            "collection_name": st.session_state.collection_name,
            "thread_id": st.session_state.thread_id,
            "session_pdf_vector_store": st.session_state.session_pdf_vector_store,
            "session_pdf_chunks_count": st.session_state.session_pdf_chunks_count,
            "session_pdf_docs": st.session_state.session_pdf_docs,
            "advanced_analysis_max_actions": config.ADVANCED_ANALYSIS_MAX_ACTIONS,
            "advanced_analysis_kb_top_k": config.ADVANCED_ANALYSIS_KB_TOP_K,
            "advanced_analysis_step_max_words": config.ADVANCED_ANALYSIS_STEP_MAX_WORDS,
            # Runtime toggle from UI (default sourced from config).
            "advanced_analysis_enable_risk_validation": st.session_state.enable_risk_validation,
            "advanced_analysis_risk_validation_kb_top_k": config.ADVANCED_ANALYSIS_RISK_VALIDATION_KB_TOP_K,
            "progress_callback": progress_callback,
        }
    }


def handle_question(question: str, logger) -> None:
    """Handle question submit, stream workflow, and render answer."""
    st.chat_message("user").markdown(question)

    with st.spinner("Calling AI..."):
        time_start = time.time()
        input_state = State(
            user_request=question,
            chat_history=get_chat_history(),
            error=None,
        )

        results = []
        error = None
        full_response = ""

        advanced_status_slot = None
        advanced_progress = None
        progress_callback = None
        if st.session_state.enable_advanced_analysis:
            advanced_status_slot = st.sidebar.empty()
            advanced_progress = st.sidebar.progress(0)

            def _on_advanced_progress(percent: int, message: str):
                """Helper for on advanced progress."""
                advanced_progress.progress(max(0, min(100, int(percent))))
                advanced_status_slot.info(f"Advanced Analysis: {message}")

            progress_callback = _on_advanced_progress

        agent_config = _build_agent_config(progress_callback)
        logger.info("")
        logger.info("Agent config: %s", redact_agent_config_for_log(agent_config))
        logger.info("")

        with zipkin_span(
            service_name=config.AGENT_NAME,
            span_name="stream",
            transport_handler=http_transport,
            encoding=Encoding.V2_JSON,
            sample_rate=100,
        ):
            for event in st.session_state.workflow.stream(
                input_state, config=agent_config
            ):
                for key, value in event.items():
                    msg = f"Completed: {key}!"
                    logger.info(msg)
                    st.toast(msg)
                    results.append(value)
                    error = value["error"]

                    if key == "QueryRewrite":
                        st.sidebar.header("Standalone question:")
                        st.sidebar.write(value["standalone_question"])
                    if key == "IntentClassifier":
                        logger.info(
                            "Intent decision: %s (has_session_pdf=%s)",
                            value.get("search_intent"),
                            value.get("has_session_pdf"),
                        )
                    if key == "Rerank":
                        st.sidebar.header("References:")
                        render_references(value["citations"])
                    if key == "AdvancedAnalysisFlow":
                        st.sidebar.header("References:")
                        render_references(value.get("citations", []))
                        # Show planner output for explainability/debugging.
                        # The subagent returns the normalized plan in the node payload.
                        render_advanced_plan(value.get("advanced_plan", []))

        if error is None:
            answer_payload = results[-1]["final_answer"]
            full_response = render_answer(answer_payload)
            elapsed_time = round((time.time() - time_start), 1)
            logger.info("Elapsed time: %s sec.", elapsed_time)
            logger.info("")

            if (
                st.session_state.enable_advanced_analysis
                and advanced_progress is not None
                and advanced_status_slot is not None
            ):
                advanced_progress.progress(100)
                advanced_status_slot.success("Advanced Analysis: completed")

            if config.ENABLE_USER_FEEDBACK:
                st.session_state.get_feedback = True
        else:
            st.error(error)
            if (
                st.session_state.enable_advanced_analysis
                and advanced_status_slot is not None
            ):
                advanced_status_slot.warning(
                    "Advanced Analysis: interrupted due to error"
                )

        add_to_chat_history(HumanMessage(content=question))
        if full_response:
            add_to_chat_history(AIMessage(content=full_response))
