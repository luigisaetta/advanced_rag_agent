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

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from py_zipkin import Encoding
from py_zipkin.zipkin import zipkin_span

import config
from agent.post_answer_evaluator import PostAnswerEvaluator
from agent.rag_agent import State
from core.transport import http_transport
from core.utils import redact_agent_config_for_log
from ui.rendering import render_advanced_plan, render_answer, render_references
from ui.session import add_to_chat_history, get_chat_history

_POST_EVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="post-answer-eval"
)


def _build_agent_config(progress_callback):
    """Helper for build agent config."""
    return {
        "configurable": {
            "model_id": st.session_state.model_id,
            "embed_model_id": config.EMBED_MODEL_ID,
            "reranker_model_id": config.RERANKER_MODEL_ID,
            "top_k": config.TOP_K,
            "enable_reranker": st.session_state.enable_reranker,
            "enable_advanced_analysis": st.session_state.enable_advanced_analysis,
            # Default is false. SESSION_DOC sets this at runtime via classifier output.
            # This key allows explicit script/config control when needed.
            "advanced_analysis_session_only": False,
            "enable_tracing": st.session_state.enable_tracing,
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
            "advanced_analysis_risk_validation_kb_top_k": (
                config.ADVANCED_ANALYSIS_RISK_VALIDATION_KB_TOP_K
            ),
            "post_answer_evaluation_enabled": st.session_state.enable_post_answer_evaluation,
            "post_answer_evaluation_model_id": config.POST_ANSWER_EVALUATION_MODEL_ID,
            "post_answer_evaluation_max_chars": config.POST_ANSWER_EVALUATION_MAX_CHARS,
            "progress_callback": progress_callback,
        }
    }


def _has_hybrid_db_signal(docs: list) -> bool:
    """Return True when retrieval provenance indicates hybrid DB retrieval."""
    retrieval_types = {
        str(((doc.get("metadata") or {}).get("retrieval_type", "") or "")).lower()
        for doc in (docs or [])
    }
    return bool({"bm25", "semantic+bm25"} & retrieval_types)


def _run_post_answer_evaluation_if_needed(
    by_step: dict, question: str, final_answer: str, agent_config: dict, logger
) -> None:
    """
    Run post-answer evaluator after UI rendering (log-only), preserving streaming UX.
    """
    if not agent_config["configurable"].get("post_answer_evaluation_enabled", True):
        return

    intent_payload = by_step.get("IntentClassifier", {}) or {}
    has_session_pdf = bool(intent_payload.get("has_session_pdf", False))
    if has_session_pdf:
        return

    retriever_docs = []
    for key in ("HybridFlow", "HybridSearch", "Search"):
        payload = by_step.get(key, {}) or {}
        docs = payload.get("retriever_docs", [])
        if docs:
            retriever_docs = docs
            break
    if not retriever_docs or not _has_hybrid_db_signal(retriever_docs):
        return

    rerank_payload = by_step.get("Rerank", {}) or {}
    eval_input = State(
        user_request=question,
        chat_history=[],
        standalone_question=(by_step.get("QueryRewrite", {}) or {}).get(
            "standalone_question", question
        ),
        retriever_docs=copy.deepcopy(retriever_docs),
        reranker_docs=copy.deepcopy(rerank_payload.get("reranker_docs", [])),
        final_answer=final_answer,
        error=None,
    )
    eval_config = copy.deepcopy(agent_config)

    def _background_eval():
        """Run post-answer evaluator in a background worker."""
        try:
            PostAnswerEvaluator().invoke(eval_input, config=eval_config)
        except Exception as exc:
            logger.warning("Post-answer evaluation async failed: %s", exc)

    try:
        _POST_EVAL_EXECUTOR.submit(_background_eval)
    except RuntimeError as exc:
        logger.warning("Post-answer evaluation async submit failed: %s", exc)


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
        by_step = {}
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

        tracing_enabled = bool(agent_config["configurable"].get("enable_tracing", True))
        tracing_context = (
            zipkin_span(
                service_name=config.AGENT_NAME,
                span_name="stream",
                transport_handler=http_transport,
                encoding=Encoding.V2_JSON,
                sample_rate=100,
            )
            if tracing_enabled
            else nullcontext()
        )
        with tracing_context:
            for event in st.session_state.workflow.stream(
                input_state, config=agent_config
            ):
                for key, value in event.items():
                    msg = f"Completed: {key}!"
                    logger.info(msg)
                    st.toast(msg)
                    results.append(value)
                    by_step[key] = value
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
            answer_payload = None
            for payload in reversed(results):
                if "final_answer" in payload:
                    answer_payload = payload["final_answer"]
                    break
            if answer_payload is None:
                raise KeyError("final_answer")
            full_response = render_answer(answer_payload)
            _run_post_answer_evaluation_if_needed(
                by_step=by_step,
                question=question,
                final_answer=full_response,
                agent_config=agent_config,
                logger=logger,
            )
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
