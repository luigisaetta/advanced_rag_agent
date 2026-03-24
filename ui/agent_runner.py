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
from core.observability import (
    annotate_current_observation,
    flush_observability,
    get_current_trace_id,
    rename_current_observation,
    langfuse_span,
)

import config
from agent.post_answer_evaluation_agent import create_post_answer_evaluation_agent
from agent.rag_agent import State
from core.agent_config import build_agent_config
from core.utils import redact_agent_config_for_log
from ui.rendering import render_advanced_plan, render_answer, render_references
from ui.session import add_to_chat_history, get_chat_history

_POST_EVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="post-answer-eval"
)
_POST_ANSWER_EVALUATION_AGENT = create_post_answer_evaluation_agent()


def _build_agent_config(progress_callback):
    """Helper for build agent config."""
    return build_agent_config(
        model_id=st.session_state.model_id,
        collection_name=st.session_state.collection_name,
        thread_id=st.session_state.thread_id,
        enable_reranker=st.session_state.enable_reranker,
        enable_advanced_analysis=st.session_state.enable_advanced_analysis,
        enable_tracing=st.session_state.enable_tracing,
        prompt_profile=st.session_state.prompt_profile,
        main_language=config.MAIN_LANGUAGE,
        # Default is false. SESSION_DOC sets this at runtime via classifier output.
        # This key allows explicit script/config control when needed.
        advanced_analysis_session_only=False,
        session_pdf_vector_store=st.session_state.session_pdf_vector_store,
        session_pdf_chunks_count=st.session_state.session_pdf_chunks_count,
        session_pdf_docs=st.session_state.session_pdf_docs,
        # Runtime toggle from UI (default sourced from config).
        advanced_analysis_enable_risk_validation=st.session_state.enable_risk_validation,
        post_answer_evaluation_enabled=st.session_state.enable_post_answer_evaluation,
        post_answer_evaluation_model_id=config.POST_ANSWER_EVALUATION_MODEL_ID,
        progress_callback=progress_callback,
    )


def _has_hybrid_db_signal(docs: list) -> bool:
    """Return True when retrieval provenance indicates hybrid DB retrieval."""
    retrieval_types = {
        str(((doc.get("metadata") or {}).get("retrieval_type", "") or "")).lower()
        for doc in (docs or [])
    }
    return bool({"bm25", "semantic+bm25"} & retrieval_types)


def _run_post_answer_evaluation_if_needed(
    by_step: dict,
    question: str,
    final_answer: str,
    agent_config: dict,
    logger,
    parent_trace_id: str | None = None,
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
    if parent_trace_id:
        eval_config.setdefault("configurable", {})[
            "post_answer_target_trace_id"
        ] = parent_trace_id

    def _background_eval():
        """Run post-answer evaluator in a background worker."""
        try:
            _POST_ANSWER_EVALUATION_AGENT.invoke(eval_input, config=eval_config)
        except Exception as exc:
            logger.warning("Post-answer evaluation async failed: %s", exc)
        finally:
            flush_observability()

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
            langfuse_span(service_name=config.AGENT_NAME, span_name="rag_request")
            if tracing_enabled
            else nullcontext()
        )
        main_trace_id = None
        with tracing_context:
            annotate_current_observation(
                input_data={"question": question},
                metadata={
                    "thread_id": st.session_state.thread_id,
                    "model_id": st.session_state.model_id,
                    "collection_name": st.session_state.collection_name,
                },
            )
            main_trace_id = get_current_trace_id()
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
                    annotate_current_observation(
                        metadata={"last_completed_node": key, "node_error": error}
                    )

                    if key == "QueryRewrite":
                        st.sidebar.header("Standalone question:")
                        st.sidebar.write(value["standalone_question"])
                    if key == "IntentClassifier":
                        intent_name = str(value.get("search_intent", "") or "").lower()
                        if intent_name:
                            rename_current_observation(f"rag_{intent_name}")
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
            annotate_current_observation(
                output_data={"final_answer": full_response},
                metadata={"workflow_status": "completed"},
            )
            _run_post_answer_evaluation_if_needed(
                by_step=by_step,
                question=question,
                final_answer=full_response,
                agent_config=agent_config,
                logger=logger,
                parent_trace_id=main_trace_id,
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
            annotate_current_observation(
                metadata={"workflow_status": "failed"},
                level="ERROR",
                status_message=str(error),
            )
            st.error(error)
            if (
                st.session_state.enable_advanced_analysis
                and advanced_status_slot is not None
            ):
                advanced_status_slot.warning(
                    "Advanced Analysis: interrupted due to error"
                )
        flush_observability()

        add_to_chat_history(HumanMessage(content=question))
        if full_response:
            add_to_chat_history(AIMessage(content=full_response))
