"""
Shared builder for LangGraph runtime config payloads.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import config


def build_agent_config(
    *,
    model_id: str,
    collection_name: str,
    thread_id: str,
    enable_reranker: bool,
    enable_advanced_analysis: bool,
    enable_tracing: bool,
    prompt_profile: str | None = None,
    main_language: str | None = None,
    advanced_analysis_session_only: bool = False,
    session_pdf_vector_store: Any = None,
    session_pdf_chunks_count: int = 0,
    session_pdf_docs: Sequence[dict] | None = None,
    advanced_analysis_enable_risk_validation: bool | None = None,
    post_answer_evaluation_enabled: bool = True,
    post_answer_evaluation_model_id: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    extra_configurable: Mapping[str, Any] | None = None,
) -> dict:
    """
    Build the `config={"configurable": ...}` payload used by workflow execution.
    """
    configurable = {
        "model_id": model_id,
        "embed_model_id": config.EMBED_MODEL_ID,
        "reranker_model_id": config.RERANKER_MODEL_ID,
        "top_k": config.TOP_K,
        "enable_reranker": bool(enable_reranker),
        "enable_advanced_analysis": bool(enable_advanced_analysis),
        "advanced_analysis_session_only": bool(advanced_analysis_session_only),
        "enable_tracing": bool(enable_tracing),
        "main_language": (
            str(main_language)
            if main_language is not None
            else str(config.MAIN_LANGUAGE)
        ),
        "prompt_profile": (
            str(prompt_profile) if prompt_profile is not None else config.PROMPT_PROFILE
        ),
        "collection_name": collection_name,
        "thread_id": thread_id,
        "session_pdf_vector_store": session_pdf_vector_store,
        "session_pdf_chunks_count": int(session_pdf_chunks_count),
        "session_pdf_docs": list(session_pdf_docs or []),
        "advanced_analysis_max_actions": config.ADVANCED_ANALYSIS_MAX_ACTIONS,
        "advanced_analysis_kb_top_k": config.ADVANCED_ANALYSIS_KB_TOP_K,
        "advanced_analysis_step_max_words": config.ADVANCED_ANALYSIS_STEP_MAX_WORDS,
        "advanced_analysis_enable_risk_validation": (
            bool(advanced_analysis_enable_risk_validation)
            if advanced_analysis_enable_risk_validation is not None
            else bool(config.ADVANCED_ANALYSIS_ENABLE_RISK_VALIDATION)
        ),
        "advanced_analysis_risk_validation_kb_top_k": (
            config.ADVANCED_ANALYSIS_RISK_VALIDATION_KB_TOP_K
        ),
        "post_answer_evaluation_enabled": bool(post_answer_evaluation_enabled),
        "post_answer_evaluation_model_id": (
            str(post_answer_evaluation_model_id)
            if post_answer_evaluation_model_id is not None
            else config.POST_ANSWER_EVALUATION_MODEL_ID
        ),
        "post_answer_evaluation_max_chars": config.POST_ANSWER_EVALUATION_MAX_CHARS,
        "progress_callback": progress_callback,
    }

    if extra_configurable:
        configurable.update(dict(extra_configurable))

    return {"configurable": configurable}
