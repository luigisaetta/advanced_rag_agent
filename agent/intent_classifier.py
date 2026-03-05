"""
File name: intent_classifier.py
Author: Luigi Saetta
Last modified: 01-03-2026
Python Version: 3.11

Description:
    This module classifies retrieval intent and routes the graph.
"""

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# integration with APM
from py_zipkin.zipkin import zipkin_span

from agent.agent_state import State
from agent.prompts import INTENT_CLASSIFIER_TEMPLATE, apply_prompt_profile
from core.oci_models import get_llm
from core.utils import get_console_logger, extract_json_from_text
from config import INTENT_MODEL_ID, AGENT_NAME

logger = get_console_logger()

ALLOWED_INTENTS = {"GLOBAL_KB", "SESSION_DOC", "HYBRID"}
INTENT_ALIASES = {
    "SESSION_PDF": "SESSION_DOC",
}


class IntentClassifier(Runnable):
    """
    Classify whether retrieval should use global KB, session PDF, or both.
    """

    @staticmethod
    def _has_session_pdf(config) -> bool:
        """Helper for has session pdf."""
        configurable = (config or {}).get("configurable", {})
        vs = configurable.get("session_pdf_vector_store")
        chunks_count = configurable.get("session_pdf_chunks_count", 0)
        return vs is not None and chunks_count > 0

    @staticmethod
    def _normalize_intent(raw_intent: str) -> str:
        """Helper for normalize intent."""
        if not isinstance(raw_intent, str):
            return "GLOBAL_KB"
        intent = raw_intent.strip().upper()
        intent = INTENT_ALIASES.get(intent, intent)
        if intent not in ALLOWED_INTENTS:
            return "GLOBAL_KB"
        return intent

    @zipkin_span(service_name=AGENT_NAME, span_name="intent_classifier")
    def invoke(self, input: State, config=None, **kwargs):
        """
        Classify request intent.
        - If no session PDF is available, skip LLM and force GLOBAL_KB.
        - If session PDF is available, call the dedicated model.
        """
        has_session_pdf = self._has_session_pdf(config)
        user_request = input.get("user_request", "")
        error = input.get("error")
        configurable = (config or {}).get("configurable", {})
        advanced_analysis_enabled = bool(
            configurable.get("enable_advanced_analysis", False)
        )

        if not has_session_pdf:
            logger.info(
                "IntentClassifier decision: GLOBAL_KB (no session PDF in memory)."
            )
            return {
                "search_intent": "GLOBAL_KB",
                "has_session_pdf": False,
                "advanced_analysis_enabled": advanced_analysis_enabled,
                "advanced_analysis_session_only": False,
                "error": error,
            }

        try:
            llm = get_llm(model_id=INTENT_MODEL_ID, temperature=0)
            prompt = PromptTemplate(
                input_variables=["user_request"],
                template=apply_prompt_profile(
                    INTENT_CLASSIFIER_TEMPLATE, config=config
                ),
            ).format(user_request=user_request)

            response = llm.invoke([HumanMessage(content=prompt)]).content
            parsed = extract_json_from_text(response)
            intent = self._normalize_intent(parsed.get("intent"))

            logger.info(
                "IntentClassifier decision: %s (session_pdf=%s).",
                intent,
                has_session_pdf,
            )

            return {
                "search_intent": intent,
                "has_session_pdf": True,
                "advanced_analysis_enabled": advanced_analysis_enabled,
                # SESSION_DOC requests are document-only by definition.
                # We propagate this as runtime setting for AdvancedAnalysisFlow
                # so planner/runner can enforce a strict "no KB" mode.
                "advanced_analysis_session_only": intent == "SESSION_DOC",
                "error": error,
            }
        except Exception as exc:
            logger.warning(
                "IntentClassifier failed, fallback to GLOBAL_KB: %s",
                exc,
            )
            return {
                "search_intent": "GLOBAL_KB",
                "has_session_pdf": True,
                "advanced_analysis_enabled": advanced_analysis_enabled,
                "advanced_analysis_session_only": False,
                "error": error,
            }
