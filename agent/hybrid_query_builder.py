"""
File name: hybrid_query_builder.py
Author: Luigi Saetta
Last modified: 02-03-2026
Python Version: 3.11

Description:
    This module builds a KB-focused query for the HYBRID flow,
    enriching the user request with session-PDF evidence.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.hybrid_query_builder import HybridQueryBuilder

License:
    This code is released under the MIT License.
"""

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from py_zipkin.zipkin import zipkin_span

from agent.agent_state import State
from agent.prompts import HYBRID_KB_QUERY_TEMPLATE
from core.oci_models import get_llm
from core.utils import get_console_logger
from config import (
    AGENT_NAME,
    HYBRID_QUERY_EXPANSION_MAX_CHARS,
    HYBRID_QUERY_EXPANSION_TOP_K,
)

logger = get_console_logger()


class HybridQueryBuilder(Runnable):
    """
    For HYBRID intent, build a KB query enriched with uploaded-document facts.
    """

    @staticmethod
    def _format_session_snippets(docs: list, max_chars: int) -> str:
        """Helper for format session snippets."""
        snippets = []
        used = 0
        for idx, doc in enumerate(docs):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            metadata = doc.metadata or {}
            source = metadata.get("source", "uploaded.pdf")
            page = metadata.get("page_label", "")
            block = f"[{idx + 1}] source={source} page={page}\n{text}"
            if used + len(block) > max_chars:
                break
            snippets.append(block)
            used += len(block)
        return "\n\n".join(snippets)

    @zipkin_span(service_name=AGENT_NAME, span_name="hybrid_query_builder")
    def invoke(self, input: State, config=None, **kwargs):
        """Invoke."""
        intent = (input.get("search_intent") or "GLOBAL_KB").upper()
        error = input.get("error")
        standalone_question = input.get("standalone_question", "")
        kb_query = standalone_question

        if intent != "HYBRID":
            return {"kb_query": kb_query, "error": error}

        configurable = (config or {}).get("configurable", {})
        session_vs = configurable.get("session_pdf_vector_store")
        model_id = configurable.get("model_id")

        if session_vs is None or not standalone_question:
            return {"kb_query": kb_query, "error": error}

        try:
            session_docs = session_vs.similarity_search(
                query=standalone_question,
                k=HYBRID_QUERY_EXPANSION_TOP_K,
            )
            session_snippets = self._format_session_snippets(
                session_docs,
                max_chars=HYBRID_QUERY_EXPANSION_MAX_CHARS,
            )
            if not session_snippets:
                return {"kb_query": kb_query, "error": error}

            llm = get_llm(model_id=model_id, temperature=0.0)
            prompt = PromptTemplate(
                input_variables=[
                    "user_request",
                    "standalone_question",
                    "session_snippets",
                ],
                template=HYBRID_KB_QUERY_TEMPLATE,
            ).format(
                user_request=input.get("user_request", ""),
                standalone_question=standalone_question,
                session_snippets=session_snippets,
            )

            response = llm.invoke([HumanMessage(content=prompt)]).content
            candidate = (response or "").strip()
            if candidate:
                kb_query = candidate

            logger.info(
                "HybridQueryBuilder built KB query. standalone_len=%d kb_query_len=%d snippets=%d",
                len(standalone_question),
                len(kb_query),
                len(session_docs),
            )
        except Exception as exc:
            logger.warning(
                "HybridQueryBuilder failed, fallback to standalone question: %s",
                exc,
            )

        return {"kb_query": kb_query, "error": error}
