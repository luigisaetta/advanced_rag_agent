"""
File name: hybrid_session_search.py
Author: Luigi Saetta
Last modified: 02-03-2026
Python Version: 3.11

Description:
    This module performs session-PDF retrieval for the HYBRID subgraph.
    It stores session candidates in a dedicated state key.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.hybrid_session_search import HybridSessionSearch

License:
    This code is released under the MIT License.
"""

from langchain_core.runnables import Runnable

from agent.agent_state import State
from core.utils import get_console_logger, docs_serializable
from config import DEBUG, HYBRID_SESSION_TOP_K

logger = get_console_logger()


class HybridSessionSearch(Runnable):
    """
    Retrieve chunks from in-memory session PDF for HYBRID branch.
    Writes them into `session_retriever_docs`.
    """

    def invoke(self, input: State, config=None, **kwargs):
        """Invoke."""
        standalone_question = input.get("standalone_question", "")
        error = input.get("error")
        relevant_docs = []

        configurable = (config or {}).get("configurable", {})
        session_vs = configurable.get("session_pdf_vector_store")

        if session_vs is None:
            logger.warning(
                "HybridSessionSearch: session vector store is missing, returning empty docs."
            )
            return {"session_retriever_docs": [], "error": error}

        try:
            relevant_docs = session_vs.similarity_search(
                query=standalone_question,
                k=HYBRID_SESSION_TOP_K,
            )
            for doc in relevant_docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_type"] = "session_pdf"

            if DEBUG:
                logger.info(
                    "HybridSessionSearch: found %d session chunks.",
                    len(relevant_docs),
                )
        except Exception as exc:
            logger.error("Error in HybridSessionSearch: %s", exc)
            error = str(exc)

        return {
            "session_retriever_docs": docs_serializable(relevant_docs),
            "error": error,
        }
