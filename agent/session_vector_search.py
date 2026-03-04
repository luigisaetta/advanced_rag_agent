"""
File name: session_vector_search.py
Author: Luigi Saetta
Last modified: 01-03-2026
Python Version: 3.11

Description:
    This module performs semantic retrieval only on the in-memory
    session PDF vector store.
"""

from langchain_core.runnables import Runnable

from agent.agent_state import State
from core.utils import get_console_logger, docs_serializable
from config import DEBUG, TOP_K

logger = get_console_logger()


class SessionVectorSearch(Runnable):
    """
    Retrieve chunks only from the in-memory session PDF vector store.
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
                "SessionVectorSearch: session vector store is missing, returning empty docs."
            )
            return {"retriever_docs": [], "error": error}

        try:
            relevant_docs = session_vs.similarity_search(
                query=standalone_question, k=TOP_K
            )
            for doc in relevant_docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_type"] = "session_pdf"

            if DEBUG:
                logger.info(
                    "SessionVectorSearch: found %d chunks from in-memory pdf.",
                    len(relevant_docs),
                )
        except Exception as exc:
            logger.error("Error in SessionVectorSearch: %s", exc)
            error = str(exc)

        return {"retriever_docs": docs_serializable(relevant_docs), "error": error}
