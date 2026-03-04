"""
File name: reranker.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module implements filtering and reranking documents
    returned by Similarity Search, using a LLM

Usage:
    Import this module into other scripts to use its functions.
    Example:
        import config

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

# Import traceback for better error logging
import traceback
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# integration with APM
from py_zipkin.zipkin import zipkin_span

from agent.agent_state import State
from agent.prompts import (
    RERANKER_TEMPLATE,
)
from core.oci_models import get_llm
from core.retry_utils import run_with_retry
from core.utils import get_console_logger, extract_json_from_text
from config import (
    DEBUG,
    AGENT_NAME,
    TOP_K,
    RERANKER_MODEL_ID,
    LLM_MAX_RETRIES,
    HYBRID_MIN_SESSION_DOCS,
)

logger = get_console_logger()


class Reranker(Runnable):
    """
    Implements a reranker using a LLM
    """

    def __init__(self):
        """
        Init
        """

    @staticmethod
    def _is_session_pdf_doc(doc: dict) -> bool:
        """Helper for is session pdf doc."""
        retrieval_type = (
            (doc.get("metadata") or {}).get("retrieval_type", "") or ""
        ).lower()
        return retrieval_type.startswith("session_pdf")

    @staticmethod
    def _doc_uid(doc: dict) -> tuple:
        """Helper for doc uid."""
        metadata = doc.get("metadata") or {}
        return (
            doc.get("page_content", ""),
            metadata.get("source", ""),
            metadata.get("page_label", ""),
            metadata.get("retrieval_type", ""),
        )

    def _enforce_hybrid_source_floors(
        self, reranked_docs: list, retriever_docs: list
    ) -> list:
        """
        Keep KB quota behavior unchanged (TOP_K) and ensure at least
        HYBRID_MIN_SESSION_DOCS session chunks are present in final context.
        """
        min_session = max(0, int(HYBRID_MIN_SESSION_DOCS))
        if min_session == 0:
            return reranked_docs

        session_count = sum(1 for d in reranked_docs if self._is_session_pdf_doc(d))
        if session_count >= min_session:
            return reranked_docs

        out = list(reranked_docs)
        seen = {self._doc_uid(d) for d in out}
        needed = min_session - session_count

        for doc in retriever_docs:
            if not self._is_session_pdf_doc(doc):
                continue
            uid = self._doc_uid(doc)
            if uid in seen:
                continue
            out.append(doc)
            seen.add(uid)
            needed -= 1
            if needed <= 0:
                break

        logger.info(
            "Reranker HYBRID floor applied: min_session=%d final_session=%d final_docs=%d",
            min_session,
            sum(1 for d in out if self._is_session_pdf_doc(d)),
            len(out),
        )
        return out

    def generate_refs(self, docs: list):
        """
        Returns a list of reference dictionaries used in the reranker.
        """
        return [
            {
                "source": (doc.get("metadata") or {}).get("source", "unknown"),
                "page": (doc.get("metadata") or {}).get("page_label", ""),
                "retrieval_type": (doc.get("metadata") or {}).get(
                    "retrieval_type", "semantic"
                ),
            }
            for doc in docs
        ]

    @staticmethod
    def get_reranked_docs(llm, query, retriever_docs):
        """
        Rerank documents using LLM based on user request.

        query: the search query (can be reformulated)
        retriever_docs: list of Langchain Documents
        """
        # Prepare chunk texts
        chunks = [doc["page_content"] for doc in retriever_docs]

        _prompt = PromptTemplate(
            input_variables=["query", "chunks"],
            template=RERANKER_TEMPLATE,
        ).format(query=query, chunks=chunks)

        messages = [HumanMessage(content=_prompt)]

        llm_response = run_with_retry(
            lambda: llm.invoke(messages),
            max_attempts=LLM_MAX_RETRIES,
            operation_name="Reranker LLM invoke",
        )

        if DEBUG:
            logger.info("Reranker LLM response: %s", llm_response)

        reranker_output = llm_response.content

        # Extract ranking order
        json_dict = extract_json_from_text(reranker_output)

        if DEBUG:
            logger.info(json_dict.get("ranked_chunks", "No ranked chunks found."))

        # Keep LLM-ranked order, validate indices against input size,
        # remove duplicates, then cap to TOP_K.
        valid_indexes = []
        seen = set()
        max_index = len(retriever_docs)

        for chunk in json_dict.get("ranked_chunks", []):
            idx = chunk.get("index")
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= max_index:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            valid_indexes.append(idx)

        return [retriever_docs[i] for i in valid_indexes[:TOP_K]]

    @zipkin_span(service_name=AGENT_NAME, span_name="reranking")
    def invoke(self, input: State, config=None, **kwargs):
        """
        Implements reranking logic.

        input: The agent state.
        """
        if DEBUG:
            logger.info("Reranker input state: %s", input)

        enable_reranker = config["configurable"]["enable_reranker"]

        user_request = input.get("standalone_question", "")
        retriever_docs = input.get("retriever_docs", [])
        error = None

        try:
            if retriever_docs:
                # there is something to rerank!
                if enable_reranker:
                    # do reranking
                    llm = get_llm(model_id=RERANKER_MODEL_ID, temperature=0.0)

                    reranked_docs = self.get_reranked_docs(
                        llm, user_request, retriever_docs
                    )

                else:
                    reranked_docs = retriever_docs
            else:
                reranked_docs = []

        except Exception as e:
            logger.error("Error in reranker: %s", e)
            # Log the full stack trace for debugging
            logger.debug(traceback.format_exc())
            error = str(e)
            # Fallback to original documents
            reranked_docs = retriever_docs

        intent = (input.get("search_intent") or "GLOBAL_KB").upper()
        if intent == "HYBRID" and reranked_docs:
            reranked_docs = self._enforce_hybrid_source_floors(
                reranked_docs=reranked_docs,
                retriever_docs=retriever_docs,
            )

        # Get reference citations
        citations = self.generate_refs(reranked_docs)

        return {"reranker_docs": reranked_docs, "citations": citations, "error": error}
