"""
File name: vector_search.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module implements the Semantic Sesarch in the agent
    using 23Ai Vector Search


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

import oracledb
from langchain_core.runnables import Runnable

# observability decorators
from core.observability import annotate_current_observation, langfuse_span

from agent.agent_state import State
from core.oci_models import get_embedding_model, get_oracle_vs
from core.utils import get_console_logger, docs_serializable

from config import AGENT_NAME, DEBUG, TOP_K

from config_private import CONNECT_ARGS

logger = get_console_logger()


class SemanticSearch(Runnable):
    """
    Implements Semantic Search for the Agent
    """

    def __init__(self):
        """
        Init
        """

    def get_connection(self):
        """
        get a connection to the DB
        """
        return oracledb.connect(**CONNECT_ARGS)

    @langfuse_span(service_name=AGENT_NAME, span_name="similarity_search")
    def invoke(self, input: State, config=None, **kwargs):
        """
        This method invokes the vector search

        input: the agent state
        """
        collection_name = config["configurable"]["collection_name"]
        relevant_docs = []
        error = None

        standalone_question = input["standalone_question"]
        search_query = input.get("kb_query") or standalone_question

        if DEBUG:
            logger.info("Search question: %s", search_query)

        try:
            embed_model = get_embedding_model()

            # get a connection to the DB and init VS
            with self.get_connection() as conn:

                v_store = get_oracle_vs(
                    conn=conn,
                    collection_name=collection_name,
                    embed_model=embed_model,
                )

                relevant_docs = v_store.similarity_search(query=search_query, k=TOP_K)

            # annotate retrieval origin to support hybrid observability in references
            for doc in relevant_docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_type"] = "semantic"

            if DEBUG:
                logger.info("Result from similarity search:")
                logger.info(relevant_docs)

        except Exception as e:
            logger.error("Error in vector_store.invoke: %s", e)
            error = str(e)

        # docs_serializable(relevant_docs)
        # convert the documents to a serializable format
        # to support the API
        annotate_current_observation(
            metadata={
                "collection_name": collection_name,
                "docs_count": len(relevant_docs),
                "error": error,
            },
            input_data={"search_query_len": len(search_query or "")},
        )
        return {"retriever_docs": docs_serializable(relevant_docs), "error": error}

    #
    #  Helper functions
    #
    def add_documents(self, docs: list, collection_name: str):
        """
        Add the chunks to an existing collection

        docs is a list of Langchain documents
        """
        try:
            embed_model = get_embedding_model()

            with self.get_connection() as conn:
                v_store = get_oracle_vs(
                    conn=conn,
                    collection_name=collection_name,
                    embed_model=embed_model,
                )
                v_store.add_documents(docs)
                logger.info("Added docs to collection %s", collection_name)

        except Exception as e:
            logger.error("Error in vector_store.add_documents: %s", e)
            raise e
