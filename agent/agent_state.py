"""
File name: agent_state.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module defines the class that handles the agent's State


Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.agent_state import State

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

from typing import Optional
from typing_extensions import TypedDict


class State(TypedDict):
    """
    The state of the graph
    """

    # the original user request
    user_request: str
    chat_history: list = []

    # the question reformulated using chat_history
    standalone_question: str = ""
    # optional KB-focused query built for HYBRID path
    kb_query: str = ""
    # routing intent chosen by classifier node
    search_intent: str = "GLOBAL_KB"
    # true when a session in-memory pdf vector store is available
    has_session_pdf: bool = False
    # true when advanced analysis is enabled from UI/runtime config
    advanced_analysis_enabled: bool = False
    # when true, advanced analysis must stay session-only and never query KB
    advanced_analysis_session_only: bool = False
    # optional plan payload for advanced analysis subgraph
    advanced_plan: Optional[list] = []

    # similarity_search
    # 30/06: modified, now they're a dict with
    # page_content and metadata
    # populated with docs_serializable (utils.py)
    retriever_docs: Optional[list] = []
    # session docs collected in HYBRID flow before merge
    session_retriever_docs: Optional[list] = []
    # reranker
    reranker_docs: Optional[list] = []
    # Answer
    final_answer: str
    # Citations
    citations: list = []

    # if any step encounter an error
    error: Optional[str] = None
