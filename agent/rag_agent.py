"""
File name: rag_agent.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module implements the orchestration in the Agent.
    Based on LanGraph.

Usage:
    Import this module into other scripts to use its functions.
    Example:


License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

from langgraph.graph import StateGraph, START, END

from agent.agent_state import State

from agent.content_moderation import ContentModerator
from agent.query_rewriter import QueryRewriter
from agent.intent_classifier import IntentClassifier
from agent.hybrid_query_builder import HybridQueryBuilder
from agent.vector_search import SemanticSearch
from agent.session_vector_search import SessionVectorSearch
from agent.hybrid_search import HybridSearch
from agent.hybrid_session_search import HybridSessionSearch
from agent.hybrid_docs_merge import HybridDocsMerge
from agent.advanced_analysis import (
    AdvancedPlanner,
    AdvancedAnalysisRunner,
    AdvancedFinalSynthesis,
    RiskValidator,
)
from agent.advanced_analysis_agent import create_advanced_analysis_agent
from agent.reranker import Reranker
from agent.answer_generator import AnswerGenerator
from core.utils import get_console_logger

logger = get_console_logger()


def _route_after_intent(state: State) -> str:
    """
    Decide the next node based on classifier output.
    """
    intent = (state.get("search_intent") or "GLOBAL_KB").upper()

    if intent == "SESSION_DOC":
        if state.get("has_session_pdf"):
            logger.info(
                "Intent SESSION_DOC selected. Routing to AdvancedAnalysisFlow in session-only mode."
            )
            return "AdvancedAnalysisFlow"
        return "SessionSearch"
    if intent == "HYBRID":
        if state.get("has_session_pdf") and state.get("advanced_analysis_enabled"):
            logger.info(
                "Intent HYBRID + session PDF + advanced-analysis enabled. Routing to advanced analysis subgraph."
            )
            return "AdvancedAnalysisFlow"
        logger.info("Intent HYBRID selected. Routing to dedicated hybrid subgraph.")
        return "HybridFlow"
    return "Search"


def _create_hybrid_subgraph(
    hybrid_query_builder,
    semantic_search,
    hybrid_search,
    hybrid_session_search,
    hybrid_docs_merge,
):
    """
    Dedicated HYBRID flow subgraph:
    HybridQueryBuilder -> Search -> HybridSearch -> HybridSessionSearch -> HybridDocsMerge
    """
    subgraph = StateGraph(State)
    subgraph.add_node("HybridQueryBuilder", hybrid_query_builder)
    subgraph.add_node("Search", semantic_search)
    subgraph.add_node("HybridSearch", hybrid_search)
    subgraph.add_node("HybridSessionSearch", hybrid_session_search)
    subgraph.add_node("HybridDocsMerge", hybrid_docs_merge)

    subgraph.add_edge(START, "HybridQueryBuilder")
    subgraph.add_edge("HybridQueryBuilder", "Search")
    subgraph.add_edge("Search", "HybridSearch")
    subgraph.add_edge("HybridSearch", "HybridSessionSearch")
    subgraph.add_edge("HybridSessionSearch", "HybridDocsMerge")
    subgraph.add_edge("HybridDocsMerge", END)
    return subgraph.compile()


def create_workflow():
    """
    Create the entire workflow
    """
    workflow = StateGraph(State)

    # create nodes (each is a a Runnable)
    # step 0: content moderation
    moderator = ContentModerator()
    # step 1: rewrite the user request using history
    query_rewriter = QueryRewriter()
    # step 2: classify retrieval intent
    intent_classifier = IntentClassifier()
    # step 3: do semantic search on global KB
    hybrid_query_builder = HybridQueryBuilder()
    # step 4: do semantic search on global KB
    semantic_search = SemanticSearch()
    # step 5: semantic search on session in-memory PDF
    session_search = SessionVectorSearch()
    # step 6: hybrid search placeholder (feature-flagged)
    hybrid_search = HybridSearch()
    # step 6b: dedicated session retrieval for HYBRID flow
    hybrid_session_search = HybridSessionSearch()
    # step 6c: merge KB/session docs in HYBRID flow
    hybrid_docs_merge = HybridDocsMerge()
    # dedicated HYBRID branch orchestration
    hybrid_flow = _create_hybrid_subgraph(
        hybrid_query_builder=hybrid_query_builder,
        semantic_search=semantic_search,
        hybrid_search=hybrid_search,
        hybrid_session_search=hybrid_session_search,
        hybrid_docs_merge=hybrid_docs_merge,
    )

    # advanced analysis branch (added 02/03/2026)
    advanced_planner = AdvancedPlanner()
    advanced_runner = AdvancedAnalysisRunner()
    advanced_final_synthesis = AdvancedFinalSynthesis()
    advanced_risk_validator = RiskValidator()
    advanced_analysis_flow = create_advanced_analysis_agent(
        advanced_planner=advanced_planner,
        advanced_runner=advanced_runner,
        advanced_final_synthesis=advanced_final_synthesis,
        advanced_risk_validator=advanced_risk_validator,
    )
    # step 7: filter and rerank, using a LLM
    reranker = Reranker()
    # step 8: generate final answer
    answer_generator = AnswerGenerator()

    # Add nodes
    workflow.add_node("Moderator", moderator)
    workflow.add_node("QueryRewrite", query_rewriter)
    workflow.add_node("IntentClassifier", intent_classifier)
    workflow.add_node("Search", semantic_search)
    workflow.add_node("SessionSearch", session_search)
    workflow.add_node("HybridSearch", hybrid_search)
    workflow.add_node("HybridFlow", hybrid_flow)
    workflow.add_node("AdvancedAnalysisFlow", advanced_analysis_flow)
    workflow.add_node("Rerank", reranker)
    workflow.add_node("Answer", answer_generator)

    # define edges
    workflow.add_edge(START, "Moderator")
    workflow.add_edge("Moderator", "QueryRewrite")
    workflow.add_edge("QueryRewrite", "IntentClassifier")

    # here we classify the intent and route accordingly (global KB search vs session KB search)
    # if any pdf has been uploaded
    workflow.add_conditional_edges(
        "IntentClassifier",
        _route_after_intent,
        {
            "Search": "Search",
            "SessionSearch": "SessionSearch",
            "HybridFlow": "HybridFlow",
            "AdvancedAnalysisFlow": "AdvancedAnalysisFlow",
        },
    )
    workflow.add_edge("Search", "HybridSearch")
    workflow.add_edge("SessionSearch", "Rerank")
    workflow.add_edge("HybridSearch", "Rerank")
    workflow.add_edge("HybridFlow", "Rerank")
    workflow.add_edge("AdvancedAnalysisFlow", END)
    workflow.add_edge("Rerank", "Answer")
    workflow.add_edge("Answer", END)

    # create workflow executor
    workflow_app = workflow.compile()

    return workflow_app
