"""
File name: post_answer_evaluation_state.py
Author: Luigi Saetta
Last modified: 13-03-2026
Python Version: 3.11

Description:
    This module defines the dedicated state for the Post Answer Evaluation subgraph.
"""

from typing import Optional
from typing_extensions import TypedDict


class PostAnswerEvaluationState(TypedDict):
    """
    State schema for post-answer evaluation subgraph.
    """

    user_request: str
    standalone_question: str
    retriever_docs: Optional[list]
    reranker_docs: Optional[list]
    final_answer: str
    citations: list
    post_answer_root_cause: str
    post_answer_reason: str
    post_answer_confidence: float
    post_answer_quality_score: int
    error: Optional[str]
