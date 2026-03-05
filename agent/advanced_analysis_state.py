"""
File name: advanced_analysis_state.py
Author: Luigi Saetta
Last modified: 02-03-2026
Python Version: 3.11

Description:
    This module defines the dedicated state for the Advanced Analysis subgraph.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.advanced_analysis_state import AdvancedAnalysisState

License:
    This code is released under the MIT License.
"""

from typing import Optional
from typing_extensions import TypedDict


class AdvancedAnalysisState(TypedDict):
    """
    State schema for advanced analysis subgraph.
    """

    user_request: str
    standalone_question: str
    search_intent: str
    has_session_pdf: bool
    advanced_analysis_enabled: bool
    advanced_analysis_session_only: bool
    retriever_docs: Optional[list]
    session_retriever_docs: Optional[list]
    advanced_plan: Optional[list]
    advanced_step_outputs: Optional[list]
    final_answer: str
    citations: list
    error: Optional[str]
