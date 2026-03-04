"""
File name: advanced_analysis_agent.py
Author: Luigi Saetta
Last modified: 04-03-2026
Python Version: 3.11

Description:
    Callable wrapper around the Advanced Analysis subgraph.
"""

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from agent.advanced_analysis_state import AdvancedAnalysisState
from agent.advanced_analysis import (
    AdvancedPlanner,
    AdvancedAnalysisRunner,
    AdvancedFinalSynthesis,
)


def create_advanced_analysis_workflow(
    advanced_planner=None,
    advanced_runner=None,
    advanced_final_synthesis=None,
):
    """
    Build and compile the Advanced Analysis subgraph.
    """
    planner = advanced_planner or AdvancedPlanner()
    runner = advanced_runner or AdvancedAnalysisRunner()
    final_synthesis = advanced_final_synthesis or AdvancedFinalSynthesis()

    subgraph = StateGraph(AdvancedAnalysisState)
    subgraph.add_node("Planner", planner)
    subgraph.add_node("AdvancedAnalysis", runner)
    subgraph.add_node("FinalSynthesis", final_synthesis)
    subgraph.add_edge(START, "Planner")
    subgraph.add_edge("Planner", "AdvancedAnalysis")
    subgraph.add_edge("AdvancedAnalysis", "FinalSynthesis")
    subgraph.add_edge("FinalSynthesis", END)
    return subgraph.compile()


class AdvancedAnalysisAgent(Runnable):
    """
    Runnable adapter that makes the advanced-analysis flow callable as an agent.
    """

    def __init__(
        self,
        advanced_planner=None,
        advanced_runner=None,
        advanced_final_synthesis=None,
    ):
        """
        Initialize the callable advanced-analysis agent.
        """
        self.workflow = create_advanced_analysis_workflow(
            advanced_planner=advanced_planner,
            advanced_runner=advanced_runner,
            advanced_final_synthesis=advanced_final_synthesis,
        )

    def invoke(self, input_state, config=None, **kwargs):
        """
        Execute the advanced-analysis subgraph and return final node outputs.
        """
        state = dict(input_state or {})
        result = self.workflow.invoke(state, config=config)
        return {
            "advanced_plan": result.get("advanced_plan", []),
            "advanced_step_outputs": result.get("advanced_step_outputs", []),
            "final_answer": result.get("final_answer", ""),
            "citations": result.get("citations", []),
            "error": result.get("error"),
        }


def create_advanced_analysis_agent(
    advanced_planner=None,
    advanced_runner=None,
    advanced_final_synthesis=None,
):
    """
    Factory for the callable advanced-analysis agent.
    """
    return AdvancedAnalysisAgent(
        advanced_planner=advanced_planner,
        advanced_runner=advanced_runner,
        advanced_final_synthesis=advanced_final_synthesis,
    )
