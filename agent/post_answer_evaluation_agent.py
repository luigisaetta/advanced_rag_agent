"""
File name: post_answer_evaluation_agent.py
Author: Luigi Saetta
Last modified: 13-03-2026
Python Version: 3.11

Description:
    Callable wrapper around the Post Answer Evaluation subgraph.
"""

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from agent.post_answer_evaluation_state import PostAnswerEvaluationState
from agent.post_answer_evaluator import PostAnswerEvaluator


def create_post_answer_evaluation_workflow(post_answer_evaluator=None):
    """
    Build and compile the Post Answer Evaluation subgraph.
    """
    evaluator = post_answer_evaluator or PostAnswerEvaluator()

    subgraph = StateGraph(PostAnswerEvaluationState)
    subgraph.add_node("PostAnswerEvaluation", evaluator)
    subgraph.add_edge(START, "PostAnswerEvaluation")
    subgraph.add_edge("PostAnswerEvaluation", END)
    return subgraph.compile()


class PostAnswerEvaluationAgent(Runnable):
    """
    Runnable adapter that makes post-answer evaluation callable as an agent.
    """

    def __init__(self, post_answer_evaluator=None):
        """
        Initialize the callable post-answer evaluation agent.
        """
        self.workflow = create_post_answer_evaluation_workflow(
            post_answer_evaluator=post_answer_evaluator
        )

    def invoke(self, input_state, config=None, **kwargs):
        """
        Execute post-answer evaluation subgraph and return normalized outputs.
        """
        state = dict(input_state or {})
        result = self.workflow.invoke(state, config=config)
        return {
            "final_answer": result.get("final_answer", ""),
            "citations": result.get("citations", []),
            "post_answer_root_cause": result.get("post_answer_root_cause", ""),
            "post_answer_reason": result.get("post_answer_reason", ""),
            "post_answer_confidence": result.get("post_answer_confidence", 0.0),
            "error": result.get("error"),
        }


def create_post_answer_evaluation_agent(post_answer_evaluator=None):
    """
    Factory for the callable post-answer evaluation agent.
    """
    return PostAnswerEvaluationAgent(post_answer_evaluator=post_answer_evaluator)
