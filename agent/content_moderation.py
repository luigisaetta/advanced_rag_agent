"""
File name: content_moderation.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module enable to add content moderation to the user request

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.content_moderation import ContentModerator

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

from langchain_core.runnables import Runnable

# observability decorators
from core.observability import annotate_current_observation, langfuse_span

from agent.agent_state import State
from core.utils import get_console_logger
from config import AGENT_NAME, DEBUG

logger = get_console_logger()


class ContentModerator(Runnable):
    """
    Takes the user request and applies some rules
    from additional content moderation
    to avoid request not permitted
    """

    def __init__(self):
        """
        Init
        """

    @langfuse_span(service_name=AGENT_NAME, span_name="content_moderation")
    def invoke(self, input: State, config=None, **kwargs):
        """
        Check if the user requst is allowed
        """
        user_request = input["user_request"]
        error = None

        if DEBUG:
            logger.debug("ContentModerator: user_request=%s", user_request)

        # for now, do nothing
        annotate_current_observation(
            metadata={"moderation_action": "allow", "error": error}
        )
        return {"error": error}
