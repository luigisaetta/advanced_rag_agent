"""
File name: answer_generator.py
Author: Luigi Saetta
Last modified: 24-02-2026
Python Version: 3.11

Description:
    This module implements the last step in the workflow: generation
    of the answer form the LLM:


Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.answer_generator import AnswerGenerator

License:
    This code is released under the MIT License.

Notes:
    This is a part of a demo showing how to implement an advanced
    RAG solution as a LangGraph agent.

Warnings:
    This module is in development, may change in future versions.
"""

from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

# observability decorators
from core.observability import annotate_current_observation, zipkin_span

from agent.agent_state import State
from agent.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    GEMINI_ANSWER_PROMPT_TEMPLATE,
    apply_prompt_profile,
)
from core.oci_models import get_llm
from core.retry_utils import stream_with_retry
from core.utils import get_console_logger
from config import AGENT_NAME, DEBUG, LLM_MAX_RETRIES

logger = get_console_logger()


class AnswerGenerator(Runnable):
    """
    Takes the user request and the chat history and rewrite the user query
    in a standalone question that is used for the semantic search
    """

    def __init__(self):
        """
        Init
        """
        self.dict_languages = {
            "en": "English",
            "fr": "French",
            "it": "Italian",
            "es": "Spanish",
        }

    def build_context_for_llm(self, docs: list):
        """
        Build the context for the final answer from LLM

        docs: list[Documents]
        """
        # more Pythonic
        _context = "\n\n".join(doc["page_content"] for doc in docs)

        return _context

    @staticmethod
    def _select_answer_template(model_id: str) -> str:
        """
        Select a dedicated answer prompt for Gemini models.
        """
        model = str(model_id or "").strip().lower()
        if model.startswith("google.gemini"):
            return GEMINI_ANSWER_PROMPT_TEMPLATE
        return ANSWER_PROMPT_TEMPLATE

    @zipkin_span(service_name=AGENT_NAME, span_name="answer_generation")
    def invoke(self, input: State, config=None, **kwargs):
        """
        Generate the final answer
        """
        # get the model_id from config
        model_id = config["configurable"]["model_id"]

        if config["configurable"]["main_language"] in self.dict_languages:
            # want to change language
            main_language = self.dict_languages.get(
                config["configurable"]["main_language"]
            )
        else:
            # "same as the question" (default)
            # answer will be in the same language as the question
            main_language = None

        if DEBUG:
            logger.info("AnswerGenerator, model_id: %s", model_id)
            logger.info("AnswerGenerator, main_language: %s", main_language)

        final_answer = ""
        error = None

        try:
            llm = get_llm(model_id=model_id)

            # docs are returned from the reranker
            _context = self.build_context_for_llm(input["reranker_docs"])
            answer_template = self._select_answer_template(model_id)

            system_prompt = PromptTemplate(
                input_variables=["context"],
                template=apply_prompt_profile(answer_template, config=config),
            ).format(context=_context)

            messages = [
                SystemMessage(content=system_prompt),
            ]
            # add the chat history
            for msg in input["chat_history"]:
                messages.append(msg)

            # to force the answer in the selected language
            if main_language is not None:
                the_question = f"{input['user_request']}. Answer in {main_language}."
            else:
                # no cross language
                the_question = input["user_request"]

            messages.append(HumanMessage(content=the_question))

            # return streaming output with bounded retries on early transient errors
            final_answer = stream_with_retry(
                lambda: llm.stream(messages),
                max_attempts=LLM_MAX_RETRIES,
                operation_name="Answer LLM stream",
            )

        except Exception as e:
            logger.error("Error in generate_answer: %s", e)
            error = str(e)

        annotate_current_observation(
            metadata={
                "model_id": model_id,
                "answer_streaming": True,
                "error": error,
            },
            input_data={"main_language": main_language or "same_as_question"},
        )
        return {"final_answer": final_answer, "error": error}
