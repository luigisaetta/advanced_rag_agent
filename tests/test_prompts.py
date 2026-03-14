"""
File name: tests/test_prompts.py
Author: Luigi Saetta
Last modified: 25-02-2026
Python Version: 3.11
License: MIT
Description: Unit tests for prompt template placeholders used by
reformulation, reranking, and answering steps.
"""

from langchain_core.prompts import PromptTemplate
from agent import prompts


def test_reformulate_prompt_has_required_placeholders():
    """Test test reformulate prompt has required placeholders."""
    tpl = prompts.REFORMULATE_PROMPT_TEMPLATE
    assert "{user_request}" in tpl
    assert "{chat_history}" in tpl


def test_answer_prompt_has_context_placeholder():
    """Test test answer prompt has context placeholder."""
    assert "{context}" in prompts.ANSWER_PROMPT_TEMPLATE


def test_gemini_answer_prompt_has_context_placeholder():
    """Test Gemini answer prompt has context placeholder."""
    assert "{context}" in prompts.GEMINI_ANSWER_PROMPT_TEMPLATE


def test_reranker_prompt_has_required_placeholders():
    """Test test reranker prompt has required placeholders."""
    tpl = prompts.RERANKER_TEMPLATE
    assert "{query}" in tpl
    assert "{chunks}" in tpl


def test_intent_classifier_prompt_has_user_request_placeholder():
    """Test test intent classifier prompt has user request placeholder."""
    assert "{user_request}" in prompts.INTENT_CLASSIFIER_TEMPLATE


def test_intent_classifier_prompt_formats_without_missing_keys():
    """Test test intent classifier prompt formats without missing keys."""
    prompt = PromptTemplate(
        input_variables=["user_request"],
        template=prompts.INTENT_CLASSIFIER_TEMPLATE,
    ).format(user_request="test")
    assert '"intent"' in prompt
