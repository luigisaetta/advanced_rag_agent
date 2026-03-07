"""
File name: post_answer_evaluator.py
Author: Luigi Saetta
Last modified: 05-03-2026
Python Version: 3.11

Description:
    This module evaluates final answer quality and logs a root-cause category.
"""

import itertools

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from py_zipkin.zipkin import zipkin_span

from agent.agent_state import State
from agent.prompts import POST_ANSWER_EVALUATION_TEMPLATE, apply_prompt_profile
from core.oci_models import get_llm
from core.post_answer_feedback import PostAnswerFeedback
from core.utils import get_console_logger, extract_json_from_text
from config import (
    AGENT_NAME,
    EMBED_MODEL_ID,
    RERANKER_MODEL_ID,
    TOP_K,
    POST_ANSWER_EVALUATION_MODEL_ID,
    POST_ANSWER_EVALUATION_MAX_CHARS,
)

logger = get_console_logger()

ALLOWED_CAUSES = {"NO_ISSUE", "RETRIEVAL", "RERANK", "GENERATION"}


class PostAnswerEvaluator(Runnable):
    """
    Post-answer evaluator node.
    It does not alter the final answer; it only logs evaluation results.
    """

    @classmethod
    def _format_docs_for_prompt(
        cls, docs: list, max_chars: int, min_chars_per_doc: int = 450
    ) -> str:
        """Helper to format docs for prompt using full content (no head/tail clipping)."""
        parts = []
        docs = list(docs or [])

        for idx, doc in enumerate(docs):
            metadata = (doc or {}).get("metadata", {}) or {}
            source = metadata.get("source", "unknown")
            page = metadata.get("page_label", "")
            retrieval_type = metadata.get("retrieval_type", "semantic")
            content = str((doc or {}).get("page_content", "") or "").strip()
            if not content:
                continue

            block = (
                f"[{idx + 1}] source={source} page={page} "
                f"type={retrieval_type}\n{content}"
            )
            parts.append(block)

        return "\n\n".join(parts)

    @staticmethod
    def _build_source_inventory(docs: list) -> str:
        """
        Build a compact, deterministic source inventory from docs metadata.
        """
        seen = set()
        lines = []
        for doc in docs or []:
            metadata = (doc or {}).get("metadata", {}) or {}
            source = str(metadata.get("source", "unknown") or "unknown")
            page = str(metadata.get("page_label", "") or "")
            retrieval_type = str(metadata.get("retrieval_type", "semantic") or "semantic")
            key = (source, page, retrieval_type)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f'- {{"source": "{source}", "page": "{page}", "retrieval_type": "{retrieval_type}"}}'
            )
        return "\n".join(lines) if lines else "- {}"

    @staticmethod
    def _normalize_cause(value: str) -> str:
        """Helper for normalize cause."""
        cause = str(value or "").strip().upper()
        if cause not in ALLOWED_CAUSES:
            return "NO_ISSUE"
        return cause

    @staticmethod
    def _normalize_confidence(value) -> float:
        """
        Normalize confidence to [0.0, 1.0]. Defaults to 0.0 on invalid input.
        """
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _split_answer_payload(answer_payload):
        """
        Return (payload_for_downstream, plain_text_for_evaluation).
        """
        if isinstance(answer_payload, str):
            return answer_payload, answer_payload

        if answer_payload is None:
            return "", ""

        if hasattr(answer_payload, "__iter__"):
            eval_stream, downstream_stream = itertools.tee(answer_payload)
            parts = []
            for chunk in eval_stream:
                content = getattr(chunk, "content", None)
                if content is None:
                    content = str(chunk)
                parts.append(str(content))
            return downstream_stream, "".join(parts).strip()

        return answer_payload, str(answer_payload).strip()

    @zipkin_span(service_name=AGENT_NAME, span_name="post_answer_evaluation")
    def invoke(self, input: State, config=None, **kwargs):
        """
        Evaluate final answer and log a root-cause class.
        """
        error = input.get("error")
        original_answer_payload = input.get("final_answer", "")
        downstream_answer_payload, final_answer_text = self._split_answer_payload(
            original_answer_payload
        )

        if not final_answer_text:
            logger.info("PostAnswerEvaluator skipped: empty final answer.")
            return {
                "final_answer": downstream_answer_payload,
                "citations": input.get("citations", []),
                "post_answer_root_cause": "",
                "post_answer_reason": "",
                "post_answer_confidence": 0.0,
                "error": error,
            }

        try:
            configurable = (config or {}).get("configurable", {})
            if not bool(configurable.get("post_answer_evaluation_enabled", True)):
                logger.info("PostAnswerEvaluator skipped: disabled by config.")
                return {
                    "final_answer": downstream_answer_payload,
                    "citations": input.get("citations", []),
                    "post_answer_root_cause": "",
                    "post_answer_reason": "",
                    "post_answer_confidence": 0.0,
                    "error": error,
                }

            answer_model_id = configurable.get("model_id")
            evaluator_model_id = configurable.get(
                "post_answer_evaluation_model_id",
                POST_ANSWER_EVALUATION_MODEL_ID,
            )
            max_chars = int(
                configurable.get(
                    "post_answer_evaluation_max_chars",
                    POST_ANSWER_EVALUATION_MAX_CHARS,
                )
            )
            max_chars = max(1200, max_chars)

            retriever_context = self._format_docs_for_prompt(
                input.get("retriever_docs", []), max_chars=max_chars
            )
            reranker_context = self._format_docs_for_prompt(
                input.get("reranker_docs", []), max_chars=max_chars
            )

            prompt = PromptTemplate(
                input_variables=[
                    "user_request",
                    "standalone_question",
                    "retriever_sources",
                    "reranker_sources",
                    "retriever_context",
                    "reranker_context",
                    "final_answer",
                ],
                template=apply_prompt_profile(
                    POST_ANSWER_EVALUATION_TEMPLATE, config=config
                ),
            ).format(
                user_request=input.get("user_request", ""),
                standalone_question=input.get("standalone_question", ""),
                retriever_sources=self._build_source_inventory(
                    input.get("retriever_docs", [])
                ),
                reranker_sources=self._build_source_inventory(
                    input.get("reranker_docs", [])
                ),
                retriever_context=retriever_context,
                reranker_context=reranker_context,
                final_answer=final_answer_text,
            )

            llm = get_llm(model_id=evaluator_model_id, temperature=0.0)
            raw = llm.invoke([HumanMessage(content=prompt)]).content
            parsed = extract_json_from_text(raw)

            root_cause = self._normalize_cause(parsed.get("root_cause"))
            reason = str(parsed.get("reason", "") or "").strip()
            confidence = self._normalize_confidence(parsed.get("confidence"))

            logger.info(
                "PostAnswerEvaluator result: root_cause=%s confidence=%.3f reason=%s",
                root_cause,
                confidence,
                reason,
            )
            try:
                question = str(
                    input.get("user_request") or input.get("standalone_question") or ""
                ).strip()
                embed_model_id = configurable.get("embed_model_id", EMBED_MODEL_ID)
                reranker_model_id = configurable.get(
                    "reranker_model_id", RERANKER_MODEL_ID
                )
                top_k = int(configurable.get("top_k", TOP_K))
                PostAnswerFeedback().insert_feedback(
                    question=question,
                    root_cause=root_cause,
                    reason=reason,
                    confidence=confidence,
                    config_data={
                        "model_id": answer_model_id,
                        "post_answer_evaluation_model_id": evaluator_model_id,
                        "embed_model_id": embed_model_id,
                        "reranker_model_id": reranker_model_id,
                        "top_k": top_k,
                    },
                )
            except Exception as db_exc:
                logger.warning(
                    "PostAnswerEvaluator persistence failed: %s",
                    db_exc,
                )
            return {
                "final_answer": downstream_answer_payload,
                "citations": input.get("citations", []),
                "post_answer_root_cause": root_cause,
                "post_answer_reason": reason,
                "post_answer_confidence": confidence,
                "error": error,
            }
        except Exception as exc:
            logger.warning("PostAnswerEvaluator failed: %s", exc)
            return {
                "final_answer": downstream_answer_payload,
                "citations": input.get("citations", []),
                "post_answer_root_cause": "",
                "post_answer_reason": "",
                "post_answer_confidence": 0.0,
                "error": error,
            }
