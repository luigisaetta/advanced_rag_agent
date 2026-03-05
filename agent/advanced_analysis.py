"""
File name: advanced_analysis.py
Author: Luigi Saetta
Last modified: 02-03-2026
Python Version: 3.11

Description:
    This module defines nodes used by the advanced analysis subgraph.

Usage:
    Import this module into other scripts to use its functions.
    Example:
        from agent.advanced_analysis import AdvancedPlanner, AdvancedAnalysisRunner

License:
    This code is released under the MIT License.
"""

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import oracledb
import time
import re
from py_zipkin.zipkin import zipkin_span

from agent.advanced_analysis_state import AdvancedAnalysisState
from agent.prompts import (
    ADVANCED_ANALYSIS_PLANNER_TEMPLATE,
    ADVANCED_ANALYSIS_STEP_TEMPLATE,
    ADVANCED_ANALYSIS_SYNTHESIS_TEMPLATE,
)
from core.oci_models import get_llm, get_embedding_model, get_oracle_vs
from core.bm25_cache import get_bm25_cache
from core.utils import get_console_logger, extract_json_from_text, docs_serializable
from config import (
    AGENT_NAME,
    ADVANCED_ANALYSIS_MAX_ACTIONS,
    ADVANCED_ANALYSIS_KB_TOP_K,
    ADVANCED_ANALYSIS_STEP_MAX_WORDS,
    ENABLE_HYBRID_SEARCH,
    HYBRID_TOP_K,
    HYBRID_SESSION_TOP_K,
)
from config_private import CONNECT_ARGS

logger = get_console_logger()

LANG_STOPWORDS = {
    "it": {
        "il",
        "lo",
        "la",
        "gli",
        "le",
        "un",
        "una",
        "di",
        "del",
        "della",
        "delle",
        "che",
        "per",
        "con",
        "nel",
        "nella",
        "dai",
        "dalle",
        "sul",
        "sulla",
        "sono",
        "come",
        "anche",
        "quindi",
        "analizza",
        "riassumi",
        "documento",
    },
    "en": {
        "the",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "in",
        "on",
        "from",
        "is",
        "are",
        "be",
        "this",
        "that",
        "these",
        "those",
        "using",
        "analyze",
        "summarize",
        "document",
        "section",
        "available",
        "knowledge",
    },
    "fr": {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "dans",
        "avec",
        "pour",
        "sur",
        "est",
        "sont",
        "ce",
        "cette",
        "ces",
        "analyse",
        "resumer",
        "document",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "de",
        "del",
        "con",
        "para",
        "en",
        "sobre",
        "es",
        "son",
        "este",
        "esta",
        "estos",
        "estas",
        "analiza",
        "resumen",
        "documento",
    },
}


def _language_name(lang: str) -> str:
    """
    Return a human-readable language name for prompt constraints.
    """
    names = {
        "it": "Italian",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
    }
    return names.get(lang, "the same language as the user request")


def _tokenize_text(text: str) -> list:
    """
    Tokenize text into lowercase alphabetic words with basic latin accents.
    """
    return re.findall(r"[a-zàèéìòóùçñ]+", str(text or "").lower())


def _detect_language_from_text(text: str) -> str:
    """
    Detect language with a simple stopword-scoring approach.
    Returns one of: it, en, fr, es.
    """
    tokens = _tokenize_text(text)
    if not tokens:
        return "en"

    scores = {
        lang: sum(1 for tok in tokens if tok in stopwords)
        for lang, stopwords in LANG_STOPWORDS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = sorted(scores.values(), reverse=True)[1]

    # conservative tie/low-confidence policy: default to English only when
    # evidence is weak; otherwise honor the best-scoring language.
    if best_score == 0:
        return "en"
    if best_score == second_score and best_score < 3:
        return "en"
    if best_score < 2:
        return "en"
    return best_lang


def _detect_question_language(user_request: str) -> str:
    """
    Detect user-request language.
    """
    return _detect_language_from_text(user_request)


def _detect_language_from_session_docs(session_docs: list) -> str:
    """
    Detect language from serialized session PDF chunks.
    Uses a bounded text sample to keep it lightweight.
    """
    parts = []
    used = 0
    max_chars = 6000
    for doc in session_docs:
        text = str((doc or {}).get("page_content", "") or "").strip()
        if not text:
            continue
        remain = max_chars - used
        if remain <= 0:
            break
        snippet = text[:remain]
        parts.append(snippet)
        used += len(snippet)
    sample = "\n".join(parts).strip()
    if not sample:
        return "en"
    return _detect_language_from_text(sample)


def _resolve_output_language(main_language: str, user_request: str) -> str:
    """
    Resolve output language from runtime config and user request.
    """
    lang = str(main_language or "").strip().lower()
    if lang in {"it", "italian", "italiano"}:
        return "it"
    if lang in {"en", "english"}:
        return "en"
    if lang in {"fr", "french", "francais", "français"}:
        return "fr"
    if lang in {"es", "spanish", "espanol", "español"}:
        return "es"
    if "same as the question" in lang or "same as question" in lang:
        return _detect_question_language(user_request)
    return _detect_question_language(user_request)


def _resolve_advanced_output_language(
    configurable: dict, user_request: str, session_docs: list, session_only: bool
) -> str:
    """
    Resolve output language for advanced-analysis execution.
    In session-only mode, prefer language inferred from PDF chunks.
    """
    if session_only and session_docs:
        return _detect_language_from_session_docs(session_docs)
    return _resolve_output_language(configurable.get("main_language", ""), user_request)


def _get_report_labels(lang: str) -> dict:
    """
    Localized labels/messages for advanced-analysis report composition.
    """
    labels = {
        "it": {
            "step_title": "Passo",
            "final_synthesis_title": "Sintesi Finale",
            "no_plan": "L'analisi avanzata non puo essere eseguita perche non e stato generato alcun piano.",
            "no_steps": "L'analisi avanzata non ha prodotto risultati intermedi da sintetizzare.",
        },
        "fr": {
            "step_title": "Etape",
            "final_synthesis_title": "Synthese Finale",
            "no_plan": "L'analyse avancee n'a pas pu etre executee car aucun plan n'a ete genere.",
            "no_steps": "L'analyse avancee n'a produit aucun resultat intermediaire a synthetiser.",
        },
        "es": {
            "step_title": "Paso",
            "final_synthesis_title": "Sintesis Final",
            "no_plan": "El analisis avanzado no pudo ejecutarse porque no se genero ningun plan.",
            "no_steps": "El analisis avanzado no genero resultados intermedios para sintetizar.",
        },
        "en": {
            "step_title": "Step",
            "final_synthesis_title": "Final Synthesis",
            "no_plan": "Advanced analysis could not run because no execution plan was generated.",
            "no_steps": "Advanced analysis did not produce step outputs to synthesize.",
        },
    }
    return labels.get(lang, labels["en"])


def _emit_progress(configurable: dict, percent: int, message: str) -> None:
    """
    Emit progress updates when a UI callback is provided.
    Callback signature: callback(percent: int, message: str)
    """
    callback = configurable.get("progress_callback")
    if callback is None:
        return
    try:
        callback(max(0, min(100, int(percent))), message)
    except Exception:
        # Progress reporting must never break execution.
        return


class AdvancedPlanner(Runnable):
    """
    First step in advanced-analysis subgraph.
    For now it initializes an empty plan.
    """

    @staticmethod
    def _serialize_all_session_chunks(
        session_docs: list, max_chars_per_chunk: int = 1400
    ) -> str:
        """
        Build a compact text view over all session chunks.
        Keeps all chunks (no global cut), truncating each chunk for prompt safety.
        """
        parts = []
        for idx, doc in enumerate(session_docs):
            metadata = doc.get("metadata") or {}
            source = metadata.get("source", "uploaded.pdf")
            page = metadata.get("page_label", "")
            text = (doc.get("page_content") or "").strip()
            if not text:
                continue
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."
            block = f"[{idx + 1}] source={source} page={page}\n{text}"
            parts.append(block)
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_plan(
        plan: list, max_actions: int, session_only: bool = False
    ) -> list:
        """Helper for normalize plan."""
        out = []
        for i, step in enumerate(plan[:max_actions], start=1):
            if not isinstance(step, dict):
                continue
            section = str(step.get("section", "")).strip() or "unknown section"
            raw_chunk_numbers = step.get("chunk_numbers", [])
            chunk_numbers = []
            if isinstance(raw_chunk_numbers, list):
                for n in raw_chunk_numbers:
                    if isinstance(n, int) and n > 0:
                        chunk_numbers.append(n)
                    elif isinstance(n, str) and n.isdigit() and int(n) > 0:
                        chunk_numbers.append(int(n))
            objective = (
                str(step.get("objective", "")).strip() or "analyze section relevance"
            )
            kb_needed = bool(step.get("kb_search_needed", False))
            kb_query = str(step.get("kb_query", "")).strip() if kb_needed else ""
            if session_only:
                # In session-only mode we must never touch KB.
                kb_needed = False
                kb_query = ""
            out.append(
                {
                    "step": i,
                    "section": section,
                    "chunk_numbers": chunk_numbers,
                    "objective": objective,
                    "kb_search_needed": kb_needed,
                    "kb_query": kb_query,
                }
            )
        return out

    @zipkin_span(service_name=AGENT_NAME, span_name="advanced_planner")
    def invoke(self, input: AdvancedAnalysisState, config=None, **kwargs):
        """Invoke."""
        error = input.get("error")
        user_request = input.get("user_request", "")
        configurable = (config or {}).get("configurable", {})
        _emit_progress(configurable, 5, "Planner started")
        model_id = configurable.get("model_id")
        # This runtime setting allows reusing AdvancedAnalysisFlow for SESSION_DOC
        # requests that must stay document-only (no KB retrieval).
        session_only = bool(
            configurable.get("advanced_analysis_session_only", False)
            or input.get("advanced_analysis_session_only", False)
        )
        session_docs = list(configurable.get("session_pdf_docs", []))
        output_lang = _resolve_output_language(
            configurable.get("main_language", ""),
            user_request,
        )
        if session_only:
            output_lang = _resolve_advanced_output_language(
                configurable=configurable,
                user_request=user_request,
                session_docs=session_docs,
                session_only=True,
            )
        max_actions = int(
            configurable.get(
                "advanced_analysis_max_actions", ADVANCED_ANALYSIS_MAX_ACTIONS
            )
        )
        max_actions = max(1, max_actions)

        # all chunks from session PDF are expected here (serialized docs)
        if not session_docs:
            logger.warning("AdvancedPlanner: no session_pdf_docs available.")
            _emit_progress(configurable, 20, "Planner completed (no session docs)")
            return {"advanced_plan": [], "error": error}

        try:
            session_chunks = self._serialize_all_session_chunks(session_docs)
            prompt = PromptTemplate(
                input_variables=["user_request", "session_chunks", "max_actions"],
                template=ADVANCED_ANALYSIS_PLANNER_TEMPLATE,
            ).format(
                user_request=user_request,
                session_chunks=session_chunks,
                max_actions=max_actions,
            )
            prompt += (
                "\n\nLanguage constraint: produce section names and objectives in "
                f"{_language_name(output_lang)}."
            )
            if session_only:
                prompt += (
                    "\n\nMandatory constraint for this run: SESSION-ONLY mode is active. "
                    "Do not require KB search in any step. "
                    "Always set kb_search_needed to false and kb_query to an empty string."
                )

            llm = get_llm(model_id=model_id, temperature=0.0)
            response = llm.invoke([HumanMessage(content=prompt)]).content
            parsed = extract_json_from_text(response)
            raw_plan = parsed.get("plan", [])
            advanced_plan = self._normalize_plan(
                raw_plan,
                max_actions=max_actions,
                session_only=session_only,
            )

            logger.info("AdvancedPlanner final plan: %s", advanced_plan)
            _emit_progress(
                configurable, 20, f"Planner completed ({len(advanced_plan)} actions)"
            )
            return {"advanced_plan": advanced_plan, "error": error}
        except Exception as exc:
            logger.exception("AdvancedPlanner failed: %s", exc)
            _emit_progress(configurable, 20, "Planner failed")
            return {"advanced_plan": [], "error": error}


class AdvancedAnalysisRunner(Runnable):
    """
    Execute advanced-analysis plan step by step.
    """

    @staticmethod
    def _select_pdf_chunks(session_docs: list, chunk_numbers: list) -> list:
        """Helper for select pdf chunks."""
        chunk_numbers = sorted(
            set(n for n in chunk_numbers if isinstance(n, int) and n > 0)
        )
        selected = []
        for n in chunk_numbers:
            idx = n - 1
            if 0 <= idx < len(session_docs):
                selected.append((n, session_docs[idx]))
        return selected

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Helper for normalize text."""
        return " ".join((text or "").split()).strip().lower()

    def _merge_docs(self, semantic_docs: list, bm25_docs: list) -> list:
        """
        Merge semantic and BM25 docs, deduplicating by normalized content.
        """
        merged = []
        index_by_content = {}

        for doc in semantic_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content:
                continue
            metadata = doc.get("metadata") or {}
            metadata.setdefault("retrieval_type", "semantic")
            doc["metadata"] = metadata
            index_by_content[content] = len(merged)
            merged.append(doc)

        for doc in bm25_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content:
                continue

            if content in index_by_content:
                existing = merged[index_by_content[content]]
                existing_meta = existing.get("metadata") or {}
                existing_type = existing_meta.get("retrieval_type", "semantic")
                if existing_type != "semantic+bm25":
                    existing_meta["retrieval_type"] = "semantic+bm25"
                    existing["metadata"] = existing_meta
                continue

            metadata = doc.get("metadata") or {}
            metadata.setdefault("retrieval_type", "bm25")
            doc["metadata"] = metadata
            index_by_content[content] = len(merged)
            merged.append(doc)

        return merged

    def _extend_with_neighbors(
        self, session_docs: list, chunk_numbers: list, radius: int = 1
    ) -> list:
        """Helper for extend with neighbors."""
        expanded = set()
        total = len(session_docs)
        for n in chunk_numbers:
            if not isinstance(n, int):
                continue
            for x in range(max(1, n - radius), min(total, n + radius) + 1):
                expanded.add(x)
        return self._select_pdf_chunks(session_docs, sorted(expanded))

    @staticmethod
    def _selected_text_len(selected_chunks: list) -> int:
        """Helper for selected text len."""
        return sum(len((doc.get("page_content") or "")) for _n, doc in selected_chunks)

    @staticmethod
    def _format_pdf_context(selected_chunks: list, max_chars: int = 12000) -> str:
        """Helper for format pdf context."""
        parts = []
        used = 0
        for n, doc in selected_chunks:
            metadata = doc.get("metadata") or {}
            source = metadata.get("source", "uploaded.pdf")
            page = metadata.get("page_label", "")
            text = (doc.get("page_content") or "").strip()
            if not text:
                continue
            block = f"[chunk {n}] source={source} page={page}\n{text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) if parts else "No PDF evidence selected."

    @staticmethod
    def _format_kb_context(kb_docs: list, max_chars: int = 8000) -> str:
        """Helper for format kb context."""
        parts = []
        used = 0
        for i, doc in enumerate(kb_docs, start=1):
            metadata = doc.get("metadata") or {}
            source = metadata.get("source", "kb")
            page = metadata.get("page_label", "")
            text = (doc.get("page_content") or "").strip()
            if not text:
                continue
            block = f"[kb {i}] source={source} page={page}\n{text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) if parts else "No KB evidence retrieved."

    @staticmethod
    def _kb_semantic_docs(query: str, collection_name: str, top_k: int) -> list:
        """Helper for kb semantic docs."""
        embed_model = get_embedding_model()
        with oracledb.connect(**CONNECT_ARGS) as conn:
            v_store = get_oracle_vs(
                conn=conn,
                collection_name=collection_name,
                embed_model=embed_model,
            )
            docs = v_store.similarity_search(query=query, k=top_k)

        out = []
        for doc in docs:
            metadata = doc.metadata or {}
            metadata["retrieval_type"] = "semantic"
            out.append({"page_content": doc.page_content, "metadata": metadata})
        return out

    @staticmethod
    def _kb_bm25_docs(query: str, collection_name: str, top_k: int) -> list:
        """Helper for kb bm25 docs."""
        cache = get_bm25_cache()
        results = cache.search_docs(
            query=query,
            table_name=collection_name,
            text_column="TEXT",
            top_n=top_k,
        )
        out = []
        for doc in results:
            metadata = doc.get("metadata") or {}
            metadata["retrieval_type"] = "bm25"
            out.append(
                {"page_content": doc.get("page_content", ""), "metadata": metadata}
            )
        return out

    def _kb_search_docs(self, query: str, collection_name: str, top_k: int) -> list:
        """Helper for kb search docs."""
        semantic_docs = self._kb_semantic_docs(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
        )
        if not ENABLE_HYBRID_SEARCH:
            return semantic_docs

        bm25_docs = self._kb_bm25_docs(
            query=query,
            collection_name=collection_name,
            top_k=max(top_k, HYBRID_TOP_K),
        )
        return self._merge_docs(semantic_docs, bm25_docs)

    @staticmethod
    def _build_citations(step_no: int, selected_chunks: list, kb_docs: list) -> list:
        """Helper for build citations."""
        citations = []
        for _n, doc in selected_chunks:
            metadata = doc.get("metadata") or {}
            citations.append(
                {
                    "step": step_no,
                    "source": metadata.get("source", "uploaded.pdf"),
                    "page": metadata.get("page_label", ""),
                    "retrieval_type": "session_pdf",
                }
            )
        for doc in kb_docs:
            metadata = doc.get("metadata") or {}
            citations.append(
                {
                    "step": step_no,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page_label", ""),
                    "retrieval_type": metadata.get("retrieval_type", "semantic"),
                }
            )
        return citations

    @staticmethod
    def _session_retrieval_fallback(configurable: dict, query: str, top_k: int) -> list:
        """Helper for session retrieval fallback."""
        session_vs = configurable.get("session_pdf_vector_store")
        if session_vs is None:
            return []
        try:
            docs = session_vs.similarity_search(query=query, k=top_k)
            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_type"] = "session_pdf"
            return docs_serializable(docs)
        except Exception:
            return []

    def _merge_selected_with_fallback(
        self, selected_chunks: list, fallback_docs: list
    ) -> list:
        """Helper for merge selected with fallback."""
        merged = list(selected_chunks)
        seen = {
            self._normalize_text((doc.get("page_content") or ""))
            for _n, doc in merged
            if self._normalize_text((doc.get("page_content") or ""))
        }
        next_idx = max([n for n, _ in merged], default=0) + 1
        for doc in fallback_docs:
            content = self._normalize_text(doc.get("page_content", ""))
            if not content or content in seen:
                continue
            merged.append((next_idx, doc))
            next_idx += 1
            seen.add(content)
        return merged

    @zipkin_span(service_name=AGENT_NAME, span_name="advanced_analysis_execution")
    def invoke(self, input: AdvancedAnalysisState, config=None, **kwargs):
        """Invoke."""
        error = input.get("error")
        plan = input.get("advanced_plan", [])
        user_request = input.get("user_request", "")
        configurable = (config or {}).get("configurable", {})
        output_lang = _resolve_output_language(
            configurable.get("main_language", ""),
            user_request,
        )
        # Keep this flag in config/state so SESSION_DOC can reuse this flow
        # while explicitly disabling KB access.
        session_only = bool(
            configurable.get("advanced_analysis_session_only", False)
            or input.get("advanced_analysis_session_only", False)
        )
        session_docs = list(configurable.get("session_pdf_docs", []))
        output_lang = _resolve_advanced_output_language(
            configurable=configurable,
            user_request=user_request,
            session_docs=session_docs,
            session_only=session_only,
        )
        labels = _get_report_labels(output_lang)
        _emit_progress(configurable, 25, "Execution started")
        model_id = configurable.get("model_id")
        collection_name = configurable.get("collection_name", "UNKNOWN")
        kb_top_k = int(
            configurable.get("advanced_analysis_kb_top_k", ADVANCED_ANALYSIS_KB_TOP_K)
        )
        step_max_words = int(
            configurable.get(
                "advanced_analysis_step_max_words", ADVANCED_ANALYSIS_STEP_MAX_WORDS
            )
        )
        kb_top_k = max(1, kb_top_k)
        step_max_words = max(120, step_max_words)

        logger.info(
            "AdvancedAnalysisFlow called. plan_steps=%d session_chunks=%d",
            len(plan),
            len(session_docs),
        )

        if not plan:
            _emit_progress(configurable, 85, "Execution skipped (empty plan)")
            return {
                "final_answer": labels["no_plan"],
                "citations": [],
                "error": error,
            }

        llm = get_llm(model_id=model_id, temperature=0.0)
        step_outputs = []
        all_citations = []

        for action in plan:
            step_start = time.time()
            step_no = action.get("step", 0)
            section = action.get("section", "unknown section")
            objective = action.get("objective", "")
            chunk_numbers = list(action.get("chunk_numbers", []))
            kb_needed = bool(action.get("kb_search_needed", False))
            kb_query = str(action.get("kb_query", "")).strip()
            if session_only:
                kb_needed = False
                kb_query = ""

            selected_chunks = self._select_pdf_chunks(session_docs, chunk_numbers)
            # include small neighborhood for better local context continuity
            if chunk_numbers:
                selected_chunks = self._extend_with_neighbors(
                    session_docs, chunk_numbers, radius=1
                )

            # robust fallback: if planner pointers are weak/missing, retrieve from session store by step query
            step_query = f"{user_request}\n{section}\n{objective}".strip()
            if kb_query:
                step_query = f"{step_query}\n{kb_query}"
            if (not selected_chunks) or (
                self._selected_text_len(selected_chunks) < 260
            ):
                fallback_docs = self._session_retrieval_fallback(
                    configurable=configurable,
                    query=step_query,
                    top_k=HYBRID_SESSION_TOP_K,
                )
                selected_chunks = self._merge_selected_with_fallback(
                    selected_chunks, fallback_docs
                )

            kb_docs = []
            if kb_needed and not session_only:
                try:
                    query = kb_query or f"{user_request} {objective}".strip()
                    kb_docs = self._kb_search_docs(
                        query=query,
                        collection_name=collection_name,
                        top_k=kb_top_k,
                    )
                except Exception as exc:
                    logger.warning(
                        "AdvancedAnalysis step %s KB search failed: %s", step_no, exc
                    )
                    kb_docs = []

            pdf_context = self._format_pdf_context(selected_chunks)
            kb_context = self._format_kb_context(kb_docs)
            prompt = PromptTemplate(
                input_variables=[
                    "max_words",
                    "user_request",
                    "step",
                    "section",
                    "objective",
                    "kb_search_needed",
                    "kb_query",
                    "pdf_context",
                    "kb_context",
                ],
                template=ADVANCED_ANALYSIS_STEP_TEMPLATE,
            ).format(
                max_words=step_max_words,
                user_request=user_request,
                step=step_no,
                section=section,
                objective=objective,
                kb_search_needed=kb_needed,
                kb_query=kb_query,
                pdf_context=pdf_context,
                kb_context=kb_context,
            )

            try:
                step_text = (
                    llm.invoke([HumanMessage(content=prompt)]).content or ""
                ).strip()
            except Exception as exc:
                logger.warning(
                    "AdvancedAnalysis step %s generation failed: %s", step_no, exc
                )
                step_text = (
                    "Unable to generate this step due to a transient model issue."
                )

            step_elapsed = round(time.time() - step_start, 2)
            logger.info(
                "AdvancedAnalysis step=%s elapsed=%ss pdf_chunks=%d kb_needed=%s kb_docs=%d",
                step_no,
                step_elapsed,
                len(selected_chunks),
                kb_needed,
                len(kb_docs),
            )
            total_steps = max(1, len(plan))
            step_progress = 25 + int((step_no / total_steps) * 60)
            _emit_progress(
                configurable,
                step_progress,
                f"Executed step {step_no}/{total_steps}",
            )

            step_outputs.append(
                f"### {labels['step_title']} {step_no} - {section}\n{step_text}"
            )
            all_citations.extend(
                self._build_citations(step_no, selected_chunks, kb_docs)
            )

        logger.info("AdvancedAnalysis steps generated. steps=%d", len(step_outputs))
        _emit_progress(configurable, 85, "Execution completed")
        return {
            "advanced_step_outputs": step_outputs,
            "citations": all_citations,
            "error": error,
        }


class AdvancedFinalSynthesis(Runnable):
    """
    Build final synthesis and compose output:
    - all step outputs
    - final synthesis section
    """

    @zipkin_span(service_name=AGENT_NAME, span_name="advanced_final_synthesis")
    def invoke(self, input: AdvancedAnalysisState, config=None, **kwargs):
        """Invoke."""
        error = input.get("error")
        user_request = input.get("user_request", "")
        step_outputs = list(input.get("advanced_step_outputs", []))
        citations = list(input.get("citations", []))
        configurable = (config or {}).get("configurable", {})
        session_only = bool(
            configurable.get("advanced_analysis_session_only", False)
            or input.get("advanced_analysis_session_only", False)
        )
        session_docs = list(configurable.get("session_pdf_docs", []))
        output_lang = _resolve_advanced_output_language(
            configurable=configurable,
            user_request=user_request,
            session_docs=session_docs,
            session_only=session_only,
        )
        labels = _get_report_labels(output_lang)
        _emit_progress(configurable, 90, "Final synthesis started")
        model_id = configurable.get("model_id")
        step_max_words = int(
            configurable.get(
                "advanced_analysis_step_max_words", ADVANCED_ANALYSIS_STEP_MAX_WORDS
            )
        )
        synthesis_max_words = max(180, int(step_max_words * 0.8))

        if not step_outputs:
            _emit_progress(configurable, 100, "Final synthesis completed (no steps)")
            return {
                "final_answer": labels["no_steps"],
                "citations": citations,
                "error": error,
            }

        synthesis_text = ""
        try:
            llm = get_llm(model_id=model_id, temperature=0.0)
            prompt = PromptTemplate(
                input_variables=["max_words", "user_request", "step_outputs"],
                template=ADVANCED_ANALYSIS_SYNTHESIS_TEMPLATE,
            ).format(
                max_words=synthesis_max_words,
                user_request=user_request,
                step_outputs="\n\n".join(step_outputs),
            )
            synthesis_text = (
                llm.invoke([HumanMessage(content=prompt)]).content or ""
            ).strip()
        except Exception as exc:
            logger.warning("AdvancedFinalSynthesis failed: %s", exc)
            synthesis_text = (
                "Unable to generate final synthesis due to a transient model issue."
            )

        final_answer = (
            "\n\n".join(step_outputs).strip()
            + f"\n\n---\n\n## {labels['final_synthesis_title']}\n"
            + synthesis_text
        )
        logger.info("AdvancedFinalSynthesis generated.")
        _emit_progress(configurable, 100, "Final synthesis completed")
        return {"final_answer": final_answer, "citations": citations, "error": error}
