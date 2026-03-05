"""
Run a small regression evaluation set against the LangGraph workflow.

Usage example:
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.vectorstores import InMemoryVectorStore

import config
from agent.agent_state import State
from agent.rag_agent import create_workflow
from core.oci_models import get_embedding_model
from core.session_pdf_vlm import scan_pdf_to_docs_with_vlm
from core.utils import docs_serializable


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Helper for load jsonl."""
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Invalid JSON at line {lineno}: {exc}") from exc
        if "id" not in row or "question" not in row:
            raise ValueError(f"Line {lineno} must include 'id' and 'question'.")
        rows.append(row)
    return rows


def _collect_answer_text(answer_stream: Any) -> str:
    """Helper for collect answer text."""
    if isinstance(answer_stream, str):
        return answer_stream
    parts: List[str] = []
    try:
        for chunk in answer_stream:
            text = getattr(chunk, "content", None)
            parts.append(text if isinstance(text, str) else str(chunk))
    except TypeError:
        return str(answer_stream)
    return "".join(parts).strip()


def _normalize_expected_sources(values: List[str]) -> set:
    """Helper for normalize expected sources."""
    out = set()
    for item in values:
        v = str(item).strip().lower()
        if v in {"session", "session_pdf", "session-doc", "session_doc"}:
            out.add("session_pdf")
        elif v in {"kb", "global_kb", "global"}:
            out.add("kb")
    return out


def _observed_sources_from_citations(citations: List[Dict[str, Any]]) -> set:
    """Helper for observed sources from citations."""
    out = set()
    for c in citations:
        rt = str(c.get("retrieval_type", "")).strip().lower()
        if rt.startswith("session_pdf"):
            out.add("session_pdf")
        elif rt:
            out.add("kb")
    return out


def _citation_expectations_ok(
    expected_citations: List[Dict[str, Any]], citations: List[Dict[str, Any]]
) -> bool:
    """Helper for citation expectations ok."""
    for exp in expected_citations:
        exp_source = str(exp.get("source", "")).strip()
        exp_page = str(exp.get("page", "")).strip()
        found = False
        for got in citations:
            got_source = str(got.get("source", "")).strip()
            got_page = str(got.get("page", "")).strip()
            source_ok = (not exp_source) or (got_source == exp_source)
            page_ok = (not exp_page) or (got_page == exp_page)
            if source_ok and page_ok:
                found = True
                break
        if not found:
            return False
    return True


def _missing_expected_citations(
    expected_citations: List[Dict[str, Any]], citations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Return expected citation entries that were not matched by observed citations.
    """
    missing: List[Dict[str, Any]] = []
    for exp in expected_citations:
        exp_source = str(exp.get("source", "")).strip()
        exp_page = str(exp.get("page", "")).strip()
        found = False
        for got in citations:
            got_source = str(got.get("source", "")).strip()
            got_page = str(got.get("page", "")).strip()
            source_ok = (not exp_source) or (got_source == exp_source)
            page_ok = (not exp_page) or (got_page == exp_page)
            if source_ok and page_ok:
                found = True
                break
        if not found:
            missing.append({"source": exp_source, "page": exp_page})
    return missing


def _citation_recall(
    expected_citations: List[Dict[str, Any]], citations: List[Dict[str, Any]]
) -> float:
    """
    Compute per-case citation recall:
    matched_expected / total_expected
    """
    total_expected = len(expected_citations)
    if total_expected == 0:
        return 0.0

    matched = 0
    used_indices = set()
    for exp in expected_citations:
        exp_source = str(exp.get("source", "")).strip()
        exp_page = str(exp.get("page", "")).strip()
        for idx, got in enumerate(citations):
            if idx in used_indices:
                continue
            got_source = str(got.get("source", "")).strip()
            got_page = str(got.get("page", "")).strip()
            source_ok = (not exp_source) or (got_source == exp_source)
            page_ok = (not exp_page) or (got_page == exp_page)
            if source_ok and page_ok:
                matched += 1
                used_indices.add(idx)
                break

    return round(matched / total_expected, 3)


def _must_contain_ok(must_contain: List[str], answer_text: str) -> bool:
    """Helper for must contain ok."""
    text = answer_text.lower()
    return all(str(term).lower() in text for term in must_contain)


def _build_or_get_session_store(
    pdf_path: str,
    cache: Dict[str, Dict[str, Any]],
    model_id: str,
) -> Dict[str, Any]:
    """Helper for build or get session store."""
    if pdf_path in cache:
        return cache[pdf_path]

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"session_pdf_path not found: {path}")

    docs, _total_pages = scan_pdf_to_docs_with_vlm(
        pdf_path=str(path),
        vlm_model_id=config.VLM_MODEL_ID,
        max_pages=config.SESSION_PDF_MAX_PAGES,
        source_name=path.name,
    )
    embed_model = get_embedding_model()
    session_vs = InMemoryVectorStore(embedding=embed_model)
    if docs:
        session_vs.add_documents(docs)

    payload = {
        "vector_store": session_vs,
        "chunks_count": len(docs),
        "source_name": path.name,
        "docs": docs_serializable(docs),
    }
    cache[pdf_path] = payload
    return payload


def _run_case(
    app,
    case: Dict[str, Any],
    model_id: str,
    collection_name: str,
    enable_reranker: bool,
    session_cache: Dict[str, Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Helper for run case."""
    session_pdf_path = str(case.get("session_pdf_path", "")).strip()
    session_vs = None
    session_chunks_count = 0
    session_source_name = ""
    session_docs = []

    if session_pdf_path:
        session_payload = _build_or_get_session_store(
            pdf_path=session_pdf_path,
            cache=session_cache,
            model_id=model_id,
        )
        session_vs = session_payload["vector_store"]
        session_chunks_count = int(session_payload["chunks_count"])
        session_source_name = session_payload["source_name"]
        session_docs = list(session_payload.get("docs", []))

    cfg = {
        "configurable": {
            "model_id": model_id,
            "enable_reranker": enable_reranker,
            "enable_advanced_analysis": False,
            "advanced_analysis_session_only": False,
            "advanced_analysis_enable_risk_validation": config.ADVANCED_ANALYSIS_ENABLE_RISK_VALIDATION,
            "advanced_analysis_risk_validation_kb_top_k": config.ADVANCED_ANALYSIS_RISK_VALIDATION_KB_TOP_K,
            "enable_tracing": False,
            "main_language": config.MAIN_LANGUAGE,
            "collection_name": collection_name,
            "thread_id": f"eval-{case['id']}",
            "session_pdf_vector_store": session_vs,
            "session_pdf_chunks_count": session_chunks_count,
            "session_pdf_docs": session_docs,
            "advanced_analysis_max_actions": config.ADVANCED_ANALYSIS_MAX_ACTIONS,
            "advanced_analysis_kb_top_k": config.ADVANCED_ANALYSIS_KB_TOP_K,
            "advanced_analysis_step_max_words": config.ADVANCED_ANALYSIS_STEP_MAX_WORDS,
        }
    }

    events = list(
        app.stream(
            State(user_request=case["question"], chat_history=[], error=None),
            config=cfg,
        )
    )

    predicted_intent: str = ""
    citations: List[Dict[str, Any]] = []
    reranker_docs_count = 0
    answer_text = ""
    node_error: Optional[str] = None

    for event in events:
        for node_name, payload in event.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                node_error = str(payload.get("error"))
            if node_name == "IntentClassifier":
                predicted_intent = str(payload.get("search_intent", ""))
            elif node_name == "Rerank":
                citations = list(payload.get("citations", []))
                reranker_docs_count = len(payload.get("reranker_docs", []))
            elif node_name == "Answer":
                answer_text = _collect_answer_text(payload.get("final_answer"))
            elif "final_answer" in payload:
                answer_text = _collect_answer_text(payload.get("final_answer"))

    expected_intent = str(case.get("expected_intent", "")).strip().upper()
    expected_sources = _normalize_expected_sources(
        list(case.get("expected_sources", []))
    )
    expected_citations = list(case.get("expected_citations", []))
    must_contain = list(case.get("must_contain", []))

    observed_sources = _observed_sources_from_citations(citations)

    intent_ok = (not expected_intent) or (predicted_intent == expected_intent)
    sources_ok = (not expected_sources) or expected_sources.issubset(observed_sources)
    citations_ok = (not expected_citations) or _citation_expectations_ok(
        expected_citations=expected_citations,
        citations=citations,
    )
    must_contain_ok = (not must_contain) or _must_contain_ok(must_contain, answer_text)
    missing_expected_citations = (
        _missing_expected_citations(expected_citations, citations)
        if expected_citations
        else []
    )
    citations_recall = (
        _citation_recall(expected_citations, citations) if expected_citations else None
    )

    if verbose and not citations_ok:
        print("")
        print(f"[citation-mismatch] case={case['id']}")
        print(f"expected_citations={json.dumps(expected_citations, ensure_ascii=True)}")
        print(f"observed_citations={json.dumps(citations, ensure_ascii=True)}")
        print(
            f"missing_expected_citations={json.dumps(missing_expected_citations, ensure_ascii=True)}"
        )

    return {
        "id": case["id"],
        "question": case["question"],
        "session_pdf_path": session_pdf_path,
        "session_pdf_source": session_source_name,
        "predicted_intent": predicted_intent,
        "expected_intent": expected_intent,
        "intent_ok": intent_ok,
        "observed_sources": sorted(observed_sources),
        "expected_sources": sorted(expected_sources),
        "sources_ok": sources_ok,
        "citations_count": len(citations),
        "reranker_docs_count": reranker_docs_count,
        "citations_ok": citations_ok,
        "expected_citations": expected_citations,
        "observed_citations": citations,
        "missing_expected_citations": missing_expected_citations,
        "citations_recall": citations_recall,
        "must_contain_ok": must_contain_ok,
        "node_error": node_error,
        "pass": all(
            [intent_ok, sources_ok, citations_ok, must_contain_ok, not node_error]
        ),
    }


def _score(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper for score."""
    total = len(results)
    passed = sum(1 for r in results if r["pass"])

    def _ratio(key: str) -> float:
        """Helper for ratio."""
        return round(sum(1 for r in results if r[key]) / total, 3) if total else 0.0

    # Recall over expected citations:
    # - macro: average per-case recall where expected citations exist
    # - micro: total matched expected / total expected across all cases
    recall_cases = [r for r in results if r.get("expected_citations")]
    macro_recall = (
        round(
            sum(float(r.get("citations_recall", 0.0)) for r in recall_cases)
            / len(recall_cases),
            3,
        )
        if recall_cases
        else None
    )
    total_expected = 0
    total_missing = 0
    for r in recall_cases:
        expected_len = len(r.get("expected_citations", []))
        missing_len = len(r.get("missing_expected_citations", []))
        total_expected += expected_len
        total_missing += missing_len
    micro_recall = (
        round((total_expected - total_missing) / total_expected, 3)
        if total_expected
        else None
    )

    reranker_counts = [
        int(r.get("reranker_docs_count", 0))
        for r in results
        if isinstance(r.get("reranker_docs_count", 0), int)
    ]
    reranker_docs_avg = (
        round(sum(reranker_counts) / len(reranker_counts), 3)
        if reranker_counts
        else 0.0
    )
    reranker_docs_min = min(reranker_counts) if reranker_counts else 0
    reranker_docs_max = max(reranker_counts) if reranker_counts else 0

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "intent_ok_rate": _ratio("intent_ok"),
        "sources_ok_rate": _ratio("sources_ok"),
        "citations_ok_rate": _ratio("citations_ok"),
        "citations_recall_avg": macro_recall,
        "citations_recall_micro": micro_recall,
        "citations_recall_cases": len(recall_cases),
        "reranker_docs_avg": reranker_docs_avg,
        "reranker_docs_min": reranker_docs_min,
        "reranker_docs_max": reranker_docs_max,
        "must_contain_ok_rate": _ratio("must_contain_ok"),
        "errors_count": sum(1 for r in results if r["node_error"]),
    }


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser(
        description="Run regression eval set for custom_rag_agent."
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--out",
        default="eval_data/regression_results.json",
        help="Path to output JSON report.",
    )
    parser.add_argument("--model-id", default=config.LLM_MODEL_ID, help="LLM model id.")
    parser.add_argument(
        "--collection-name",
        default=config.DEFAULT_COLLECTION,
        help="Oracle VS collection name used for KB retrieval.",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranker for all test cases.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional max number of cases to run (0 = all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed mismatch diagnostics (including expected/observed citations).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(dataset_path)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]

    app = create_workflow()
    session_cache: Dict[str, Dict[str, Any]] = {}

    results: List[Dict[str, Any]] = []
    print(f"Running {len(rows)} cases from {dataset_path} ...")
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row['id']}")
        try:
            result = _run_case(
                app=app,
                case=row,
                model_id=args.model_id,
                collection_name=args.collection_name,
                enable_reranker=not args.disable_reranker,
                session_cache=session_cache,
                verbose=args.verbose,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = {
                "id": row.get("id", f"case_{idx}"),
                "question": row.get("question", ""),
                "pass": False,
                "node_error": str(exc),
                "intent_ok": False,
                "sources_ok": False,
                "citations_ok": False,
                "must_contain_ok": False,
                "reranker_docs_count": 0,
            }
        results.append(result)

    summary = _score(results)
    report = {
        "dataset": str(dataset_path),
        "model_id": args.model_id,
        "collection_name": args.collection_name,
        "reranker_enabled": not args.disable_reranker,
        "summary": summary,
        "results": results,
    }
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print("")
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
