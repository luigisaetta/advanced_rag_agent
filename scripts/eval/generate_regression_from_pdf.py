"""
Generate a regression JSONL dataset from a single PDF.

Output format matches existing files in eval_data:
{"id": "...", "question": "...", "expected_intent": "GLOBAL_KB",
 "expected_sources": ["kb"], "expected_citations": [{"source": "...", "page": "..."}]}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage

import config
from core.oci_models import get_llm
from core.retry_utils import run_with_retry
from core.session_pdf_vlm import scan_pdf_to_docs_with_vlm
from core.utils import get_console_logger, remove_path_from_ref

logger = get_console_logger()

FORBIDDEN_TOKENS = {
    "page",
    "pagina",
    "pag.",
    "pag",
    "section",
    "sections",
    "sezione",
    "sezioni",
    "chapter",
    "capitolo",
    "file",
    "document",
    "documento",
}


def _clean_page_text(text: str) -> str:
    """Remove synthetic chunk header if present."""
    txt = str(text or "").strip()
    txt = re.sub(r"^# Doc\. title:.*?\n", "", txt, flags=re.IGNORECASE)
    return txt.strip()


def _is_not_too_empty(text: str, min_chars: int) -> bool:
    """Return True when text has enough meaningful content."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) < min_chars:
        return False
    # Require a minimum number of alphabetic chars as anti-noise filter.
    alpha_chars = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", normalized))
    return alpha_chars >= int(min_chars * 0.5)


def _extract_pages(pdf_path: Path, min_page_chars: int) -> List[Dict[str, str]]:
    """OCR PDF per page and return candidate non-empty pages."""
    logger.info(
        "Starting PDF scan: file=%s min_page_chars=%d",
        pdf_path,
        min_page_chars,
    )
    docs, total_pages = scan_pdf_to_docs_with_vlm(
        pdf_path=str(pdf_path),
        vlm_model_id=config.VLM_MODEL_ID,
        max_pages=config.SESSION_PDF_MAX_PAGES,
        source_name=pdf_path.name,
        metadata_retrieval_type="kb_seed",
        split_per_page=True,
    )

    page_rows: List[Dict[str, str]] = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        page_label = str(metadata.get("page_label", "")).strip()
        cleaned = _clean_page_text(doc.page_content)
        if not page_label or not _is_not_too_empty(cleaned, min_page_chars):
            if idx == 1 or idx % 10 == 0 or idx == len(docs):
                logger.info(
                    "Page scan progress: %d/%d (kept=%d skipped=%d)",
                    idx,
                    len(docs),
                    len(page_rows),
                    idx - len(page_rows),
                )
            continue
        page_rows.append({"page": page_label, "text": cleaned})
        if idx == 1 or idx % 10 == 0 or idx == len(docs):
            logger.info(
                "Page scan progress: %d/%d (kept=%d skipped=%d)",
                idx,
                len(docs),
                len(page_rows),
                idx - len(page_rows),
            )

    logger.info(
        "Extracted candidate pages: %d/%d from %s (declared total pages: %d)",
        len(page_rows),
        len(docs),
        pdf_path.name,
        total_pages,
    )
    return page_rows


def _build_generation_prompt(
    selected_pages: List[Dict[str, str]],
    questions_count: int,
    file_stem: str,
) -> str:
    """Prompt to generate constrained questions + citation chunk/page."""
    pages_block = []
    for row in selected_pages:
        pages_block.append(
            f'{{"chunk_number": "{row["page"]}", "text": "{row["text"].replace(chr(34), chr(39))}"}}'
        )
    pages_block_text = ",\n".join(pages_block)

    return f"""
You are generating a regression test set for a RAG system.

Generate exactly {questions_count} high-quality questions based ONLY on the provided chunks.
Each question must be answerable from one chunk only.

Hard rules:
- Do NOT mention file name, document name, chunk number, page number, section number, chapter number.
- Do NOT use phrases like "in this document", "on page X", "in section 3.1".
- Do NOT refer to numbered sections (e.g., 1.2, 3.4.5).
- Questions must be grounded strictly in concepts/information from the chunks.
- Keep questions specific and factual.
- Avoid duplicates.

Return ONLY valid JSON with this exact shape:
{{
  "questions": [
    {{"question": "...", "chunk_number": "N"}},
    {{"question": "...", "chunk_number": "N"}}
  ]
}}

Where:
- chunk_number must be one of the provided chunk numbers.
- Return exactly {questions_count} items.

Forbidden document stem (must never appear in questions):
{file_stem}

Chunks:
[
{pages_block_text}
]
""".strip()


def _parse_questions_json(raw: str) -> List[Dict[str, str]]:
    """Parse LLM output expecting {'questions': [...]}."""
    raw = str(raw or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        logger.warning("LLM response did not contain a valid JSON object.")
        return []
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception as exc:
        logger.warning("Failed to parse JSON from LLM response: %s", exc)
        return []
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        logger.warning("Parsed JSON missing list field 'questions'.")
        return []
    return questions if isinstance(questions, list) else []


def _question_is_valid(question: str, file_stem: str) -> bool:
    """Apply hard constraints on generated question text."""
    q = str(question or "").strip()
    if not q:
        return False

    q_low = q.lower()
    file_tokens = [
        t for t in re.split(r"[^a-zA-Z0-9]+", file_stem.lower()) if len(t) > 2
    ]
    if any(tok in q_low for tok in file_tokens):
        return False
    if any(tok in q_low for tok in FORBIDDEN_TOKENS):
        return False
    # Reject explicit page/section range patterns.
    if re.search(r"\b\d+\s*[-–]\s*\d+\b", q):
        return False
    if re.search(r"\b\d+\.\d+(\.\d+)?\b", q):
        return False
    return True


def _generate_questions(
    selected_pages: List[Dict[str, str]],
    questions_count: int,
    model_id: str,
    file_stem: str,
    max_attempts: int = 4,
) -> List[Tuple[str, str]]:
    """Generate and validate questions from selected page snippets."""
    logger.info(
        "Starting question generation: target=%d model=%s max_attempts=%d",
        questions_count,
        model_id,
        max_attempts,
    )
    llm = get_llm(model_id=model_id, temperature=0.2)
    allowed_pages = {row["page"] for row in selected_pages}
    unique_questions = set()
    out: List[Tuple[str, str]] = []

    for attempt in range(1, max_attempts + 1):
        missing = questions_count - len(out)
        if missing <= 0:
            break

        prompt = _build_generation_prompt(
            selected_pages=selected_pages,
            questions_count=max(questions_count, missing),
            file_stem=file_stem,
        )
        response = run_with_retry(
            lambda: llm.invoke([HumanMessage(content=prompt)]),
            max_attempts=config.LLM_MAX_RETRIES,
            operation_name="Generate regression questions",
        )
        candidates = _parse_questions_json(getattr(response, "content", ""))
        accepted_this_attempt = 0

        for item in candidates:
            if len(out) >= questions_count:
                break
            question = str((item or {}).get("question", "")).strip()
            chunk_number = str((item or {}).get("chunk_number", "")).strip()
            if chunk_number not in allowed_pages:
                continue
            if not _question_is_valid(question, file_stem=file_stem):
                continue
            signature = question.lower()
            if signature in unique_questions:
                continue
            unique_questions.add(signature)
            out.append((question, chunk_number))
            accepted_this_attempt += 1

        logger.info(
            "Question generation attempt %d/%d -> candidates=%d accepted=%d collected=%d/%d",
            attempt,
            max_attempts,
            len(candidates),
            accepted_this_attempt,
            len(out),
            questions_count,
        )

    return out


def _to_jsonl_rows(
    qa_rows: List[Tuple[str, str]],
    source_name: str,
    id_prefix: str,
    start_id: int,
) -> List[Dict[str, object]]:
    """Convert (question, chunk_number) tuples to eval_data JSONL schema."""
    rows: List[Dict[str, object]] = []
    for idx, (question, chunk_number) in enumerate(qa_rows, start=start_id):
        rows.append(
            {
                "id": f"{id_prefix}_{idx:03d}",
                "question": question,
                "expected_intent": "GLOBAL_KB",
                "expected_sources": ["kb"],
                "expected_citations": [
                    {
                        "source": source_name,
                        # Single chunk/page only, never a range.
                        "page": str(chunk_number),
                    }
                ],
            }
        )
    return rows


def _write_jsonl(rows: List[Dict[str, object]], out_path: Path) -> None:
    """Write dataset in JSONL format."""
    logger.info("Writing %d rows to %s", len(rows), out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(rows, start=1):
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            if idx == 1 or idx % 25 == 0 or idx == len(rows):
                logger.info("Write progress: %d/%d rows", idx, len(rows))
    logger.info("Dataset write completed: %s", out_path)


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser(
        description="Generate regression JSONL test cases from a single PDF."
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Absolute path of input PDF.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=6,
        help="Random non-empty pages sampled from the PDF.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        help="Number of questions to generate.",
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=450,
        help="Minimum chars for a page to be considered non-empty.",
    )
    parser.add_argument(
        "--model-id",
        default=config.LLM_MODEL_ID,
        help="LLM model used to generate questions.",
    )
    parser.add_argument(
        "--id-prefix",
        default="kb_auto",
        help="Prefix for generated id values.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting numeric id (zero-padded to 3 digits).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for page sampling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    for handler in logger.handlers:
        handler.setLevel(args.log_level)
    logger.info("Regression dataset generation started.")
    logger.info(
        "Input args: pdf_path=%s out=%s sample_pages=%d questions=%d min_page_chars=%d model_id=%s id_prefix=%s start_id=%d seed=%d log_level=%s",
        args.pdf_path,
        args.out,
        args.sample_pages,
        args.questions,
        args.min_page_chars,
        args.model_id,
        args.id_prefix,
        args.start_id,
        args.seed,
        args.log_level,
    )

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    random.seed(args.seed)
    source_name = remove_path_from_ref(str(pdf_path))
    file_stem = pdf_path.stem

    candidate_pages = _extract_pages(
        pdf_path=pdf_path,
        min_page_chars=max(50, int(args.min_page_chars)),
    )
    if len(candidate_pages) < 1:
        raise ValueError("No non-empty pages found in PDF.")

    sample_size = min(max(1, int(args.sample_pages)), len(candidate_pages))
    selected_pages = random.sample(candidate_pages, k=sample_size)
    selected_pages.sort(key=lambda row: int(row["page"]))

    logger.info(
        "Selected %d pages for generation: %s",
        len(selected_pages),
        [row["page"] for row in selected_pages],
    )

    qa_rows = _generate_questions(
        selected_pages=selected_pages,
        questions_count=max(1, int(args.questions)),
        model_id=args.model_id,
        file_stem=file_stem,
    )
    if len(qa_rows) < int(args.questions):
        raise RuntimeError(
            f"Could not generate enough valid questions: {len(qa_rows)}/{args.questions}."
        )

    dataset_rows = _to_jsonl_rows(
        qa_rows=qa_rows[: int(args.questions)],
        source_name=source_name,
        id_prefix=str(args.id_prefix).strip(),
        start_id=int(args.start_id),
    )
    logger.info(
        "Converted generated QA pairs to JSONL schema: %d rows", len(dataset_rows)
    )
    _write_jsonl(dataset_rows, out_path=out_path)

    logger.info("Regression dataset generation completed successfully.")
    print(f"Generated {len(dataset_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
