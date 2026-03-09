# Evaluation Techniques in `advanced_rag_agent`: A Technical Deep Dive

## Abstract

This article documents the evaluation stack implemented in `advanced_rag_agent`, focusing on three complementary layers:

1. **Offline regression evaluation** (`scripts/eval/run_regression_eval.py`) for deterministic, repeatable quality checks over curated JSONL cases.
2. **Synthetic evaluation-set generation from PDFs** (`scripts/eval/generate_regression_from_pdf.py`) to bootstrap high-signal test suites with citation-aware expectations.
3. **Post-answer root-cause diagnosis** (`agent/post_answer_evaluator.py`) for production-like forensic classification of answer failures into retrieval, reranking, or generation stages.

Unlike generic “single-score RAG evaluation”, this design explicitly decomposes quality by pipeline stage and preserves evidence provenance. The resulting methodology is practical for iterative engineering: each failed case can be traced to a concrete subsystem and configuration profile.

---

## 1. Why this evaluation architecture exists

The runtime graph in `agent/rag_agent.py` combines multiple retrieval modes (global KB, session PDF, hybrid merging, optional advanced analysis subgraph), then applies LLM reranking and final answer generation. In such a non-trivial graph, aggregate accuracy alone is insufficient because:

- a bad answer can originate from **wrong routing** (intent classifier),
- from **insufficient candidate retrieval**,
- from **over-filtering during reranking**,
- or from **hallucination/omission during generation**.

The implemented evaluation strategy therefore separates concerns:

- **Offline regression** measures observable end-to-end behavior against expected intent, sources, citations, and lexical constraints.
- **Post-answer evaluation** acts as an LLM judge to infer likely root cause using both pre-rerank and post-rerank evidence.
- **Persistence + UI analytics** support longitudinal monitoring by model/config slice.

This is a pipeline-oriented quality model, not just an answer-grading model.

---

## 2. Offline Regression Evaluation (`run_regression_eval.py`)

### 2.1 Dataset contract

Regression datasets are JSONL rows with required fields:

- `id`
- `question`

Optional supervision fields:

- `session_pdf_path` (for `SESSION_DOC` / `HYBRID` cases)
- `expected_intent` in `{GLOBAL_KB, SESSION_DOC, HYBRID}`
- `expected_sources` subset of `{kb, session_pdf}`
- `expected_citations` as `[{"source": "...", "page": "..."}]`
- `must_contain` list of mandatory answer substrings

This schema intentionally mixes **routing supervision**, **provenance supervision**, and **weak answer-content constraints**.

### 2.2 Execution path

For each case, the runner:

1. Builds workflow with `create_workflow()`.
2. Optionally builds/reuses in-memory vector store for `session_pdf_path` (cached by path to amortize OCR+embedding cost).
3. Streams graph events and extracts:
   - predicted intent (`IntentClassifier`),
   - reranker outputs and citations (`Rerank`),
   - final answer text (`Answer` or fallback payloads).
4. Computes boolean checks + citation recall metrics.

Notably, evaluation is done on the **real graph**, with runtime config aligned to production defaults (unless overridden by CLI flags).

### 2.3 Scoring dimensions

Per-case pass criteria:

`pass = intent_ok AND sources_ok AND citations_ok AND must_contain_ok AND no node_error`

Global summary includes:

- `pass_rate`
- `intent_ok_rate`
- `sources_ok_rate`
- `citations_ok_rate`
- `must_contain_ok_rate`
- `errors_count`
- reranker document statistics (`avg/min/max`)
- citation recall:
  - **macro**: average per-case recall over cases with expected citations
  - **micro**: total matched expected citations / total expected citations

This dual recall view is important:

- **Macro recall** penalizes uneven behavior across cases.
- **Micro recall** rewards total matched evidence volume.

### 2.4 Citation matching semantics

Citation expectation matching is exact on `(source, page)` when provided (empty fields behave as wildcards). Missing expected citations are explicitly listed per case (`missing_expected_citations`), which is useful for root-cause inspection and report diffing.

### 2.5 Strengths and limits

Strengths:

- deterministic schema and machine-readable report,
- stage-aware checks (intent/source/citation),
- easy A/B testing via flags like `--disable-reranker`.

Limits:

- `must_contain` is lexical and brittle to paraphrase,
- citation comparison currently lacks fuzzy normalization,
- no semantic answer-quality score beyond substring checks.

---

## 3. Synthetic Dataset Generation from PDFs (`generate_regression_from_pdf.py`)

### 3.1 Motivation

Manual curation of regression sets is expensive. The generator creates citation-supervised test cases from a single PDF, reducing cold-start cost for new domains.

### 3.2 Generation pipeline

1. **Page extraction** with `scan_pdf_to_docs_with_vlm(..., split_per_page=True)`.
2. **Content filtering**: pages must pass a `min_page_chars` threshold and an alphabetic-density check.
3. **Random page sampling** (`sample_pages`, controlled by seed).
4. **LLM question generation** constrained to selected page snippets.
5. **Validation filters** reject leaked metadata and structurally invalid questions.
6. Conversion to regression schema with one expected citation per generated question.

### 3.3 Constraint engineering

The generation prompt enforces hard anti-leakage rules:

- no explicit file/page/section/chapter references,
- no “in this document” style phrasing,
- question must be answerable from a single chunk/page.

Post-generation filters (`_question_is_valid`) additionally reject:

- file-stem token leakage,
- forbidden lexical markers (`page`, `section`, etc.),
- explicit numeric range and section-like patterns (`x-y`, `1.2.3`).

This two-layer design (prompt constraints + programmatic filters) is critical in practice because LLM-only constraint following is not perfectly reliable.

### 3.4 Failure handling and reliability

Generation runs with retries and multiple attempts. The script fails hard if it cannot collect enough valid questions, preventing silent quality degradation in generated benchmarks.

### 3.5 Evaluation-theoretic caveat

Because questions are synthesized from sampled pages, the dataset primarily measures **retrieval/citation alignment** and less the full distribution of real user intents. It is best used for regression safety nets, not as the sole benchmark for business-quality claims.

---

## 4. Post-Answer Root-Cause Evaluation (`PostAnswerEvaluator`)

### 4.1 Operational role

This component is a **diagnostic judge**, not a response rewriter. It runs after answer rendering (UI path), classifies likely primary failure stage, and persists the result for analysis.

Root-cause labels:

- `NO_ISSUE`
- `RETRIEVAL`
- `RERANK`
- `GENERATION`

### 4.2 Trigger gating

In `ui/agent_runner.py`, evaluation runs only when:

1. `post_answer_evaluation_enabled` is true,
2. request is not session-PDF mode (`has_session_pdf == False`),
3. retrieval provenance indicates hybrid DB signal (`bm25` or `semantic+bm25`).

This avoids evaluating flows where evidence semantics differ (e.g., session-only PDF pipelines).

### 4.3 Evidence packaging

The evaluator receives and injects into prompt:

- `user_request`, `standalone_question`,
- source inventory for pre-rerank and post-rerank docs,
- full formatted retrieved context,
- final answer text.

Important implementation detail: current formatting uses full chunk text (no head/tail clipping), so diagnosis fidelity is higher but token cost scales with retrieved context size.

### 4.4 Decision policy and conservatism

The prompt embeds a strict decision tree with explicit conservative bias:

- default to `NO_ISSUE` under ambiguity,
- assign `GENERATION` only with strong direct evidence that reranker context was sufficient.

This reduces false blame assignment but may under-report subtle failures.

### 4.5 Output normalization and resilience

Model output is parsed from JSON-like text. Safety measures:

- unknown root cause values are normalized to `NO_ISSUE`,
- confidence is clamped to `[0.0, 1.0]`,
- failures are non-blocking (answer flow continues).

The component also handles streamed answer payloads via iterator duplication (`itertools.tee`) to avoid consuming downstream streams.

---

## 5. Persistence and Observability

### 5.1 Oracle persistence model

`core/post_answer_feedback.py` stores evaluator rows in `POST_ANSWER_EVALUATION` with schema evolution support:

- table auto-create if missing,
- additive migration for missing `CONFIDENCE` column.

Stored fields include timestamp, question, root cause, reason, confidence, and config JSON (model IDs, top-k, etc.).

### 5.2 Query surface for analysis

`list_feedback(...)` supports filtering by:

- `root_cause`,
- date interval,
- `llm_model_id` (with backward-compatible fallback to `model_id` in JSON).

This allows controlled comparisons across model versions and periods.

### 5.3 Streamlit inspection page

`pages/post_answer_eval_ui.py` provides:

- filtered table view,
- class distribution statistics,
- CSV export,
- per-record deep inspection (question, reason, config JSON).

This turns evaluation logs into an actionable debugging console.

---

## 6. Methodological interpretation

The current stack approximates a layered evaluation taxonomy:

- **L0 Structural correctness**: execution without node errors.
- **L1 Routing correctness**: expected intent.
- **L2 Evidence provenance**: expected sources and citations.
- **L3 Surface answer constraints**: must-contain rules.
- **L4 Causal diagnosis**: post-answer root-cause inference.

Most RAG projects stop at L3. `advanced_rag_agent` adds L4 with persisted diagnostics, which is the main differentiator for iterative quality engineering.

---

## 7. Practical tuning patterns enabled by this framework

1. **Reranker A/B sensitivity**
Run regression with and without reranker (`--disable-reranker`) and inspect shifts in `citations_recall_micro` plus post-answer `RERANK` incidence.

2. **Model-switch impact audits**
Change answer model or evaluator model and compare root-cause distributions filtered by `llm_model_id`.

3. **Retrieval coverage audits**
Track cases with high intent correctness but low citation recall; this usually indicates retrieval recall gaps rather than generator behavior.

4. **Hybrid-floor validation**
Correlate `HYBRID_MIN_SESSION_DOCS` behavior (from reranker logic) with observed source mix and citation recall on hybrid benchmarks.

---

## 8. Known gaps and technically grounded extensions

High-impact extensions consistent with current architecture:

- Add semantic answer metrics (LLM-as-judge rubric or embedding-based similarity) alongside `must_contain`.
- Record richer provenance IDs (chunk hash / chunk index) to improve citation matching robustness.
- Introduce per-class calibration set for post-answer evaluator to estimate confusion matrix (especially `NO_ISSUE` vs mild failures).
- Add cost/latency telemetry columns for evaluator runs to quantify diagnostic overhead.
- Build nightly regression trend reports from JSON outputs + Oracle logs.

---

## 9. Conclusion

`advanced_rag_agent` implements a mature, engineering-oriented evaluation stack that goes beyond single-score benchmarking. Offline regression provides repeatable gates; synthetic dataset generation accelerates benchmark creation; post-answer root-cause classification provides operational diagnostics with persistent observability.

The core technical value is **stage-aware evaluability**: failures are not only detected, but attributed to likely components. This significantly reduces debugging entropy and supports disciplined iteration on retrieval, reranking, and generation subsystems.
