# Uploaded PDF Handling

This document consolidates:

1. Current implementation details for uploaded PDF handling.
2. Recommended retrieval strategy and hardening notes for PoC evolution.

Related document:
- [Advanced Analysis](./advanced_analysis.md)
- [Post Answer Evaluation](./post_answer_evaluation.md)
- [Generate Regression Dataset](./generate_regression.md)
- [Langfuse Observability Setup](./README_langfuse_observability.md)

## Overview

The app supports an optional session-scoped PDF workflow:

1. User uploads a PDF from the Streamlit sidebar.
2. The PDF is scanned page-by-page with a VLM (OCR-like extraction).
3. Extracted text is split into chunks and stored in an in-memory vector store.
4. At question time, the graph decides retrieval strategy (`GLOBAL_KB`, `SESSION_DOC`, `HYBRID`).
5. References are shown in the sidebar with source/page metadata.

The session PDF store is not persistent: it lives only in the current app session.

## Current Implementation

### Upload and Scan Flow

Main entrypoint: [assistant_ui.py](../assistant_ui.py)

1. User uploads file (`st.sidebar.file_uploader`).
2. On `Scan PDF in memory`:
   - uploaded bytes are saved to a temporary file;
   - `scan_pdf_to_docs_with_vlm(...)` is called;
   - chunks are embedded and indexed in `InMemoryVectorStore`;
   - session state is updated with:
     - `session_pdf_vector_store`
     - `session_pdf_name`
     - `session_pdf_chunks_count`

Important: chunk metadata `source` uses the real uploaded filename (not temp filename).

### VLM Extraction and Chunk Metadata

Core module: [session_pdf_vlm.py](../core/session_pdf_vlm.py)

`scan_pdf_to_docs_with_vlm(...)`:

1. Opens the PDF with `pypdfium2`.
2. Renders each page to PNG.
3. Sends page image to VLM.
4. Splits extracted text into chunks.
5. Produces `Document` objects with metadata:
   - `source`: uploaded filename
   - `page_label`: page number (1-based)
   - `retrieval_type`: `session_pdf_vlm`

Guardrail: page count is limited by `SESSION_PDF_MAX_PAGES` from [config.py](../config.py).

### Graph Routing With Intent Classification

Workflow module: [rag_agent.py](../agent/rag_agent.py)

Current high-level graph:

`Moderator -> QueryRewrite -> IntentClassifier -> (Search | SessionSearch | HybridFlow) -> Rerank -> Answer`

#### Intent Definition and Call Sequences

Intent is defined by [intent_classifier.py](../agent/intent_classifier.py):

- If no session PDF is present in memory, intent is forced to `GLOBAL_KB`.
- If session PDF is present, a dedicated intent model classifies into one of:
  - `GLOBAL_KB`
  - `SESSION_DOC`
  - `HYBRID`

Graph-style sequences for each intent:

```text
GLOBAL_KB
START -> Moderator -> QueryRewrite -> IntentClassifier -> Search -> HybridSearch -> Rerank -> Answer -> END

SESSION_DOC
START -> Moderator -> QueryRewrite -> IntentClassifier -> SessionSearch -> Rerank -> Answer -> END

HYBRID
START -> Moderator -> QueryRewrite -> IntentClassifier -> HybridFlow -> Rerank -> Answer -> END
HybridFlow subgraph: HybridQueryBuilder -> Search -> HybridSearch -> HybridSessionSearch -> HybridDocsMerge
```

Intent classifier node: [intent_classifier.py](../agent/intent_classifier.py)

- If no session PDF is available, it forces `GLOBAL_KB` (no classifier LLM call).
- If session PDF is available, it calls the dedicated intent model and returns:
  - `GLOBAL_KB`
  - `SESSION_DOC`
  - `HYBRID`

Current behavior by intent:

1. `GLOBAL_KB`: existing DB retrieval path (`Search` + `HybridSearch`) is used.
2. `SESSION_DOC`: session-only retrieval path is used (`SessionSearch`).
3. `HYBRID`: routed through dedicated `HybridFlow` subgraph (`HybridQueryBuilder -> Search -> HybridSearch -> HybridSessionSearch -> HybridDocsMerge`); `HybridSearch` remains KB-only (semantic + BM25), session chunks are retrieved in `HybridSessionSearch`, then merged in `HybridDocsMerge`.

### Session-Only Retrieval

Node: [session_vector_search.py](../agent/session_vector_search.py)

- Uses `session_pdf_vector_store.similarity_search(...)`.
- Annotates chunks with `retrieval_type = session_pdf`.
- Returns docs in the same serializable shape as other retrieval nodes.

### HYBRID Retrieval (Implemented)

Node: [hybrid_search.py](../agent/hybrid_search.py)

1. Starts from DB semantic candidates produced by `Search`.
2. If `ENABLE_HYBRID_SEARCH` is enabled, adds BM25 candidates and deduplicates.
3. If intent is `HYBRID`, also queries the in-memory session vector store and appends non-duplicate session chunks.
4. Uses conservative session budget via `HYBRID_SESSION_TOP_K` in [config.py](../config.py).
5. Logs merged counts by source (`semantic`, `bm25`, `session`, `merged`).

This is an additive strategy and keeps current DB retrieval behavior stable.

### BM25 Cache Persistence

BM25 indexes are persisted to a serialized cache file (`bm25_cache.pkl`) under `BM25_CACHE_DIR`.

- If the file exists at startup, cache entries are loaded from disk.
- If the file is missing (or incomplete), indexes are built from DB for configured collections and then saved.

Default local path is `bm25_cache` (see `BM25_CACHE_DIR` in [config.py](../config.py)).
In Docker Compose, UI and MCP use separate host folders:
- `bm25_cache/ui`
- `bm25_cache/mcp`

### References Behavior

Reference rendering is in [assistant_ui.py](../assistant_ui.py).

- For DB-backed references: source/page/link are shown (when page is parseable).
- For session-PDF references (`retrieval_type` starts with `session_pdf`):
  - only `source`, `page`, and `retrieval_type` are shown;
  - no link is generated (by design, in-memory docs have no static citation URL).

### API Notes

The FastAPI endpoint in [rag_agent_api.py](../rag_agent_api.py) does not currently pass an in-memory session vector store, so it follows `GLOBAL_KB` behavior.

## Recommended Retrieval Sequence and Hardening

This section defines the recommended end-to-end sequence to handle user requests when an uploaded PDF is available in session memory, while protecting existing behavior and avoiding regressions.

Note: this section is forward-looking. The current implementation already uses
`GLOBAL_KB` / `SESSION_DOC` / `HYBRID` intent labels and `HYBRID_SESSION_TOP_K`.
A deterministic rule-based classifier is an optional future refinement.

### Goals

1. Preserve current retrieval quality and behavior for existing DB-backed RAG.
2. Add uploaded-PDF support as an optional enhancement, not a replacement.
3. Keep latency controlled and predictable.
4. Avoid brittle routing decisions that could skip important retrieval paths.
5. Keep architecture maintainable with minimal disruption to the current agent.

### Non-Goals

1. Replacing the existing semantic/BM25 pipeline.
2. Building persistent per-user document memory across restarts.
3. Hard-routing all requests exclusively to uploaded PDF or DB sources.

### Current Baseline (Before Session-PDF Retrieval Integration)

1. User question enters the graph.
2. Query rewrite.
3. Semantic search (+ BM25/hybrid path when enabled).
4. Global reranker.
5. Final answer generation.

Session PDF scanning and in-memory indexing exist in UI, but those chunks are not yet fused into retrieval in the baseline.

### Recommended Sequence (Target Behavior)

#### 1) Pre-Check

1. If no session PDF is loaded, execute the current pipeline unchanged.
2. If session PDF exists, continue with the augmented flow below.

#### 2) Query Intent Classification (Current: LLM-Based)

1. Classify request as one of:
   - `GLOBAL_KB`
   - `SESSION_DOC`
   - `HYBRID`
2. Current implementation uses a dedicated LLM (`INTENT_MODEL_ID`) when a session PDF is present.
3. If no session PDF is present, classifier is skipped and intent is forced to `GLOBAL_KB`.
4. A deterministic rule-based classifier can be considered later as an optimization.

#### 3) Multi-Source Retrieval (Always Keep Baseline Path)

1. Always run existing DB retrieval:
   - semantic path
   - BM25/hybrid path (as currently configured)
2. If session PDF exists, also query the session in-memory vector store.
3. Session retrieval `k` is currently conservative and fixed for `HYBRID`:
   - `HYBRID_SESSION_TOP_K` (default: `3`)

Important: DB retrieval budget must not be reduced by introducing session retrieval.

#### 4) Merge and Deduplicate

1. Merge candidates from:
   - semantic retrieval
   - BM25/hybrid retrieval
   - session-PDF retrieval
2. Deduplicate by normalized content (same approach as existing hybrid dedupe logic).
3. Preserve source metadata so references remain explainable:
   - `retrieval_type = semantic`
   - `retrieval_type = bm25`
   - `retrieval_type = session_pdf_vlm`

#### 5) Candidate Capping (Latency Guardrail)

1. Apply a hard cap to total candidates before reranker.
2. Cap should be configurable (example: 20 or 25).
3. Cap strategy should keep DB non-regression:
   - first keep DB baseline candidates
   - then add session candidates up to cap
   - do not shrink existing DB baseline below current settings

#### 6) Single Reranker Pass

1. Use one global reranker call over merged candidates.
2. Avoid double reranking in initial design to control latency/cost.
3. Let reranker perform final relevance selection across all sources.

#### 7) Final Answer Generation

1. Use reranker-selected chunks to build context.
2. Generate final answer as in current flow.
3. References should include source and retrieval type, without exposing raw chunk text in UI.

### Why This Sequence

1. Avoids hard routing errors:
   - misclassification does not skip DB retrieval.
2. Minimizes regression risk:
   - existing DB path remains intact.
3. Controls performance:
   - intent-based session `k` + total cap + one reranker pass.
4. Keeps implementation incremental:
   - no full redesign required.

### Risk Analysis

#### Accuracy Risks

1. Intent misclassification:
   - Mitigation: classification adjusts budgets only; does not disable DB path.
2. Reranker dropping useful context:
   - Mitigation: tune `HYBRID_SESSION_TOP_K` and add configurable caps.
3. VLM OCR imperfections:
   - Mitigation: page limits, prompt refinement, observability on OCR yield.

#### Performance Risks

1. Larger candidate set increases reranker latency.
   - Mitigation: intent-based budgets and total cap.
2. Session retrieval overhead.
   - Mitigation: small fixed `HYBRID_SESSION_TOP_K`; query session store only for `HYBRID`.

#### Operational Risks

1. Session memory growth with large PDFs.
   - Mitigation: `SESSION_PDF_MAX_PAGES`, optional file size cap, single active session doc in POC.
2. Non-portability across replicas/restarts.
   - Accepted for current session-scoped design.

### Non-Regression Policy

To ensure current behavior is not degraded:

1. DB retrieval settings remain unchanged.
2. Session retrieval is additive, never substitutive.
3. Feature flag available for quick disablement.
4. Instrumentation required before/after enablement.

### Suggested Config Controls

Implemented now:
1. `ENABLE_HYBRID_SEARCH` (bool)
2. `HYBRID_TOP_K` (BM25 budget)
3. `HYBRID_SESSION_TOP_K` (session chunk budget when intent is `HYBRID`)
4. `SESSION_PDF_MAX_PAGES` (already present)

Proposed next hardening controls:
1. `MAX_CANDIDATES_BEFORE_RERANK` (global cap)
2. Optional: `ENABLE_SESSION_PDF_INTENT_CLASSIFIER` (bool)

### Observability Requirements

Log per request:

1. Session PDF presence (yes/no)
2. Intent decision (`GLOBAL_KB`, `SESSION_DOC`, or `HYBRID`)
3. Candidate counts by source:
   - semantic
   - bm25/hybrid
   - session
4. Merged count and post-dedupe count
5. Pre-reranker capped count
6. Step timings:
   - session retrieval
   - reranker
   - final generation
7. Retry attempts/errors (already partially implemented for LLM calls)

### Rollout Strategy

#### Phase 1 (Safe Intro)

1. Keep feature disabled by default.
2. Enable in controlled test environment.
3. Use conservative settings:
   - low `HYBRID_SESSION_TOP_K` (e.g. `3`)
4. Validate:
   - no regression on normal Q/A
   - improved behavior on uploaded-doc queries

#### Phase 2 (Tune)

1. Analyze logs and latency distributions.
2. Tune session budgets and cap.
3. Expand summary budget only if needed.

#### Phase 3 (Production Hardening)

1. Add tests for intent rules and merge/dedupe behavior.
2. Add regression suite with and without session PDF.
3. Evaluate multi-instance strategy if session continuity is needed.

### Test Scenarios

1. No uploaded PDF:
   - behavior identical to baseline.
2. Uploaded PDF + targeted Q/A:
   - answer uses session doc when relevant.
3. Uploaded PDF + generic unrelated question:
   - DB path still returns correct context.
4. Uploaded PDF + summary request:
   - enough session chunks included to support broad summary.
5. Large uploaded PDF near page limit:
   - accepted up to limit, rejected beyond limit.
6. Error paths:
   - session retrieval failures do not break baseline DB response path.

### Decision Summary

Use a hybrid additive retrieval strategy:

1. Always keep existing DB retrieval.
2. Add session retrieval only when PDF exists.
3. Tune session contribution by intent.
4. Merge + dedupe + cap.
5. Single reranker pass.
6. Maintain strict non-regression guarantees.
