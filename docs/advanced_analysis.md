# Advanced Analysis

This document explains the **Advanced Analysis** feature end-to-end:
- when it is activated,
- how routing works,
- how planning and execution are performed,
- what data is passed between nodes,
- how final output is generated.

## Purpose

Advanced Analysis is a dedicated execution path for complex requests that require:
1. Evidence from the uploaded in-session PDF, and
2. Complementary evidence from the global KB.

It is designed for structured multi-step reasoning (plan + execution), instead of a single retrieval/answer pass.

## Activation Conditions

The graph routes to `AdvancedAnalysisFlow` only when all of the following are true:
1. A session PDF is loaded in memory (`has_session_pdf = true`)
2. Intent classifier output is `HYBRID`
3. UI checkbox **Advanced Analysis** is enabled (`enable_advanced_analysis = true`)

Otherwise:
1. `GLOBAL_KB` and `SESSION_DOC` follow their standard paths
2. `HYBRID` follows `HybridFlow` (non-advanced)

## Graph Integration

In the parent graph (`agent/rag_agent.py`), `AdvancedAnalysisFlow` is a dedicated subgraph node.

Current subgraph structure:
1. `Planner`
2. `AdvancedAnalysis`
3. `FinalSynthesis`

The first step is intentionally called **Planner** as requested.

## Dedicated Subgraph State

Advanced Analysis uses a dedicated state schema:
- `agent/advanced_analysis_state.py`

This state is separate from the main graph schema and includes:
1. request/routing fields (`user_request`, `standalone_question`, `search_intent`, ...)
2. execution fields (`advanced_plan`, `advanced_step_outputs`, `final_answer`, `citations`, `error`)

## Inputs Passed to Advanced Analysis

At runtime, the UI passes the following relevant config values:
1. `session_pdf_docs`: full serialized list of uploaded PDF chunks (not top-k subset)
2. `session_pdf_vector_store`: in-memory vector index (used for robust fallback retrieval)
3. `collection_name`: KB target collection
4. `model_id`: active LLM
5. planner/executor controls:
   - `advanced_analysis_max_actions`
   - `advanced_analysis_kb_top_k`
   - `advanced_analysis_step_max_words`

## Planner Node

Node: `AdvancedPlanner` in `agent/advanced_analysis.py`

### Planner Responsibilities

1. Read user request
2. Read **all session PDF chunks** from config (`session_pdf_docs`)
3. Build a planning prompt (`ADVANCED_ANALYSIS_PLANNER_TEMPLATE`)
4. Call LLM and parse JSON plan
5. Normalize and validate plan actions
6. Log final plan

### Plan Output Schema

Each action includes:
1. `step` (sequence number)
2. `section` (human-readable section label)
3. `chunk_numbers` (pointers to chunk indices from the serialized session input)
4. `objective` (what to produce in this step)
5. `kb_search_needed` (`true/false`)
6. `kb_query` (query string if KB search is required)

### Action Limit

The planner enforces max actions from:
- `ADVANCED_ANALYSIS_MAX_ACTIONS` (default in `config.py`)

## Execution Node (AdvancedAnalysisRunner)

Node: `AdvancedAnalysisRunner` in `agent/advanced_analysis.py`

It executes the plan step-by-step.

For each step:
1. Select PDF chunks by `chunk_numbers`
2. Expand local context with neighbor chunks (`±1`)
3. If chunk selection is weak/empty, run robust fallback retrieval against session vector store
4. If `kb_search_needed = true`, perform KB vector search (`top_k` controlled by config)
5. Build step prompt (`ADVANCED_ANALYSIS_STEP_TEMPLATE`) using:
   - user request
   - step metadata
   - selected PDF evidence
   - retrieved KB evidence
6. Generate concise step output (max words controlled by config)
7. Collect citations from both sources (session + KB)

### Per-step Logging

A compact line is logged per step:
1. step id
2. elapsed seconds
3. number of PDF chunks used
4. whether KB search was required
5. number of KB docs retrieved

## Final Synthesis Node

Node: `AdvancedFinalSynthesis` in `agent/advanced_analysis.py`

Responsibilities:
1. Read all per-step outputs (`advanced_step_outputs`)
2. Generate a concise cross-step synthesis with a dedicated LLM prompt
3. Compose final output as:
   - all step outputs
   - `## Final Synthesis` section
4. Return final answer and propagated citations

## Robustness in PDF Chunk Handling

To reduce failure cases where the wrong section is passed:
1. `chunk_numbers` are normalized and validated
2. Neighbor chunk expansion adds continuity for section context
3. If selected evidence is empty/too short, session retrieval fallback is used
4. Fallback docs are merged with deduplication by normalized content

This improves resilience when planner pointers are incomplete or noisy.

## Final Output Generation

Current behavior:
1. Each step produces a short markdown section:
   - `### Step N - <section>`
2. `AdvancedFinalSynthesis` adds a final section:
   - `## Final Synthesis`
3. Final answer contains both:
   - all step outputs in order
   - the final synthesis section
4. Final citations are the union of per-step citations

## Configuration

Current advanced-analysis parameters in `config.py`:
1. `ADVANCED_ANALYSIS_MAX_ACTIONS`
2. `ADVANCED_ANALYSIS_KB_TOP_K`
3. `ADVANCED_ANALYSIS_STEP_MAX_WORDS`

Related retrieval settings still reused:
1. `HYBRID_SESSION_TOP_K` (used in session fallback retrieval)

## UI Behavior

In Streamlit:
1. `Advanced Analysis` checkbox is available in sidebar
2. Default is `False`
3. When active and routing conditions are met, Advanced Analysis path is used

## Regression Runner Notes

The regression runner currently passes:
1. `enable_advanced_analysis = false` by default
2. Session docs payload and advanced-analysis config values

This preserves backward compatibility for existing regression sets unless explicitly enabled later.

## Current Limitations

1. Planner quality depends on LLM JSON compliance and chunk indexing quality
2. Final synthesis quality depends on per-step output quality
3. KB search per step is single-pass semantic retrieval (no reranker in this subgraph)
4. Advanced analysis path is UI-enabled; production policy controls may still be needed

## Suggested Next Improvements

1. Add confidence/evidence scoring per step and in final synthesis
2. Add explicit validation for chunk pointers against page/section metadata
3. Add dedicated regression suite with advanced-analysis enabled
4. Add telemetry dashboard for plan quality and step-level latency
5. Add policy controls for advanced-analysis activation by tenant/use case

## Callable Usage From Python

Advanced Analysis can also be executed directly from a Python script, without going through the main graph routing.

Use:
1. `create_advanced_analysis_agent(...)` from `agent/advanced_analysis_agent.py`
2. `agent.invoke(initial_state, config=...)`

Important behavior note (current implementation):
1. Planner may still set `kb_search_needed=true` for some steps.
2. If that happens, runner performs KB retrieval.
3. So a "session-only" run is possible in practice, but not yet enforced by a hard runtime switch.

In other words:
1. If your prompt leads planner to keep `kb_search_needed=false`, no KB access is used.
2. If you need strict "never access KB", add a dedicated guard flag in code (future change).

## Complete Python Example

The example below:
1. Reads a local PDF from filesystem.
2. Scans it with the same VLM pipeline used in UI.
3. Builds serialized chunks (`session_pdf_docs`).
4. Builds an in-memory vector store for fallback retrieval.
5. Calls the advanced-analysis agent with initial state + config.

```python
import os

from langchain_core.vectorstores import InMemoryVectorStore

from agent.advanced_analysis_agent import create_advanced_analysis_agent
from core.session_pdf_vlm import scan_pdf_to_docs_with_vlm
from core.oci_models import get_embedding_model
from core.utils import docs_serializable
from config import (
    LLM_MODEL_ID,
    VLM_MODEL_ID,
    DEFAULT_COLLECTION,
    ADVANCED_ANALYSIS_MAX_ACTIONS,
    ADVANCED_ANALYSIS_KB_TOP_K,
    ADVANCED_ANALYSIS_STEP_MAX_WORDS,
)


def run_advanced_analysis_on_file(pdf_path: str, question: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Scan full document with VLM OCR + chunking (no page limit here).
    docs, page_count = scan_pdf_to_docs_with_vlm(
        pdf_path=pdf_path,
        vlm_model_id=VLM_MODEL_ID,
        max_pages=-1,
        source_name=os.path.basename(pdf_path),
    )
    print(f"Scanned pages={page_count}, chunks={len(docs)}")

    # 2) Build serialized chunks expected by advanced-analysis config.
    session_pdf_docs = docs_serializable(docs)

    # 3) Build in-memory vector store used by advanced-analysis fallback retrieval.
    embed_model = get_embedding_model()
    session_pdf_vector_store = InMemoryVectorStore(embedding=embed_model)
    if docs:
        session_pdf_vector_store.add_documents(docs)

    # 4) Build callable advanced-analysis agent.
    advanced_agent = create_advanced_analysis_agent()

    # 5) Initial state (AdvancedAnalysisState-compatible fields).
    initial_state = {
        "user_request": question,
        "standalone_question": question,
        "search_intent": "SESSION_DOC",
        "has_session_pdf": True,
        "advanced_analysis_enabled": True,
        "retriever_docs": [],
        "session_retriever_docs": [],
        "advanced_plan": [],
        "advanced_step_outputs": [],
        "final_answer": "",
        "citations": [],
        "error": None,
    }

    # 6) Config payload consumed by planner/runner/synthesis nodes.
    run_config = {
        "configurable": {
            "model_id": LLM_MODEL_ID,
            "main_language": "same as the question",
            "collection_name": DEFAULT_COLLECTION,
            "session_pdf_docs": session_pdf_docs,
            "session_pdf_vector_store": session_pdf_vector_store,
            "advanced_analysis_max_actions": ADVANCED_ANALYSIS_MAX_ACTIONS,
            "advanced_analysis_kb_top_k": ADVANCED_ANALYSIS_KB_TOP_K,
            "advanced_analysis_step_max_words": ADVANCED_ANALYSIS_STEP_MAX_WORDS,
        }
    }

    result = advanced_agent.invoke(initial_state, config=run_config)

    print("----- FINAL ANSWER -----")
    print(result.get("final_answer", ""))
    print("----- CITATIONS -----")
    print(result.get("citations", []))
    print("----- ERROR -----")
    print(result.get("error"))

    return result


if __name__ == "__main__":
    PDF_PATH = "/absolute/path/to/your/document.pdf"
    QUESTION = "Analyze the entire document and provide a structured synthesis."
    run_advanced_analysis_on_file(PDF_PATH, QUESTION)
```

### Notes For Session-Only Runs

If you want to minimize KB usage today:
1. Ask explicitly for document-only analysis in the user question.
2. Keep `search_intent` as `SESSION_DOC` in initial state.
3. Still provide a valid `collection_name` in config because current runner can query KB if planner requests it.
