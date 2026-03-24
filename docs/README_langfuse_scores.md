# Langfuse Score Integration (Post-Answer Evaluation)

This document explains how the project writes post-answer evaluation scores to Langfuse, including:

1. `post_answer_root_cause` (categorical)
2. `post_answer_confidence` (numeric)
3. `post_answer_quality_score` (numeric, 1-10)

## Goal

For each user interaction, keep two traces:

1. Main RAG trace (user answer generation)
2. Post-answer evaluation trace

Then attach evaluation scores to the **main RAG trace** so answer quality can be analyzed directly where the user-facing response is produced.

## High-Level Flow

1. UI opens the main trace (`rag_request`) and captures its `trace_id`.
2. UI passes that `trace_id` into runtime config as `post_answer_target_trace_id`.
3. Post-answer evaluator computes:
   - `root_cause`
   - `confidence`
   - `quality_score` (1-10)
4. Post-answer evaluator writes all three scores onto the target main trace via Langfuse API.

## Prompt Contract (LLM Output)

The post-answer evaluator prompt enforces JSON output including `quality_score`:

```json
{
  "root_cause": "NO_ISSUE|RETRIEVAL|RERANK|GENERATION",
  "reason": "short evidence-based reason",
  "confidence": 0.0,
  "quality_score": 1
}
```

`quality_score` is normalized to integer range `[1, 10]` in code.

## How Post-Answer Evaluation Is Computed

The evaluation is implemented as **LLM-as-a-judge** in the `PostAnswerEvaluator` node.

Code location:
- `agent/post_answer_evaluator.py`

Prompt location (source of criteria/rubric):
- `agent/prompts.py` -> `POST_ANSWER_EVALUATION_TEMPLATE`

Execution model:
1. The evaluator receives:
   - user request
   - standalone question
   - retriever docs (before reranker)
   - reranker docs (used for final answer)
   - final answer text
2. It builds a structured evaluation prompt and calls a dedicated evaluation model:
   - config key: `post_answer_evaluation_model_id`
   - fallback config: `POST_ANSWER_EVALUATION_MODEL_ID`
3. The model must return strict JSON with:
   - `root_cause` (`NO_ISSUE|RETRIEVAL|RERANK|GENERATION`)
   - `reason` (short evidence-based explanation)
   - `confidence` (`0.0..1.0`)
   - `quality_score` (`1..10`)
4. Output is normalized in code to avoid invalid values before logging and scoring.

### Root-Cause Criteria (Prompt Decision Tree)

The rubric in `POST_ANSWER_EVALUATION_TEMPLATE` applies this order:
1. `NO_ISSUE` if retrieval/rerank evidence is relevant and answer is grounded.
2. `RETRIEVAL` if retrieved evidence is missing/off-topic.
3. `RERANK` if retrieval has key evidence but reranker drops/under-prioritizes it.
4. `GENERATION` only when reranker evidence is sufficient but final answer is wrong/incomplete.
5. Default to `NO_ISSUE` when evidence is ambiguous.

### Numeric Quality Score Criteria (1-10)

The prompt instructs the judge to rate overall answer quality using:
1. Accuracy
2. Grounding in provided evidence
3. Completeness
4. Relevance to request
5. Clarity

Reference anchors in prompt:
- `1` very poor
- `5` acceptable/partial
- `7` good
- `9` excellent
- `10` near-perfect

## Key Code Snippets

### 1) Capture main trace id and pass it forward

Source: `ui/agent_runner.py`

```python
tracing_context = (
    langfuse_span(service_name=config.AGENT_NAME, span_name="rag_request")
    if tracing_enabled
    else nullcontext()
)
main_trace_id = None
with tracing_context:
    annotate_current_observation(
        input_data={"question": question},
        metadata={
            "thread_id": st.session_state.thread_id,
            "model_id": st.session_state.model_id,
            "collection_name": st.session_state.collection_name,
        },
    )
    main_trace_id = get_current_trace_id()
    if main_trace_id:
        agent_config.setdefault("configurable", {})[
            "langfuse_parent_trace_id"
        ] = main_trace_id

# later, when scheduling post-answer eval
_run_post_answer_evaluation_if_needed(
    by_step=by_step,
    question=question,
    final_answer=full_response,
    agent_config=agent_config,
    logger=logger,
    parent_trace_id=main_trace_id,
)
```

### 2) Compute and write scores in post-answer evaluator

Source: `agent/post_answer_evaluator.py`

```python
root_cause = self._normalize_cause(parsed.get("root_cause"))
reason = str(parsed.get("reason", "") or "").strip()
confidence = self._normalize_confidence(parsed.get("confidence"))
quality_score = self._normalize_quality_score(parsed.get("quality_score"))

# write scores on main RAG trace
target_trace_id = str(configurable.get("post_answer_target_trace_id", "") or "").strip()
if target_trace_id:
    create_trace_score(
        trace_id=target_trace_id,
        name="post_answer_root_cause",
        value=root_cause,
        data_type="CATEGORICAL",
        comment=reason or None,
        metadata={"confidence": confidence},
    )
    create_trace_score(
        trace_id=target_trace_id,
        name="post_answer_confidence",
        value=confidence,
        data_type="NUMERIC",
        metadata={"root_cause": root_cause},
    )
    create_trace_score(
        trace_id=target_trace_id,
        name="post_answer_quality_score",
        value=quality_score,
        data_type="NUMERIC",
        metadata={"root_cause": root_cause, "confidence": confidence},
    )
    flush_observability()
```

### 3) Langfuse helper that creates trace-bound scores

Source: `core/observability.py`

```python
def create_trace_score(
    *,
    trace_id: str,
    name: str,
    value: str | float,
    data_type: str = "CATEGORICAL",
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not (
        trace_id and name and _LANGFUSE_AVAILABLE and _is_enabled() and _is_configured()
    ):
        return
    try:
        client = _get_client()
        if client is None:
            return
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
            metadata=metadata,
        )
    except Exception:
        return
```

## Data Types and Semantics

1. `post_answer_root_cause`
   - Type: `CATEGORICAL`
   - Values: `NO_ISSUE`, `RETRIEVAL`, `RERANK`, `GENERATION`
2. `post_answer_confidence`
   - Type: `NUMERIC`
   - Range: `0.0` to `1.0`
3. `post_answer_quality_score`
   - Type: `NUMERIC`
   - Range: integer `1` to `10`

## Why both categorical and numeric

1. Categorical score (`root_cause`) is excellent for diagnostics and slicing by failure mode.
2. Numeric scores (`confidence`, `quality_score`) are better for trend charts, thresholds, alerts, and aggregate KPIs.

## Verification Checklist

1. Run one user query in UI.
2. Confirm two traces are visible:
   - main RAG trace
   - post-answer evaluation trace
3. Open the main RAG trace and verify scores:
   - `post_answer_root_cause`
   - `post_answer_confidence`
   - `post_answer_quality_score`
4. Confirm `post_answer_quality_score` is between `1` and `10`.
