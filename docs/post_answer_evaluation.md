# Post Answer Evaluation

This document explains how the **Post Answer Evaluation** feature is implemented, when it runs, what it stores, and the current limitations.

## Purpose

Post Answer Evaluation is a **diagnostic step** executed after the final answer is produced.  
It does not modify the final answer shown to the user.  
Its goal is to classify the most likely primary cause of answer quality issues into:

- `NO_ISSUE`
- `RETRIEVAL`
- `RERANK`
- `GENERATION`

It also stores the result in Oracle for later analysis.

## Where It Is Implemented

- Evaluator logic: `agent/post_answer_evaluator.py`
- Evaluation prompt: `agent/prompts.py` (`POST_ANSWER_EVALUATION_TEMPLATE`)
- UI trigger path: `ui/agent_runner.py`
- Oracle persistence: `core/post_answer_feedback.py`
- UI toggle: `ui/sidebar.py`, session init in `ui/session.py`

## Runtime Flow

### 1) Answer generation and rendering

The standard graph produces a final answer.  
In Streamlit, the answer is rendered first.

### 2) Conditional trigger

After rendering, `ui/agent_runner.py` calls `_run_post_answer_evaluation_if_needed(...)`.

Evaluation runs only if all conditions are true:

1. `post_answer_evaluation_enabled` is `True` (UI checkbox in **Options**).
2. The request is not in session-PDF mode (`has_session_pdf == False`).
3. Retrieved docs include hybrid DB signal (`bm25` or `semantic+bm25` retrieval type).

If one condition is false, the evaluator is skipped.

### 3) Input assembly

The evaluator receives:

- user request (`user_request`)
- rewritten question (`standalone_question`)
- retrieved docs before rerank (`retriever_docs`)
- docs after rerank (`reranker_docs`)
- final answer text (`final_answer`)

### 4) Prompt construction

`PostAnswerEvaluator` builds:

- a compact source inventory (source/page/retrieval_type)
- full retrieved context text
- full reranker context text
- final answer text

Important: current implementation uses **full chunk text** (no head/tail clipping).

### 5) LLM evaluation

The evaluator calls the configured model (`model_id`, temperature `0.0`) with `POST_ANSWER_EVALUATION_TEMPLATE`.

Prompt logic is conservative:

- `GENERATION` is selected only with clear direct evidence.
- if evidence is ambiguous/insufficient, fallback is `NO_ISSUE`.

### 6) JSON parse and normalization

The model must return JSON:

```json
{
  "root_cause": "NO_ISSUE|RETRIEVAL|RERANK|GENERATION",
  "reason": "short evidence-based reason"
}
```

If `root_cause` is outside allowed values, code normalizes it to `NO_ISSUE`.

### 7) Oracle persistence

After successful evaluation, code writes one row to Oracle table `POST_ANSWER_EVALUATION`.

If the table does not exist, it is created automatically.

## Oracle Schema and Persistence

Implementation is in `core/post_answer_feedback.py`.

Table name:

- `POST_ANSWER_EVALUATION`

Columns:

- `ID` (identity primary key)
- `CREATED_AT` (`DATE`)
- `QUESTION` (`CLOB`)
- `ROOT_CAUSE` (`VARCHAR2(30)`)
- `REASON` (`CLOB`)
- `CONFIG_JSON` (`CLOB`)

Saved payload:

- `QUESTION`: from `user_request` (fallback to `standalone_question`)
- `ROOT_CAUSE`: normalized class
- `REASON`: short evaluator justification
- `CONFIG_JSON`: essential config JSON (currently includes `model_id`)

Current create/insert behavior:

1. Check `user_tables` for `POST_ANSWER_EVALUATION`.
2. Create table if missing.
3. Insert row.
4. Commit.

If DB write fails, the failure is logged and does not break the user answer flow.

## UI Controls

In Streamlit sidebar:

- **Options** → `Enable Post Answer Evaluation` (default `True`)

This value is propagated into runtime config key:

- `post_answer_evaluation_enabled`

## Configuration Keys

From runtime config:

- `post_answer_evaluation_enabled`: enables/disables feature
- `post_answer_evaluation_max_chars`: legacy key still passed in config
- `model_id`: evaluator model used for scoring

Note:
`post_answer_evaluation_max_chars` currently does not reduce context length because the evaluator now sends full chunk text.

## Error Handling and Resilience

The feature is designed to be non-blocking:

- if evaluator fails, it logs warning and returns empty post-answer fields
- if persistence fails, it logs warning and continues

Therefore, user-facing answer generation remains unaffected.

## Known Limitations

1. **Cost/latency growth**
   Full context increases token usage and evaluation latency, especially with many/large chunks.

2. **Judge-model bias**
   Classification quality depends on LLM judge behavior and prompt interpretation.

3. **UI-path scope**
   Current trigger is in Streamlit runner; API behavior may differ if this helper is not called the same way.

4. **Single-label simplification**
   Only one primary root cause is stored; multi-factor failures are collapsed to one class.

5. **Limited stored diagnostics**
   Only essential config is saved (`model_id` now). Deep forensic analysis may need more metadata later.

## Suggested Extensions

1. Add confidence score and/or secondary root cause.
2. Save compact retrieval diagnostics (`top_k`, source counts, selected doc IDs).
3. Add a lightweight offline calibration set to measure false positives over time.
4. Add dashboard queries on Oracle table to monitor class distribution by model/version.

## Quick Validation Checklist

1. Enable checkbox in UI.
2. Run a question that triggers hybrid DB retrieval.
3. Check logs for `PostAnswerEvaluator result`.
4. Query Oracle table:

```sql
SELECT
  id,
  created_at,
  root_cause,
  SUBSTR(reason, 1, 200) AS reason_preview,
  SUBSTR(config_json, 1, 200) AS config_preview
FROM post_answer_evaluation
ORDER BY id DESC;
```

