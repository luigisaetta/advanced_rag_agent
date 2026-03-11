# Regression Evaluation (JSONL)

This document explains how to run regression tests with `eval_data` and how to run the script with or without post-answer evaluation.

## Files

- Dataset example: `eval_data/regression_sample.jsonl`
- Runner script: `scripts/eval/run_regression_eval.py`

## Dataset Format (JSONL)

One JSON object per line.

Required fields:

- `id`: unique test id
- `question`: user request

Optional fields:

- `session_pdf_path`: absolute path of a PDF loaded in memory for that case
- `expected_intent`: `GLOBAL_KB | SESSION_DOC | HYBRID`
- `expected_sources`: list with `kb` and/or `session_pdf`
- `expected_citations`: list like `{"source":"Doc.pdf","page":"12"}`
- `must_contain`: list of terms expected in the final answer

## Standard Regression Run (No Post-Answer Evaluation)

Use this when you want only the existing regression metrics (`pass_rate`, citations, intent/source checks):

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl \
  --disable-post-answer-evaluation
```

Optional:

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl \
  --disable-post-answer-evaluation \
  --out eval_data/regression_results.json \
  --model-id openai.gpt-5.2 \
  --collection-name COLL01 \
  --max-cases 10
```

## Regression Run With Post-Answer Evaluation

Use this when you also want root-cause classification from the same post-answer evaluator logic (`NO_ISSUE`, `RETRIEVAL`, `RERANK`, `GENERATION`):

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl
```

You can choose a dedicated evaluator model:

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl \
  --post-answer-model-id openai.gpt-5.4
```

## Categories-Only Mode (Post-Answer Evaluation)

If you only want one category per question:

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/regression_sample.jsonl \
  --post-answer-categories-only \
  --out eval_data/post_answer_categories.json
```

Output items include:

- `id`
- `question`
- `root_cause` (`NO_ISSUE | RETRIEVAL | RERANK | GENERATION`)

Note: `--post-answer-categories-only` requires post-answer evaluation enabled (you cannot combine it with `--disable-post-answer-evaluation`).

## Other Useful Flags

- `--disable-reranker`: run A/B checks without reranking
- `--verbose`: print detailed mismatch diagnostics

## Output

Default output file: `eval_data/regression_results.json`.

Standard mode report includes:

- summary metrics (`pass_rate`, `intent_ok_rate`, `sources_ok_rate`, `citations_ok_rate`, citation recall metrics, reranker docs stats, `errors_count`)
- per-case details (`predicted_intent`, `observed_sources`, `node_error`, `pass`)
- `post_answer_root_cause` when post-answer evaluation is enabled

Categories-only mode report includes:

- dataset and model metadata
- per-question category list only

## Notes

- `SESSION_DOC` and `HYBRID` cases require a valid `session_pdf_path`.
- Run commands from project root.
- The script uses values from `config.py` unless overridden by CLI flags.
