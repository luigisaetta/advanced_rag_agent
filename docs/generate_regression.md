# Generate Regression Dataset From PDF

This document explains how to generate a regression JSONL file (same format as files in `eval_data/`) from a single PDF.

Script:

- `scripts/eval/generate_regression_from_pdf.py`

## What The Script Does

1. Takes one PDF full path as input.
2. Extracts text page-by-page using the VLM OCR pipeline.
3. Filters out pages that are too empty.
4. Randomly samples `N` candidate pages.
5. Generates questions grounded on sampled pages.
6. Writes a JSONL dataset compatible with `run_regression_eval.py`.

Output row format:

```json
{
  "id": "kb_auto_001",
  "question": "....",
  "expected_intent": "GLOBAL_KB",
  "expected_sources": ["kb"],
  "expected_citations": [{"source": "MyDoc.pdf", "page": "12"}]
}
```

## Rules Enforced

- Questions must not mention file/document name.
- Questions must not mention page numbers.
- Questions must not mention section/chapter numbering.
- Questions must be based on concepts/information present in sampled page text.
- Expected citations are single page/chunk values (no ranges).

## Complete Example

Run from project root:

```bash
PYTHONPATH=$(pwd) python scripts/eval/generate_regression_from_pdf.py \
  --pdf-path /absolute/path/to/PRG_1_07_LineeGuidaCIG_2020.pdf \
  --out eval_data/generated_prg107.jsonl \
  --sample-pages 6 \
  --questions 10 \
  --min-page-chars 450 \
  --model-id openai.gpt-oss-120b \
  --id-prefix kb_auto \
  --start-id 101 \
  --seed 42
```

## Parameters

- `--pdf-path` (required): absolute PDF path.
- `--out` (required): output JSONL file.
- `--sample-pages` (default `6`): number of random non-empty pages sampled.
- `--questions` (default `10`): number of questions to generate.
- `--min-page-chars` (default `450`): minimum characters for a page to be considered non-empty.
- `--model-id` (default `config.LLM_MODEL_ID`): model used to generate questions.
- `--id-prefix` (default `kb_auto`): id prefix for generated rows.
- `--start-id` (default `1`): starting numeric id.
- `--seed` (default `42`): random seed for page sampling.

## Important Note About Models

- OCR page extraction uses `config.VLM_MODEL_ID` (currently `openai.gpt-5.2` unless changed in config).
- `--model-id` controls only the question-generation model.

## Suggested Validation

After generating the JSONL, run regression evaluation:

```bash
PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/generated_prg107.jsonl \
  --collection-name COLL01 \
  --verbose
```

