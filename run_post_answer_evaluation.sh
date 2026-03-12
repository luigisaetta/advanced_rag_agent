PYTHONPATH=$(pwd) python scripts/eval/run_regression_eval.py \
  --dataset eval_data/file14.jsonl \
  --model-id google.gemini-2.5-pro \
  --post-answer-categories-only \
  --post-answer-model-id openai.gpt-5.4

