PYTHONPATH=$(pwd) python scripts/eval/generate_regression_from_pdf.py \
  --pdf-path /Users/lsaetta/Progetti/work-iren/download_feb2026/PRG_1_UNI_9571-1_2012.pdf \
  --out eval_data/file11.jsonl \
  --sample-pages 10 \
  --questions 10 \
  --min-page-chars 450 \
  --model-id openai.gpt-5.2 \
  --id-prefix kb_auto \
  --start-id 101 \
  --seed 42
