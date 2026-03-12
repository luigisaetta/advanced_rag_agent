PYTHONPATH=$(pwd) python scripts/eval/generate_regression_from_pdf.py \
  --pdf-path /Users/lsaetta/Progetti/work-iren/download_feb2026/PRG_1_UNI_10611_1997.pdf \
  --out eval_data/file14.jsonl \
  --sample-pages 10 \
  --questions 10 \
  --min-page-chars 450 \
  --model-id openai.gpt-5.2 \
  --id-prefix kb_auto \
  --start-id 131 \
  --seed 42
