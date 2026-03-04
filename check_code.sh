#!/usr/bin/env bash
set -euo pipefail

py_files=()

if command -v rg >/dev/null 2>&1; then
  while IFS= read -r file; do
    py_files+=("$file")
  done < <(rg --files -g "*.py")
else
  while IFS= read -r file; do
    py_files+=("$file")
  done < <(find . -type f -name "*.py" -not -path "*/.*/*" | sed 's|^\./||')
fi

if [[ ${#py_files[@]} -eq 0 ]]; then
  echo "No Python files found."
  exit 0
fi

black "${py_files[@]}"
pylint "${py_files[@]}"
