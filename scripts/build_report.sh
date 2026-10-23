#!/usr/bin/env bash
# Regenerate every table and figure of the report from results/predictions, then compile.
# xgboost is not imported here, so one process is fine (see CLAUDE.md on libomp).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m pm25.evaluation.report > /dev/null
uv run python -m pm25.evaluation.analysis > /dev/null
uv run python -m pm25.evaluation.figures
uv run python -m pm25.evaluation.latex
cp results/figures/*.png report/fig/

if command -v tectonic > /dev/null; then
  (cd report && tectonic -X compile main.tex)
  echo "report/main.pdf"
else
  echo "tectonic not found: upload report/ to Overleaf and compile with XeLaTeX"
fi
