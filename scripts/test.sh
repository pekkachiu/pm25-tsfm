#!/usr/bin/env bash
# torch and xgboost bundle different libomp builds on macOS and crash (or deadlock) when
# both are used in one process, so the XGBoost tests run in their own pytest process.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest --ignore=tests/test_xgboost_model.py --cov-report= "$@"
uv run pytest tests/test_xgboost_model.py --cov-append "$@"
