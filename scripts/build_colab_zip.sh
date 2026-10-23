#!/usr/bin/env bash
# Package the committed code plus processed data and NWP wind for Colab Enterprise.
# Uses `git archive HEAD`, so uncommitted changes and .env are never included.
set -euo pipefail
cd "$(dirname "$0")/.."

out=dist/pm25_project.zip
mkdir -p dist
rm -f "$out"
git archive --format=zip -o "$out" HEAD
zip -q -r "$out" data/processed data/wind_forecast -x '*.DS_Store'
if unzip -l "$out" | grep -qE '\.env$|data/raw'; then
  echo "refusing: secrets or raw data in $out" >&2
  rm -f "$out"
  exit 1
fi
cp notebooks/*.ipynb dist/
ls -lh "$out"
