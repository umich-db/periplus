#!/usr/bin/env bash
# Build v7 base digests for the 10 CUAD contracts. Resumable (cli.py skips built docs).
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
LOGS="methods/wiki_v7/logs"
cat methods/wiki_v7_window/results/cuad_keys.txt | xargs -P 4 -I {} bash -c '
  k="$1"
  safe=$(echo "$k" | tr -c "A-Za-z0-9_-" "_" | sed "s/_*$//")
  '"$PY"' methods/wiki_v7/cli.py compile "$k" > "'"$LOGS"'/compile_${safe}.log" 2>&1
  echo "[done] $k"
' _ {}
echo "CUAD_COMPILES_DONE"
