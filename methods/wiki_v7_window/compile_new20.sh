#!/usr/bin/env bash
# Build v7 base digests for the 20 NEW movies. Resumable: cli.py skips any doc whose
# _manifest.json already exists. One process per movie, bounded parallelism.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
LOGS="methods/wiki_v7/logs"
KEYS="methods/wiki_v7_window/results/new20_keys.txt"
cat "$KEYS" | xargs -P 4 -I {} bash -c '
  k="$1"
  safe=$(echo "$k" | tr -c "A-Za-z0-9_-" "_" | sed "s/_*$//")
  '"$PY"' methods/wiki_v7/cli.py compile "$k" > "'"$LOGS"'/compile_${safe}.log" 2>&1
  echo "[done] $k"
' _ {}
echo "NEW20_COMPILES_DONE"
