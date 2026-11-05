#!/bin/bash
# Judge each routing mode and save a per-mode summary. Run after answer_all.sh.
# The judge verdict cache (keyed by qid+prediction) is kept across modes so that
# identical predictions are not re-judged.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY=envs/base/bin/python
RES=methods/wiki_v8/results
EVAL=results/eval

for mode in keyword cosine hybrid llm; do
  [ -f "$RES/wiki_v8_${mode}.jsonl" ] || { echo "[eval_all] skip $mode (no rows)"; continue; }
  cp "$RES/wiki_v8_${mode}.jsonl" "$RES/wiki_v8.jsonl"
  echo "[eval_all] ===== judging $mode ====="
  $PY shared/eval.py wiki_v8 2>&1 | grep -vE "HTTP Request|Retrying" | tail -2
  cp "$EVAL/wiki_v8_summary.json" "$EVAL/wiki_v8_${mode}_summary.json"
done
echo "[eval_all] per-mode summaries written: $EVAL/wiki_v8_<mode>_summary.json"
