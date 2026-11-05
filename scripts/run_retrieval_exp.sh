#!/usr/bin/env bash
# Retrieval experiment: embedding retrieval vs raised-budget baseline, all over the
# SAME reused wiki_v7 digests + ORIGINAL strict QA + raw-doc fallback.
#   wiki_v7_collapse  - RAPTOR collapsed-tree (embedding top-k detail pages)
#   wiki_v7_sem       - semantic traversal (capped consolidated seed + embedding walk)
#   wiki_v7_bigbudget - control: v7 native, budget 12k->24k (full consolidated + kw pages)
# Sequential (embedding variants each load SBERT; avoid CPU contention). Then eval all 3.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
ST="results/eval/retrieval_exp_status.txt"
: > "$ST"
stamp(){ date '+%H:%M:%S'; }

for m in wiki_v7_collapse wiki_v7_sem wiki_v7_bigbudget; do
  echo "[$(stamp)] START answer $m" | tee -a "$ST"
  bash methods/$m/answer_all.sh > methods/$m/logs/answer_all.log 2>&1
  echo "[$(stamp)] DONE  answer $m -> $(tail -1 methods/$m/logs/answer_all.log)" | tee -a "$ST"
done

echo "[$(stamp)] START eval (3 methods)" | tee -a "$ST"
$PY shared/eval.py wiki_v7_collapse wiki_v7_sem wiki_v7_bigbudget > results/eval/eval_retrieval_exp.log 2>&1
grep -E '^\[wiki_v7_' results/eval/eval_retrieval_exp.log | tee -a "$ST"
echo "[$(stamp)] ALL DONE" | tee -a "$ST"
