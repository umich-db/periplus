#!/usr/bin/env bash
# Re-run answer generation at 2048-token cap. naive+rag re-answered (were 4096);
# wiki_v7/wiki_v6 QA already 2048 (kept). Re-judge all four at 2048.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"

echo "### archive 4096 results ###"
mkdir -p archive/default_4096
for m in naive_gpt rag wiki_v7 wiki_v6; do cp -r methods/$m/results archive/default_4096/${m}_results 2>/dev/null; done
cp -r results/eval archive/default_4096/eval 2>/dev/null

echo "### clear naive+rag answers + ALL verdict caches (re-judge at 2048) ###"
rm -f methods/naive_gpt/results/naive_gpt.jsonl methods/rag/results/rag.jsonl
rm -f results/eval/*_verdicts.jsonl results/eval/*_summary.json

echo "### STAGE B: re-answer naive + rag at 2048 (parallel) ###"
$PY methods/naive_gpt/run.py > methods/naive_gpt/logs/run_2048.log 2>&1 &
$PY methods/rag/run.py        > methods/rag/logs/run_2048.log 2>&1 &
wait
cat methods/naive_gpt/results/naive_gpt.jsonl >/dev/null 2>&1
echo "[stageB] naive+rag done"

echo "### STAGE C: judge all four at 2048 ###"
$PY shared/eval.py naive_gpt rag wiki_v7 wiki_v6 > results/eval/eval_2048.log 2>&1

echo "### SUMMARIES (2048 cap) ###"
for m in naive_gpt rag wiki_v7 wiki_v6; do
  f="results/eval/${m}_summary.json"
  [ -f "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"{d['method']:12s} acc={d['judge_accuracy']:.1%} fab={d['fabrication_pct']:>5}%  ans_in={d['total_input_tokens']:>11,} ans_out={d['total_output_tokens']:>8,} build={d['build_tokens']:>11,} tok/q={d['tokens_per_question']:.0f}\")"
done
echo "paper: naive 49.1% / wiki 53.9% / rag 17.0%"
echo "RERUN_2048_DONE"
