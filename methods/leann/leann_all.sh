#!/usr/bin/env bash
# LEANN answer phase (indexes already built on GPU) + judge. -P3 parallel.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/leann/bin/python"
PYB="envs/base/bin/python"
# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
mapfile -t SLUGS < <(corpus_slugs)
echo "### LEANN answer (gpt-5 QA over local-index retrieval), -P3 ###"
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/leann/run.py answer --movie "$1" > methods/leann/logs/answer_$1.log 2>&1; echo "[answered] $1"' _ {}
# concat ONLY the per-movie slug files — NOT leann_judged.jsonl / leann.jsonl,
# which the old leann_*.jsonl glob picked up and duplicated (590 = 2x295 bug).
: > methods/leann/results/leann.jsonl
for s in "${SLUGS[@]}"; do cat "methods/leann/results/leann_$s.jsonl" >> methods/leann/results/leann.jsonl; done
echo "### JUDGE leann ###"
$PYB shared/eval.py leann > results/eval/eval_leann.log 2>&1
$PYB -c "import json;d=json.load(open('results/eval/leann_summary.json'));print(f\"leann: acc={d['judge_accuracy']:.1%} F1halluc={d['hallucination_f1_pct']}% miss={d['miss_pct']}% ans_tok={d['total_input_tokens']+d['total_output_tokens']:,} (build=0, local GPU embed)\")"
echo "LEANN_ALL_DONE"
