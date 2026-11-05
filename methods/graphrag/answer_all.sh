#!/usr/bin/env bash
# Re-run graphrag ANSWER phase (indexes intact) with retry-on-500, then judge.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/graphrag/bin/python"; PYB="envs/base/bin/python"
# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
mapfile -t SLUGS < <(corpus_slugs)
echo "### clear corrupted graphrag answers + verdicts ###"
rm -f methods/graphrag/results/graphrag_local_*.jsonl methods/graphrag/results/graphrag_global_*.jsonl
rm -f methods/graphrag/results/graphrag_local.jsonl methods/graphrag/results/graphrag_global.jsonl
rm -f methods/graphrag_local/results/*.jsonl methods/graphrag_global/results/*.jsonl
rm -f results/eval/graphrag_local_verdicts.jsonl results/eval/graphrag_global_verdicts.jsonl
rm -f results/eval/graphrag_local_summary.json results/eval/graphrag_global_summary.json

echo "### re-answer (local+global) per movie, -P3, with retry ###"
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/graphrag/run.py answer --movie "$1" > methods/graphrag/logs/answer_$1.log 2>&1; echo "[answered] $1"' _ {}
ls methods/graphrag/results/graphrag_local_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_local.jsonl
ls methods/graphrag/results/graphrag_global_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_global.jsonl
for m in graphrag_local graphrag_global; do
  mkdir -p methods/$m/results; cp methods/graphrag/results/$m.jsonl methods/$m/results/$m.jsonl
  cp methods/graphrag/build_costs.json methods/$m/build_costs.json 2>/dev/null || true
done

echo "### judge ###"
$PYB shared/eval.py graphrag_local graphrag_global > results/eval/eval_graphrag2.log 2>&1
for m in graphrag_local graphrag_global; do
  $PYB -c "import json;d=json.load(open('results/eval/${m}_summary.json'));print(f\"${m}: acc={d['judge_accuracy']:.1%} F1halluc={d['hallucination_f1_pct']}% miss={d['miss_pct']}%\")" 2>/dev/null
done
echo GRAPHRAG_REANSWER_DONE
