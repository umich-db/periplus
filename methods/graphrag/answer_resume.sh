#!/usr/bin/env bash
# GraphRAG ANSWER-ONLY resume: indexes already built (20/20). Re-answer only the
# partial/missing movies (per-slug files opened with "w", so partials are cleanly
# redone). Does NOT touch indexing. Then concat all 30 -> local/global.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/graphrag/bin/python"
TODO="highlander_endgame miami_vice_film minority_report_film mission_impossible_film raiders_of_the_lost_ark supergirl_film sweet_smell_of_success the_breakfast_club tinker_tailor_soldier_spy_film up_in_the_air_1940_film white_christmas_film"

echo "### ANSWER-resume ${TODO} (-P3) ###"
printf '%s\n' $TODO | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/graphrag/run.py answer --movie "$1" > methods/graphrag/logs/answer_$1.log 2>&1; echo "[answered] $1"' _ {}

echo "### CONCAT all 30 -> local/global ###"
ls methods/graphrag/results/graphrag_local_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_local.jsonl
ls methods/graphrag/results/graphrag_global_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_global.jsonl
for m in graphrag_local graphrag_global; do
  mkdir -p "methods/$m/results"
  cp "methods/graphrag/results/$m.jsonl" "methods/$m/results/$m.jsonl"
  cp methods/graphrag/build_costs.json "methods/$m/build_costs.json" 2>/dev/null || true
done
echo "local -> $(wc -l < methods/graphrag/results/graphrag_local.jsonl) rows; global -> $(wc -l < methods/graphrag/results/graphrag_global.jsonl) rows"
echo "GRAPHRAG_ANSWER_RESUME_DONE"
