#!/usr/bin/env bash
# Full re-run at gpt-5 DEFAULT reasoning, consistent across naive_gpt / rag /
# wiki_v7 / wiki_v6. Stages: build (parallel) -> answer (parallel) -> judge.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)

echo "########## CLEAR old (minimal-reasoning) outputs (already archived) ##########"
rm -f methods/naive_gpt/results/naive_gpt.jsonl
rm -f methods/rag/results/rag.jsonl                       # keep output/emb (embeddings reasoning-independent)
rm -f methods/wiki_v7/results/wiki_v7*.jsonl
rm -rf methods/wiki_v7/output/digests_v7                  # force rebuild at default reasoning
rm -f methods/wiki_v6/results/wiki_v6*.jsonl
rm -rf methods/wiki_v6/output/wikis methods/wiki_v6/output/_master_template.json
rm -f results/eval/*_verdicts.jsonl results/eval/*_summary.json   # re-judge fresh

echo "########## STAGE A: BUILDS (wiki_v7 rebuild + wiki_v6 build, parallel) ##########"
bash methods/wiki_v7/compile_all.sh > methods/wiki_v7/logs/compile_all_default.log 2>&1 &
V7B=$!
$PY methods/wiki_v6/run.py build > methods/wiki_v6/logs/build.log 2>&1 &
V6B=$!
wait $V7B; echo "[stageA] wiki_v7 build done"
wait $V6B; echo "[stageA] wiki_v6 build done"

echo "########## STAGE B: ANSWERS (naive, rag, v7 per-movie, v6 per-movie) ##########"
$PY methods/naive_gpt/run.py > methods/naive_gpt/logs/run_default.log 2>&1 &
$PY methods/rag/run.py > methods/rag/logs/run_default.log 2>&1 &
printf '%s\n' "${SLUGS[@]}" | xargs -P 4 -I {} bash -c '
  $0 methods/wiki_v7/answer.py --movie "$1" > methods/wiki_v7/logs/answer_$1.log 2>&1
' "$PY" {} &
V7A=$!
printf '%s\n' "${SLUGS[@]}" | xargs -P 4 -I {} bash -c '
  $0 methods/wiki_v6/run.py answer --movie "$1" > methods/wiki_v6/logs/answer_$1.log 2>&1
' "$PY" {} &
V6A=$!
wait
echo "[stageB] all answers done"
ls methods/wiki_v7/results/wiki_v7_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v7/results/wiki_v7.jsonl
ls methods/wiki_v6/results/wiki_v6_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v6/results/wiki_v6.jsonl

echo "########## STAGE C: JUDGE all four ##########"
$PY shared/eval.py naive_gpt rag wiki_v7 wiki_v6 > results/eval/eval_default.log 2>&1

echo "########## SUMMARIES ##########"
for m in naive_gpt rag wiki_v7 wiki_v6; do
  f="results/eval/${m}_summary.json"
  [ -f "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"{d['method']:12s} acc={d['judge_accuracy']:.1%} fab={d['fabrication_pct']:>5}%  ans_in={d['total_input_tokens']:>11,} ans_out={d['total_output_tokens']:>8,} build={d['build_tokens']:>11,} tok/q={d['tokens_per_question']:.0f}\")"
done
echo "paper: naive 49.1% / wiki 53.9% / rag 17.0%"
echo "RERUN_DEFAULT_DONE"
