#!/usr/bin/env bash
# Re-run ALL answers with standardized "<=10 words" prompt (naive@1024, rag@2048,
# wiki_v7@2048, wiki_v6@2048). Reuses builds/embeddings/digests. Re-judge all @2048.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)

echo "### archive prior + clear answers + verdict caches ###"
mkdir -p archive/pre_10word
for m in naive_gpt rag wiki_v7 wiki_v6; do cp -r methods/$m/results archive/pre_10word/${m}_results 2>/dev/null; done
rm -f methods/naive_gpt/results/naive_gpt.jsonl methods/rag/results/rag.jsonl
rm -f methods/wiki_v7/results/wiki_v7*.jsonl methods/wiki_v6/results/wiki_v6*.jsonl
rm -f results/eval/*_verdicts.jsonl results/eval/*_summary.json

echo "### ANSWERS (reuse builds): naive@1024 + rag@2048 + wiki_v7 + wiki_v6 ###"
$PY methods/naive_gpt/run.py > methods/naive_gpt/logs/run_10w.log 2>&1 &
$PY methods/rag/run.py        > methods/rag/logs/run_10w.log 2>&1 &
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '$0 methods/wiki_v7/answer.py --movie "$1" > methods/wiki_v7/logs/answer_$1.log 2>&1' "$PY" {} &
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '$0 methods/wiki_v6/run.py answer --movie "$1" > methods/wiki_v6/logs/answer_$1.log 2>&1' "$PY" {} &
wait
ls methods/wiki_v7/results/wiki_v7_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v7/results/wiki_v7.jsonl
ls methods/wiki_v6/results/wiki_v6_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v6/results/wiki_v6.jsonl
echo "[answers] done"

echo "### JUDGE all four @2048 ###"
$PY shared/eval.py naive_gpt rag wiki_v7 wiki_v6 > results/eval/eval_10w.log 2>&1

echo "### SUMMARIES (<=10 words, naive 1024 / rest 2048) ###"
for m in naive_gpt rag wiki_v7 wiki_v6; do
  f="results/eval/${m}_summary.json"
  [ -f "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"{d['method']:12s} acc={d['judge_accuracy']:.1%} fab={d['fabrication_pct']:>5}%  ans_in={d['total_input_tokens']:>11,} ans_out={d['total_output_tokens']:>8,} build={d['build_tokens']:>11,} tok/q={d['tokens_per_question']:.0f}\")"
done
echo "paper: naive 49.1% / wiki 53.9% / rag 17.0%"
echo "RERUN_10W_DONE"
