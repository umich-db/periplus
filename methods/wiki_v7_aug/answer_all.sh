#!/usr/bin/env bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
mkdir -p methods/wiki_v7_aug/logs methods/wiki_v7_aug/results
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '
  s="$1"
  '"$PY"' methods/wiki_v7_aug/answer.py --movie "$s" > methods/wiki_v7_aug/logs/answer_${s}.log 2>&1
  echo "[done] $s ($(grep -c "^  " methods/wiki_v7_aug/logs/answer_${s}.log) qs)"
' _ {}
# concat per-movie -> results/wiki_v7_aug.jsonl
ls methods/wiki_v7_aug/results/wiki_v7_aug_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v7_aug/results/wiki_v7_aug.jsonl
echo "WIKI_V7_GUESS_ANSWER_DONE total=$(wc -l < methods/wiki_v7_aug/results/wiki_v7_aug.jsonl)"
