#!/bin/bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY=envs/base/bin/python
MOVIES="10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men"
FILT='HTTP Request|Retrying|Batches|it/s|Some weights|newly init|You should|warnings|Using collapsed|TreeRetriever|Tokenizer:|Threshold:|Top K:|Selection|Context Embed|Embedding Model|Num Layers|Start Layer|No device'
for slug in $MOVIES; do
  echo "### answering $slug ###"
  $PY methods/raptor_forced/answer.py --movie "$slug" 2>&1 | grep -avE "$FILT"
done
ls methods/raptor_forced/results/raptor_forced_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/raptor_forced/results/raptor_forced.jsonl
echo "COMBINED -> $(wc -l < methods/raptor_forced/results/raptor_forced.jsonl) rows"
