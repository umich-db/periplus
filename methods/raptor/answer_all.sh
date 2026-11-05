#!/bin/bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY=envs/base/bin/python
# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
MOVIES="$(corpus_slugs | tr "\n" " ")"
FILT='HTTP Request|Retrying|Batches|it/s|Some weights|newly init|You should|warnings|Using collapsed|TreeRetriever|Tokenizer:|Threshold:|Top K:|Selection|Context Embed|Embedding Model|Num Layers|Start Layer'
for slug in $MOVIES; do
  echo "### answering $slug ###"
  $PY methods/raptor/raptor_bench.py answer --movie "$slug" 2>&1 | grep -avE "$FILT"
done
ls methods/raptor/results/raptor_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/raptor/results/raptor.jsonl
echo "COMBINED -> $(wc -l < methods/raptor/results/raptor.jsonl) rows"
