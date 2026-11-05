#!/bin/bash
# Full wiki_v8 answering: all routing modes x all 10 movies, then combine per mode.
# Waits until all 10 packed indexes are built. Resumable (answer.py skips done qids).
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY=envs/base/bin/python
STORES=methods/wiki_v8/output/packed_v8
RES=methods/wiki_v8/results
FILT='HTTP Request|Retrying|No device|HF_TOKEN|Loading|Batches|weights|encoder_attention'

# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
MOVIES="$(corpus_slugs | tr "\n" " ")"
MODES="keyword cosine hybrid llm"

echo "[answer_all] waiting for all 10 packed indexes..."
until [ "$(ls $STORES/*/packed.json 2>/dev/null | wc -l)" -eq 10 ]; do sleep 30; done
echo "[answer_all] all 10 built; starting answering at $(date +%H:%M:%S)"

for mode in $MODES; do
  echo "[answer_all] ===== MODE: $mode ====="
  for slug in $MOVIES; do
    echo "[answer_all] $mode / $slug"
    $PY methods/wiki_v8/answer.py --movie "$slug" --mode "$mode" 2>&1 | grep -vE "$FILT"
  done
  ls $RES/wiki_v8_${mode}_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > $RES/wiki_v8_${mode}.jsonl
  echo "[answer_all] combined $mode -> $(wc -l < $RES/wiki_v8_${mode}.jsonl) rows"
done
echo "[answer_all] ALL MODES DONE at $(date +%H:%M:%S)"
