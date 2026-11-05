#!/usr/bin/env bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
mapfile -t SLUGS < <(corpus_slugs)
SFX=""; [ -n "${CUAD:-}" ] && SFX="_cuad"
export SFX
printf '%s\n' "${SLUGS[@]}" | xargs -P 6 -I {} bash -c '
  s="$1"
  '"$PY"' methods/llm_index/answer.py --movie "$s" > "methods/llm_index/logs/answer${SFX}_${s}.log" 2>&1
  echo "[done] $s ($(grep -c "^  " "methods/llm_index/logs/answer${SFX}_${s}.log") qs)"
' _ {}

# Concat this corpus's per-document shards only. The glob was llm_index_*.jsonl,
# which also matches llm_index_cuad_<slug>.jsonl, so running both corpora merged
# 410 contract rows into the 885-question screenplay file. Build the list from
# the corpus instead of globbing.
OUT="methods/llm_index/results/llm_index${SFX}.jsonl"
: > "$OUT"
for s in "${SLUGS[@]}"; do
  f="methods/llm_index/results/llm_index${SFX}_${s}.jsonl"
  [ -f "$f" ] && cat "$f" >> "$OUT"
done
echo "WIKI_V7_ANSWER_DONE${SFX} total=$(wc -l < "$OUT") over ${#SLUGS[@]} documents"
