#!/usr/bin/env bash
# Build the remaining llm_index digests with bounded parallelism (one movie per
# process; cli.py skips already-built movies). Logs per movie under logs/.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
LOGS="methods/llm_index/logs"

# Corpus from shared/corpus.py (30 screenplays, or the 10 contracts under
# CUAD=1), not a list frozen in this file. See scripts/corpus_lib.sh.
source scripts/corpus_lib.sh
mapfile -t DOCIDS < <(corpus_docids)
# Wall-clock stamps for the build-time record (the per-process timer sums three
# parallel processes and would overstate a 3-way build about 3x).
# One JSON line per build run, appended: a resumed build is a second segment,
# and the recorder sums the segments' wall-clock rather than trusting one run.
WALL="methods/llm_index/logs/_wall${CUAD:+_cuad}.jsonl"
T0="$(date +%s)"
printf '%s\n' "${DOCIDS[@]}" | xargs -P 3 -I {} bash -c '
  slug="$1"
  safe=$(echo "$slug" | tr -c "A-Za-z0-9_-" "_" | sed "s/_*$//")
  log="'"$LOGS"'/compile_${safe}.log"
  echo "[launch] $slug -> $log"
  '"$PY"' methods/llm_index/cli.py compile "$slug" > "$log" 2>&1
  echo "[done]   $slug ($(grep -c "HTTP/1.1 200" "$log") ok calls)"
' _ {}
printf '{"start": %s, "end": %s, "docs_in_corpus": %s, "source": "stamp"}\n' "$T0" "$(date +%s)" "${#DOCIDS[@]}" >> "$WALL"
echo "ALL_COMPILES_DONE"

# Section 3.4: the index is balanced. Splitting bounds fanout at f = C/s = 33 and
# borrow/merge hold every non-root, non-category internal node inside [17, 33].
# The pass is local (all-mpnet embeddings, zero LLM tokens) and it is part of the
# method, not an ablation, so a build that stops before it is not the index the
# paper describes. Resumable: it skips a document already balanced.
echo "=== balancing (paper 3.4) ==="
"$PY" methods/llm_index/balance_window.py
echo "ALL_BALANCE_DONE"
# Record this corpus's build wall-clock from the stamped segments.
"$PY" methods/llm_index/record_build_time.py
