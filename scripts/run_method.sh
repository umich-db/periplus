#!/usr/bin/env bash
# One entrypoint for every benchmark method.
#
#   bash scripts/run_method.sh --list
#   bash scripts/run_method.sh <method> [build|answer|all]     # default: all
#   CUAD=1 bash scripts/run_method.sh <method>                 # contracts instead of screenplays
#
# Each method has its own venv (see scripts/setup_envs.sh) and its own natural
# split between a one-off build phase and a per-question answer phase. Stateless
# baselines have no build phase and report build_tokens=0.
#
# Every phase is resumable: answer loops skip question ids already in the output
# JSONL, and build phases skip documents whose index already exists. Killing a
# run and re-issuing the same command picks up where it stopped.
set -uo pipefail
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT

BASE="envs/base/bin/python"
DOCETL="envs/docetl/bin/python"
GR="envs/graphrag/bin/python"
LEANN="envs/leann/bin/python"

# The corpus is whatever is in data/, via shared/corpus.py, rather than a list
# frozen in this file. It said ten movies while the benchmark had grown to
# thirty, which is why every method sprouted a run_new20.sh and why the results
# tables carry three different row counts. One list, derived from disk.
if [ -n "${CUAD:-}" ]; then
  mapfile -t MOVIES < <(envs/base/bin/python -c \
    "import sys; sys.path.insert(0,'shared'); from corpus import DOCS; print('\n'.join(DOCS['cuad'].slugs))")
else
  mapfile -t MOVIES < <(envs/base/bin/python -c \
    "import sys; sys.path.insert(0,'shared'); from corpus import DOCS; print('\n'.join(DOCS['nqa'].slugs))")
fi
[ "${#MOVIES[@]}" -gt 0 ] || { echo "corpus is empty -- is data/ populated?" >&2; exit 2; }

METHODS="naive_gpt docetl_standard leann rag graphrag raptor doctopus llm_index"

usage() {
  cat <<'USAGE'
Methods (env, build cost):

  naive_gpt        base     stateless   whole document in the prompt, every question
  rag              base     embeddings  fixed-size chunks, top-5 cosine retrieval
  raptor           base     LLM         recursive clustering + summary tree (upstream impl)
  leann            leann    local GPU   graph index, embeddings recomputed at query time
  graphrag         graphrag LLM         Microsoft GraphRAG; emits graphrag_local + graphrag_global
  docetl_standard  docetl   stateless   DocETL map-reduce over chunks
  moar             docetl   stateless   DocETL with the MOAR optimizer (no search)
  shed             base     stateless   SHED-style hierarchical extraction/distillation
  doctopus         base     embeddings  budget-aware group-knapsack strategy selection
  wiki_v6          base     LLM         our method, v6 (superseded, kept for the ablation)
  llm_index          base     LLM         OUR METHOD: the paper's hierarchical plain-text index
  wiki_v8          base     LLM         our method, v8 (routing-mode ablation)

Phases:  build | answer | all (default)
Corpus:  CUAD=1 switches from the 30 screenplays to the 10 contracts.

Judge the output afterwards:  envs/base/bin/python shared/eval.py <method>
USAGE
}

case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  --list) echo "$METHODS" | tr ' ' '\n'; exit 0 ;;
esac

M="$1"
PHASE="${2:-all}"
CUAD_SFX=""
[ -n "${CUAD:-}" ] && CUAD_SFX="_cuad"
mkdir -p "methods/$M/logs"

do_build=false; do_answer=false
case "$PHASE" in
  build)  do_build=true ;;
  answer) do_answer=true ;;
  all)    do_build=true; do_answer=true ;;
  *) echo "unknown phase '$PHASE' (want build|answer|all)" >&2; exit 2 ;;
esac

echo "=== $M / $PHASE ${CUAD:+(CUAD)} ==="

case "$M" in
  # --- single-script methods: build and answer happen inside one run.py -----
  naive_gpt)       $do_answer && $BASE   methods/naive_gpt/run.py ;;
  rag)             $do_answer && $BASE   methods/rag/run.py ;;
  shed)            $do_answer && $BASE   methods/shed/run.py ;;
  doctopus)        $do_answer && $BASE   methods/doctopus/run.py ;;
  docetl_standard) $do_answer && $DOCETL methods/docetl_standard/run.py ;;
  moar)            $do_answer && $DOCETL methods/moar/run.py ;;

  # --- explicit build/answer split ------------------------------------------
  raptor)
    if $do_build; then
      for s in "${MOVIES[@]}"; do
        echo "[build] $s"; $BASE methods/raptor/raptor_bench.py build "$s" > "methods/raptor/logs/build_$s.log" 2>&1
      done
    fi
    if $do_answer; then
      if [ -n "${CUAD:-}" ]; then bash methods/raptor/run_cuad.sh
      else bash methods/raptor/answer_all.sh; fi
    fi ;;

  leann)
    if $do_build; then
      for s in "${MOVIES[@]}"; do
        echo "[build] $s"; $LEANN methods/leann/run.py build --movie "$s" > "methods/leann/logs/build_$s.log" 2>&1
      done
    fi
    if $do_answer; then
      if [ -n "${CUAD:-}" ]; then bash methods/leann/run_cuad.sh
      else bash methods/leann/leann_all.sh; fi
    fi ;;

  graphrag)
    # graphrag_all.sh does setup + index + local/global answer + judge in one pass.
    if [ -n "${CUAD:-}" ]; then bash methods/graphrag/run_cuad.sh
    elif [ "$PHASE" = "answer" ]; then bash methods/graphrag/answer_all.sh
    else bash methods/graphrag/graphrag_all.sh; fi ;;

  llm_index)
    $do_build  && bash methods/llm_index/compile_all.sh
    $do_answer && bash methods/llm_index/answer_all.sh ;;

  wiki_v6)
    if $do_build; then
      $BASE methods/wiki_v6/run.py bootstrap
      for s in "${MOVIES[@]}"; do $BASE methods/wiki_v6/run.py compile --movie "$s"; done
    fi
    if $do_answer; then
      for s in "${MOVIES[@]}"; do $BASE methods/wiki_v6/run.py answer --movie "$s"; done
      ls methods/wiki_v6/results/wiki_v6_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v6/results/wiki_v6.jsonl
    fi ;;

  wiki_v8)
    $do_build  && $BASE methods/wiki_v8/cli.py compile
    $do_answer && bash methods/wiki_v8/answer_all.sh ;;

  *) echo "unknown method '$M'. Try --list." >&2; exit 2 ;;
esac

RESULT="methods/$M/results/$M${CUAD_SFX}.jsonl"
if [ -f "$RESULT" ]; then
  echo "=== $M done: $(wc -l < "$RESULT") rows -> $RESULT"
  echo "    judge it with:  $BASE shared/eval.py $M"
fi
