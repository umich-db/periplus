#!/usr/bin/env bash
# Rerun all 10 methods to regenerate the new latency markers:
#   results/eval/build_times/<m>.json   (buildtimer.py, every method)
#   results/eval/latency_probe/<m>.json (latencyprobe.py, doctopus/moar/docetl)
# Sequenced cheap -> expensive so probe data lands first; destructive index
# rebuilds (graphrag/leann/wiki) run last. Continue-on-error; per-method logs.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
BASE="envs/base/bin/python"
DOCETL="envs/docetl/bin/python"
GR="envs/graphrag/bin/python"
LEANN="envs/leann/bin/python"
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)
LOGD="logs/rerun_latency"; mkdir -p "$LOGD"
STATUS="$LOGD/STATUS.txt"; : > "$STATUS"
mkdir -p methods/doctopus/logs methods/rag/logs methods/shed/logs methods/naive_gpt/logs \
         methods/docetl_standard/logs methods/moar/logs methods/leann/logs methods/wiki_v6/logs

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
run(){ # run <name> <logfile> -- <cmd...>
  local name="$1" log="$2"; shift 3
  echo "[$(stamp)] START  $name" | tee -a "$STATUS"
  if "$@" > "$log" 2>&1; then echo "[$(stamp)] OK     $name" | tee -a "$STATUS"
  else echo "[$(stamp)] FAIL   $name (rc=$? see $log)" | tee -a "$STATUS"; fi
}

echo "[$(stamp)] ===== rerun_all_latency START =====" | tee -a "$STATUS"

# ---------------- PHASE 1: cheap, emits markers fast ----------------
# doctopus: answer cache makes calibrate/execute fast; probe = 25 fresh calls.
run doctopus        methods/doctopus/logs/rerun.log        -- $BASE   methods/doctopus/run.py
# docetl_standard: full map (docetl caches internally) + 25-call probe.
run docetl_standard methods/docetl_standard/logs/rerun.log -- $DOCETL methods/docetl_standard/run.py
# naive_gpt / rag / shed: resumable answers (skip done) -> build_times only.
run naive_gpt       methods/naive_gpt/logs/rerun.log       -- $BASE   methods/naive_gpt/run.py
run rag             methods/rag/logs/rerun.log             -- $BASE   methods/rag/run.py
run shed            methods/shed/logs/rerun.log            -- $BASE   methods/shed/run.py

# ---------------- PHASE 2: moar (~82min optimize) ----------------
run moar            methods/moar/logs/rerun.log            -- $DOCETL methods/moar/run.py

# ---------------- PHASE 3: destructive index rebuilds (fresh build_min) ----------------
# graphrag: graphrag_all.sh wipes runs/*/output and reindexes all 10 fresh.
run graphrag        "$LOGD/graphrag_all.log"               -- bash methods/graphrag/graphrag_all.sh

# leann: clear indexes -> rebuild on GPU (no API cost) -> answer (gpt-5).
echo "[$(stamp)] START  leann_build" | tee -a "$STATUS"
rm -f methods/leann/output/indexes/*.leann*
for s in "${SLUGS[@]}"; do
  $LEANN methods/leann/run.py build --movie "$s" > "methods/leann/logs/build_${s}.log" 2>&1 || true
done
echo "[$(stamp)] OK     leann_build" | tee -a "$STATUS"
run leann_answer    "$LOGD/leann_answer.log"               -- bash methods/leann/leann_all.sh

# wiki_v7: clear digests -> compile all 10 fresh -> answer.
echo "[$(stamp)] START  wiki_v7_build" | tee -a "$STATUS"
rm -rf methods/wiki_v7/output/digests_v7
run wiki_v7_compile "$LOGD/wiki_v7_compile.log"            -- bash methods/wiki_v7/compile_all.sh
run wiki_v7_answer  "$LOGD/wiki_v7_answer.log"             -- bash methods/wiki_v7/answer_all.sh

# wiki_v6: clear wikis -> bootstrap -> compile x10 -> answer x10.
echo "[$(stamp)] START  wiki_v6 (bootstrap+compile+answer)" | tee -a "$STATUS"
rm -rf methods/wiki_v6/output/wikis methods/wiki_v6/output/_master_template.json
rm -f methods/wiki_v6/results/wiki_v6*.jsonl
$BASE methods/wiki_v6/run.py bootstrap > methods/wiki_v6/logs/bootstrap.log 2>&1 || true
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c \
  "$BASE methods/wiki_v6/run.py compile --movie \"\$1\" > methods/wiki_v6/logs/compile_\$1.log 2>&1" _ {}
printf '%s\n' "${SLUGS[@]}" | xargs -P 4 -I {} bash -c \
  "$BASE methods/wiki_v6/run.py answer --movie \"\$1\" > methods/wiki_v6/logs/answer_\$1.log 2>&1" _ {}
ls methods/wiki_v6/results/wiki_v6_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v6/results/wiki_v6.jsonl 2>/dev/null || true
echo "[$(stamp)] OK     wiki_v6" | tee -a "$STATUS"

# ---------------- refresh the charts ----------------
run make_latency    "$LOGD/make_latency.log"               -- $BASE shared/make_latency.py

echo "[$(stamp)] ===== rerun_all_latency DONE =====" | tee -a "$STATUS"
echo "--- build_times ---"  | tee -a "$STATUS"; ls -1 results/eval/build_times 2>/dev/null   | tee -a "$STATUS"
echo "--- latency_probe ---"| tee -a "$STATUS"; ls -1 results/eval/latency_probe 2>/dev/null | tee -a "$STATUS"
