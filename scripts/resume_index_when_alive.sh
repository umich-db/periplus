#!/usr/bin/env bash
# Resume the index's NarrativeQA chain the moment the API key has quota, and stop
# at the first step that does not fully succeed, publishing nothing.
#
#   BENCH_ROOT=/path/to/run_tree bash scripts/resume_index_when_alive.sh
#
# Every step is followed by a check of what it was supposed to produce. A key
# that dies mid-run has twice turned a half-finished step into published numbers
# (a 0.0% accuracy from rows that were API errors), so nothing downstream runs,
# and the results files are not touched, unless the step before it is whole.
set -u
BENCH_ROOT="${BENCH_ROOT:?set BENCH_ROOT to the run tree}"; cd "$BENCH_ROOT"
PY=envs/base/bin/python
alive() { "$PY" -c '
import importlib.util
s=importlib.util.spec_from_file_location("ac","shared/azure_client.py"); ac=importlib.util.module_from_spec(s); s.loader.exec_module(ac)
ac.chat([{"role":"user","content":"ok"}],max_tokens=8,bucket="probe",max_retries=2)' >/dev/null 2>&1; }
stop() { echo "$(date -Is) STOPPED: $1. Nothing published; re-run this script to resume."; exit 1; }
n=0; until alive; do n=$((n+1)); [ $((n % 12)) -eq 0 ] && echo "$(date -Is) probe $n: key still refused"; sleep 300; done
echo "$(date -Is) key alive after $n probe(s)"

echo "$(date -Is) build: compile missing documents -> balance -> record"
bash methods/llm_index/compile_all.sh
built=$(ls methods/llm_index/output/digests/*/_manifest.json 2>/dev/null | grep -vc '/CUAD_')
bal=$(ls methods/llm_index/output/digests_balanced/*/_manifest.json 2>/dev/null | grep -vc '/CUAD_')
[ "$built" -eq 30 ] && [ "$bal" -eq 30 ] || stop "build incomplete (built $built/30, balanced $bal/30)"

echo "$(date -Is) answer: 885 questions"
bash methods/llm_index/answer_all.sh
rows=$(wc -l < methods/llm_index/results/llm_index.jsonl 2>/dev/null || echo 0)
errs=$(grep -c '"prediction": "\[ERROR' methods/llm_index/results/llm_index.jsonl 2>/dev/null || true)
[ "$rows" -eq 885 ] && [ "${errs:-0}" -eq 0 ] || stop "answers incomplete ($rows/885 rows, ${errs:-0} error rows)"

echo "$(date -Is) score: LLM judge"
"$PY" shared/eval.py llm_index
"$PY" - <<'P' || stop "scoring incomplete (no summary, or unjudged rows)"
import json, sys
d = json.load(open("results/eval/llm_index_summary.json"))
sys.exit(0 if d.get("n_questions") == 885 and not d.get("unjudged") else 1)
P
echo "$(date -Is) refresh: output.txt / latency.txt"
"$PY" scripts/refresh_measurements.py
echo "$(date -Is) INDEX_NQA_COMPLETE"
