#!/usr/bin/env bash
# Poll the API key; launch the full sweep the moment it has quota.
#   BENCH_ROOT=/path/to/run_tree bash scripts/sweep_when_alive.sh
set -u
BENCH_ROOT="${BENCH_ROOT:?set BENCH_ROOT to the run tree}"
cd "$BENCH_ROOT"
PY="envs/base/bin/python"
n=0
while true; do
  n=$((n+1))
  if "$PY" -c '
import importlib.util
s=importlib.util.spec_from_file_location("ac","shared/azure_client.py"); ac=importlib.util.module_from_spec(s); s.loader.exec_module(ac)
ac.chat([{"role":"user","content":"ok"}],max_tokens=8,bucket="probe",max_retries=2)
' >/dev/null 2>&1; then
    echo "$(date -Is) key alive after $n probe(s); launching sweep"
    "$PY" scripts/rerun_all.py > logs/rerun/sweep3.log 2>&1
    echo "$(date -Is) sweep exited $?"
    exit 0
  fi
  [ $((n % 12)) -eq 0 ] && echo "$(date -Is) probe $n: still refused (412)"
  sleep 300
done
