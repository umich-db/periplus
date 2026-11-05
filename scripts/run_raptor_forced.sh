#!/usr/bin/env bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
# wait for the single-movie smoke test to finish so we don't double-write 8mm
while pgrep -f "raptor_forced/answer.py --movie" >/dev/null; do sleep 5; done
echo "[start] raptor_forced answer_all"
bash methods/raptor_forced/answer_all.sh > methods/raptor_forced/logs/answer_all.log 2>&1
echo "[start] raptor_forced eval"
envs/base/bin/python shared/eval.py raptor_forced > results/eval/eval_raptor_forced.log 2>&1
grep -E '^\[raptor_forced\]' results/eval/eval_raptor_forced.log
echo "[done] raptor_forced"
