#!/usr/bin/env bash
# Create the five per-method virtualenvs and fetch the one upstream checkout that
# is imported at runtime (RAPTOR). Idempotent: re-running skips what already exists.
#
#   bash scripts/setup_envs.sh          # envs + required upstream
#   bash scripts/setup_envs.sh --refs   # also clone the reference checkouts
#
# The methods are mutually incompatible on their pinned dependency versions
# (graphrag, docetl and leann each want a different pydantic/llama-index), which
# is why each gets its own env rather than one shared requirements file.
set -euo pipefail
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT

PY_VERSION=3.11

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

for e in base docetl graphrag leann raptor; do
  if [ -x "envs/$e/bin/python" ]; then
    echo "[skip] envs/$e already exists"
    continue
  fi
  echo "[create] envs/$e (python $PY_VERSION)"
  uv venv --python "$PY_VERSION" "envs/$e"
  uv pip install --python "envs/$e/bin/python" -r "requirements/$e.txt"
done

# naive_gpt and rag run in the base env; the symlinks keep the paths in the
# older rerun_*.sh scripts working.
[ -e envs/naive_gpt ] || ln -s base envs/naive_gpt
[ -e envs/rag ]       || ln -s base envs/rag

# RAPTOR's own package is imported directly by methods/raptor/raptor_bench.py.
if [ ! -d methods/raptor/upstream/raptor ]; then
  echo "[clone] RAPTOR upstream"
  git clone https://github.com/parthsarthi03/raptor.git methods/raptor/upstream
  git -C methods/raptor/upstream checkout -q 7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767
fi

# Reference-only checkouts. Not imported: leann runs from its PyPI package and
# shed is reimplemented in methods/shed/. Clone them only to read the originals.
if [ "${1:-}" = "--refs" ]; then
  [ -d methods/leann/repo ] || { git clone https://github.com/StarTrail-org/LEANN.git methods/leann/repo
    git -C methods/leann/repo checkout -q cfb1382eccde2de8cd671a10ae0f251e35cf498b; }
  [ -d methods/shed/repo ]  || { git clone https://github.com/Ruiying-Ma/SHED.git methods/shed/repo
    git -C methods/shed/repo checkout -q 232df269463caef4e9c23f91e379e18c9bd69f49; }
  [ -d data/nqa_repo ]      || { git clone https://github.com/deepmind/narrativeqa.git data/nqa_repo
    git -C data/nqa_repo checkout -q 904246f6d1fe99a99a08a03501fe3e619af2cee5; }
fi

echo
echo "envs ready. Next:"
echo "  cp .env.example .env    # then fill in your Azure credentials"
echo "  bash scripts/run_method.sh --list"
