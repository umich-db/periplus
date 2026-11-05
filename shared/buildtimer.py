"""Record one-time build wall-time per method for shared/make_latency.py.

Writes results/eval/build_times/<method>.json:
    {"method", "build_min", "build_coverage", "n_built", "n_total"}

make_latency.py reads these to fill build_min/build_coverage instead of the
hand-measured/recovered numbers it uses today.

Two ways to use it:

    from buildtimer import time_build
    with time_build("rag", n_total=len(movies)) as bt:
        for slug, text in movies.items():
            build_movie(slug, text); bt.built()      # count what actually built
    # build_min written on clean exit; coverage = n_built/n_total

or, for runners that build one movie per process invocation (leann, graphrag),
accumulate into the same file across calls:

    from buildtimer import add_build
    add_build("leann", minutes, n_total=len(SLUGS))
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, time, pathlib

BENCH = _BENCH
OUTD = BENCH / "results" / "eval" / "build_times"


def _coverage(n_built, n_total):
    if not n_total:
        return "full"
    return "full" if n_built >= n_total else f"{n_built}/{n_total}"


def write(method, build_min, n_built=None, n_total=None, coverage=None):
    """Overwrite this method's build record."""
    OUTD.mkdir(parents=True, exist_ok=True)
    rec = {"method": method, "build_min": round(build_min, 3),
           "build_coverage": coverage or _coverage(n_built or 0, n_total),
           "n_built": n_built, "n_total": n_total}
    (OUTD / f"{method}.json").write_text(json.dumps(rec, indent=2))
    print(f"[buildtimer] {method}: {build_min:.3f} min ({rec['build_coverage']})", flush=True)
    return rec


def add_build(method, minutes, n_total=None, count=True, inc=1):
    """Accumulate a per-invocation build (one or more movies) into the method file.

    inc   = how many movies this call built (default 1).
    count = False adds wall-time without bumping coverage (e.g. a corpus-level
            bootstrap that isn't one of the n_total movies).
    """
    OUTD.mkdir(parents=True, exist_ok=True)
    p = OUTD / f"{method}.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    bmin = round((prev.get("build_min") or 0) + minutes, 3)
    nb = (prev.get("n_built") or 0) + (inc if count else 0)
    nt = n_total if n_total is not None else prev.get("n_total")
    return write(method, bmin, n_built=nb, n_total=nt)


class time_build:
    """Context manager timing the build phase; writes on clean exit only."""
    def __init__(self, method, n_total=None, coverage=None):
        self.method, self.n_total, self.coverage, self.n_built = method, n_total, coverage, 0

    def built(self, n=1):
        self.n_built += n

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            return False  # build failed/partial -> don't claim a number
        mins = (time.time() - self._t0) / 60
        write(self.method, mins, n_built=self.n_built or self.n_total,
              n_total=self.n_total, coverage=self.coverage)
        return False
