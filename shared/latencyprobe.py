"""Single-thread per-question answer latency for methods that answer in a
batched/concurrent pass and so can't record a faithful latency_s inline:
doctopus (ThreadPoolExecutor), moar / docetl_standard (whole-dataset docetl run).

After such a runner has produced its answers, it calls measure(method, answer_one,
items): answer_one(item) runs the REAL answer LLM call(s) for ONE question,
uncached. measure() times it sequentially over a stratified sample of N questions
(env LATENCY_N, default 25; 0 disables) and writes
results/eval/latency_probe/<method>[_cuad].json:
    {answer_median_s, answer_p95_s, answer_n, answer_src}
make_latency.py overlays these over the hand-set numbers in latency.json.

Concurrency is held at 1 by construction (a plain sequential loop), matching how
the other 7 methods recorded latency_s.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import os, json, time, pathlib

BENCH = _BENCH
OUTD = BENCH / "results" / "eval" / "latency_probe"


def sample_n():
    try:
        return int(os.environ.get("LATENCY_N", "25"))
    except ValueError:
        return 25


def _pct(sv, q):
    """Linear-interpolation percentile (matches numpy.percentile default)."""
    if not sv:
        return 0.0
    if len(sv) == 1:
        return sv[0]
    pos = q / 100 * (len(sv) - 1)
    lo = int(pos); hi = min(lo + 1, len(sv) - 1); f = pos - lo
    return sv[lo] * (1 - f) + sv[hi] * f


def _stratified(items, n):
    """Evenly spaced sample across the list (spreads movies, not first-N)."""
    if n >= len(items):
        return list(items)
    step = max(1, len(items) // n)
    return items[::step][:n]


def measure(method, answer_one, items, n=None):
    n = sample_n() if n is None else n
    if n <= 0:
        print(f"[latency] {method}: skipped (LATENCY_N=0)", flush=True)
        return None
    sub = _stratified(items, n)
    lats = []
    for i, it in enumerate(sub):
        t0 = time.time()
        try:
            answer_one(it)
        except Exception as e:  # noqa: BLE001
            print(f"[latency] {method}: sample {i} error {repr(e)[:80]}", flush=True)
            continue
        lats.append(time.time() - t0)
    return record(method, lats)


def _suffix():
    """Corpus suffix, so a CUAD probe does not overwrite the NarrativeQA one.

    The probe wrote results/eval/latency_probe/<method>.json regardless of
    corpus, so whichever corpus ran last owned the file and the other silently
    reported its number. Same convention as the results files: "" or "_cuad".
    """
    return "_cuad" if os.environ.get("CUAD") else ""


def record(method, latencies):
    OUTD.mkdir(parents=True, exist_ok=True)
    sv = sorted(latencies); n = len(sv)
    summ = {"method": method,
            "answer_median_s": round(_pct(sv, 50), 2),
            "answer_p95_s": round(_pct(sv, 95), 2),
            "answer_n": n,
            "answer_src": f"single-thread N={n}"}
    (OUTD / f"{method}{_suffix()}.json").write_text(json.dumps(summ, indent=2))
    print(f"[latency] {method}: median {summ['answer_median_s']}s "
          f"p95 {summ['answer_p95_s']}s (N={n}, single-thread)", flush=True)
    return summ
