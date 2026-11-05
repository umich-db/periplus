"""collect_metrics.py — one tidy table of every metric, recomputed from raw rows.

Reads, for each method and each corpus:
  accuracy        the scorer's summary (shared/eval.py for NarrativeQA,
                  shared/eval_cuad_baselines.py for CUAD, the same scorer for
                  every method on a given corpus)
  answer tokens   summed from the per-question rows, not from a summary
  build tokens    methods/<m>/build_costs*.json summed over documents, else rows
  answer latency  median / p95 / total, from the recorded latency_s
  build latency   results/eval/build_times/<m>[_cuad].json

Everything except accuracy is derived here from the raw JSONL, so a chart can
never disagree with the rows it came from. Nothing is hardcoded and nothing is
overridden: if a number is missing it is reported as null and the method is shown
as incomplete rather than filled in.

    envs/base/bin/python shared/collect_metrics.py            # write + print
    envs/base/bin/python shared/collect_metrics.py --print    # print only

Writes results/eval/rerun_metrics.json.
"""

import os, sys, json, math, pathlib, statistics

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS, SCORED, DISPLAY  # noqa: E402

EVAL = BENCH / "results" / "eval"
OUT = EVAL / "rerun_metrics.json"

# Which directory each scored name's rows live in (graphrag fans out to two).
METHOD_DIR = {m: m for m in SCORED}
METHOD_DIR["graphrag_local"] = "graphrag"
METHOD_DIR["graphrag_global"] = "graphrag"


def _rows(method, corpus):
    p = BENCH / "methods" / METHOD_DIR[method] / "results" / f"{method}{corpus.result_suffix}.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return round(s[i], 2)


def _build_tokens(method, corpus, rows):
    """Build cost for the whole corpus.

    The per-document build_costs file wins, because it is a sum over documents
    and that is the quantity being reported. The build_tokens field on a result
    row is whatever that method's runner chose to stamp there, usually one
    document's cost, so taking a max over rows understates a 30-document build by
    more than an order of magnitude. Rows are the last resort."""
    d = BENCH / "methods" / METHOD_DIR[method]
    for name in (f"build_costs{corpus.result_suffix}.json", "build_costs.json"):
        f = d / name
        if f.exists():
            try:
                blob = json.loads(f.read_text())
            except ValueError:
                continue
            if isinstance(blob, dict):
                tot = sum(v.get("build_tokens_total", 0) for v in blob.values()
                          if isinstance(v, dict))
                if tot:
                    return tot
    if rows:
        return max((r.get("build_tokens") or 0) for r in rows)
    return 0


def _latency(method, corpus, rows):
    """Per-question answer latency, and where the number came from.

    Most methods time each answer inline and write latency_s on the row. Three
    cannot: doctopus answers through a ThreadPoolExecutor and docetl/moar push
    the whole dataset through one batched run, so an inline number would be
    wall-clock divided by concurrency rather than the cost of answering one
    question. Those call shared/latencyprobe.py afterwards, which re-answers a
    stratified sample sequentially and writes results/eval/latency_probe/<m>.json.

    Inline timings win when present; the probe is the fallback. The source is
    reported either way, because a sampled sequential probe and a full inline
    measurement are not the same measurement and a chart should not imply they
    are.
    """
    inline = [r["latency_s"] for r in (rows or [])
              if isinstance(r.get("latency_s"), (int, float)) and r["latency_s"] > 0]
    if inline:
        return {"median_s": round(statistics.median(inline), 2),
                "p95_s": _pct(inline, 0.95),
                "total_min": round(sum(inline) / 60, 2),
                "n": len(inline), "source": "inline per-question"}
    # Corpus-strict: an unsuffixed probe file belongs to NarrativeQA only.
    # Falling back to it for CUAD would report one corpus's latency as the
    # other's, which is how doctopus briefly showed its NarrativeQA number in
    # the CUAD column.
    f = EVAL / "latency_probe" / f"{METHOD_DIR[method]}{corpus.result_suffix}.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
        except ValueError:
            d = {}
        if d.get("answer_n"):
            return {"median_s": d.get("answer_median_s"), "p95_s": d.get("answer_p95_s"),
                    "total_min": None, "n": d.get("answer_n"),
                    "source": f"sampled probe ({d.get('answer_src', 'n/a')})"}
    return {"median_s": None, "p95_s": None, "total_min": None, "n": 0,
            "source": "not measured"}


def _accuracy(method, corpus):
    """Whatever the corpus's single scorer wrote. None if it has not run."""
    f = EVAL / f"{method}{corpus.result_suffix}_summary.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    if corpus.key == "cuad":
        return d.get("overall_accuracy"), d.get("n_questions")
    acc = d.get("judge_accuracy")
    return (round(100 * acc, 2) if acc is not None else None), d.get("n_questions")


def _build_min(method, corpus):
    for name in (f"{METHOD_DIR[method]}{corpus.result_suffix}.json",
                 f"{METHOD_DIR[method]}.json"):
        f = EVAL / "build_times" / name
        if f.exists():
            d = json.loads(f.read_text())
            return d.get("build_min"), d.get("build_coverage")
    return None, None


def collect():
    out = {"corpora": {}}
    for key, corpus in DOCS.items():
        n_expected = len(corpus.questions)
        recs = {}
        for method in SCORED:
            rows = _rows(method, corpus)
            if rows is None:
                recs[method] = {"display": DISPLAY.get(method, method), "status": "not run"}
                continue
            lat = _latency(method, corpus, rows)
            in_tok = sum(r.get("input_tokens") or 0 for r in rows)
            out_tok = sum(r.get("output_tokens") or 0 for r in rows)
            bmin, bcov = _build_min(method, corpus)
            acc, n_scored = _accuracy(method, corpus)
            recs[method] = {
                "display": DISPLAY.get(method, method),
                "status": "complete" if len(rows) >= n_expected else
                          f"partial {len(rows)}/{n_expected}",
                "n_rows": len(rows),
                "n_expected": n_expected,
                "n_scored": n_scored,
                "accuracy_pct": acc,
                "answer_in_tokens": in_tok,
                "answer_out_tokens": out_tok,
                "answer_tokens": in_tok + out_tok,
                "build_tokens": _build_tokens(method, corpus, rows),
                "answer_latency_median_s": lat["median_s"],
                "answer_latency_p95_s": lat["p95_s"],
                "answer_latency_total_min": lat["total_min"],
                "answer_latency_n": lat["n"],
                "answer_latency_source": lat["source"],
                "build_min": bmin,
                "build_coverage": bcov,
            }
        out["corpora"][key] = {"name": corpus.name, "n_questions": n_expected,
                               "n_docs": len(corpus.slugs), "methods": recs}
    return out


def show(data):
    for key, c in data["corpora"].items():
        print(f"\n=== {c['name']}  ({c['n_docs']} docs, {c['n_questions']} questions) ===")
        hdr = (f"{'method':18s} {'acc%':>6s} {'ans tok':>12s} {'bld tok':>12s} "
               f"{'ans med s':>10s} {'ans p95':>9s} {'ans min':>9s} {'bld min':>9s}  status")
        print(hdr); print("-" * len(hdr))
        rows = [(m, r) for m, r in c["methods"].items()]
        rows.sort(key=lambda t: (t[1].get("accuracy_pct") is None,
                                 t[1].get("accuracy_pct") or 0))
        for m, r in rows:
            if r.get("status") == "not run":
                print(f"{r['display'][:18]:18s} {'-':>6s} {'-':>12s} {'-':>12s} "
                      f"{'-':>10s} {'-':>9s} {'-':>9s} {'-':>9s}  not run")
                continue
            f = lambda v, w, d=0: (f"{v:>{w},.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
            print(f"{r['display'][:18]:18s} {f(r['accuracy_pct'],6,1)} {f(r['answer_tokens'],12)} "
                  f"{f(r['build_tokens'],12)} {f(r['answer_latency_median_s'],10,2)} "
                  f"{f(r['answer_latency_p95_s'],9,2)} {f(r['answer_latency_total_min'],9,1)} "
                  f"{f(r['build_min'],9,1)}  {r['status']}")


if __name__ == "__main__":
    data = collect()
    show(data)
    if "--print" not in sys.argv:
        EVAL.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(data, indent=2))
        print(f"\nwrote {OUT.relative_to(BENCH)}")
