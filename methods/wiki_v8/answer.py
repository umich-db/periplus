"""wiki_v8 answer pass -> benchmark JSONL.

Answers one movie's questions over the packed index using packed.query_packed
(via cli.py, which routes every LLM call through the shared Azure client and
injects the cluster embedder). Snapshots the packed_answer token bucket per
question for clean attribution, and records the Phase 3 instrumentation
(nodes_scored, leaf_count, depth_reached, level_ups, hops, llm_route_calls).

Usage:
  answer.py --movie <slug> [--mode keyword|cosine|hybrid|llm]
Writes results/wiki_v8_<mode>_<slug>.jsonl (resumable).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cli  # noqa: E402  triggers setup: ac, cold_start, pipeline, packed; patches llm; injects embed

BENCH = _BENCH
import sys as _sys; _sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS as _DOCS  # noqa: E402
_CORPUS_KEY = "cuad" if __import__("os").environ.get("CUAD") else "nqa"

ac, packed = cli.ac, cli.packed
TXT = BENCH / "data" / "movies" / "txt"


def _answer_bucket(counters):
    c = counters.get("packed_answer", {})
    return c.get("prompt", 0), c.get("completion", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--mode", default="keyword",
                    choices=["keyword", "cosine", "hybrid", "llm"])
    args = ap.parse_args()
    slug, mode = args.movie, args.mode
    packed.configure(packed.CFG, route_mode=mode)

    store_dir = cli.OUTPUT / slug
    if not (store_dir / "packed.json").exists():
        print(f"NO packed index for {slug} at {store_dir}"); sys.exit(1)
    raw_doc = TXT / f"{slug}.txt"

    # amortized one-time build cost per question (from compile_stats.json)
    stats_path = HERE / "output" / "compile_stats.json"
    build_total = 0
    if stats_path.exists():
        build_total = json.loads(stats_path.read_text()).get(slug, {}).get("build_tokens", 0)

    questions = list(_DOCS[_CORPUS_KEY].questions)
    qs = [q for q in questions if q["movie"] == slug]
    amort = round(build_total / max(1, len(qs)))

    out = HERE / "results" / f"wiki_v8_{mode}_{slug}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["qid"])

    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done:
            continue
        bp, bc = _answer_bucket(ac.get_counters())
        t0 = time.time()
        try:
            pred, source, m = packed.query_packed(q["question"], str(store_dir),
                                                  str(raw_doc), route_mode=mode)
        except Exception as e:  # noqa: BLE001
            pred, source, m = f"[ERROR: {repr(e)[:120]}]", "error", {}
        lat = round(time.time() - t0, 2)
        ap2, ac2 = _answer_bucket(ac.get_counters())
        in_tok, out_tok = ap2 - bp, ac2 - bc
        rec = {"method": "wiki_v8", "route_mode": mode, "movie": slug, "qid": q["qid"],
               "question": q["question"], "gold_answers": q["gold_answers"],
               "prediction": pred, "source": source,
               "input_tokens": in_tok, "output_tokens": out_tok,
               "total_tokens": in_tok + out_tok, "latency_s": lat, "build_tokens": amort,
               "nodes_scored": m.get("nodes_scored"), "leaf_count": m.get("leaf_count"),
               "depth_reached": m.get("depth_reached"), "level_ups": m.get("level_ups"),
               "hops": m.get("hops"), "llm_route_calls": m.get("llm_route_calls")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"  {q['qid']} src={source} scored={m.get('nodes_scored')} "
              f"in={in_tok} out={out_tok} {lat}s :: {pred[:50]}", flush=True)
    f.close()
    print(f"DONE {slug} [{mode}]: {len(qs)} questions -> {out}", flush=True)


if __name__ == "__main__":
    main()
