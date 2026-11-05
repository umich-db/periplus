"""record_build_time.py — the index's one-time build wall-clock, per corpus.

Sums the build segments compile_all.sh appends to
methods/llm_index/logs/_wall[_cuad].jsonl (one line per build run, so a resumed
build is a second segment) and writes results/eval/build_times/llm_index[_cuad].json
in the buildtimer schema the collector and figures read.

Why not the per-process timer: cli.py records each document's own process time
and add_build sums them, but compile_all.sh runs three documents at once, so that
sum overstates a 3-way build about threefold. The figure reports wall-clock for
the corpus, like every baseline, so that is what this records.

    BENCH_ROOT=/path/to/run_tree envs/base/bin/python methods/llm_index/record_build_time.py
"""

import os, sys, json, pathlib

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[2])
LOGS = BENCH / "methods" / "llm_index" / "logs"
DIGESTS = BENCH / "methods" / "llm_index" / "output" / "digests"
OUT = BENCH / "results" / "eval" / "build_times"


def segments(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue                      # a run killed mid-stamp leaves a partial line
        if "start" in d and "end" in d and d["end"] >= d["start"]:
            out.append(d)
    return out


def built(corpus_key):
    if not DIGESTS.exists():
        return 0
    docs = [p for p in DIGESTS.iterdir() if (p / "_manifest.json").exists()]
    if corpus_key == "cuad":
        return sum(1 for p in docs if p.name.startswith("CUAD_"))
    return sum(1 for p in docs if not p.name.startswith("CUAD_"))


def record(corpus_key):
    sfx = "_cuad" if corpus_key == "cuad" else ""
    segs = segments(LOGS / f"_wall{sfx}.jsonl")
    if not segs:
        print(f"  {corpus_key}: no build segments recorded yet")
        return None
    minutes = sum(s["end"] - s["start"] for s in segs) / 60
    n_total = max(s.get("docs_in_corpus", 0) for s in segs) or (10 if corpus_key == "cuad" else 30)
    n_built = built(corpus_key)
    rec = {"method": "llm_index", "corpus": corpus_key, "build_min": round(minutes, 3),
           "build_coverage": "full" if n_built >= n_total else f"{n_built}/{n_total}",
           "n_built": n_built, "n_total": n_total, "segments": len(segs),
           "sources": sorted({s.get("source", "stamp") for s in segs})}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm_index{sfx}.json").write_text(json.dumps(rec, indent=2))
    print(f"  {corpus_key}: {minutes:.1f} min wall-clock over {len(segs)} run(s), "
          f"{n_built}/{n_total} documents built")
    return rec


if __name__ == "__main__":
    print("llm_index build wall-clock:")
    r = [record("nqa"), record("cuad")]
    sys.exit(0 if any(r) else 1)
