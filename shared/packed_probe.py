"""Phase 3 instrumentation: nodes_scored for v8 (packed) vs v7 (flat manifest),
measured the same way, on the same 295 questions.

v8 nodes_scored comes straight from the answering run (recorded per question).
v7 nodes_scored is measured here by counting how many manifest entries v7's
traverse() scores per question: we monkeypatch pipeline._score to increment a
counter, then run pipeline.traverse (which makes NO LLM calls, so this is free)
over each question against v7's existing digests. v7 rescans the whole manifest
every step, so its count grows with document size; v8 scores only the children
along its path, so its count stays about f times depth.

Writes:
  results/eval/packed_probe/wiki_v8.json   per-movie medians + per-question rows
  results/eval/packed_probe/nodes_scored.csv
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import sys, json, pathlib, statistics, csv

BENCH = _BENCH
OUTD = BENCH / "results" / "eval" / "packed_probe"
MODE = sys.argv[1] if len(sys.argv) > 1 else "keyword"   # v8 mode to read

SLUG2TITLE = {
    "10_things_i_hate_about_you": "10 Things I Hate About You",
    "12_and_holding": "12 and Holding",
    "1492_conquest_of_paradise": "1492: Conquest of Paradise",
    "17_again_film": "17 Again (film)", "2012_film": "2012 (film)",
    "50_50_2011_film": "50/50 (2011 film)", "8mm_film": "8mm (film)",
    "9_2005_film": "9 (2005 film)", "a_dry_white_season": "A Dry White Season",
    "a_few_good_men": "A Few Good Men",
}

# --- set up v7 (paths + digests) via its cli; traverse makes no LLM calls ------
sys.path.insert(0, str(BENCH / "methods" / "wiki_v7"))
import cli as v7cli  # noqa: E402  sets run.DIGESTS, patches llm (unused by traverse)
run, pipeline = v7cli.run, v7cli.pipeline

# count every manifest entry v7 scores
_orig_score = pipeline._score
_counter = {"n": 0}
def _counting_score(qwords, text):  # noqa: E302
    _counter["n"] += 1
    return _orig_score(qwords, text)
pipeline._score = _counting_score


def measure_v7(question, wiki_dir):
    """v7 as actually run: reads consolidated.md (which exceeds the 12k budget) and
    stops, so it scores ~0 manifest entries."""
    _counter["n"] = 0
    try:
        pipeline.traverse(question, str(wiki_dir))
    except Exception as e:  # noqa: BLE001
        print(f"  v7 traverse error: {repr(e)[:80]}")
    return _counter["n"]


def measure_v7_flat(question, wiki_dir):
    """v7's flat index WITHOUT the consolidated-overview shortcut: the manifest scan
    it would do to route to a page. Mirrors traverse's loop (score every unvisited
    entry each step, read the best, expand the query) minus the consolidated/index
    pre-reads. Counts total entries scored. This is the O(N) flat-scan baseline."""
    import re
    manifest = json.loads((pathlib.Path(wiki_dir) / "_manifest.json").read_text())
    expanded = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    visited, total, scored, maxc = set(), 0, 0, 12000
    while total < maxc:
        best_s, best = -1, None
        for e in manifest:
            if e["path"] in visited:
                continue
            scored += 1
            s = len(expanded & set(re.findall(r"[a-zA-Z]{3,}", e["summary"].lower())))
            if e.get("level") == "branch":
                s += 1
            if s > best_s:
                best_s, best = s, e
        if best is None or best_s <= 0:
            break
        visited.add(best["path"])
        full = pathlib.Path(wiki_dir) / best["path"]
        if full.exists():
            c = full.read_text(encoding="utf-8", errors="ignore")[:maxc - total]
            total += len(c)
            expanded.update(re.findall(r"[a-zA-Z]{3,}", c.lower()))
    return scored


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(l) for l in
                 (BENCH / "data" / "questions" / "questions.jsonl").read_text().splitlines()
                 if l.strip()]

    # v8 per-question metrics from the answering run
    v8_rows = [json.loads(l) for l in
               (BENCH / "methods" / "wiki_v8" / "results" / f"wiki_v8_{MODE}.jsonl")
               .read_text().splitlines() if l.strip()]
    v8_by_qid = {r["qid"]: r for r in v8_rows}

    rows = []
    for q in questions:
        qid, slug = q["qid"], q["movie"]
        v8 = v8_by_qid.get(qid, {})
        wiki_dir = run.find_digest_dir(SLUG2TITLE[slug])
        v7_scored = measure_v7(q["question"], wiki_dir) if wiki_dir else None
        v7_flat = measure_v7_flat(q["question"], wiki_dir) if wiki_dir else None
        rows.append({
            "qid": qid, "movie": slug,
            "leaf_count": v8.get("leaf_count"),
            "v8_nodes_scored": v8.get("nodes_scored"),
            "v8_depth": v8.get("depth_reached"),
            "v7_nodes_scored": v7_scored,
            "v7_flat_nodes_scored": v7_flat,
        })
    print(f"measured {len(rows)} questions")

    # per-movie medians
    per_movie = {}
    by_movie = {}
    for r in rows:
        by_movie.setdefault(r["movie"], []).append(r)
    for slug, rs in by_movie.items():
        lc = [r["leaf_count"] for r in rs if r["leaf_count"]]
        v8s = [r["v8_nodes_scored"] for r in rs if r["v8_nodes_scored"] is not None]
        v7s = [r["v7_nodes_scored"] for r in rs if r["v7_nodes_scored"] is not None]
        v7f = [r["v7_flat_nodes_scored"] for r in rs if r["v7_flat_nodes_scored"] is not None]
        per_movie[slug] = {
            "n": len(rs),
            "leaf_count": statistics.median(lc) if lc else None,
            "v8_nodes_scored_med": statistics.median(v8s) if v8s else None,
            "v7_nodes_scored_med": statistics.median(v7s) if v7s else None,
            "v7_flat_nodes_scored_med": statistics.median(v7f) if v7f else None,
        }

    (OUTD / "wiki_v8.json").write_text(json.dumps(
        {"mode": MODE, "n_questions": len(rows), "per_movie": per_movie,
         "per_question": rows}, indent=2))

    with open(OUTD / "nodes_scored.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "movie", "leaf_count", "v8_nodes_scored",
                                          "v8_depth", "v7_nodes_scored", "v7_flat_nodes_scored"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n  movie                            leaves   v8_scored  v7_actual  v7_flat")
    for slug in sorted(per_movie, key=lambda s: per_movie[s]["leaf_count"] or 0):
        p = per_movie[slug]
        print(f"  {slug:32s} {p['leaf_count'] or 0:6.0f}   {p['v8_nodes_scored_med'] or 0:8.0f}   "
              f"{p['v7_nodes_scored_med'] or 0:8.0f}   {p['v7_flat_nodes_scored_med'] or 0:8.0f}")
    print(f"\nwrote {OUTD/'wiki_v8.json'} and {OUTD/'nodes_scored.csv'}")


if __name__ == "__main__":
    main()
