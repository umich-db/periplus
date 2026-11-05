"""nav_probe.py — deterministic navigation-cost comparison, v7 vs v7_balanced.

v7's traverse() is pure Python keyword scoring (no LLM); the LLM is only called
AFTER traverse, to answer over the returned digest. So the navigation cost of a
question, nodes_scored (how many manifest summaries get scored) and pages_read,
is a deterministic function of (question, tree) and can be measured with ZERO
tokens by calling pipeline.traverse directly on each tree. This isolates the
structural effect of bounded fanout from the stochastic answer model.

For each of the 295 questions we run the UNCHANGED v7 traverse on both the v7
tree and the balanced tree, recording nodes_scored, pages_read, and whether the
returned digest is byte-identical (it must be for the 9 docs whose manifest did
not change). We also compute the flat-scan cost (v7 scores every unvisited entry
each round) and the true B+ descent cost (worst-case sibling-set size on a
root-to-leaf path, the thing bounded fanout actually caps).

Usage: python nav_probe.py   -> writes results/nav_probe.json
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
V7 = HERE.parent / "wiki_v7"
BENCH = _BENCH
sys.path.insert(0, str(V7))
import cli  # noqa: E402  setup (paths, llm routing); we never call the LLM here
pipeline, run = cli.pipeline, cli.run

V7_DIGESTS = BENCH / "methods" / "wiki_v7" / "output" / "digests_v7"
BAL_DIGESTS = HERE / "output" / "digests_v7_balanced"

SLUG2TITLE = {
 "10_things_i_hate_about_you": "10 Things I Hate About You", "12_and_holding": "12 and Holding",
 "1492_conquest_of_paradise": "1492: Conquest of Paradise", "17_again_film": "17 Again (film)",
 "2012_film": "2012 (film)", "50_50_2011_film": "50/50 (2011 film)", "8mm_film": "8mm (film)",
 "9_2005_film": "9 (2005 film)", "a_dry_white_season": "A Dry White Season", "a_few_good_men": "A Few Good Men",
}
SLUG2DOC = {
 "10_things_i_hate_about_you": "10_Things_I_Hate_About_You", "12_and_holding": "12_and_Holding",
 "1492_conquest_of_paradise": "1492__Conquest_of_Paradise", "17_again_film": "17_Again__film",
 "2012_film": "2012__film", "50_50_2011_film": "50-50__2011_film", "8mm_film": "8mm__film",
 "9_2005_film": "9__2005_film", "a_dry_white_season": "A_Dry_White_Season", "a_few_good_men": "A_Few_Good_Men",
}


def instrument():
    """Wrap _score to count scoring ops per traverse call. Behavior unchanged."""
    state = {"n": 0}
    orig = pipeline._score

    def counting(qw, text):
        state["n"] += 1
        return orig(qw, text)

    pipeline._score = counting
    return state


def descent_cost(manifest):
    """Worst-case B+ descent cost = max sibling-set size encountered on any
    root-to-leaf path = the max fanout of any node on the deepest path. For a
    bounded-fanout tree this is <= f; for v7 it is the widest node's child count."""
    children = Counter(e.get("parent") for e in manifest if e.get("parent") is not None)
    # per-node fanout, then the max over all internal nodes (any descent passes
    # through at most this many siblings at its worst level)
    return max(children.values()) if children else 0


def probe_tree(question, digest_dir, state):
    state["n"] = 0
    digest = pipeline.traverse(question, str(digest_dir))
    return {"nodes_scored": state["n"], "digest_len": len(digest), "digest": digest}


def main():
    state = instrument()
    questions = [json.loads(l) for l in (BENCH / "data" / "questions" / "questions.jsonl").read_text().splitlines() if l.strip()]

    rows = []
    for q in questions:
        slug = q["movie"]
        doc = SLUG2DOC[slug]
        v7dir = V7_DIGESTS / doc
        baldir = BAL_DIGESTS / doc
        v = probe_tree(q["question"], v7dir, state)
        b = probe_tree(q["question"], baldir, state)
        rows.append({
            "qid": q["qid"], "movie": slug,
            "v7_nodes_scored": v["nodes_scored"], "bal_nodes_scored": b["nodes_scored"],
            "v7_digest_len": v["digest_len"], "bal_digest_len": b["digest_len"],
            "digest_identical": v["digest"] == b["digest"],
        })

    # per-tree structural descent cost (worst-case), per doc
    descent = {}
    for slug, doc in SLUG2DOC.items():
        v7m = json.loads((V7_DIGESTS / doc / "_manifest.json").read_text())
        balm = json.loads((BAL_DIGESTS / doc / "_manifest.json").read_text())
        descent[slug] = {"v7_max_fanout": descent_cost(v7m), "bal_max_fanout": descent_cost(balm)}

    def avg(key):
        return round(sum(r[key] for r in rows) / len(rows), 1)

    changed = [r for r in rows if not r["digest_identical"]]
    out = {
        "n_questions": len(rows),
        "avg_v7_nodes_scored": avg("v7_nodes_scored"),
        "avg_bal_nodes_scored": avg("bal_nodes_scored"),
        "max_v7_nodes_scored": max(r["v7_nodes_scored"] for r in rows),
        "max_bal_nodes_scored": max(r["bal_nodes_scored"] for r in rows),
        "n_digest_identical": sum(1 for r in rows if r["digest_identical"]),
        "n_digest_changed": len(changed),
        "changed_qids": [r["qid"] for r in changed],
        "descent_max_fanout": descent,
        "worst_descent_v7": max(d["v7_max_fanout"] for d in descent.values()),
        "worst_descent_bal": max(d["bal_max_fanout"] for d in descent.values()),
        "rows": rows,
    }
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "nav_probe.json").write_text(json.dumps(out, indent=2))
    print(f"n={out['n_questions']}")
    print(f"flat-scan nodes_scored  v7 avg={out['avg_v7_nodes_scored']} max={out['max_v7_nodes_scored']}  "
          f"| balanced avg={out['avg_bal_nodes_scored']} max={out['max_bal_nodes_scored']}")
    print(f"B+ descent worst-case sibling-set (max fanout on a path)  v7={out['worst_descent_v7']}  balanced={out['worst_descent_bal']}")
    print(f"digests identical for {out['n_digest_identical']}/{out['n_questions']} questions; "
          f"changed only for: {out['changed_qids'][:5]}{'...' if len(changed)>5 else ''}")


if __name__ == "__main__":
    main()
