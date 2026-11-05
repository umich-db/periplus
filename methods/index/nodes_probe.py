"""Phase 7 instrumentation: nodes_scored per question for `index` vs `v7`, measured the
same way, on the same 295 questions.

index nodes_scored comes straight from the answering run (recorded per question in
results/index_<slug>.jsonl). v7 has no bounded router: its traverse rescans the whole
manifest to route to a page, so its per-question nodes_scored is the flat-index width it
scans. That is measured here read-only from v7's stored digests by replaying its traverse
loop (which makes NO LLM calls), reusing shared/packed_probe.measure_v7_flat unchanged. v7's
code is not modified.

Writes:
  results/eval/nodes_probe/index.json    per-movie medians + per-question rows
  results/eval/nodes_probe/index.csv     one row per question (for the paper)
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, csv, pathlib, statistics

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
OUTD = BENCH/"results"/"eval"/"nodes_probe"

sys.path.insert(0, str(HERE))
import config as _cfg  # noqa: E402

# reuse the shared v7 read-only measurement (sets up v7 paths on import; makes no LLM calls)
sys.path.insert(0, str(BENCH/"shared"))
import packed_probe as _pp  # noqa: E402
measure_v7_flat, SLUG2TITLE, run = _pp.measure_v7_flat, _pp.SLUG2TITLE, _pp.run

SLUGS = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
         "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
         "a_dry_white_season", "a_few_good_men"]


def _index_rows():
    rows = {}
    for slug in SLUGS:
        p = HERE/"results"/f"index_{slug}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); rows[r["qid"]] = r
    return rows


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    cfg = _cfg.load()
    idx_by_qid = _index_rows()
    questions = [json.loads(l) for l in
                 (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip()]

    rows = []
    for q in questions:
        qid, slug = q["qid"], q["movie"]
        ir = idx_by_qid.get(qid, {})
        wiki_dir = run.find_digest_dir(SLUG2TITLE[slug])
        v7_flat = measure_v7_flat(q["question"], wiki_dir) if wiki_dir else None
        rows.append({
            "qid": qid, "movie": slug,
            "leaf_count": ir.get("leaf_count"),
            "nodes_scored": ir.get("nodes_scored"),
            "answer_tokens": ir.get("answer_tokens"),
            "depth_reached": ir.get("depth_reached"),
            "backtracks": ir.get("backtracks"),
            "hops": ir.get("hops"),
            "budget_spent": ir.get("budget_spent"),
            "fallback": ir.get("fallback"),
            "v7_nodes_scored": v7_flat,
        })
    print(f"measured {len(rows)} questions")

    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    per_movie, by_movie = {}, {}
    for r in rows:
        by_movie.setdefault(r["movie"], []).append(r)
    for slug, rs in by_movie.items():
        per_movie[slug] = {
            "n": len(rs),
            "leaf_count": med([r["leaf_count"] for r in rs]),
            "index_nodes_scored_med": med([r["nodes_scored"] for r in rs]),
            "v7_nodes_scored_med": med([r["v7_nodes_scored"] for r in rs]),
            "answer_tokens_med": med([r["answer_tokens"] for r in rs]),
            "backtracks_med": med([r["backtracks"] for r in rs]),
            "hops_med": med([r["hops"] for r in rs]),
            "budget_spent_med": med([r["budget_spent"] for r in rs]),
            "fallback_rate": round(sum(1 for r in rs if r["fallback"])/len(rs), 4),
        }

    (OUTD/"index.json").write_text(json.dumps(
        {"method": "index", "routing_mode": cfg.routing_mode, "fanout": cfg.f,
         "n_questions": len(rows), "per_movie": per_movie, "per_question": rows}, indent=2))

    with open(OUTD/"index.csv", "w", newline="") as f:
        cols = ["slug", "qid", "leaf_count", "nodes_scored", "answer_tokens",
                "depth_reached", "backtracks", "hops", "budget_spent", "fallback", "v7_nodes_scored"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({"slug": r["movie"], "qid": r["qid"], "leaf_count": r["leaf_count"],
                        "nodes_scored": r["nodes_scored"], "answer_tokens": r["answer_tokens"],
                        "depth_reached": r["depth_reached"], "backtracks": r["backtracks"],
                        "hops": r["hops"], "budget_spent": r["budget_spent"],
                        "fallback": r["fallback"], "v7_nodes_scored": r["v7_nodes_scored"]})

    print(f"\n  movie                            leaves  index_scored  v7_flat_scored")
    for slug in sorted(per_movie, key=lambda s: per_movie[s]["leaf_count"] or 0):
        p = per_movie[slug]
        print(f"  {slug:32s} {p['leaf_count'] or 0:6.0f}   {p['index_nodes_scored_med'] or 0:10.0f}   "
              f"{p['v7_nodes_scored_med'] or 0:12.0f}")
    print(f"\nwrote {OUTD/'index.json'} and {OUTD/'index.csv'}")


if __name__ == "__main__":
    main()
