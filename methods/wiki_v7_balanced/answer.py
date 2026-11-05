"""wiki_v7_balanced answer pass -> benchmark JSONL.

Identical to wiki_v7/answer.py (v7's traversal + digest + fallback, UNCHANGED),
except run.DIGESTS is pointed at the bounded-fanout balanced tree produced by
balance.py. This is the accuracy gate: v7's own answer path, run on the balanced
tree, judged by the shared judge.

Reuses wiki_v7/cli.py for setup (shared Azure client, llm routing, path
overrides). Adds nodes_scored / pages_read instrumentation by wrapping
pipeline._score and pipeline traverse logging (behavior unchanged; the wrapper
only counts). Checkpointed/resumable per movie.

Usage: answer.py --movie <slug>   (writes results/wiki_v7_balanced_<slug>.jsonl)
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
V7 = HERE.parent / "wiki_v7"
sys.path.insert(0, str(V7))            # import wiki_v7/cli (setup) unchanged
import cli  # noqa: E402  triggers: ac, cold_start, pipeline, run; patches llm; overrides paths

BENCH = _BENCH
ac, pipeline, run = cli.ac, cli.pipeline, cli.run

# --- point v7's traversal at the balanced digests (only change vs wiki_v7) ---
run.DIGESTS = HERE / "output" / "digests_v7_balanced"

SLUG2TITLE = {
 "10_things_i_hate_about_you": "10 Things I Hate About You", "12_and_holding": "12 and Holding",
 "1492_conquest_of_paradise": "1492: Conquest of Paradise", "17_again_film": "17 Again (film)",
 "2012_film": "2012 (film)", "50_50_2011_film": "50/50 (2011 film)", "8mm_film": "8mm (film)",
 "9_2005_film": "9 (2005 film)", "a_dry_white_season": "A Dry White Season", "a_few_good_men": "A Few Good Men",
}

# --- nodes-scored instrumentation: wrap _score, count per question (no behavior change) ---
_score_count = 0
_orig_score = pipeline._score


def _counting_score(query_words, text):
    global _score_count
    _score_count += 1
    return _orig_score(query_words, text)


pipeline._score = _counting_score


def wiki_answer_tot(counters):
    c = counters.get("wiki_answer", {})
    return c.get("prompt", 0), c.get("completion", 0)


def main():
    global _score_count
    ap = argparse.ArgumentParser(); ap.add_argument("--movie", required=True)
    slug = ap.parse_args().movie
    title = SLUG2TITLE[slug]
    wiki_dir = run.find_digest_dir(title)
    if wiki_dir is None:
        print(f"NO BALANCED DIGEST for {slug}/{title}"); sys.exit(1)
    raw_doc = BENCH / "data" / "movies" / "txt" / f"{slug}.txt"

    build_costs = json.loads((HERE / "build_costs.json").read_text())
    digest_key = wiki_dir.name
    build_total = build_costs.get(digest_key, {}).get("build_tokens_total", 0)

    questions = [json.loads(l) for l in (BENCH / "data" / "questions" / "questions.jsonl").read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]
    amort = round(build_total / max(1, len(qs)))

    out = HERE / "results" / f"wiki_v7_balanced_{slug}.jsonl"
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
        bp, bc = wiki_answer_tot(ac.get_counters())
        _score_count = 0
        t0 = time.time()
        try:
            pred, source, _ = pipeline.query_with_traversal(q["question"], str(wiki_dir), str(raw_doc))
        except Exception as e:  # noqa: BLE001
            pred, source = f"[ERROR: {repr(e)[:120]}]", "error"
        lat = round(time.time() - t0, 2)
        ap2, ac2 = wiki_answer_tot(ac.get_counters())
        in_tok, out_tok = ap2 - bp, ac2 - bc
        rec = {"method": "wiki_v7_balanced", "movie": slug, "qid": q["qid"], "question": q["question"],
               "gold_answers": q["gold_answers"], "prediction": pred, "source": source,
               "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
               "latency_s": lat, "build_tokens": amort, "nodes_scored": _score_count}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"  {q['qid']} src={source} in={in_tok} out={out_tok} scored={_score_count} {lat}s :: {pred[:50]}", flush=True)
    f.close()
    print(f"DONE {slug}: {len(qs)} questions -> {out}", flush=True)


if __name__ == "__main__":
    main()
