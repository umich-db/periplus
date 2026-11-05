"""Part 2 eval: v7_balanced vs v7_window on the full 885-question NarrativeQA set (295 old + 590 new).

Both builds answer through the IDENTICAL v7 digest path. Nothing about digest compilation, the answer
path, or the judge is changed: the digest is produced by pipeline.traverse and the answer by
pipeline.query_with_traversal, both called unmodified. The only addition is a cost optimisation that
cannot change a result: v7 reads consolidated.md first and rarely navigates, so for most questions the
two trees hand the answer model a BYTE-IDENTICAL digest. We compute both digests locally (no LLM),
and when they match we make one answer call and assign it to both builds. When they differ we answer
each build separately. The fallback reads the same raw document for both builds, so an identical
digest implies an identical final record.

Judging reuses bench/shared/eval.py judge_one (the shared NarrativeQA judge), cached per
(qid, prediction) so an identical prediction on both builds is judged once.

Checkpointed and resumable via eval_harness.run_checkpointed.

Writes results/nqa_eval_records.jsonl and results/window_gate_nqa.json.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, math, pathlib, threading, argparse
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
V7 = BENCH / "methods" / "wiki_v7"
sys.path.insert(0, str(V7))
import cli  # noqa: E402  (wires shared azure client, llm routing, paths)
ac, pipeline, run = cli.ac, cli.pipeline, cli.run

sys.path.insert(0, str(BENCH / "shared"))
sys.path.insert(0, str(BENCH / "methods" / "index"))
import eval as _eval          # noqa: E402  shared judge (unchanged)
import eval_harness as EH     # noqa: E402

BAL_DIGESTS = BENCH / "methods" / "wiki_v7_balanced" / "output" / "digests_v7_balanced"
WIN_DIGESTS = HERE / "output" / "digests_v7_window"
QFILE = BENCH / "data" / "questions" / "questions_30.jsonl"
TXT = BENCH / "data" / "movies" / "txt"

OLD10 = {
    "10_things_i_hate_about_you": "10_Things_I_Hate_About_You", "12_and_holding": "12_and_Holding",
    "1492_conquest_of_paradise": "1492:_Conquest_of_Paradise", "17_again_film": "17_Again_(film)",
    "2012_film": "2012_(film)", "50_50_2011_film": "50-50_(2011_film)", "8mm_film": "8mm_(film)",
    "9_2005_film": "9_(2005_film)", "a_dry_white_season": "A_Dry_White_Season",
    "a_few_good_men": "A_Few_Good_Men",
}

# ---- per-thread token accounting by bucket (answer vs judge), via the shared client ----
_tl = threading.local()
_orig_chat = ac.chat


def _chat(msgs, max_tokens=2048, bucket="wiki_answer", **kw):
    text, pt, ct = _orig_chat(msgs, max_tokens=max_tokens, bucket=bucket, **kw)
    acc = getattr(_tl, "acc", None)
    if acc is not None:
        a = acc.setdefault(bucket, [0, 0]); a[0] += pt; a[1] += ct
    return text, pt, ct


ac.chat = _chat
_eval.ac.chat = _chat

# ---- reuse the real traverse for digest computation, stub it for the answer call ----
_orig_traverse = pipeline.traverse


def _traverse_stub(question, wiki_dir, max_chars=None):
    return _tl.digest


pipeline.traverse = _traverse_stub

_judge_cache, _jlock = {}, threading.Lock()


def judge_cached(qid, question, gold, pred, source):
    k = (qid, pred)
    with _jlock:
        if k in _judge_cache:
            return _judge_cache[k]
    v = _eval.judge_one({"qid": qid, "question": question, "gold_answers": gold,
                         "prediction": pred, "source": source})
    with _jlock:
        _judge_cache[k] = v
    return v


def slug_to_dirname():
    m = dict(OLD10)
    prep = json.loads((HERE / "results" / "nqa_prepared.json").read_text())
    for r in prep:
        m[r["slug"]] = r["wiki_key"]
    return {s: run.safe_filename(k) for s, k in m.items()}


def mcnemar(a, b):
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    if n == 0:
        return {"b": x, "c": y, "n": 0, "p": 1.0}
    k = min(x, y)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return {"b": x, "c": y, "n": n, "p": round(p, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="v7_window_vs_balanced__nqa885")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    dirname = slug_to_dirname()
    qs = [json.loads(l) for l in QFILE.read_text().splitlines() if l.strip()]
    have = {s for s in dirname
            if (BAL_DIGESTS / dirname[s] / "_manifest.json").exists()
            and (WIN_DIGESTS / dirname[s] / "_manifest.json").exists()}
    missing = sorted({q["movie"] for q in qs} - have)
    if missing:
        print(f"[warn] {len(missing)} movies missing a build, their questions are skipped: {missing[:5]}")
    qs = [q for q in qs if q["movie"] in have]
    qs.sort(key=lambda q: (q["movie"], q["qid"]))
    if a.limit:
        qs = qs[:a.limit]
    print(f"evaluating {len(qs)} questions over {len(have)} movies, both builds")

    def key(q):
        return f"{q['movie']}::{q['qid']}"

    def work(q):
        slug = q["movie"]
        d = dirname[slug]
        raw = str(TXT / f"{slug}.txt")
        _tl.acc = {}
        bal_dir, win_dir = str(BAL_DIGESTS / d), str(WIN_DIGESTS / d)
        dg_b = _orig_traverse(q["question"], bal_dir)
        dg_w = _orig_traverse(q["question"], win_dir)
        same = (dg_b == dg_w)

        _tl.digest = dg_b
        pb, sb, _ = pipeline.query_with_traversal(q["question"], bal_dir, raw)
        if same:
            pw, sw = pb, sb
        else:
            _tl.digest = dg_w
            pw, sw, _ = pipeline.query_with_traversal(q["question"], win_dir, raw)

        vb = judge_cached(q["qid"], q["question"], q["gold_answers"], pb, sb)
        vw = vb if (same or pw == pb) else judge_cached(q["qid"], q["question"], q["gold_answers"], pw, sw)

        acc = _tl.acc
        ans_in, ans_out = acc.get("wiki_answer", [0, 0])
        j_in, j_out = acc.get("judge", [0, 0])
        return {"qid": q["qid"], "movie": slug, "question": q["question"],
                "gold_answers": q["gold_answers"], "digest_identical": same,
                "digest_chars_balanced": len(dg_b), "digest_chars_window": len(dg_w),
                "balanced": {"prediction": pb, "source": sb, "verdict": vb["verdict"], "halluc": vb["halluc"]},
                "window": {"prediction": pw, "source": sw, "verdict": vw["verdict"], "halluc": vw["halluc"]},
                "answer_tokens": ans_in + ans_out, "judge_tokens": j_in + j_out}

    recs = [r for r in EH.run_checkpointed(a.tag, qs, key, work, workers=a.workers) if r]
    if a.no_report:
        for r in recs[:3]:
            print(json.dumps({k: r[k] for k in ("qid", "movie", "digest_identical", "balanced", "window")},
                             ensure_ascii=False)[:400])
        print(f"smoke ok: {len(recs)} records")
        return
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    with open(HERE / "results" / "nqa_eval_records.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    newslugs = {r["slug"] for r in json.loads((HERE / "results" / "nqa_prepared.json").read_text())}

    def acc_of(rows, build):
        return round(100 * sum(1 for r in rows if r[build]["verdict"] == "correct") / len(rows), 2) if rows else None

    def fb_of(rows, build):
        return round(100 * sum(1 for r in rows if r[build]["source"] in ("fallback", "miss")) / len(rows), 2) if rows else None

    old = [r for r in recs if r["movie"] not in newslugs]
    new = [r for r in recs if r["movie"] in newslugs]
    per_movie = {}
    for slug in sorted({r["movie"] for r in recs}):
        rows = [r for r in recs if r["movie"] == slug]
        per_movie[slug] = {"n": len(rows), "balanced": acc_of(rows, "balanced"),
                           "window": acc_of(rows, "window"),
                           "digest_identical": sum(1 for r in rows if r["digest_identical"]),
                           "is_new": slug in newslugs}

    b_ok = [r["balanced"]["verdict"] == "correct" for r in recs]
    w_ok = [r["window"]["verdict"] == "correct" for r in recs]
    out = {
        "n_questions": len(recs), "n_movies": len({r["movie"] for r in recs}),
        "digest_identical_rate": round(100 * sum(1 for r in recs if r["digest_identical"]) / len(recs), 2),
        "overall": {"balanced": acc_of(recs, "balanced"), "window": acc_of(recs, "window")},
        "old10": {"n": len(old), "balanced": acc_of(old, "balanced"), "window": acc_of(old, "window")},
        "new20": {"n": len(new), "balanced": acc_of(new, "balanced"), "window": acc_of(new, "window")},
        "mcnemar_window_vs_balanced": mcnemar(w_ok, b_ok),
        "mcnemar_old10": mcnemar([r["window"]["verdict"] == "correct" for r in old],
                                 [r["balanced"]["verdict"] == "correct" for r in old]),
        "mcnemar_new20": mcnemar([r["window"]["verdict"] == "correct" for r in new],
                                 [r["balanced"]["verdict"] == "correct" for r in new]),
        "fallback_rate": {"balanced": fb_of(recs, "balanced"), "window": fb_of(recs, "window")},
        "answer_tokens_total": sum(r["answer_tokens"] for r in recs),
        "judge_tokens_total": sum(r["judge_tokens"] for r in recs),
        "per_movie": per_movie,
    }
    (HERE / "results" / "window_gate_nqa.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"PART 2 — v7_window vs v7_balanced, {out['n_questions']} questions / {out['n_movies']} movies")
    print("=" * 78)
    print(f"digest identical on both trees: {out['digest_identical_rate']}% of questions")
    print(f"{'split':10s} {'n':>5s} {'balanced':>10s} {'window':>10s}   McNemar")
    for k, mk in (("overall", "mcnemar_window_vs_balanced"), ("old10", "mcnemar_old10"), ("new20", "mcnemar_new20")):
        d = out[k] if k != "overall" else {"n": out["n_questions"], **out["overall"]}
        m = out[mk]
        print(f"{k:10s} {d['n']:5d} {str(d['balanced']):>10s} {str(d['window']):>10s}   "
              f"b={m['b']} c={m['c']} p={m['p']}")
    print(f"fallback rate: balanced={out['fallback_rate']['balanced']}%  window={out['fallback_rate']['window']}%")
    print(f"answer tokens={out['answer_tokens_total']:,}  judge tokens={out['judge_tokens_total']:,}")


if __name__ == "__main__":
    main()
