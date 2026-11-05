"""Part 3 eval: v7_balanced vs v7_window on CUAD (10 contracts x 41 clause questions = 410).

Same v7 digest answer path as NarrativeQA, unchanged, and the same identical-digest answer reuse
(when both trees hand the answer model a byte-identical digest, one call serves both builds).

Judging follows the two CUAD buckets:
  - has_answer: the shared NarrativeQA LLM judge (bench/shared/eval.py judge_one), with the merged
    gold spans passed as the reference. Correct if the answer is semantically consistent with any span.
  - no-answer:  correct ONLY if the system abstains or states the clause is not present. Any
    substantive clause text is a fabrication under absence and scores wrong. This bucket is decided
    by an explicit, deterministic matcher (no LLM), so every verdict is auditable, and the raw
    predictions are persisted so the matcher can be re-checked.

Writes results/cuad_eval_records.jsonl and results/window_gate_cuad.json.

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

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
sys.path.insert(0, str(BENCH / "methods" / "wiki_v7"))
import cli  # noqa: E402
ac, pipeline, run = cli.ac, cli.pipeline, cli.run

sys.path.insert(0, str(BENCH / "shared"))
sys.path.insert(0, str(BENCH / "methods" / "index"))
import eval as _eval          # noqa: E402
import eval_harness as EH     # noqa: E402

BAL = BENCH / "methods" / "wiki_v7_balanced" / "output" / "digests_v7_balanced"
WIN = HERE / "output" / "digests_v7_window"
QFILE = BENCH / "data" / "questions" / "cuad_questions.jsonl"
CTXT = BENCH / "data" / "contracts" / "txt"
GOLD_CHAR_BUDGET = 4000

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
_orig_traverse = pipeline.traverse
pipeline.traverse = lambda q, wd, max_chars=None: _tl.digest

# ---- deterministic abstention matcher for the no-answer bucket ----
_MISS = ("", "not found", "none", "n/a", "not applicable")
_ABSENT = ("not found", "not present", "not contain", "does not contain", "no such",
           "not specified", "not mentioned", "not stated", "not included", "absent",
           "no clause", "not addressed", "not in the contract", "not in this contract",
           "silent on", "no provision")


def is_abstention(pred, source):
    p = (pred or "").strip().lower().rstrip(" .")
    if source == "miss" or p in _MISS:
        return True
    return any(x in p for x in _ABSENT)


def mcnemar(a, b):
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    if n == 0:
        return {"b": x, "c": y, "n": 0, "p": 1.0}
    k = min(x, y)
    return {"b": x, "c": y, "n": n,
            "p": round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n), 5)}


_jcache, _jlock = {}, threading.Lock()


def judge_has_answer(qid, question, gold, pred, source):
    k = (qid, pred)
    with _jlock:
        if k in _jcache:
            return _jcache[k]
    ref, used = [], 0
    for g in gold:                                            # cap the reference so judge prompts stay sane
        if used + len(g) > GOLD_CHAR_BUDGET:
            break
        ref.append(g); used += len(g)
    v = _eval.judge_one({"qid": qid, "question": question, "gold_answers": ref or gold[:1],
                         "prediction": pred, "source": source})
    with _jlock:
        _jcache[k] = v
    return v


def verdict_for(q, pred, source):
    """Returns (correct: bool, kind: str). no-answer is decided deterministically."""
    if not q["has_answer"]:
        ab = is_abstention(pred, source)
        return ab, "abstained" if ab else "fabricated_under_absence"
    v = judge_has_answer(q["qid"], q["question"], q["gold_answers"], pred, source)
    return v["verdict"] == "correct", v["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="v7_window_vs_balanced__cuad410")
    a = ap.parse_args()

    sel = json.loads((HERE / "results" / "cuad_selection.json").read_text())
    dirname = {s["slug"]: run.safe_filename(s["wiki_key"]) for s in sel}
    qs = [json.loads(l) for l in QFILE.read_text().splitlines() if l.strip()]
    have = {s for s, d in dirname.items()
            if (BAL / d / "_manifest.json").exists() and (WIN / d / "_manifest.json").exists()}
    missing = sorted(set(dirname) - have)
    if missing:
        print(f"[warn] {len(missing)} contracts missing a build, skipped: {missing}")
    qs = [q for q in qs if q["movie"] in have]
    qs.sort(key=lambda q: (q["movie"], q["qid"]))
    if a.limit:
        qs = qs[:a.limit]
    print(f"evaluating {len(qs)} CUAD questions over {len(have)} contracts, both builds")

    def key(q):
        return f"{q['movie']}::{q['qid']}"

    def work(q):
        d = dirname[q["movie"]]
        raw = str(CTXT / f"{q['movie']}.txt")
        _tl.acc = {}
        bdir, wdir = str(BAL / d), str(WIN / d)
        dg_b = _orig_traverse(q["question"], bdir)
        dg_w = _orig_traverse(q["question"], wdir)
        same = dg_b == dg_w
        _tl.digest = dg_b
        pb, sb, _ = pipeline.query_with_traversal(q["question"], bdir, raw)
        if same:
            pw, sw = pb, sb
        else:
            _tl.digest = dg_w
            pw, sw, _ = pipeline.query_with_traversal(q["question"], wdir, raw)
        cb, kb = verdict_for(q, pb, sb)
        cw, kw = (cb, kb) if (same or pw == pb) else verdict_for(q, pw, sw)
        acc = _tl.acc
        ai, ao = acc.get("wiki_answer", [0, 0]); ji, jo = acc.get("judge", [0, 0])
        return {"qid": q["qid"], "contract": q["contract_title"], "movie": q["movie"],
                "clause_type": q["clause_type"], "question": q["question"],
                "has_answer": q["has_answer"], "n_gold_spans": len(q.get("gold_spans", [])),
                "digest_identical": same,
                "balanced": {"prediction": pb, "source": sb, "correct": cb, "kind": kb},
                "window": {"prediction": pw, "source": sw, "correct": cw, "kind": kw},
                "answer_tokens": ai + ao, "judge_tokens": ji + jo}

    recs = [r for r in EH.run_checkpointed(a.tag, qs, key, work, workers=a.workers) if r]
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    with open(HERE / "results" / "cuad_eval_records.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def pct(rows, b):
        return round(100 * sum(1 for r in rows if r[b]["correct"]) / len(rows), 2) if rows else None

    def fb(rows, b):
        return round(100 * sum(1 for r in rows if r[b]["source"] in ("fallback", "miss")) / len(rows), 2) if rows else None

    has = [r for r in recs if r["has_answer"]]
    noa = [r for r in recs if not r["has_answer"]]
    per_contract = {}
    for c in sorted({r["movie"] for r in recs}):
        rows = [r for r in recs if r["movie"] == c]
        ident = sum(1 for r in rows if r["digest_identical"])
        per_contract[c] = {
            "n": len(rows), "balanced": pct(rows, "balanced"), "window": pct(rows, "window"),
            "digest_identical": ident, "digest_differ": len(rows) - ident,
            "digest_differ_rate": round(100 * (len(rows) - ident) / len(rows), 2),
            "has_answer": {"n": sum(1 for r in rows if r["has_answer"]),
                           "balanced": pct([r for r in rows if r["has_answer"]], "balanced"),
                           "window": pct([r for r in rows if r["has_answer"]], "window")},
            "no_answer": {"n": sum(1 for r in rows if not r["has_answer"]),
                          "balanced": pct([r for r in rows if not r["has_answer"]], "balanced"),
                          "window": pct([r for r in rows if not r["has_answer"]], "window")}}
    out = {
        "n_questions": len(recs), "n_contracts": len({r["movie"] for r in recs}),
        "digest_identical_rate": round(100 * sum(1 for r in recs if r["digest_identical"]) / len(recs), 2) if recs else None,
        "overall": {"balanced": pct(recs, "balanced"), "window": pct(recs, "window")},
        "has_answer": {"n": len(has), "balanced": pct(has, "balanced"), "window": pct(has, "window")},
        "no_answer_abstention": {"n": len(noa), "balanced": pct(noa, "balanced"), "window": pct(noa, "window")},
        "fabrication_under_absence": {
            "balanced": sum(1 for r in noa if r["balanced"]["kind"] == "fabricated_under_absence"),
            "window": sum(1 for r in noa if r["window"]["kind"] == "fabricated_under_absence")},
        "mcnemar_overall": mcnemar([r["window"]["correct"] for r in recs], [r["balanced"]["correct"] for r in recs]),
        "mcnemar_has_answer": mcnemar([r["window"]["correct"] for r in has], [r["balanced"]["correct"] for r in has]),
        "mcnemar_no_answer": mcnemar([r["window"]["correct"] for r in noa], [r["balanced"]["correct"] for r in noa]),
        "fallback_rate": {"balanced": fb(recs, "balanced"), "window": fb(recs, "window")},
        "answer_tokens_total": sum(r["answer_tokens"] for r in recs),
        "judge_tokens_total": sum(r["judge_tokens"] for r in recs),
        "per_contract": per_contract,
    }
    (HERE / "results" / "window_gate_cuad.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print(f"PART 3 — CUAD, {out['n_questions']} questions / {out['n_contracts']} contracts")
    print("=" * 78)
    print(f"digest identical on both trees: {out['digest_identical_rate']}%")
    print("  per-contract digest differ rate:")
    for c, v in sorted(per_contract.items(), key=lambda kv: -kv[1]["digest_differ_rate"]):
        print(f"    {v['digest_differ_rate']:6.2f}%  ({v['digest_differ']:3d}/{v['n']})  {c[:52]}")
    for k in ("overall", "has_answer", "no_answer_abstention"):
        d = out[k] if k != "overall" else {"n": out["n_questions"], **out["overall"]}
        print(f"  {k:22s} n={d['n']:4d}  balanced={d['balanced']}%  window={d['window']}%")
    print(f"  fabrication under absence: balanced={out['fabrication_under_absence']['balanced']} "
          f"window={out['fabrication_under_absence']['window']} (of {len(noa)})")
    for k in ("mcnemar_overall", "mcnemar_has_answer", "mcnemar_no_answer"):
        m = out[k]
        print(f"  {k:22s} b={m['b']} c={m['c']} n={m['n']} p={m['p']}")
    print(f"  answer tokens={out['answer_tokens_total']:,}  judge tokens={out['judge_tokens_total']:,}")


if __name__ == "__main__":
    main()
