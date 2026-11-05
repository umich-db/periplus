"""Diagnostic (not part of the shipped method): of the questions the default config MISSES,
how many are recoverable by the wide fallback (whole source document re-read)? This isolates
whether the accuracy gap is the index/tree layer (routing reaches the wrong or too-thin leaf)
versus genuinely unanswerable questions. Judged with the shared eval judge. Read-only on v7.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as _cfg, store as _store, answer  # noqa: E402
sys.path.insert(0, str(_BENCH/"shared"))
import eval as _eval  # noqa: E402  (judge_one)

BENCH = _BENCH
SLUGS = answer.SLUGS
cfg = _cfg.load(); cfg.fallback_wide = True                     # force wide re-read for the probe

qmap = {q["qid"]: q for q in (json.loads(l) for l in
        (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip())}

recovered = still_miss = 0
per_movie = {}
for slug in SLUGS:
    rows = [json.loads(l) for l in (HERE/"results"/f"index_{slug}.jsonl").read_text().splitlines() if l.strip()]
    misses = [r for r in rows if r["source"] == "miss"]
    if not misses:
        continue
    idx = _store.load_index(slug); emb = {}
    rec = 0
    for r in misses:
        q = qmap[r["qid"]]
        res = answer.answer_question(idx, slug, q["question"], cfg, emb)
        v = _eval.judge_one({"qid": r["qid"], "question": q["question"],
                             "gold_answers": q["gold_answers"], "prediction": res["prediction"],
                             "source": res["source"]})
        if v["verdict"] == "correct":
            rec += 1; recovered += 1
        else:
            still_miss += 1
    per_movie[slug] = {"misses": len(misses), "recovered_by_wide": rec}
    print(f"{slug:32s} misses={len(misses):2d} recovered_by_wide={rec}", flush=True)

n = 295
base_correct = round(0.356 * n)
print(f"\nDEFAULT misses: {recovered+still_miss} | recovered by wide re-read: {recovered} | still miss: {still_miss}")
print(f"accuracy default = 35.6%  ->  ceiling with wide fallback = {(base_correct+recovered)/n*100:.1f}%")
(HERE/"results"/"diag_wide_recovery.json").write_text(json.dumps(
    {"recovered": recovered, "still_miss": still_miss, "per_movie": per_movie}, indent=2))
