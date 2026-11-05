"""Stratified sampler for LLM-judge validation.

Draws a stratified random sample across (dataset, method) strata and writes a
CSV with one row per (question, prediction, gold, judge_verdict) triple, plus
an empty `human_verdict` column for manual labeling.

Sampling target defaults to ~100 rows total, ~6-7 per stratum (8 methods x 2
datasets = 16 strata). CUAD rows are drawn from the answerable subset only,
since no-answer abstention is scored deterministically and doesn't need judge
validation.

Usage (from bench/):
  envs/base/bin/python shared/judge_validation_sampler.py

Outputs: bench/results/eval/judge_validation_sample.csv
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import csv, json, pathlib, random

BENCH = _BENCH
EVAL  = BENCH/"results"/"eval"
OUT   = EVAL/"judge_validation_sample.csv"
random.seed(42)
PER_STRATUM = 7   # 8 methods x 2 datasets = 16 strata; 7 each = 112 rows

METHODS = ["naive", "docetl", "leann", "rag", "graphrag", "raptor", "doctopus", "index"]

# ---------- NarrativeQA: per-method judged files ----------
NQA_JUDGED = {
    "naive":    "methods/naive_gpt/results/naive_gpt_judged.jsonl",
    "docetl":   "methods/docetl_standard/results/docetl_standard_judged.jsonl",
    "leann":    "methods/leann/results/leann_judged.jsonl",
    "rag":      "methods/rag/results/rag_judged.jsonl",
    "graphrag": "methods/graphrag_local/results/graphrag_local_judged.jsonl",
    "raptor":   "methods/raptor/results/raptor_judged.jsonl",
    "doctopus": "methods/doctopus/results/doctopus_judged.jsonl",
}
def load_nqa():
    """Return {method: [{qid, question, gold, prediction, judge_verdict}, ...]}"""
    out = {m: [] for m in METHODS}
    for m, rel in NQA_JUDGED.items():
        for L in open(BENCH/rel):
            r = json.loads(L)
            out[m].append({
                "qid": r["qid"], "question": r["question"],
                "gold": " | ".join(r.get("gold_answers", [])),
                "prediction": r.get("prediction", ""),
                "judge_verdict": r["verdict"],
            })
    # ours: from nqa_eval_records.jsonl (window field carries verdict + prediction)
    for L in open(BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl"):
        r = json.loads(L)
        w = r["window"]
        out["index"].append({
            "qid": r["qid"], "question": r["question"],
            "gold": " | ".join(r.get("gold_answers", [])),
            "prediction": w.get("prediction", ""),
            "judge_verdict": w["verdict"],
        })
    return out

# ---------- CUAD: answerable subset only ----------
CUAD_JUDGED = {
    "naive":    "results/eval/cuad_halluc/naive.jsonl",
    "docetl":   "results/eval/cuad_halluc/docetl.jsonl",
    "leann":    "results/eval/cuad_halluc/leann.jsonl",
    "rag":      "results/eval/cuad_halluc/rag.jsonl",
    "graphrag": "results/eval/cuad_halluc/graphrag.jsonl",
    "raptor":   "results/eval/cuad_halluc/raptor.jsonl",
    "doctopus": "results/eval/cuad_halluc/doctopus.jsonl",
}
CUAD_PRED_FILES = {
    "naive":    "methods/naive_gpt/results/naive_gpt_cuad.jsonl",
    "docetl":   "methods/docetl_standard/results/docetl_standard_cuad.jsonl",
    "leann":    "methods/leann/results/leann_cuad.jsonl",
    "rag":      "methods/rag/results/rag_cuad.jsonl",
    "graphrag": "methods/graphrag/results/graphrag_local_cuad.jsonl",
    "raptor":   "methods/raptor/results/raptor_cuad.jsonl",
    "doctopus": "methods/doctopus/results/doctopus_cuad.jsonl",
}
CUAD_QUESTIONS = "data/questions/cuad_questions.jsonl"

def load_cuad():
    """Return {method: [{qid, question, gold, prediction, judge_verdict}, ...]}
    Restricted to has_answer questions (192 no-answer are scored deterministically)."""
    q_by_qid = {}
    for L in open(BENCH/CUAD_QUESTIONS):
        r = json.loads(L)
        q_by_qid[r["qid"]] = r
    out = {m: [] for m in METHODS}
    for m, pred_rel in CUAD_PRED_FILES.items():
        pred_by_qid = {}
        p = BENCH/pred_rel
        if not p.exists(): continue
        for L in open(p):
            r = json.loads(L)
            pred_by_qid[r["qid"]] = r.get("prediction", "")
        for L in open(BENCH/CUAD_JUDGED[m]):
            r = json.loads(L)
            qid = r["qid"]
            qmeta = q_by_qid.get(qid)
            if not qmeta: continue
            out[m].append({
                "qid": qid, "question": qmeta["question"],
                "gold": " | ".join(qmeta.get("gold_answers", [])),
                "prediction": pred_by_qid.get(qid, ""),
                "judge_verdict": r["verdict"],
            })
    # ours: from cuad_eval_records.jsonl (window field, has_answer subset only)
    for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
        r = json.loads(L)
        if not r.get("has_answer"): continue
        w = r["window"]
        out["index"].append({
            "qid": r["qid"], "question": r["question"],
            "gold": " | ".join(r.get("gold_answers", [])),
            "prediction": w.get("prediction", ""),
            "judge_verdict": "correct" if w.get("correct") else "incorrect",
        })
    return out

def stratified_sample(pool_by_method, k):
    """Balanced sample: k rows per method, random within method."""
    out = []
    for m in METHODS:
        rows = pool_by_method.get(m, [])
        if not rows:
            print(f"  WARN: {m} has 0 rows, skipping")
            continue
        picks = random.sample(rows, min(k, len(rows)))
        for r in picks:
            out.append({**r, "method": m})
    return out

nqa  = load_nqa()
cuad = load_cuad()
print("Pool sizes:")
for m in METHODS:
    print(f"  {m:10s}  NQA={len(nqa.get(m,[])):3d}  CUAD={len(cuad.get(m,[])):3d}")

sample_nqa  = stratified_sample(nqa,  PER_STRATUM)
sample_cuad = stratified_sample(cuad, PER_STRATUM)
for r in sample_nqa:  r["dataset"] = "NarrativeQA"
for r in sample_cuad: r["dataset"] = "CUAD"
rows = sample_nqa + sample_cuad
random.shuffle(rows)   # so labeler isn't biased by grouping

# reindex for easy reference
for i, r in enumerate(rows, 1):
    r["idx"] = i

cols = ["idx", "dataset", "method", "qid",
        "question", "gold", "prediction",
        "judge_verdict", "human_verdict", "notes"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({**{k: r.get(k, "") for k in cols}, "human_verdict": "", "notes": ""})

print(f"\nwrote {OUT}  ({len(rows)} rows for labeling)")
print("Fill in `human_verdict` as `correct` or `incorrect` for each row, then run:")
print("  envs/base/bin/python shared/judge_validation_score.py")
