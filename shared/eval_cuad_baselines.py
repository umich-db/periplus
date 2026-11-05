"""eval_cuad_baselines.py — score baseline CUAD predictions with the SAME methodology
as methods/wiki_v7_window/eval_cuad.py, so baselines are directly comparable to v7.

no-answer bucket (192 q): correct iff the system abstains ("not present"/"not found"/...),
  decided deterministically (no LLM). Any substantive clause text = fabrication_under_absence.
has-answer bucket (218 q): shared/eval.py LLM judge (correct/incorrect).

Reads methods/<m>/results/<m>_cuad.jsonl (or an explicit path). Writes
results/eval/<m>_cuad_summary.json. Usage: eval_cuad_baselines.py <method> [<method> ...]
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib, importlib.util, sys
from concurrent.futures import ThreadPoolExecutor

BENCH = _BENCH
EVAL = BENCH/"results"/"eval"; EVAL.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("ev", BENCH/"shared"/"eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
WORKERS = 32

# deterministic abstention matcher (verbatim from eval_cuad.py)
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

Q = {q["qid"]: q for q in (json.loads(l) for l in (BENCH/"data/questions/cuad_questions.jsonl").read_text().splitlines() if l.strip())}

def results_path(method):
    return BENCH/"methods"/method.replace("graphrag_local","graphrag").replace("graphrag_global","graphrag")/"results"/f"{method}_cuad.jsonl"

def score(method, path=None):
    p = pathlib.Path(path) if path else results_path(method)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    bad = ev.quota_contaminated(rows)
    if bad:
        print(f"[{method}] NOT SCORED: {len(bad)}/{len(rows)} rows are API-failure rows, not "
              f"answers. Remove them and resume the answer pass.", flush=True)
        return {"method": method, "n_questions": 0, "not_scored": f"{len(bad)} API-failure rows"}
    by_qid = {r["qid"]: r for r in rows}
    in_tok = sum(r.get("input_tokens",0) for r in rows); out_tok = sum(r.get("output_tokens",0) for r in rows)
    build = max((r.get("build_tokens",0) for r in rows), default=0)  # amortized-per-q; keep max as indicator

    has_q  = [q for q in Q.values() if q["has_answer"] and q["qid"] in by_qid]
    noa_q  = [q for q in Q.values() if not q["has_answer"] and q["qid"] in by_qid]

    # no-answer: deterministic
    noa_correct = 0; fabricated = 0
    for q in noa_q:
        r = by_qid[q["qid"]]
        if is_abstention(r.get("prediction",""), r.get("source")): noa_correct += 1
        else: fabricated += 1

    # has-answer: LLM judge (parallel)
    def judge(q):
        r = by_qid[q["qid"]]
        try:
            v = ev.judge_one({"qid": q["qid"], "question": q["question"],
                              "gold_answers": q["gold_answers"] or [""],
                              "prediction": r.get("prediction", ""),
                              "source": r.get("source")})
        except Exception:                            # noqa: BLE001
            # Same rule as shared/eval.py: a judge call that exhausts its retries
            # is a transport failure, not a wrong answer. It is excluded from the
            # denominator rather than scored incorrect, and reported.
            return None
        return v["verdict"] == "correct"
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        verdicts = list(ex.map(judge, has_q))
    unjudged = sum(1 for v in verdicts if v is None)
    has_correct = sum(1 for v in verdicts if v)
    has_n = len(has_q) - unjudged
    if unjudged:
        print(f"[{method}] WARNING: {unjudged}/{len(has_q)} answerable questions could "
              f"not be judged (endpoint errors); excluded from accuracy, not counted "
              f"as incorrect. Re-run to fill them in.", flush=True)

    n = has_n + len(noa_q)
    overall = 100*(has_correct + noa_correct)/n if n else 0
    summary = {
        "method": method, "n_questions": n,
        "overall_accuracy": round(overall, 2),
        "has_answer": {"n": has_n, "accuracy": round(100*has_correct/has_n, 2) if has_n else 0},
        "unjudged": unjudged,
        "no_answer_abstention": {"n": len(noa_q), "abstention_rate": round(100*noa_correct/len(noa_q), 2) if noa_q else 0},
        "fabrication_under_absence": {"n": len(noa_q), "count": fabricated,
                                      "pct": round(100*fabricated/len(noa_q), 2) if noa_q else 0},
        "answer_in_tokens": in_tok, "answer_out_tokens": out_tok, "answer_tokens": in_tok+out_tok,
    }
    (EVAL/f"{method}_cuad_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{method}] overall={overall:.1f}% has_ans={summary['has_answer']['accuracy']}% "
          f"abstain={summary['no_answer_abstention']['abstention_rate']}% "
          f"fab_under_absence={fabricated}/{len(noa_q)} -> {method}_cuad_summary.json", flush=True)
    return summary

if __name__ == "__main__":
    for m in sys.argv[1:]:
        try: score(m)
        except FileNotFoundError as e: print(f"[{m}] no results yet: {e}", flush=True)
