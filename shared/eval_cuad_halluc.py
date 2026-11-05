"""eval_cuad_halluc.py — score CUAD with the SAME hallucination metric NarrativeQA uses:
"wrong & fabricated" = the judge calls the answer incorrect AND the fabrication judge calls it
fabricated, over all 410 clauses.

CUAD needs one adaptation, because half its questions have no gold answer at all:
  has-answer bucket (218 q): shared/eval.py judge_one -> verdict + halluc, verbatim.
  no-answer bucket (192 q):  abstaining is the CORRECT answer, so an abstention is
      correct + supported (never a hallucination, same as eval.py's miss rule). Answering
      without abstaining asserts a clause the contract does not contain, which is exactly
      incorrect + fabricated.
That keeps the definition identical to NarrativeQA's while respecting CUAD's no-answer gold.

Verdicts are cached per method in results/eval/cuad_halluc/<method>.jsonl (keyed by
qid+prediction), so reruns cost nothing. Writes results/eval/cuad_halluc_summary.json.

Usage from bench/:  envs/base/bin/python shared/eval_cuad_halluc.py [method ...]
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import sys, json, pathlib, importlib.util, hashlib, threading
from concurrent.futures import ThreadPoolExecutor

BENCH = _BENCH
EVAL = BENCH/"results"/"eval"
CACHE_DIR = EVAL/"cuad_halluc"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
_spec = importlib.util.spec_from_file_location("ev", BENCH/"shared"/"eval.py")
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)
WORKERS = 32

# deterministic abstention matcher (verbatim from eval_cuad_baselines.py)
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


Q = {q["qid"]: q for q in (json.loads(l) for l in
     (BENCH/"data/questions/cuad_questions.jsonl").read_text().splitlines() if l.strip())}

# display -> predictions file. graphrag_local lives under methods/graphrag/.
METHODS = {
    "naive":    BENCH/"methods/naive_gpt/results/naive_gpt_cuad.jsonl",
    "docetl":   BENCH/"methods/docetl_standard/results/docetl_standard_cuad.jsonl",
    "leann":    BENCH/"methods/leann/results/leann_cuad.jsonl",
    "rag":      BENCH/"methods/rag/results/rag_cuad.jsonl",
    "raptor":   BENCH/"methods/raptor/results/raptor_cuad.jsonl",
    "doctopus": BENCH/"methods/doctopus/results/doctopus_cuad.jsonl",
    "graphrag": BENCH/"methods/graphrag/results/graphrag_local_cuad.jsonl",
    "index":    None,   # special: predictions come from cuad_eval_records.jsonl (window)
}

_lock = threading.Lock()


def _key(qid, pred):
    return hashlib.sha1((qid + "||" + str(pred)).encode()).hexdigest()


def load_rows(disp):
    if disp != "index":
        p = METHODS[disp]
        return [{"qid": r["qid"], "prediction": r.get("prediction", ""), "source": r.get("source")}
                for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())]
    p = BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"
    return [{"qid": r["qid"], "prediction": r["window"]["prediction"], "source": r["window"].get("source")}
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())]


def score(disp):
    rows = [r for r in load_rows(disp) if r["qid"] in Q]
    cf_path = CACHE_DIR/f"{disp}.jsonl"
    cache = {}
    if cf_path.exists():
        for l in cf_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); cache[d["key"]] = d
    cf = open(cf_path, "a", encoding="utf-8")

    def process(r):
        q = Q[r["qid"]]
        pred, src = r["prediction"], r["source"]
        if not q["has_answer"]:
            # no-answer clause: abstaining IS correct; answering fabricates a clause
            if is_abstention(pred, src):
                return {"verdict": "correct", "halluc": "supported"}
            return {"verdict": "incorrect", "halluc": "fabricated"}
        k = _key(r["qid"], pred)
        if k in cache:
            return cache[k]
        v = ev.judge_one({"qid": r["qid"], "question": q["question"],
                          "gold_answers": q["gold_answers"] or [""],
                          "prediction": pred, "source": src})
        v = {"key": k, "qid": r["qid"], "verdict": v["verdict"], "halluc": v["halluc"],
             "halluc_reason": v.get("halluc_reason", "")}
        with _lock:
            cf.write(json.dumps(v, ensure_ascii=False) + "\n"); cf.flush(); cache[k] = v
        return v

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        judged = list(ex.map(process, rows))
    cf.close()

    n = len(judged)
    wf = sum(1 for v in judged if v["verdict"] == "incorrect" and v["halluc"] == "fabricated")
    acc = sum(1 for v in judged if v["verdict"] == "correct")
    out = {"method": disp, "n": n,
           "wrong_and_fabricated_pct": round(100*wf/n, 2) if n else 0,
           "wrong_and_fabricated_n": wf,
           "judge_accuracy_pct": round(100*acc/n, 2) if n else 0}
    print(f"[{disp}] n={n} wrong&fabricated={wf} ({out['wrong_and_fabricated_pct']}%) "
          f"acc={out['judge_accuracy_pct']}%", flush=True)
    return out


if __name__ == "__main__":
    want = sys.argv[1:] or list(METHODS)
    res = {}
    for m in want:
        res[m] = score(m)
    outp = EVAL/"cuad_halluc_summary.json"
    if outp.exists():                                   # merge with earlier partial runs
        old = json.loads(outp.read_text()); old.update(res); res = old
    outp.write_text(json.dumps(res, indent=2))
    print("wrote", outp)
