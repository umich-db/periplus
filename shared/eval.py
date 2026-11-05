"""Uniform evaluation harness for the long-document QA benchmark.

Scores every methods/<method>/results/<method>.jsonl with:
  - judge_accuracy: LLM judge (correct/incorrect) over all questions
  - fabrication_pct: 2nd LLM judge pass (supported/fabricated)
  - token cost: input/output (answer) + build_tokens, build/answer split, per-q

Judging is concurrent (thread pool). ALL judge calls go through Azure (shared
client), tagged to the `judge` bucket so they never contaminate method costs.
Verdicts are cached per method.

Per-question detail is written so every answer can be inspected:
  methods/<method>/results/<method>_judged.jsonl — full row: question, gold,
      prediction, verdict, judge_reason, halluc verdict + reason, tokens, source
  methods/<method>/results/<method>_errors.md    — human-readable list of WRONG
      answers with the judge's reason, plus fabrications

The summary goes to results/eval/<method>_summary.json instead.

Usage: eval.py [method ...]   (default: all methods with a results file)
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import sys, json, pathlib, importlib.util, hashlib, threading, re
from concurrent.futures import ThreadPoolExecutor
import collections


def _tok(t):
    return set(re.findall(r"[a-z0-9']+", str(t).lower()))


def _f1(pred, gold):
    P, G = _tok(pred), _tok(gold)
    if not P or not G:
        return 0.0
    tp = len(P & G)
    pr, rc = tp/len(P), tp/len(G)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

METHODS_DIR = BENCH/"methods"
EVAL_DIR = BENCH/"results"/"eval"; EVAL_DIR.mkdir(parents=True, exist_ok=True)
WORKERS = 32   # raised 12->32: endpoint shows no 429s, headroom for faster judging

JUDGE_PROMPT = """You are an answer-equivalence judge. Given a question, the \
ground-truth answer(s), and a predicted answer, decide if the prediction is \
essentially correct (conveys the same core fact, even if worded differently).

Question: {q}
Ground truth answer(s): {gold}
Predicted answer: {pred}

Reply with a single line: "correct" or "incorrect", then a one-line reason."""

HALLUC_PROMPT = """You check whether a predicted answer is FABRICATED. A \
prediction is "fabricated" if it asserts specific details (names, numbers, \
events) that are NOT entailed by the gold answer(s) and would not be supported \
by the source document. If the prediction is vague, hedged, says "not found", \
or is consistent with the gold answer, it is "supported".

Question: {q}
Ground truth answer(s): {gold}
Predicted answer: {pred}

Reply with a single line: "supported" or "fabricated", then a one-line reason."""

_cache_lock = threading.Lock()


def _key(rec):
    return hashlib.sha1((rec["qid"] + "||" + str(rec["prediction"])).encode()).hexdigest()


def load_cache(method):
    p = EVAL_DIR/f"{method}_verdicts.jsonl"; cache = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if "result" not in d:  # backfill older verdicts
                    d["result"] = "right" if d.get("verdict") == "correct" else "wrong"
                cache[d["key"]] = d
    return cache


def judge_one(rec):
    gold = " | ".join(rec["gold_answers"])
    pred = str(rec["prediction"] or "").strip()
    pl = pred.lower()
    # A miss/abstention (empty, "not found", error, or source=miss) is NEVER a
    # hallucination — the system declined to answer, it didn't fabricate.
    is_miss = (pl in ("", "not found", "none", "n/a") or pred.startswith("[ERROR")
               or "not found" in pl or rec.get("source") in ("miss", "none"))
    if is_miss:
        return {"result": "wrong", "verdict": "incorrect", "judge_reason": "miss / not-found / error (abstained)",
                "halluc": "supported", "halluc_reason": "abstained — not a fabrication", "miss": 1}
    jt, _, _ = ac.chat([{"role": "user", "content": JUDGE_PROMPT.format(q=rec["question"], gold=gold, pred=pred)}],
                       max_tokens=2048, bucket="judge")
    line0 = jt.strip().split("\n")[0].lower()
    correct = "incorrect" not in line0 and "correct" in line0
    ht, _, _ = ac.chat([{"role": "user", "content": HALLUC_PROMPT.format(q=rec["question"], gold=gold, pred=pred)}],
                       max_tokens=2048, bucket="judge")
    h0 = ht.strip().split("\n")[0].lower()
    fabricated = "fabricated" in h0
    return {"result": "right" if correct else "wrong",
            "verdict": "correct" if correct else "incorrect",
            "judge_reason": jt.strip().split("\n")[-1][:240] if "\n" in jt.strip() else jt.strip()[:240],
            "halluc": "fabricated" if fabricated else "supported",
            "halluc_reason": ht.strip().split("\n")[-1][:240] if "\n" in ht.strip() else ht.strip()[:240],
            "miss": 0}



def quota_contaminated(rows):
    """Rows written while the API key was dead.

    Every runner catches an exception from the client and records it as the
    prediction, so a key that runs out mid-pass leaves rows like
    "[ERROR: QuotaExhausted(...)]" with zero tokens. Scored, they are misses:
    GraphRAG was published at 0.0% from 528 of them, and the index at 0.0% from
    676, with the real answers excluded as unjudgeable. They are not answers and
    must never reach a score, so scoring refuses while any are present.
    """
    return [r for r in rows
            if str(r.get("prediction", "")).startswith("[ERROR")
            and ("QuotaExhausted" in str(r.get("prediction")) or "usage limit" in str(r.get("prediction")).lower()
                 or not ((r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)))]


def eval_method(method):
    rp = METHODS_DIR/method/"results"/f"{method}.jsonl"
    rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
    bad = quota_contaminated(rows)
    if bad:
        print(f"[{method}] NOT SCORED: {len(bad)}/{len(rows)} rows are API-failure rows "
              f"(e.g. {str(bad[0].get('prediction'))[:60]!r}), not answers. Remove them and "
              f"resume the answer pass; scoring them would publish misses as a result.", flush=True)
        return {"method": method, "n_questions": 0, "not_scored": f"{len(bad)} API-failure rows"}
    cache = load_cache(method)
    cf = open(EVAL_DIR/f"{method}_verdicts.jsonl", "a", encoding="utf-8")

    def process(rec):
        k = _key(rec)
        if k in cache:
            return {**rec, **cache[k]}
        try:
            v = judge_one(rec)
        except Exception as e:                       # noqa: BLE001
            # One judge call that exhausts its retries must not abort the pass.
            # ThreadPoolExecutor.map re-raises on iteration, so before this a
            # single failure threw away every verdict already computed for the
            # method -- on a 42%-per-call 404 rate that happened reliably, and a
            # whole scoring run died with nothing written. An unjudged record is
            # recorded AS unjudged and excluded from the denominator, because
            # scoring it "incorrect" would silently understate accuracy by
            # whatever fraction the endpoint happened to drop.
            return {**rec, "result": "unjudged", "verdict": "unjudged",
                    "judge_reason": f"judge call failed: {repr(e)[:160]}",
                    "halluc": "unjudged", "halluc_reason": "", "miss": 0,
                    "key": k, "qid": rec["qid"], "_unjudged": True}
        v["key"] = k; v["qid"] = rec["qid"]
        with _cache_lock:
            cf.write(json.dumps(v, ensure_ascii=False) + "\n"); cf.flush()
            cache[k] = v
        return {**rec, **v}

    judged = [None]*len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(process, rows)):
            judged[i] = res
    cf.close()
    # normalize the verdict cache file: dedup + ensure `result` on every line
    with open(EVAL_DIR/f"{method}_verdicts.jsonl", "w", encoding="utf-8") as vf:
        for v in cache.values():
            vf.write(json.dumps(v, ensure_ascii=False) + "\n")

    # --- detailed per-question log, kept WITH the system it belongs to ---
    method_dir = METHODS_DIR/method/"results"
    detail_cols = ["qid", "movie", "question", "gold_answers", "prediction", "source",
                   "result", "verdict", "judge_reason", "halluc", "halluc_reason",
                   "input_tokens", "output_tokens", "build_tokens"]
    with open(method_dir/f"{method}_judged.jsonl", "w", encoding="utf-8") as jf:
        for x in judged:
            jf.write(json.dumps({c: x.get(c) for c in detail_cols}, ensure_ascii=False) + "\n")

    # --- human-readable errors file (wrong + fabricated), also per-system ---
    wrong = [x for x in judged if x["verdict"] == "incorrect"]
    fabs = [x for x in judged if x["halluc"] == "fabricated"]
    with open(method_dir/f"{method}_errors.md", "w", encoding="utf-8") as ef:
        ef.write(f"# {method}: {len(wrong)} incorrect / {len(judged)} ({len(fabs)} fabricated)\n\n")
        for x in wrong:
            ef.write(f"## {x['qid']} ({x['movie']})\n")
            ef.write(f"- **Q:** {x['question']}\n")
            ef.write(f"- **Gold:** {x['gold_answers']}\n")
            ef.write(f"- **Predicted:** {x['prediction']}\n")
            ef.write(f"- **Source:** {x.get('source','-')}\n")
            ef.write(f"- **Judge:** {x['judge_reason']}\n")
            if x["halluc"] == "fabricated":
                ef.write(f"- **Fabricated:** {x['halluc_reason']}\n")
            ef.write("\n")

    unjudged = [x for x in judged if x.get("_unjudged")]
    if unjudged:
        print(f"[{method}] WARNING: {len(unjudged)}/{len(judged)} records could not be "
              f"judged (endpoint errors). They are EXCLUDED from accuracy, not counted "
              f"as incorrect. Re-run to fill them in.", flush=True)
    judged = [x for x in judged if not x.get("_unjudged")]
    if not judged:
        print(f"[{method}] every judge call failed — writing no summary", flush=True)
        return {"method": method, "n_questions": 0, "unjudged": len(unjudged)}

    n = len(judged)
    correct = sum(1 for x in judged if x["verdict"] == "correct")
    fab = len(fabs)
    # token-F1==0 hallucination (paper-comparable metric): answered but zero word overlap
    f1s = [max((_f1(x["prediction"], g) for g in x["gold_answers"]), default=0.0) for x in judged]
    f1_halluc = sum(1 for x, s in zip(judged, f1s) if s == 0 and not x.get("miss"))
    avg_f1 = round(sum(f1s)/n, 4) if n else 0
    miss_n = sum(1 for x in judged if x.get("miss"))
    in_tok = sum(x.get("input_tokens", 0) for x in judged)
    out_tok = sum(x.get("output_tokens", 0) for x in judged)
    bc_path = METHODS_DIR/method/"build_costs.json"
    if bc_path.exists():
        bc = json.loads(bc_path.read_text())
        build_tok = sum(v.get("build_tokens_total", v.get("build_tokens", 0)) for v in bc.values())
    else:
        build_tok = sum(x.get("build_tokens", 0) for x in judged)
    per_movie = {}
    bym = collections.defaultdict(list)
    for x in judged:
        bym[x["movie"]].append(x)
    for m, xs in sorted(bym.items()):
        per_movie[m] = {"n": len(xs),
                        "judge_accuracy": round(sum(1 for y in xs if y["verdict"] == "correct")/len(xs), 4),
                        "fabrication_pct": round(100*sum(1 for y in xs if y["halluc"] == "fabricated")/len(xs), 2)}
    summary = {"method": method, "n_questions": n,
               "judge_accuracy": round(correct/n, 4) if n else 0,
               "fabrication_pct": round(100*fab/n, 2) if n else 0,        # LLM-judge (strict, spec §4)
               "hallucination_f1_pct": round(100*f1_halluc/n, 2) if n else 0,  # token-F1==0 (paper-comparable)
               "avg_token_f1": avg_f1, "miss_pct": round(100*miss_n/n, 2) if n else 0,
               "total_input_tokens": in_tok, "total_output_tokens": out_tok,
               "answer_tokens": in_tok+out_tok, "build_tokens": build_tok,
               "tokens_per_question": round((in_tok+out_tok)/n, 1) if n else 0,
               "unjudged": len(unjudged),
               "per_movie": per_movie}
    (EVAL_DIR/f"{method}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{method}] acc={summary['judge_accuracy']:.1%} fab={summary['fabrication_pct']}% "
          f"in={in_tok:,} out={out_tok:,} build={build_tok:,} | wrong={len(wrong)} -> {method}_errors.md", flush=True)
    return summary


def find_methods():
    return [m.name for m in sorted(METHODS_DIR.iterdir())
            if (m/"results"/f"{m.name}.jsonl").exists() and (m/"results"/f"{m.name}.jsonl").stat().st_size > 0]


def main():
    methods = sys.argv[1:] or find_methods()
    print(f"evaluating: {methods}")
    all_sum = [eval_method(m) for m in methods]
    jt = ac.get_counters().get("judge", {})
    print(f"\njudge bucket: in={jt.get('prompt',0):,} out={jt.get('completion',0):,} calls={jt.get('calls',0)}")
    (EVAL_DIR/"all_summaries.json").write_text(json.dumps(all_sum, indent=2))


if __name__ == "__main__":
    main()
