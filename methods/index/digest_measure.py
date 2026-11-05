"""Measure index (ours) in `digest` answer mode vs raw_chunk vs v7. Same abstain prompt, same judge,
same 295 questions; the only change from the shipped method is answer_context=digest (the answer model
reads a compiled digest of index's extracted subject content over the reached + doc_leaves region,
instead of the reached leaf's raw source chunk). Tree, retrieval, model, reasoning, and fallback
unchanged. Checkpointed + resumable via eval_harness. Writes results/digest_measure.json.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, pathlib, threading, copy, statistics, time
from math import comb
import concurrent.futures as cf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_BENCH/"shared"))
import client, config as _cfg, store as _store, pack_global as pg  # noqa: E402
import answer as _answer, eval as _eval, eval_harness as EH  # noqa: E402

SLUGS = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
         "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
         "a_dry_white_season", "a_few_good_men"]
CFG = _cfg.load()
CFG_DIGEST = copy.copy(CFG); CFG_DIGEST.fallback_mode = "doc_leaves"; CFG_DIGEST.answer_context = "digest"

_lock = threading.Lock()
_orig = client.embed
def _le(*a, **k):
    with _lock:
        return _orig(*a, **k)
client.embed = _le


def prewarm(idx):
    nodes = list(idx["nodes"].values())
    return {n["id"]: v for n, v in zip(nodes, client.embed([_answer._routing_text(n) for n in nodes],
                                                           model_name=CFG.embedding_model))}


def mcnemar(a_correct, b_correct):
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": 0, "p": 1.0}
    k = min(b, c)
    return {"b": b, "c": c, "n": n, "p": round(min(1.0, 2.0*sum(comb(n, i) for i in range(k+1))*0.5**n), 4)}


def main():
    client.embed(["warmup"])
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass
    built = {}
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        def b(slug):
            idx = _store.load_index(slug)
            return slug, {"idx": idx, "emb": prewarm(idx)}
        for slug, d in ex.map(b, SLUGS):
            built[slug] = d
            print(f"built {slug}", flush=True)

    qmap = {f"{s}::{q['qid']}": (s, q) for s in SLUGS for q in pg._questions(s)}
    tasks = [qmap[k] for k in sorted(qmap)]

    def key(t):
        s, q = t
        return f"{s}::{q['qid']}"

    def work(t):
        slug, q = t
        d = built[slug]
        t0 = time.time()
        r = _answer.answer_question(d["idx"], slug, q["question"], CFG_DIGEST, d["emb"])
        latency = round(time.time() - t0, 2)                 # per-question wall time (parallel run)
        v = _eval.judge_one({"qid": q["qid"], "question": q["question"], "gold_answers": q["gold_answers"],
                             "prediction": r["prediction"], "source": r["source"]})
        return {"slug": slug, "qid": q["qid"], "question": q["question"], "gold_answers": q["gold_answers"],
                "prediction": r["prediction"], "source": r["source"], "verdict": v["verdict"],
                "halluc": v["halluc"], "fallback": r["fallback"],
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "latency_s": latency}

    tag = EH.config_tag(CFG_DIGEST, extra="answerctx-digest")
    recs = [r for r in EH.run_checkpointed(tag, tasks, key, work, workers=12) if r]
    byqid = {f"{r['slug']}::{r['qid']}": r for r in recs}

    # comparison sets (saved verdicts, same judge)
    raw = {}
    rp = HERE/"methods"/"index_doc_leaves"/"results"/"index_doc_leaves_judged.jsonl"
    if rp.exists():
        for l in rp.read_text().splitlines():
            if l.strip():
                x = json.loads(l); raw[f"{x['movie']}::{x['qid']}"] = x
    v7 = {}
    for slug in SLUGS:
        f = HERE/"methods"/"wiki_v7"/"results"/"wiki_v7_judged.jsonl"
    vf = HERE/"methods"/"wiki_v7"/"results"/"wiki_v7_judged.jsonl"
    if vf.exists():
        for l in vf.read_text().splitlines():
            if l.strip():
                x = json.loads(l); v7[x["qid"]] = x

    n = len(recs)
    def acc(rs): return round(100*sum(1 for r in rs if r["verdict"] == "correct")/len(rs), 1)
    def miss(rs): return round(100*sum(1 for r in rs if r.get("source") == "miss")/len(rs), 1)
    def fab(rs): return round(100*sum(1 for r in rs if r["verdict"] == "incorrect" and r.get("halluc") == "fabricated")/len(rs), 1)
    # cost + latency for the digest run (build tokens are the per_category tree's, unchanged: digest is
    # selection-only, no extra LLM). Latency is per-question wall time from a PARALLEL run, so it is a
    # rough number (includes contention), not comparable to the report's summed sequential latencies.
    dig_in = sum(r.get("input_tokens", 0) for r in recs); dig_out = sum(r.get("output_tokens", 0) for r in recs)
    lats = [r.get("latency_s", 0) for r in recs if r.get("latency_s")]
    cost = {"answer_input_tokens": dig_in, "answer_output_tokens": dig_out, "answer_tokens": dig_in + dig_out,
            "answer_tokens_per_q": round((dig_in + dig_out)/n) if n else 0, "build_tokens": 116717,
            "latency_median_s_parallel": round(statistics.median(lats), 2) if lats else None,
            "latency_note": "per-question wall time from a parallel run; not comparable to sequential latency"}

    # align on common qids for paired stats
    qids = sorted(byqid)
    dig_c = [byqid[q]["verdict"] == "correct" for q in qids]
    out = {"n": n,
           "index_digest": {"accuracy": acc(recs), "miss_rate": miss(recs), "fabrication": fab(recs)},
           "digest_cost": cost}
    if raw:
        rl = [raw[q] for q in qids if q in raw]
        out["index_raw_chunk"] = {"accuracy": acc(rl), "miss_rate": miss(rl), "fabrication": fab(rl)}
        raw_c = [raw[q]["verdict"] == "correct" for q in qids if q in raw]
        dig_c_r = [byqid[q]["verdict"] == "correct" for q in qids if q in raw]
        out["mcnemar_digest_vs_raw"] = mcnemar(dig_c_r, raw_c)
    if v7:
        vqids = [q for q in qids if q.split("::", 1)[1] in v7]
        vl = [v7[q.split("::", 1)[1]] for q in vqids]
        out["v7"] = {"accuracy": acc(vl), "miss_rate": miss(vl)}
        dig_v = [byqid[q]["verdict"] == "correct" for q in vqids]
        v7_v = [v7[q.split("::", 1)[1]]["verdict"] == "correct" for q in vqids]
        out["mcnemar_digest_vs_v7"] = mcnemar(dig_v, v7_v)  # b = digest right/v7 wrong

    (HERE/"results"/"digest_measure.json").write_text(json.dumps(out, indent=2))
    print("\n=== DIGEST ANSWERING (295 questions) ===")
    for k in ("index_raw_chunk", "index_digest", "v7"):
        if k in out:
            m = out[k]
            print(f"  {k:16s} acc={m['accuracy']}%  miss={m.get('miss_rate','?')}%  fab={m.get('fabrication','n/a')}")
    if "mcnemar_digest_vs_v7" in out:
        m = out["mcnemar_digest_vs_v7"]
        print(f"  McNemar digest vs v7: b(digest>v7)={m['b']} c(v7>digest)={m['c']} n={m['n']} p={m['p']}")
    if "mcnemar_digest_vs_raw" in out:
        m = out["mcnemar_digest_vs_raw"]
        print(f"  McNemar digest vs raw_chunk: b(digest>raw)={m['b']} c(raw>digest)={m['c']} n={m['n']} p={m['p']}")
    print(f"  digest cost: answer_tok={cost['answer_tokens']:,} ({cost['answer_tokens_per_q']}/q) + build_tok={cost['build_tokens']:,}"
          f"  latency_med(parallel)={cost['latency_median_s_parallel']}s")


if __name__ == "__main__":
    main()
