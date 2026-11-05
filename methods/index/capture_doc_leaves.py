"""Capture the shipped method (per_category tree + doc_leaves fallback) over the 295 questions to
produce real hallucination and token data for the per-method charts. Answers with the existing answer
path (fallback_mode=doc_leaves), judges with the existing judge (verdict + halluc), and logs answer
tokens. Writes methods/index_doc_leaves/results/index_doc_leaves_judged.jsonl (what the hallucination
chart reads) and prints the token totals. Checkpointed + resumable via eval_harness. No build, routing,
or judge-logic change. Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, pathlib, threading, copy
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
CFG_DOC = copy.copy(CFG); CFG_DOC.fallback_mode = "doc_leaves"   # the shipped method
OUT_DIR = HERE/"methods"/"index_doc_leaves"/"results"

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
        r = _answer.answer_question(d["idx"], slug, q["question"], CFG_DOC, d["emb"])
        v = _eval.judge_one({"qid": q["qid"], "question": q["question"], "gold_answers": q["gold_answers"],
                             "prediction": r["prediction"], "source": r["source"]})
        return {"movie": slug, "qid": q["qid"], "question": q["question"], "gold_answers": q["gold_answers"],
                "prediction": r["prediction"], "source": r["source"], "verdict": v["verdict"],
                "halluc": v["halluc"], "halluc_reason": v.get("halluc_reason", ""),
                # both key styles so downstream (make_graphs reads input_tokens/output_tokens) just works
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "in_tok": r["input_tokens"], "out_tok": r["output_tokens"], "fallback": r["fallback"]}

    tag = EH.config_tag(CFG_DOC, extra="method-capture")
    recs = [r for r in EH.run_checkpointed(tag, tasks, key, work, workers=12) if r]

    # assemble the judged file the hallucination chart reads (verdict + halluc per question)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR/"index_doc_leaves_judged.jsonl"
    with open(jpath, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    correct = sum(1 for r in recs if r["verdict"] == "correct")
    fab = sum(1 for r in recs if r["verdict"] == "incorrect" and r["halluc"] == "fabricated")
    in_tok = sum(r.get("in_tok", r.get("input_tokens", 0)) for r in recs)
    out_tok = sum(r.get("out_tok", r.get("output_tokens", 0)) for r in recs)
    summ = {"method": "index (ours) = per_category + doc_leaves", "n": n,
            "accuracy": round(100*correct/n, 1), "wrong_fab_pct": round(100*fab/n, 2),
            "in_tok": in_tok, "out_tok": out_tok, "answer_tok": in_tok + out_tok,
            "build_tok": 116717, "tok_per_q": round((in_tok + out_tok)/n),
            "judged_file": str(jpath)}
    (OUT_DIR/"summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== doc_leaves method capture ===")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
