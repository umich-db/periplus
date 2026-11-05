"""Answer-step span diagnostic: does giving the answer model MORE source text than the current chunk
help, or is the ceiling the answer step / extraction? Runs on the oracle set (perfect routing to the
answer leaf), varying only how much source text the answer model sees. Analysis only; reuses the
per_category leaves, the all-mpnet embedder, the real answer call, and the judge. Built on
eval_harness so every condition is checkpointed and resumable.

Data note: in this corpus a subject's recorded source span EQUALS its chunk (ratio 1.00), and the
answer path already reads that full chunk. So "current span" and "full chunk" are the same text here;
the real question is whether widening BEYOND the current chunk (into adjacent document, up to the full
source) beats it. Conditions, narrow -> wide:
  current   : the answer leaf's chunk (what the system reads now, ~6k chars)
  pad<N>    : that chunk extended by N characters each side into the document
  fulldoc   : the whole source document (the wide extreme)

Two guardrails (per the request): (1) each condition+pad gets its OWN checkpoint tag, so a resume can
never mix conditions. (2) the oracle answer leaf is selected ONCE per question (deterministic argmax of
all-mpnet cosine between (question + gold) and leaf content), cached to results/oracle_leaves.json, and
every condition is fed the SAME leaf's span variants.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, pathlib, argparse, statistics, threading
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_BENCH/"shared"))
import client, config as _cfg, store as _store, chunk as _chunk, pack_global as pg  # noqa: E402
import answer as _answer, eval as _eval, eval_harness as EH  # noqa: E402

SLUGS = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
         "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
         "a_dry_white_season", "a_few_good_men"]
CFG = _cfg.load()
COVER_BAR = 0.5269          # same calibrated bar as the diagnostics: answer captured in a leaf
FULLDOC_CAP = 120000        # char cap for the fulldoc extreme (keeps token cost bounded)
ORACLE_PATH = HERE/"results"/"oracle_leaves.json"

_lock = threading.Lock()
_orig = client.embed
def _le(*a, **k):
    with _lock:
        return _orig(*a, **k)
client.embed = _le


def load_movie(slug):
    idx = _store.load_index(slug)
    doc = _chunk.load_source(slug)
    leaves = sorted((n for n in idx["nodes"].values() if n["kind"] == "leaf"), key=lambda n: n["id"])
    chunks = idx["chunks"]
    return {"leaves": leaves, "chunks": chunks, "doc": doc}


def select_oracle(movies, fresh=False):
    """Pick the answer leaf once per question and cache it. Deterministic: leaves are in id order and
    argmax takes the first (lowest-id) leaf on a tie; all-mpnet is deterministic."""
    if ORACLE_PATH.exists() and not fresh:
        return json.loads(ORACLE_PATH.read_text())
    oracle = {}
    for slug in SLUGS:
        m = movies[slug]
        lc = [lf["content"] for lf in m["leaves"]]
        lvecs = client.embed(lc, model_name=CFG.embedding_model)
        for q in pg._questions(slug):
            gold = " | ".join(q["gold_answers"])
            gv = client.embed([q["question"] + " " + gold], model_name=CFG.embedding_model)[0]
            cos = lvecs @ gv
            best = int(np.argmax(cos))
            lf = m["leaves"][best]; src = lf["source"]
            oracle[f"{slug}::{q['qid']}"] = {"slug": slug, "qid": q["qid"], "leaf_id": lf["id"],
                                             "chunk_id": src["chunk_id"], "start_char": src["start_char"],
                                             "end_char": src["end_char"], "maxcos": float(cos[best])}
        print(f"selected oracle leaves for {slug}", flush=True)
    ORACLE_PATH.write_text(json.dumps(oracle, indent=2))
    return oracle


def span_text(condition, pad, oc, movie):
    """The source text the answer model sees for this leaf under this condition."""
    doc = movie["doc"]; s, e = oc["start_char"], oc["end_char"]
    if condition == "current":                              # the leaf's chunk (== current span here)
        ck = movie["chunks"].get(oc["chunk_id"]) or movie["chunks"].get(str(oc["chunk_id"]))
        return ck["text"] if ck else doc[s:e]
    if condition == "widened":
        return doc[max(0, s - pad):min(len(doc), e + pad)]
    if condition == "fulldoc":
        return doc[:FULLDOC_CAP]
    raise ValueError(condition)


def run_condition(condition, pad, oracle, movies, fresh=False, workers=10):
    tag = EH.config_tag(CFG, extra=f"span-{condition}" + (f"-pad{pad}" if condition == "widened" else ""))
    qs = {f"{s}::{q['qid']}": q for s in SLUGS for q in pg._questions(s)}
    tasks = [oracle[k] for k in sorted(oracle)]              # deterministic order

    def work(oc):
        q = qs[f"{oc['slug']}::{oc['qid']}"]
        text = span_text(condition, pad, oc, movies[oc["slug"]])
        pred, _, _ = _answer._answer_call(q["question"], text)
        src = "index" if pred.lower() not in _answer._MISS else "miss"
        v = _eval.judge_one({"qid": q["qid"], "question": q["question"], "gold_answers": q["gold_answers"],
                             "prediction": pred, "source": src})
        return {"slug": oc["slug"], "qid": oc["qid"], "condition": condition, "pad": pad,
                "prediction": pred, "verdict": v["verdict"], "maxcos": oc["maxcos"],
                "span_chars": len(text)}

    recs = EH.run_checkpointed(tag, tasks, lambda t: f"{t['slug']}::{t['qid']}", work,
                               fresh=fresh, workers=workers)
    recs = [r for r in recs if r]
    n = len(recs)
    cov = [r for r in recs if r["maxcos"] >= COVER_BAR]
    acc = lambda rs: round(sum(1 for r in rs if r["verdict"] == "correct")/len(rs), 4) if rs else 0
    return {"condition": condition, "pad": pad, "tag": tag, "n": n, "n_covered": len(cov),
            "accuracy_all": acc(recs), "accuracy_covered": acc(cov),
            "mean_span_chars": round(statistics.mean(r["span_chars"] for r in recs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="current,fulldoc",
                    help="comma list: current, fulldoc, or widened:PAD (e.g. widened:2000)")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    movies = {s: load_movie(s) for s in SLUGS}
    oracle = select_oracle(movies, fresh=a.fresh)
    print(f"oracle set: {len(oracle)} questions, "
          f"{sum(1 for v in oracle.values() if v['maxcos'] >= COVER_BAR)} covered (maxcos >= {COVER_BAR})\n", flush=True)

    results = []
    for spec in a.conditions.split(","):
        spec = spec.strip()
        if spec.startswith("widened:"):
            cond, pad = "widened", int(spec.split(":")[1])
        else:
            cond, pad = spec, 0
        print(f">>> running condition {spec}", flush=True)
        r = run_condition(cond, pad, oracle, movies, fresh=a.fresh)
        results.append(r)
        print(f"    {spec}: acc_all={r['accuracy_all']} acc_covered={r['accuracy_covered']} "
              f"(mean {r['mean_span_chars']} chars, n={r['n']})", flush=True)

    out = HERE/"results"/"span_diag.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    for r in results:
        prev[r["tag"]] = r
    out.write_text(json.dumps(prev, indent=2))
    print("\n=== SPAN DIAGNOSTIC (oracle set) ===")
    print(f"{'condition':16s} {'mean_chars':>10s} {'acc_all':>8s} {'acc_covered':>12s}")
    for r in sorted(prev.values(), key=lambda r: r["mean_span_chars"]):
        name = r["condition"] + (f":{r['pad']}" if r["condition"] == "widened" else "")
        print(f"{name:16s} {r['mean_span_chars']:>10} {r['accuracy_all']:>8} {r['accuracy_covered']:>12}")


if __name__ == "__main__":
    main()
