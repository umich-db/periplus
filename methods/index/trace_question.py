"""Trace how ONE question is answered, step by step. Analysis tool, read-only: it reuses the real
answer.py helpers (Router, traverse, the digest builder, the fallback path) so the trace matches a
real run, and adds logging around them. Shows, in order: the routing expansions (each scored child
with its cosine score, the chosen one marked), the leaves reached, the exact digest (overview plus
source spans) sent to the answer LLM, the primary prediction, the sibling-span fallback if it fired,
and the judge verdict. Also reports whether the gold answer is captured in some leaf at all.

Usage:
  trace_question.py --movie 10_things_i_hate_about_you --qid 10_things_i_hate_about_you_015 --mode balanced
  trace_question.py --movie 12_and_holding --match "tree house fire" --mode per_category

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_BENCH/"shared"))
import numpy as np  # noqa: E402
import client, config as _cfg, store as _store, pack_global as pg, pack_balanced as pb  # noqa: E402
import answer as _answer, eval as _eval  # noqa: E402


def build(mode, slug, cfg):
    if mode == "per_category":
        return _store.load_index(slug)
    if mode == "global_sort":
        return pg.build_index(slug, cfg)[0]
    if mode == "balanced":
        return pb.build_index(slug, cfg)[0]
    raise ValueError(mode)


def trace(slug, q, mode, cfg):
    idx = build(mode, slug, cfg)
    nodes = idx["nodes"]
    question, gold = q["question"], q["gold_answers"]

    # is the answer captured in any leaf at all (coverage), independent of routing
    leaves = [n for n in nodes.values() if n["kind"] == "leaf"]
    lv = client.embed([lf["content"] for lf in leaves], model_name=cfg.embedding_model)
    gv = client.embed([question + " " + " | ".join(gold)], model_name=cfg.embedding_model)[0]
    cos = lv @ gv
    bi = int(np.argmax(cos))

    # instrument routing: log each expansion's scored children
    router = _answer.Router(question, cfg, {})
    expansions = []
    orig = router.score_nodes

    def traced(kids):
        scores = orig(kids)
        expansions.append([(k["id"], _answer._routing_text(k), float(s)) for k, s in zip(kids, scores)])
        return scores

    router.score_nodes = traced
    tr = _answer.traverse(idx, router, cfg)
    reached = tr["reached_leaves"]

    # reproduce the answer exactly as answer_question does (primary, then sibling-span fallback)
    char_budget = cfg.C * 4
    overview = idx.get("overview")
    primary = _answer._digest(overview, _answer._leaf_chunks(idx, reached), char_budget)
    pred, _, _ = _answer._answer_call(question, primary)
    source, fallback, fb_digest, fb_leaves = "index", None, None, []
    if pred.lower() in _answer._MISS:
        fallback = "span"
        fb_leaves = _answer._sibling_leaves(idx, reached)
        fb_digest = _answer._digest(overview, _answer._leaf_chunks(idx, fb_leaves), char_budget)
        pred, _, _ = _answer._answer_call(question, fb_digest)
        source = "fallback" if pred.lower() not in _answer._MISS else "miss"
    if pred.lower() in _answer._MISS:
        pred, source = "not found", "miss"

    verdict = _eval.judge_one({"qid": q["qid"], "question": question, "gold_answers": gold,
                               "prediction": pred, "source": source})

    _print(slug, q, mode, cfg, nodes, leaves, cos, bi, expansions, tr, reached,
           primary, pred, source, fallback, fb_leaves, fb_digest, verdict)


def _short(t, n=90):
    return " ".join(str(t).split())[:n]


def _print(slug, q, mode, cfg, nodes, leaves, cos, bi, expansions, tr, reached,
           primary, pred, source, fallback, fb_leaves, fb_digest, verdict):
    W = "=" * 96
    print(W)
    print(f"TRACE  movie={slug}  qid={q['qid']}  mode={mode}  f={cfg.f}  routing={cfg.routing_mode}")
    print(W)
    print(f"Q: {q['question']}")
    print(f"gold: {' | '.join(q['gold_answers'])}")

    print(f"\nCOVERAGE (is the answer captured as any leaf, regardless of routing):")
    print(f"  best leaf cosine={cos[bi]:.3f}  ->  {_short(leaves[bi]['content'], 140)}")
    top = np.argsort(-cos)[:5]
    print("  top-5 leaves by cosine to (question+gold):")
    for j in top:
        print(f"    {cos[j]:.3f}  {_short(leaves[int(j)]['content'], 120)}")

    print(f"\nROUTING ({len(expansions)} expansions, nodes_scored={tr['nodes_scored']}, "
          f"depth_reached={tr['depth_reached']}, backtracks={tr['backtracks']}, hops={tr['hops']}, "
          f"budget {tr['budget_spent']}/{tr['budget_total']}):")
    reached_ids = {r["id"] for r in reached}
    for i, exp in enumerate(expansions):
        ranked = sorted(exp, key=lambda e: -e[2])
        print(f"  expansion {i+1}: scored {len(exp)} children (top {min(6,len(ranked))} by score):")
        for nid, txt, sc in ranked[:6]:
            kind = nodes[nid]["kind"]
            mark = " <== REACHED LEAF" if (kind == "leaf" and nid in reached_ids) else ""
            print(f"    {sc:.3f} [{kind}] {_short(txt, 100)}{mark}")

    print(f"\nLEAVES REACHED ({len(reached)}), whose source spans feed the answer:")
    for r in reached:
        print(f"  {_short(r['content'], 140)}")

    print(f"\nPRIMARY DIGEST sent to the answer LLM ({len(primary)} chars; overview + source spans):")
    print("  " + primary.replace("\n", "\n  ")[:1200] + (" ...[truncated]" if len(primary) > 1200 else ""))

    if fallback:
        print(f"\nPRIMARY MISSED -> sibling-span fallback over {len(fb_leaves)} leaves. Fallback digest "
              f"({len(fb_digest)} chars):")
        print("  " + fb_digest.replace("\n", "\n  ")[:800] + (" ...[truncated]" if len(fb_digest) > 800 else ""))

    print(f"\nRESULT: source={source} fallback={fallback}")
    print(f"  prediction: {pred}")
    print(f"  judge verdict: {verdict['verdict']}")
    print(W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--match", default=None, help="substring of the question text")
    ap.add_argument("--mode", default="balanced", choices=["per_category", "global_sort", "balanced"])
    a = ap.parse_args()
    cfg = _cfg.load()
    qs = pg._questions(a.movie)
    if a.qid:
        q = next((x for x in qs if x["qid"] == a.qid), None)
    elif a.match:
        q = next((x for x in qs if a.match.lower() in x["question"].lower()), None)
    else:
        q = qs[0]
    if q is None:
        print("no matching question"); sys.exit(1)
    trace(a.movie, q, a.mode, cfg)
