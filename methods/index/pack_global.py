"""Path 1 baseline (build_mode = global_sort): a real B+ tree over subjects.

This is a BASELINE to measure, not the shipped method. The per-category pack (pack.py) is
left fully intact; this runs only behind build_mode = global_sort. It answers Lin's question:
does a uniform B+ tree built by embedding order keep nodes coherent, given that subjects have
no single natural order and text similarity can mislead (two subjects about different people
can read alike)?

The build ignores category boundaries as structure and uses every subject in the source
document: embed each subject, lay them on one global order by a greedy nearest-neighbor chain
(adjacent subjects most similar), group every f adjacent into a leaf-level node, write one
routing summary per node, then recurse on the summaries the same way until the top has at most
f. Every leaf ends at the same depth and every node holds f (the last takes the remainder):
the uniform shape Lin described.

Each subject still carries its Phase 4 category, so we measure node coherence: category purity
per leaf-level node (fraction in the node's majority category) and distinct categories per
node. High purity means the global order kept categories together; low purity is the evidence
that category structure should drive the build, not embedding order.

Nothing here is stored as a query index; the embeddings are transient. gpt-5 client and its
reasoning settings, the all-mpnet embedder, the judge, and the loaders are reused unchanged.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib, argparse, statistics
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import store as _store, pack as _pack  # noqa: E402  (reuse node + routing-summary helpers)

OVERVIEW_PROMPT = (
    "Below are routing summaries for the top-level nodes of an index over one source document. "
    "Using ONLY these, write a concise overview of the whole source document in 4 to 8 "
    "sentences: what kind of document it is, and the main people, places, and events across it. "
    "Base it on the full set together.\n\n{blocks}\n\nOverview:"
)


# ---------- global order: greedy nearest-neighbor chain (deterministic) ----------
def _nn_order(vecs):
    """One global order: start at index 0 (the lowest-id item), then repeatedly append the
    nearest not-yet-used item by cosine similarity. Deterministic; ties break to the lowest
    index. Rows of `vecs` are L2-normalized, so the dot product is cosine."""
    import numpy as np
    n = len(vecs)
    sim = vecs @ vecs.T
    used = [False] * n
    order = [0]; used[0] = True; cur = 0
    for _ in range(n - 1):
        row = sim[cur]
        best, best_s = -1, -2.0
        for j in range(n):
            if not used[j] and row[j] > best_s:      # strict >: first (lowest) index wins ties
                best_s, best = float(row[j]), j
        order.append(best); used[best] = True; cur = best
    return order


def _ctx(cfg, cache):
    return {"nodes": {}, "f": cfg.f, "s": cfg.s, "embed_model": cfg.embedding_model,
            "summ_batch": max(1, cfg.B // cfg.C), "workers": cfg.propose_workers,
            "cache": cache, "stats": {"calls": 0, "cached": 0, "in": 0, "out": 0}}


def _group_layer(items, ctx, summarize=True):
    """Order `items` by the nearest-neighbor chain, group every f adjacent, and return the new
    layer of internal nodes (each with cat=None, since a global-sort node spans whatever
    categories the embedding order placed adjacent). With summarize=True a routing summary is
    written per group (LLM); with False the summary is left None (zero LLM, for the shape and
    purity instrumentation only)."""
    vecs = client.embed([_pack._routing_text(it) for it in items], model_name=ctx["embed_model"])
    ordered = [items[i] for i in _nn_order(vecs)]
    groups = [ordered[i:i + ctx["f"]] for i in range(0, len(ordered), ctx["f"])]
    summaries = _pack.write_summaries(groups, None, ctx) if summarize else [None] * len(groups)
    boxes = []
    for g, summ in zip(groups, summaries):
        b = _pack._make_internal(g, None, summ)
        ctx["nodes"][b["id"]] = b
        for c in g:
            c["parent"] = b["id"]
        boxes.append(b)
    return boxes


def pack_global(subjects, cfg, cache_path=None, summarize=True):
    """Bottom-up uniform B+ tree. Returns (nodes, root_id, first_boxes, stats). first_boxes is
    the leaf-level grouping layer (used for the category-purity instrumentation). summarize=False
    skips routing summaries entirely (zero LLM) for a shape/purity-only build."""
    cache = {}
    if cache_path and pathlib.Path(cache_path).exists():
        cache = json.loads(pathlib.Path(cache_path).read_text())
    ctx = _ctx(cfg, cache)

    # leaves = subjects, in a deterministic id order so the chain start is well defined
    leaves = sorted((_pack._make_leaf(s) for s in subjects), key=lambda n: n["id"])
    for lf in leaves:
        ctx["nodes"][lf["id"]] = lf

    items, first_boxes = leaves, None
    while len(items) > cfg.f:
        items = _group_layer(items, ctx, summarize=summarize)
        if first_boxes is None:
            first_boxes = items
    if first_boxes is None:                                   # tiny document: leaves are the top
        first_boxes = leaves

    if len(items) == 1:
        root = items[0]; root["kind"] = "root"
    else:
        root = _pack._make_internal(items, None, None); root["kind"] = "root"
        ctx["nodes"][root["id"]] = root
        for c in items:
            c["parent"] = root["id"]

    if cache_path:
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(cache_path).write_text(json.dumps(ctx["cache"], ensure_ascii=False))
    return ctx["nodes"], root["id"], first_boxes, ctx["stats"]


def _overview(nodes, root_id, cfg):
    top = [nodes[cid] for cid in nodes[root_id]["children"]]
    blocks = "\n".join(f"- {(n.get('summary') or n.get('content') or '').strip()[:200]}" for n in top)
    text, pt, ct = client.chat(OVERVIEW_PROMPT.format(blocks=blocks), max_tokens=16384, bucket="index_build")
    return (text or "").strip(), pt + ct


def build_index(slug, cfg):
    """Build the global_sort index for `slug` as an in-memory idx (same shape store.load_index
    returns), reusing the exact Phase 4 subjects the per-category build uses. No files needed:
    the answer path takes an idx dict directly."""
    chunks = _store._get_chunks(slug, cfg)
    subjects = _store._get_subjects(slug, cfg, chunks)
    nodes, root_id, first_boxes, stats = pack_global(
        subjects, cfg, cache_path=HERE/"output"/slug/"routing_cache_global.json")
    overview, ov_tok = _overview(nodes, root_id, cfg)
    nodes[root_id]["overview"] = overview
    nodes[root_id]["summary"] = overview
    chunks_d = {c["id"]: {"chunk_id": c["id"], "start_char": c["start_char"],
                          "end_char": c["end_char"], "text": c["text"]} for c in chunks}
    idx = {"nodes": nodes, "root": root_id, "chunks": chunks_d, "overview": overview,
           "meta": {"slug": slug, "root_id": root_id, "n_nodes": len(nodes), "build_mode": "global_sort"}}
    return idx, first_boxes, stats, ov_tok


# ---------- tree-shape + coherence instrumentation ----------
def _leaf_depths(nodes, root_id):
    """Root-to-leaf distance for every leaf (the real depth-uniformity test, unlike the leaf
    `level` field which is 0 for every leaf regardless of how far it sits from the root)."""
    from collections import deque
    depths, dq = [], deque([(root_id, 0)])
    while dq:
        nid, d = dq.popleft()
        n = nodes[nid]
        if n["kind"] == "leaf":
            depths.append(d)
        else:
            for c in n["children"]:
                dq.append((c, d + 1))
    return depths


def tree_stats(nodes, root_id, f):
    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    levels = Counter(n["level"] for n in nodes.values())
    fanout = Counter(len(n["children"]) for n in internal)
    ld = Counter(_leaf_depths(nodes, root_id))
    return {"depth": nodes[root_id]["level"], "n_nodes": len(nodes),
            "root_children": len(nodes[root_id]["children"]),
            "nodes_per_level (0=leaf)": {str(k): levels[k] for k in sorted(levels)},
            "fanout_distribution (children -> count)": {str(k): fanout[k] for k in sorted(fanout)},
            "root_to_leaf_depth (depth -> #leaves)": {str(k): ld[k] for k in sorted(ld)},
            "uniform_depth": len(ld) == 1, "distinct_fanout_values": len(fanout)}


def purity(nodes, first_boxes):
    """Category purity per leaf-level node: fraction of a node's subject-leaves in its majority
    category. Reports mean and distribution, distinct categories per node, and the worst nodes."""
    per_node = []
    for b in first_boxes:
        leaves = [nodes[cid] for cid in b["children"] if nodes[cid]["kind"] == "leaf"]
        cats = [lf["category"] for lf in leaves]
        if not cats:
            continue
        maj, maj_n = Counter(cats).most_common(1)[0]
        per_node.append({"node": b["id"], "size": len(cats), "majority": maj,
                         "purity": maj_n/len(cats), "distinct": len(set(cats)),
                         "members": [(lf["category"], lf["content"].split(" :: ")[0]) for lf in leaves]})
    pur = [p["purity"] for p in per_node]
    dis = [p["distinct"] for p in per_node]
    worst = sorted(per_node, key=lambda p: (p["purity"], -p["distinct"]))[:3]
    return {"n_leaf_nodes": len(per_node),
            "mean_purity": round(statistics.mean(pur), 4) if pur else None,
            "purity_min": round(min(pur), 4) if pur else None,
            "purity_median": round(statistics.median(pur), 4) if pur else None,
            "purity_max": round(max(pur), 4) if pur else None,
            "mean_distinct_categories": round(statistics.mean(dis), 3) if dis else None,
            "max_distinct_categories": max(dis) if dis else None,
            "worst": worst}


# ---------- compare: run the Phase 7 answer path on both build modes ----------
def _questions(slug):
    BENCH = _BENCH
    return [q for q in (json.loads(l) for l in
            (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip())
            if q["movie"] == slug]


def _run_and_judge(idx, slug, qs, cfg):
    import answer as _answer
    sys.path.insert(0, str(_BENCH/"shared"))
    import eval as _eval
    emb, nscored, correct, fb = {}, [], 0, 0
    for q in qs:
        r = _answer.answer_question(idx, slug, q["question"], cfg, emb)
        nscored.append(r["nodes_scored"]); fb += 1 if r["fallback"] else 0
        v = _eval.judge_one({"qid": q["qid"], "question": q["question"],
                             "gold_answers": q["gold_answers"], "prediction": r["prediction"],
                             "source": r["source"]})
        correct += 1 if v["verdict"] == "correct" else 0
    n = len(qs)
    return {"n": n, "judge_accuracy": round(correct/n, 4) if n else 0,
            "nodes_scored_median": statistics.median(nscored) if nscored else 0,
            "fallback_rate": round(fb/n, 4) if n else 0}


def compare(slug, cfg):
    # global_sort (in-memory) vs per_category (the shipped index, loaded from disk). Both share
    # the same Phase 4 subjects for slug, so the only difference is the tree shape.
    g_idx, first_boxes, gstats, _ = build_index(slug, cfg)
    p_idx = _store.load_index(slug)
    qs = _questions(slug)

    g_tree = tree_stats(g_idx["nodes"], g_idx["root"], cfg.f)
    p_tree = tree_stats(p_idx["nodes"], p_idx["root"], cfg.f)
    pu = purity(g_idx["nodes"], first_boxes)
    g_ans = _run_and_judge(g_idx, slug, qs, cfg)
    p_ans = _run_and_judge(p_idx, slug, qs, cfg)
    return {"slug": slug, "f": cfg.f, "global_tree": g_tree, "per_category_tree": p_tree,
            "purity": pu, "global_answer": g_ans, "per_category_answer": p_ans}


def _ensure_subjects(slug, cfg):
    """Return the Phase 4 subjects for slug, persisting subjects.jsonl if it is missing so the
    per_category and global_sort builds share an identical subject set."""
    out = HERE/"output"/slug
    sp = out/"subjects.jsonl"
    if sp.exists():
        return [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
    import chunk as _chunk, categories as _cat, extract as _extract
    chunks = _chunk.chunk_document(_chunk.load_source(slug), cfg.chunk_target, cfg.B)
    union, _, _ = _cat.propose(chunks, workers=cfg.propose_workers, cache_path=out/"propose_cache.json")
    categories, mapping, _ = _cat.merge(union, cfg.embedding_model, cfg.merge_similarity)
    subjects, _, _ = _extract.extract(chunks, categories, cfg)
    _extract.write_dump(slug, categories, mapping, subjects, cfg.f)
    return subjects


def sweep(cfg):
    """Shape + category purity for global_sort across all 10 screenplays (zero LLM for the
    grouping; only ensures subjects exist). The build-side coherence evidence for Lin."""
    slugs = _store.SLUGS if hasattr(_store, "SLUGS") else [
        "10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
        "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
        "a_dry_white_season", "a_few_good_men"]
    rows = []
    for slug in slugs:
        subs = _ensure_subjects(slug, cfg)
        nodes, root_id, first_boxes, _ = pack_global(subs, cfg, summarize=False)
        ts = tree_stats(nodes, root_id, cfg.f)
        pu = purity(nodes, first_boxes)
        rows.append({"slug": slug, "n_subjects": len(subs), "depth": ts["depth"],
                     "uniform_depth": ts["uniform_depth"], "leaf_nodes": pu["n_leaf_nodes"],
                     "mean_purity": pu["mean_purity"], "mean_distinct_cats": pu["mean_distinct_categories"],
                     "max_distinct_cats": pu["max_distinct_categories"]})
        print(f"  {slug:32s} subj={len(subs):4d} depth={ts['depth']} uniform={ts['uniform_depth']} "
              f"mean_purity={pu['mean_purity']} distinct_cats/node={pu['mean_distinct_categories']}", flush=True)
    mp = [r["mean_purity"] for r in rows if r["mean_purity"] is not None]
    dc = [r["mean_distinct_cats"] for r in rows if r["mean_distinct_cats"] is not None]
    print(f"\nCORPUS (10 screenplays): mean category purity = {statistics.mean(mp):.3f} "
          f"| mean distinct categories per leaf node = {statistics.mean(dc):.1f}")
    (HERE/"results"/"global_sort_purity_sweep.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="10_things_i_hate_about_you")
    ap.add_argument("--mode", choices=["stats", "compare", "sweep"], default="stats")
    a = ap.parse_args()
    cfg = _cfg.load()

    if a.mode == "sweep":
        sweep(cfg); sys.exit(0)

    if a.mode == "stats":
        idx, first_boxes, stats, ov_tok = build_index(a.movie, cfg)
        ts = tree_stats(idx["nodes"], idx["root"], cfg.f)
        pu = purity(idx["nodes"], first_boxes)
        print(json.dumps({"movie": a.movie, "f": cfg.f, "tree": ts,
                          "purity": {k: v for k, v in pu.items() if k != "worst"},
                          "routing_summary_calls": stats["calls"]}, indent=2, ensure_ascii=False))
        print("\nworst 3 leaf-level nodes (lowest purity):")
        for w in pu["worst"]:
            print(f"  purity={w['purity']:.2f} distinct={w['distinct']} majority={w['majority']!r} size={w['size']}")
            for cat, name in w["members"][:12]:
                print(f"      [{cat}] {name}")
        sys.exit(0)

    # compare mode
    r = compare(a.movie, cfg)
    print("=" * 70)
    print(f"BUILD-MODE COMPARISON: {a.movie}  (f={cfg.f})")
    print("=" * 70)
    print("\nglobal_sort tree:", json.dumps(r["global_tree"], ensure_ascii=False))
    print("per_category tree:", json.dumps(r["per_category_tree"], ensure_ascii=False))
    pu = r["purity"]
    print(f"\ncategory purity (global_sort leaf-level nodes): mean={pu['mean_purity']} "
          f"min={pu['purity_min']} median={pu['purity_median']} max={pu['purity_max']} | "
          f"distinct cats/node mean={pu['mean_distinct_categories']} max={pu['max_distinct_categories']}")
    print("worst 3 nodes:")
    for w in pu["worst"]:
        print(f"  purity={w['purity']:.2f} distinct={w['distinct']} majority={w['majority']!r}: "
              + ", ".join(f"[{c}] {n}" for c, n in w["members"][:8]))
    print(f"\n{'metric':22s} {'global_sort':>14s} {'per_category':>14s}")
    for k, lab in [("judge_accuracy", "judge_accuracy"), ("nodes_scored_median", "nodes_scored_med"),
                   ("fallback_rate", "fallback_rate")]:
        print(f"  {lab:20s} {r['global_answer'][k]:>14} {r['per_category_answer'][k]:>14}")
