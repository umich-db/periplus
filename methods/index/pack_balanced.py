"""Path 2 (build_mode = balanced): a category-ordered B+ bulk load with balanced nodes.

This keeps per_category (pack.py, the shipped default) and global_sort (pack_global.py, the
baseline) fully intact and runs only behind build_mode = balanced. It is the fix for what
global_sort proved: a real B+ tree needs a sort key whose order has locality (adjacent keys
related), and raw subject text has none (global_sort landed at 0.189 purity, ~17 categories per
leaf, because two subjects in different categories can read alike).

The fix is to sort at the level where a meaningful order exists: the category. Category names are
clean semantic labels, so they embed and order well. So we order the categories by similarity,
keep each category's subjects together, and bulk load fixed-size nodes over that order exactly
like a B+ tree. The sort key now has locality (adjacent subjects share a category or a related
one), so we get the B+ shape (uniform depth, fixed fanout, no thin levels) AND coherent nodes.

Only the ordering differs from global_sort: it is hierarchical (category first via the same
nearest-neighbor chain, then subject within the category) instead of a flat embedding chain. The
bulk load, occupancy rule, routing summaries, gpt-5 client and reasoning settings, all-mpnet
embedder, judge, and loaders are reused unchanged. Nothing is stored as a query index.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib, argparse, statistics, math
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import store as _store, pack as _pack  # noqa: E402  (reuse node + routing-summary helpers)
import pack_global as pg  # noqa: E402  (reuse the nearest-neighbor chain and all instrumentation)


# ---------- hierarchical order: categories, then subjects within each ----------
def _ordered_leaves(subjects, cfg):
    """One global subject order with locality from the category, not raw text. Group subjects by
    category; order the categories by a nearest-neighbor chain over their NAME embeddings (clean
    labels, unlike noisy subject text); within each category order its subject-leaves by the same
    chain over subject embeddings; concatenate category by category. Deterministic: category names
    sorted (lowest-id start / tie-break) before the chain, subject-leaves sorted by leaf id."""
    by_cat = defaultdict(list)
    for s in subjects:
        by_cat[s["category"]].append(s)

    cat_names = sorted(by_cat)                                   # deterministic base order
    if len(cat_names) > 1:
        cat_vecs = client.embed(cat_names, model_name=cfg.embedding_model)
        cat_names = [cat_names[i] for i in pg._nn_order(cat_vecs)]

    ordered = []
    for cat in cat_names:
        leaves = sorted((_pack._make_leaf(s) for s in by_cat[cat]), key=lambda n: n["id"])
        if len(leaves) > 1:
            vecs = client.embed([_pack._routing_text(lf) for lf in leaves], model_name=cfg.embedding_model)
            leaves = [leaves[i] for i in pg._nn_order(vecs)]
        ordered.extend(leaves)
    return ordered


# ---------- B+ bulk load with the minimum-occupancy rule ----------
def _balanced_groups(items, f):
    """Pack `items` (already in bulk-load order) into fixed-size groups of f, B+ style. Enforce the
    standard B+ minimum occupancy: every group holds between ceil(f/2) and f. If the trailing group
    would fall below ceil(f/2), redistribute it with its neighbor (merge the last two, split evenly)
    so both land inside [ceil(f/2), f]; a plain merge could push the neighbor above f, which the
    occupancy window forbids. With a single group (tiny input) the rule cannot apply and is skipped."""
    lo = math.ceil(f / 2)
    groups = [items[i:i + f] for i in range(0, len(items), f)]
    if len(groups) >= 2 and len(groups[-1]) < lo:
        merged = groups[-2] + groups[-1]
        half = len(merged) // 2
        groups[-2], groups[-1] = merged[:half], merged[half:]
    return groups


def _pack_layer(items, ctx, summarize=True):
    """Bulk-load one layer: group `items` into balanced nodes IN THE EXISTING ORDER (no re-sort, so
    the category locality carried by the order is preserved up the tree), write one routing summary
    per node, and return the new layer."""
    groups = _balanced_groups(items, ctx["f"])
    summaries = _pack.write_summaries(groups, None, ctx) if summarize else [None] * len(groups)
    boxes = []
    for g, summ in zip(groups, summaries):
        b = _pack._make_internal(g, None, summ)
        ctx["nodes"][b["id"]] = b
        for c in g:
            c["parent"] = b["id"]
        boxes.append(b)
    return boxes


def pack_balanced(subjects, cfg, cache_path=None, summarize=True):
    """Bottom-up balanced B+ tree over the category-then-subject order. Returns
    (nodes, root_id, first_boxes, stats); first_boxes is the leaf-level grouping layer for the
    purity + occupancy instrumentation. summarize=False skips routing summaries (zero LLM)."""
    cache = {}
    if cache_path and pathlib.Path(cache_path).exists():
        cache = json.loads(pathlib.Path(cache_path).read_text())
    ctx = pg._ctx(cfg, cache)

    leaves = _ordered_leaves(subjects, cfg)
    for lf in leaves:
        ctx["nodes"][lf["id"]] = lf

    items, first_boxes = leaves, None
    while len(items) > cfg.f:
        items = _pack_layer(items, ctx, summarize=summarize)
        if first_boxes is None:
            first_boxes = items
    if first_boxes is None:                                       # tiny document: leaves are the top
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


def build_index(slug, cfg):
    """Build the balanced index for `slug` as an in-memory idx (same shape store.load_index
    returns), reusing the exact Phase 4 subjects the other builds use."""
    chunks = _store._get_chunks(slug, cfg)
    subjects = _store._get_subjects(slug, cfg, chunks)
    nodes, root_id, first_boxes, stats = pack_balanced(
        subjects, cfg, cache_path=HERE/"output"/slug/"routing_cache_balanced.json")
    overview, ov_tok = pg._overview(nodes, root_id, cfg)
    nodes[root_id]["overview"] = overview
    nodes[root_id]["summary"] = overview
    chunks_d = {c["id"]: {"chunk_id": c["id"], "start_char": c["start_char"],
                          "end_char": c["end_char"], "text": c["text"]} for c in chunks}
    idx = {"nodes": nodes, "root": root_id, "chunks": chunks_d, "overview": overview,
           "meta": {"slug": slug, "root_id": root_id, "n_nodes": len(nodes), "build_mode": "balanced"}}
    return idx, first_boxes, stats, ov_tok


# ---------- occupancy instrumentation (the B+ half-full-to-full check) ----------
def occupancy(nodes, root_id, f):
    """Every non-root internal node must hold between ceil(f/2) and f children (the B+ minimum
    occupancy rule). The root is exempt (a B+ root may hold as few as 2)."""
    lo = math.ceil(f / 2)
    internal = [n for n in nodes.values() if n["kind"] == "internal"]
    sizes = [len(n["children"]) for n in internal]
    bad = [(n["id"], len(n["children"])) for n in internal if not (lo <= len(n["children"]) <= f)]
    return {"window": [lo, f], "nonroot_internal_nodes": len(sizes),
            "min_children": min(sizes) if sizes else None,
            "max_children": max(sizes) if sizes else None,
            "all_within_window": len(bad) == 0,
            "root_children": len(nodes[root_id]["children"]),
            "violations": bad[:5]}


# ---------- three-way compare: per_category vs global_sort vs balanced ----------
def _build_all(slug, cfg):
    per = _store.load_index(slug)
    g_idx, g_boxes, _, _ = pg.build_index(slug, cfg)
    b_idx, b_boxes, _, _ = build_index(slug, cfg)
    return per, (g_idx, g_boxes), (b_idx, b_boxes)


def compare(slug, cfg):
    per, (g_idx, g_boxes), (b_idx, b_boxes) = _build_all(slug, cfg)
    qs = pg._questions(slug)
    trees = {"per_category": pg.tree_stats(per["nodes"], per["root"], cfg.f),
             "global_sort": pg.tree_stats(g_idx["nodes"], g_idx["root"], cfg.f),
             "balanced": pg.tree_stats(b_idx["nodes"], b_idx["root"], cfg.f)}
    pur = {"global_sort": pg.purity(g_idx["nodes"], g_boxes),
           "balanced": pg.purity(b_idx["nodes"], b_boxes)}
    occ = occupancy(b_idx["nodes"], b_idx["root"], cfg.f)
    ans = {"per_category": pg._run_and_judge(per, slug, qs, cfg),
           "global_sort": pg._run_and_judge(g_idx, slug, qs, cfg),
           "balanced": pg._run_and_judge(b_idx, slug, qs, cfg)}
    return {"slug": slug, "f": cfg.f, "trees": trees, "purity": pur, "occupancy": occ, "answer": ans}


def _run_raw(idx, slug, qs, cfg):
    """Raw per-question rows (correct, nodes_scored, fallback) so the full-10 driver can pool the
    real 295-question distribution instead of averaging per-movie summaries."""
    import answer as _answer
    sys.path.insert(0, str(_BENCH/"shared"))
    import eval as _eval
    emb, rows = {}, []
    for q in qs:
        r = _answer.answer_question(idx, slug, q["question"], cfg, emb)
        v = _eval.judge_one({"qid": q["qid"], "question": q["question"],
                             "gold_answers": q["gold_answers"], "prediction": r["prediction"],
                             "source": r["source"]})
        rows.append((v["verdict"] == "correct", r["nodes_scored"], bool(r["fallback"])))
    return rows


def _summ(rows):
    n = len(rows)
    return {"n": n, "judge_accuracy": round(sum(c for c, _, _ in rows)/n, 4) if n else 0,
            "nodes_scored_median": statistics.median([ns for _, ns, _ in rows]) if n else 0,
            "fallback_rate": round(sum(f for _, _, f in rows)/n, 4) if n else 0}


def full(cfg):
    """All three modes across all 10 screenplays, pooled over the 295 questions."""
    slugs = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
             "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
             "a_dry_white_season", "a_few_good_men"]
    pooled = {"per_category": [], "global_sort": [], "balanced": []}
    g_pur, b_pur, per_movie = [], [], []
    for slug in slugs:
        per, (g_idx, g_boxes), (b_idx, b_boxes) = _build_all(slug, cfg)
        qs = pg._questions(slug)
        gp = pg.purity(g_idx["nodes"], g_boxes); bp = pg.purity(b_idx["nodes"], b_boxes)
        occ = occupancy(b_idx["nodes"], b_idx["root"], cfg.f)
        bt = pg.tree_stats(b_idx["nodes"], b_idx["root"], cfg.f)
        rows = {m: _run_raw(idx, slug, qs, cfg) for m, idx in
                (("per_category", per), ("global_sort", g_idx), ("balanced", b_idx))}
        for m in pooled:
            pooled[m] += rows[m]
        for b in b_boxes:
            leaves = [b_idx["nodes"][c] for c in b["children"] if b_idx["nodes"][c]["kind"] == "leaf"]
            if leaves:
                cats = [lf["category"] for lf in leaves]
                b_pur.append(Counter(cats).most_common(1)[0][1]/len(cats))
        for b in g_boxes:
            leaves = [g_idx["nodes"][c] for c in b["children"] if g_idx["nodes"][c]["kind"] == "leaf"]
            if leaves:
                cats = [lf["category"] for lf in leaves]
                g_pur.append(Counter(cats).most_common(1)[0][1]/len(cats))
        row = {"slug": slug, "n_questions": len(qs),
               "balanced_depth": bt["depth"], "balanced_uniform": bt["uniform_depth"],
               "balanced_occupancy_ok": occ["all_within_window"],
               "purity_global": gp["mean_purity"], "purity_balanced": bp["mean_purity"],
               "answer": {m: _summ(rows[m]) for m in rows}}
        per_movie.append(row)
        print(f"{slug:32s} q={len(qs):2d} occ_ok={occ['all_within_window']} unif={bt['uniform_depth']} | "
              f"purity g={gp['mean_purity']} b={bp['mean_purity']} | acc "
              f"p={row['answer']['per_category']['judge_accuracy']} "
              f"g={row['answer']['global_sort']['judge_accuracy']} "
              f"b={row['answer']['balanced']['judge_accuracy']}", flush=True)

    corpus = {m: _summ(pooled[m]) for m in pooled}
    corpus["mean_purity_global"] = round(statistics.mean(g_pur), 4) if g_pur else None
    corpus["mean_purity_balanced"] = round(statistics.mean(b_pur), 4) if b_pur else None
    out = {"per_movie": per_movie, "corpus": corpus, "n_leaf_nodes": {"global": len(g_pur), "balanced": len(b_pur)}}
    (HERE/"results"/"balanced_compare_all.json").write_text(json.dumps(out, indent=2))
    _print_corpus(corpus)
    return out


def _print_corpus(corpus):
    print("\n" + "=" * 82)
    print("CORPUS (295 questions across 10 screenplays)")
    print("=" * 82)
    print(f"mean category purity (leaf nodes): global_sort={corpus['mean_purity_global']} "
          f"balanced={corpus['mean_purity_balanced']}")
    print(f"{'metric':22s} {'per_category':>14s} {'global_sort':>14s} {'balanced':>14s}")
    for k in ["judge_accuracy", "nodes_scored_median", "fallback_rate"]:
        print(f"  {k:20s} {corpus['per_category'][k]:>14} {corpus['global_sort'][k]:>14} "
              f"{corpus['balanced'][k]:>14}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="10_things_i_hate_about_you")
    ap.add_argument("--mode", choices=["stats", "compare", "full"], default="stats")
    a = ap.parse_args()
    cfg = _cfg.load()

    if a.mode == "full":
        full(cfg); sys.exit(0)

    if a.mode == "stats":
        idx, first_boxes, stats, _ = build_index(a.movie, cfg)
        ts = pg.tree_stats(idx["nodes"], idx["root"], cfg.f)
        pu = pg.purity(idx["nodes"], first_boxes)
        occ = occupancy(idx["nodes"], idx["root"], cfg.f)
        print(json.dumps({"movie": a.movie, "f": cfg.f, "tree": ts,
                          "occupancy": occ,
                          "purity": {k: v for k, v in pu.items() if k != "worst"},
                          "routing_summary_calls": stats["calls"]}, indent=2, ensure_ascii=False))
        print("\nworst 3 leaf-level nodes (lowest purity):")
        for w in pu["worst"]:
            print(f"  purity={w['purity']:.2f} distinct={w['distinct']} majority={w['majority']!r} size={w['size']}")
            for cat, name in w["members"][:12]:
                print(f"      [{cat}] {name}")
        sys.exit(0)

    # compare mode: one movie, all three
    r = compare(a.movie, cfg)
    print("=" * 78)
    print(f"THREE-WAY COMPARISON: {a.movie}  (f={cfg.f})")
    print("=" * 78)
    for m in ("per_category", "global_sort", "balanced"):
        print(f"\n{m} tree:", json.dumps(r["trees"][m], ensure_ascii=False))
    print("\noccupancy (balanced):", json.dumps(r["occupancy"], ensure_ascii=False))
    for m in ("global_sort", "balanced"):
        pu = r["purity"][m]
        print(f"\ncategory purity ({m}): mean={pu['mean_purity']} min={pu['purity_min']} "
              f"median={pu['purity_median']} max={pu['purity_max']} | distinct cats/node "
              f"mean={pu['mean_distinct_categories']} max={pu['max_distinct_categories']}")
    print("\nworst 3 balanced nodes:")
    for w in r["purity"]["balanced"]["worst"]:
        print(f"  purity={w['purity']:.2f} distinct={w['distinct']} majority={w['majority']!r}: "
              + ", ".join(f"[{c}] {n}" for c, n in w["members"][:8]))
    print(f"\n{'metric':22s} {'per_category':>14s} {'global_sort':>14s} {'balanced':>14s}")
    for k in ["judge_accuracy", "nodes_scored_median", "fallback_rate"]:
        print(f"  {k:20s} {r['answer']['per_category'][k]:>14} {r['answer']['global_sort'][k]:>14} "
              f"{r['answer']['balanced'][k]:>14}")
