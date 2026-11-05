"""build_mode = per_category_rebalanced: a post-hoc rebalance of the already-built per_category tree.

Starts from the per_category tree (loaded from disk, not rebuilt from scratch) and fixes only the
off nodes, B+ style, keeping the category-aligned structure that routes well. per_category (default)
and balanced are left intact; this runs only behind build_mode = per_category_rebalanced.

The hard rule from Lin: balancing is an optimization and must not cost accuracy, and it must not
lower the single-category rate (0.96 for per_category). Coherence beats occupancy, the opposite of a
textbook B+ tree, because the routing key here is semantic (a category name), not a sorted number.
So merges are coherence-gated: an underfull node merges with a sibling ONLY if their category names
are closely related (all-mpnet cosine above a strict bar) and the result stays at or below f. A thin
but coherent node is kept over a full but mixed one. Splits reuse pack's clustering. Only the changed
(dirty) nodes are re-summarized; the untouched good parts keep their summaries and clean labels.

Part 1 also implements insert(leaf), the reusable B+ primitive (route, attach, split-and-cascade),
which the readjustment shares and which later folds fallback misses back into the tree.

Vocabulary: root, category, subject, field, source document.
"""
import sys, json, math, pathlib, argparse
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import client, config as _cfg, store as _store, pack as _pack  # noqa: E402
import cluster as _cluster, answer as _answer  # noqa: E402

COHERENCE_BAR = 0.70   # strict: two category names must embed this close to fold together (all-mpnet
                       # cosine). Calibrated so only genuine near-synonyms qualify (Athletics/Sports,
                       # Riots/Urban riots, Township life/setting), not loosely-related themes.


# ---------- build context (routing-summary call + cache), reusing pack's machinery ----------
def _ctx(cfg, cache):
    return {"nodes": None, "f": cfg.f, "seed": cfg.cluster_seed, "s": cfg.s,
            "embed_model": cfg.embedding_model, "summ_batch": max(1, cfg.B // cfg.C),
            "workers": cfg.propose_workers, "cache": cache,
            "stats": {"calls": 0, "cached": 0, "in": 0, "out": 0}}


def _child_nodes(nodes, node):
    return [nodes[c] for c in node["children"]]


def _cat_set(nodes, node):
    """Categories among the leaf descendants of a node."""
    out = Counter()
    stack = [node["id"]]
    while stack:
        n = nodes[stack.pop()]
        if n["kind"] == "leaf":
            out[n["category"]] += 1
        else:
            stack.extend(n["children"])
    return out


def _relevel(nodes, nid):
    """Recompute a node's level from its children (bottom-up consistency after edits)."""
    n = nodes[nid]
    if n["kind"] == "leaf":
        return 0
    n["level"] = 1 + max(_relevel(nodes, c) for c in n["children"])
    return n["level"]


def _new_internal(nodes, children, cat, summary):
    node = _pack._make_internal(children, cat, summary)
    nodes[node["id"]] = node
    for c in children:
        c["parent"] = node["id"]
    return node


# ---------- the split primitive (reused by insert and by the overfull fix) ----------
def split_node(nodes, node, cfg, ctx):
    """Split an overfull node (> f children) into groups on a coherent embedding boundary (pack's
    clustering over the children's routing text), keep the first group in place, and return the NEW
    sibling nodes to add to the parent. Every resulting node is <= f. Marks results dirty."""
    kids = _child_nodes(nodes, node)
    vecs = client.embed([_pack._routing_text(k) for k in kids], model_name=cfg.embedding_model)
    groups = [[kids[i] for i in cl] for cl in _cluster.partition(vecs, cfg.f, cfg.cluster_seed)]
    # keep group 0 in `node`; spin the rest into new sibling nodes
    node["children"] = [c["id"] for c in groups[0]]
    node["dirty"] = True
    for c in groups[0]:
        c["parent"] = node["id"]
    new_sibs = []
    for g in groups[1:]:
        cats = _cat_set(nodes, {"children": [c["id"] for c in g], "kind": "internal"})
        cat = next(iter(cats)) if len(cats) == 1 else None
        b = _new_internal(nodes, g, cat, summary=None)
        b["dirty"] = True
        new_sibs.append(b)
    return new_sibs


def _add_children_to_parent(nodes, parent_id, new_children, cfg, ctx):
    """Attach new_children to a parent and cascade a split upward if it overflows. If the root
    overflows, grow a new root so leaves stay at equal depth."""
    while True:
        parent = nodes[parent_id]
        parent["children"].extend(c["id"] for c in new_children)
        for c in new_children:
            c["parent"] = parent_id
        parent["dirty"] = True
        if len(parent["children"]) <= cfg.f:
            return parent_id if parent["kind"] != "root" else parent_id
        # overflow -> split
        gp_id = parent.get("parent")
        sibs = split_node(nodes, parent, cfg, ctx)
        if gp_id is None:                                    # parent is the root: grow a new root
            parent["kind"] = "internal"
            new_root = _new_internal(nodes, [parent] + sibs, None, summary=None)
            new_root["kind"] = "root"; new_root["dirty"] = True
            return new_root["id"]
        parent_id, new_children = gp_id, sibs


# ---------- Part 1: insert(leaf), B+ style ----------
def insert(nodes, root_id, leaf, cfg, ctx, emb_cache=None):
    """Route from the root to the target parent by the leaf's embedding (the same cosine routing the
    answer path uses), attach the leaf, and split-and-cascade on overflow. Returns the (possibly new)
    root id. Marks the path dirty; re-summarize lazily."""
    nodes[leaf["id"]] = leaf
    router = _answer.Router(_pack._routing_text(leaf), cfg, emb_cache if emb_cache is not None else {})
    cur = nodes[root_id]
    path = [cur["id"]]
    while cur["kind"] != "internal" or any(nodes[c]["kind"] != "leaf" for c in cur["children"]):
        inner = [nodes[c] for c in cur["children"] if nodes[c]["kind"] != "leaf"]
        if not inner:
            break
        scores = router.score_nodes(inner)
        cur = inner[int(np.argmax(scores))]
        path.append(cur["id"])
    new_root = _add_children_to_parent(nodes, cur["id"], [leaf], cfg, ctx)
    for nid in path:
        if nid in nodes:
            nodes[nid]["dirty"] = True
    return new_root if new_root else root_id


# ---------- Part 2: the readjustment pass (coherence-gated merge + split overfull) ----------
def _category_tops(nodes, root_id):
    """Category-top nodes: the highest node carrying each category (parent is a different or no
    category). These are the single-category coherent units per_category routes on."""
    tops = {}
    for n in nodes.values():
        c = n.get("category")
        if not c or n["kind"] == "leaf":
            continue
        p = nodes.get(n.get("parent") or "")
        if p is None or p.get("category") != c:
            tops[n["id"]] = c
    return tops


def _components(n, edges):
    """Union-find connected components over `n` items given edge list of (i, j)."""
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges:
        parent[find(i)] = find(j)
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values())


def readjust(nodes, root_id, cfg, ctx):
    """Coherence-PRESERVING cleanup. The single-category leaf-level nodes and their clean category-name
    labels are never touched (merging them into a multi-category summary node is the very thing that
    made balanced route worse). Instead:
      - fold: group a parent's small (underfull) single-category children that are closely related
        (category-name cosine >= COHERENCE_BAR) under a new shared parent, keeping each child a clean
        single-category node. This organizes the many thin top-level categories without mixing them.
      - merge: combine underfull SAME-category siblings (safe, no coherence loss).
      - split: divide anything left over f on a coherent boundary.
    Returns counts of what changed."""
    lo = math.ceil(cfg.f / 2)
    tops = _category_tops(nodes, root_id)
    by_parent = defaultdict(list)
    for nid, cat in tops.items():
        by_parent[nodes[nid].get("parent")].append((nid, cat))

    folded_groups = 0; same_cat_merges = 0
    for parent_id, members in by_parent.items():
        if parent_id is None or len(members) < 2:
            continue
        small = [(nid, cat) for nid, cat in members if len(nodes[nid]["children"]) < lo]
        if len(small) < 2:
            continue
        names = [cat for _, cat in small]
        vecs = client.embed(names, model_name=cfg.embedding_model)
        sim = vecs @ vecs.T
        n = len(small)
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if float(sim[a, b]) >= COHERENCE_BAR]
        for comp in _components(n, edges):
            if len(comp) < 2:
                continue
            comp = comp[:cfg.f]                              # keep the shared parent within occupancy
            child_nodes = [nodes[small[i][0]] for i in comp]
            same_cat = len({small[i][1] for i in comp}) == 1
            if same_cat:                                     # same category split-boxes: merge their leaves
                kids = [ck for cn in child_nodes for ck in cn["children"]]
                new = _new_internal(nodes, [nodes[k] for k in kids], small[comp[0]][1], summary=None)
                same_cat_merges += 1
            else:                                            # related categories: shared PARENT, children stay clean
                new = _new_internal(nodes, child_nodes, None, summary=None)
                folded_groups += 1
            new["dirty"] = True
            gids = {small[i][0] for i in comp}
            par = nodes[parent_id]
            par["children"] = [c for c in par["children"] if c not in gids] + [new["id"]]
            new["parent"] = parent_id
            if same_cat:
                for i in comp:
                    nodes.pop(small[i][0], None)

    split = 0
    for nid in list(nodes.keys()):
        node = nodes.get(nid)
        if node and node["kind"] in ("internal", "root") and len(node["children"]) > cfg.f:
            sibs = split_node(nodes, node, cfg, ctx)
            if node.get("parent"):
                _add_children_to_parent(nodes, node["parent"], sibs, cfg, ctx)
            split += 1

    _relevel(nodes, root_id)
    return {"folded_groups": folded_groups, "same_category_merges": same_cat_merges, "split": split}


# ---------- Part 3: lazy re-summarization (only dirty nodes) ----------
def resummarize_dirty(nodes, cfg, ctx):
    """Regenerate routing summaries only for dirty internal nodes whose children are NOT a single
    clean category (those route on the category name and keep summary=None). Batched + cached."""
    dirty = [n for n in nodes.values()
             if n.get("dirty") and n["kind"] in ("internal", "root")]
    need = [n for n in dirty if n["category"] is None]        # multi-category -> needs a summary
    if need:
        groups = [_child_nodes(nodes, n) for n in need]
        summs = _pack.write_summaries(groups, None, ctx)
        for n, s in zip(need, summs):
            if n["kind"] != "root":                           # the root's summary is the overview, set later
                n["summary"] = s
    for n in nodes.values():
        n["dirty"] = False
    return len(need)


# ---------- structural metrics ----------
def structure(nodes, root_id, cfg):
    lo = math.ceil(cfg.f / 2)
    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    tops = _category_tops(nodes, root_id)
    # depth of each category-top (root-to-node) = spread across categories
    depth_from_root = {}
    from collections import deque
    dq = deque([(root_id, 0)])
    while dq:
        nid, d = dq.popleft()
        depth_from_root[nid] = d
        for c in nodes[nid]["children"]:
            if nodes[c]["kind"] != "leaf":
                dq.append((c, d + 1))
    cat_depths = [depth_from_root.get(nid, 0) for nid in tops]
    fan = Counter(len(n["children"]) for n in internal)
    underfull = sum(1 for n in internal if n["kind"] != "root" and len(n["children"]) < lo)
    multi = sum(1 for n in internal if len(_cat_set(nodes, n)) > 1)
    # leaf-level nodes: internal nodes whose children are all subject leaves -- the routing targets
    leaf_level = [n for n in internal if n["children"] and all(nodes[c]["kind"] == "leaf" for c in n["children"])]
    ll_multi = sum(1 for n in leaf_level if len({nodes[c]["category"] for c in n["children"]}) > 1)
    return {"n_nodes": len(nodes), "n_internal": len(internal),
            "depth": nodes[root_id]["level"],
            "cat_depth_spread_var": round(float(np.var(cat_depths)), 3) if cat_depths else 0.0,
            "cat_depth_min": min(cat_depths) if cat_depths else 0,
            "cat_depth_max": max(cat_depths) if cat_depths else 0,
            "underfull_nodes": underfull,
            "one_child_nodes": fan.get(1, 0),
            "fanout_distribution": {str(k): fan[k] for k in sorted(fan)},
            "single_category_rate_all_internal": round(1 - multi/len(internal), 3) if internal else None,
            "single_category_rate_leaf_level": round(1 - ll_multi/len(leaf_level), 3) if leaf_level else None,
            "n_leaf_level_nodes": len(leaf_level)}


# ---------- build the rebalanced index from the built per_category tree ----------
def build_index(movie, cfg):
    per = _store.load_index(movie)
    nodes = {nid: dict(n) for nid, n in per["nodes"].items()}   # copy so per_category on disk is untouched
    for n in nodes.values():
        n.setdefault("dirty", False)
    root_id = per["root"]
    cache_path = HERE/"output"/movie/"routing_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ctx = _ctx(cfg, cache)
    before = structure(nodes, root_id, cfg)
    changed = readjust(nodes, root_id, cfg, ctx)
    # readjust may have grown a new root; find it (node with kind root)
    root_id = next(nid for nid, n in nodes.items() if n["kind"] == "root")
    resummarize_dirty(nodes, cfg, ctx)
    after = structure(nodes, root_id, cfg)
    idx = {"nodes": nodes, "root": root_id, "chunks": per["chunks"], "overview": per.get("overview", ""),
           "meta": {"slug": movie, "build_mode": "per_category_rebalanced"}}
    return idx, before, after, changed, ctx["stats"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="a_dry_white_season")
    a = ap.parse_args()
    cfg = _cfg.load()
    idx, before, after, changed, stats = build_index(a.movie, cfg)
    print(json.dumps({"movie": a.movie, "f": cfg.f, "changed": changed,
                      "resummarize_calls": stats["calls"],
                      "before": before, "after": after}, indent=2))
