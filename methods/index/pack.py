"""Phase 5: pack the subjects into a balanced, bounded-fanout tree (bottom-up).

No node is ever wider than f. Grouping and clustering are the embedding's job
(cluster.partition, the ceil(n/f) routine, pack only). The LLM is called only to write a
routing summary for each SPLIT box (a node whose children exceeded f and had to be
divided), not for flat nodes. Runs at two scopes: within each category over its subjects,
then over the category top-nodes, since there are more categories than f. The root points
at the packed category boxes; the whole-document overview is written later in Phase 6.

Vocabulary: root, category, subject, field, source document.
"""
import sys, re, json, hashlib, pathlib, argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import cluster as _cluster  # noqa: E402
import chunk as _chunk, categories as _cat, extract as _extract  # noqa: E402

SUMM_PROMPT = (
    "You are writing routing summaries for {label}. Below are {k} sibling groups of items. "
    "For EACH group, write a routing summary of at most {s} tokens that says what "
    "DISTINGUISHES this group from the other groups, so a question can be routed to the "
    "right group. Do not recap the contents; state what sets it apart. Return ONLY a JSON "
    'object mapping each group number to its summary: {{"0": "...", "1": "..."}}.\n\n'
    "{blocks}\n\nJSON object:"
)


# ---------- nodes ----------
def _routing_text(node):
    if node["kind"] == "leaf":
        return node["content"]
    return node["summary"] or node["category"] or ""


def _leaf_id(subject):
    h = hashlib.sha1(json.dumps([subject["category"], subject["name"], subject["fields"],
                                 subject["source"]], sort_keys=True).encode()).hexdigest()
    return "leaf_" + h[:16]


def _int_id(child_ids):
    return "n_" + hashlib.sha1("|".join(sorted(child_ids)).encode()).hexdigest()[:16]


def _make_leaf(subject):
    content = subject["name"] + " :: " + "; ".join(f"{k}: {v}" for k, v in subject["fields"].items())
    return {"id": _leaf_id(subject), "level": 0, "kind": "leaf", "category": subject["category"],
            "summary": None, "content": content, "children": [], "parent": None, "dirty": False,
            "source": subject["source"]}


def _make_internal(children, cat, summary):
    cids = [c["id"] for c in children]
    return {"id": _int_id(cids), "level": 1 + max(c["level"] for c in children),
            "kind": "internal", "category": cat, "summary": summary, "content": None,
            "children": cids, "parent": None, "dirty": False}


# ---------- routing summaries (the only LLM in pack) ----------
def _skey(group):
    return hashlib.sha1("|".join(sorted(c["id"] for c in group)).encode()).hexdigest()


def _group_repr(group, cap_items=40, cap_chars=160):
    lines = []
    for c in group[:cap_items]:
        lines.append("- " + " ".join(str(_routing_text(c)).split())[:cap_chars])
    if len(group) > cap_items:
        lines.append(f"- (and {len(group) - cap_items} more)")
    return "\n".join(lines)


def _parse_obj(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        return {str(k): str(v).strip() for k, v in json.loads(m.group(0)).items()}
    except Exception:  # noqa: BLE001
        return {}


def _truncate_tokens(text, s):
    words = text.split()
    cap = max(1, int(s / 1.3))                 # ~s tokens
    return " ".join(words[:cap])


def _summ_call(batch_groups, cat, ctx):
    label = f"category '{cat}'" if cat else "the top level of the index"
    blocks = "\n\n".join(f"Group {i}:\n{_group_repr(g)}" for i, g in enumerate(batch_groups))
    text, pt, ct = client.chat(
        SUMM_PROMPT.format(label=label, k=len(batch_groups), s=ctx["s"], blocks=blocks),
        max_tokens=16384, bucket="index_build")
    ctx["stats"]["in"] += pt; ctx["stats"]["out"] += ct
    obj = _parse_obj(text)
    return [_truncate_tokens(obj.get(str(i)) or _group_repr(batch_groups[i], 3, 60), ctx["s"])
            for i in range(len(batch_groups))]


def write_summaries(groups, cat, ctx):
    """One routing summary per split box. Cache keyed by the sorted child-id set (persisted),
    so a rebuild with unchanged membership makes no call. Batches floor(B/C) groups per call."""
    cache = ctx["cache"]
    keys = [_skey(g) for g in groups]
    res = [None] * len(groups)
    todo = []
    for i, k in enumerate(keys):
        if k in cache:
            res[i] = cache[k]; ctx["stats"]["cached"] += 1
        else:
            todo.append(i)
    batches = [todo[b:b + ctx["summ_batch"]] for b in range(0, len(todo), ctx["summ_batch"])]
    def _run(idxs):
        return idxs, _summ_call([groups[i] for i in idxs], cat, ctx)
    if batches:
        with ThreadPoolExecutor(max_workers=ctx["workers"]) as ex:
            for idxs, summs in ex.map(_run, batches):
                for i, s in zip(idxs, summs):
                    res[i] = s; cache[keys[i]] = s
        ctx["stats"]["calls"] += len(batches)
    return res


# ---------- pack ----------
def pack_level(items, cat, ctx):
    """Return the single top node for `items`. Split boxes get routing summaries; the flat
    wrap (len <= f) does not (its routing text is its category name)."""
    f = ctx["f"]
    if len(items) <= f:
        node = _make_internal(items, cat, summary=None)
        ctx["nodes"][node["id"]] = node
        for c in items:
            c["parent"] = node["id"]
        return node
    vecs = client.embed([_routing_text(it) for it in items], model_name=ctx["embed_model"])
    groups = [[items[i] for i in cl] for cl in _cluster.partition(vecs, f, ctx["seed"])]
    summaries = write_summaries(groups, cat, ctx)
    boxes = []
    for g, summ in zip(groups, summaries):
        b = _make_internal(g, cat, summary=summ)
        ctx["nodes"][b["id"]] = b
        for c in g:
            c["parent"] = b["id"]
        boxes.append(b)
    return pack_level(boxes, cat, ctx)


def pack(subjects, cfg, cache_path=None):
    cache = {}
    if cache_path and pathlib.Path(cache_path).exists():
        cache = json.loads(pathlib.Path(cache_path).read_text())
    ctx = {"nodes": {}, "f": cfg.f, "seed": cfg.cluster_seed, "s": cfg.s,
           "embed_model": cfg.embedding_model, "summ_batch": max(1, cfg.B // cfg.C),
           "workers": cfg.propose_workers, "cache": cache,
           "stats": {"calls": 0, "cached": 0, "in": 0, "out": 0}}
    by_cat = defaultdict(list)
    for s in subjects:
        by_cat[s["category"]].append(s)

    cat_tops, cat_top_id = [], {}
    for cat in sorted(by_cat):                              # deterministic order
        leaves = [_make_leaf(s) for s in by_cat[cat]]
        for lf in leaves:
            ctx["nodes"][lf["id"]] = lf
        top = pack_level(leaves, cat, ctx)
        cat_tops.append(top); cat_top_id[cat] = top["id"]

    root = pack_level(cat_tops, None, ctx)                  # category layer + root
    root["kind"] = "root"
    if cache_path:                                          # persist routing-summary cache
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(cache_path).write_text(json.dumps(ctx["cache"], ensure_ascii=False))
    return ctx["nodes"], root["id"], cat_top_id, ctx["stats"]


# ---------- acceptance ----------
def _load_phase4(slug, cfg):
    """Reuse persisted Phase 4 output if present (subjects.jsonl + categories.json); else
    run the pipeline. Avoids re-running the extract LLM for a pack acceptance."""
    out = HERE/"output"/slug
    sp, cp = out/"subjects.jsonl", out/"categories.json"
    if sp.exists() and cp.exists():
        subjects = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
        return subjects
    chunks = _chunk.chunk_document(_chunk.load_source(slug), cfg.chunk_target, cfg.B)
    union, _, _ = _cat.propose(chunks, workers=cfg.propose_workers, cache_path=out/"propose_cache.json")
    categories, _, _ = _cat.merge(union, cfg.embedding_model, cfg.merge_similarity)
    subjects, _, _ = _extract.extract(chunks, categories, cfg)
    return subjects


if __name__ == "__main__":
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    a = ap.parse_args()
    cfg = _cfg.load()
    subjects = _load_phase4(a.movie, cfg)
    nodes, root_id, cat_top_id, stats = pack(subjects, cfg)

    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    # asserts
    for n in internal:
        assert len(n["children"]) <= cfg.f, f"node {n['id']} has {len(n['children'])} > f children"
    assert len(nodes[root_id]["children"]) <= cfg.f, "root wider than f (category layer not packed)"
    by_cat = Counter(s["category"] for s in subjects)
    for cat, tid in cat_top_id.items():
        n = by_cat[cat]; depth = nodes[tid]["level"]
        expect = max(1, math.ceil(math.log(n, cfg.f))) if n > 1 else 1
        assert abs(depth - expect) <= 1, f"category {cat!r}: depth {depth} vs expected ~{expect} (n={n})"

    levels = Counter(n["level"] for n in nodes.values())
    fanout = Counter(len(n["children"]) for n in internal)
    packed = {c: nodes[t]["level"] for c, t in cat_top_id.items() if by_cat[c] > cfg.f}
    print(json.dumps({
        "movie": a.movie, "f": cfg.f, "total_subjects": len(subjects),
        "n_nodes": len(nodes), "depth": nodes[root_id]["level"],
        "root_children": len(nodes[root_id]["children"]),
        "nodes_per_level (0=leaf)": {str(k): levels[k] for k in sorted(levels)},
        "fanout_distribution (children -> count)": {str(k): fanout[k] for k in sorted(fanout)},
        "packed_categories (depth>1)": dict(sorted(packed.items(), key=lambda kv: -kv[1])),
        "routing_summary_calls": stats["calls"], "served_from_cache": stats["cached"],
        "pack_build_tokens": stats["in"] + stats["out"],
    }, indent=2, ensure_ascii=False))
