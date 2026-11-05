"""Inspect one movie's tree: per_category vs balanced. Analysis only, no rebuild of a new mode.
per_category loads from disk; balanced rebuilds from its cached routing summaries (identical tree,
zero LLM). Reuses the all-mpnet embedder for the traversal traces and the saved per-question judged
verdicts. Writes results/tree_inspection.md and results/tree_structure_<movie>.png.

Part 1: indented tree + structural stats + a two-panel figure.
Part 2: sampled node contents side by side + the granularity numbers.
Part 3: traces of questions right under per_category but wrong under balanced.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, pathlib, argparse, statistics
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_BENCH/"shared"))
import client, config as _cfg, store as _store, pack as _pack, pack_balanced as pb  # noqa: E402
import answer as _answer  # noqa: E402

COL = {"per_category": "#4C72B0", "balanced": "#C44E52"}


# ---------- load both trees (no LLM) ----------
def load_trees(movie, cfg):
    per = _store.load_index(movie)
    chunks = _store._get_chunks(movie, cfg)
    subjects = _store._get_subjects(movie, cfg, chunks)
    bn, broot, _boxes, _stats = pb.pack_balanced(
        subjects, cfg, cache_path=HERE/"output"/movie/"routing_cache_balanced.json")
    bal = {"nodes": bn, "root": broot}
    return {"per_category": per, "balanced": bal}


# ---------- per-node derived facts (subtree leaves, leaf categories) ----------
def derive(nodes, root):
    leaves_under, cats_under = {}, {}

    def visit(nid):
        n = nodes[nid]
        if n["kind"] == "leaf":
            leaves_under[nid] = 1
            cats_under[nid] = Counter([n["category"]])
            return
        lu, cu = 0, Counter()
        for c in n["children"]:
            visit(c)
            lu += leaves_under[c]; cu += cats_under[c]
        leaves_under[nid] = lu; cats_under[nid] = cu
    visit(root)
    return leaves_under, cats_under


def _cats_str(counter, cap=4):
    items = counter.most_common()
    head = ", ".join(f"{c} x{n}" if n > 1 else c for c, n in items[:cap])
    return head + (f", +{len(items)-cap} more cats" if len(items) > cap else "")


# ---------- Part 1: indented tree + stats ----------
def indent_tree(nodes, root, leaves_under, cats_under, out, max_lines=200):
    lines = []

    def rec(nid, d):
        if len(lines) >= max_lines:
            return
        n = nodes[nid]
        kids = n.get("children", [])
        leaf_kids = [k for k in kids if nodes[k]["kind"] == "leaf"]
        inner_kids = [k for k in kids if nodes[k]["kind"] != "leaf"]
        tag = f"{'  '*d}{n['kind']:8s} {nid[:14]:14s} children={len(kids):2d} leaves={leaves_under[nid]:3d}  [{_cats_str(cats_under[nid])}]"
        lines.append(tag)
        for k in inner_kids:
            rec(k, d + 1)
        if leaf_kids:
            lc = Counter(nodes[k]["category"] for k in leaf_kids)
            lines.append(f"{'  '*(d+1)}({len(leaf_kids)} leaves: {_cats_str(lc)})")
    rec(root, 0)
    out.extend(lines)
    if len(lines) >= max_lines:
        out.append(f"  ... (truncated at {max_lines} lines)")


def stats(nodes, root, leaves_under):
    depth = nodes[root]["level"]
    per_level = Counter(n["level"] for n in nodes.values())
    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    fan = [len(n["children"]) for n in internal]
    fanhist = Counter(fan)
    # subjects (leaf descendants) per node, per level (internal+root only)
    size_by_level = defaultdict(list)
    for n in internal:
        size_by_level[n["level"]].append(leaves_under[n["id"]])
    return {"depth": depth, "per_level": dict(sorted(per_level.items())),
            "fan_min": min(fan), "fan_med": statistics.median(fan), "fan_max": max(fan),
            "fanhist": dict(sorted(fanhist.items())),
            "size_by_level": {lv: (round(statistics.mean(v), 1), min(v), max(v)) for lv, v in sorted(size_by_level.items())}}


# ---------- Part 2: granularity ----------
def leaf_level_nodes(nodes):
    return [n for n in nodes.values()
            if n["kind"] == "internal" and n["children"] and all(nodes[c]["kind"] == "leaf" for c in n["children"])]


def granularity(nodes, cats_under):
    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    lln = leaf_level_nodes(nodes)
    subj_per_lln = [sum(1 for c in n["children"] if nodes[c]["kind"] == "leaf") for n in lln]
    children_per_internal = [len(n["children"]) for n in internal]
    multi_cat = sum(1 for n in internal if len(cats_under[n["id"]]) > 1)
    summ_tokens = [client.count_tokens(n["summary"]) for n in internal if n.get("summary")]
    return {"n_leaf_level_nodes": len(lln),
            "subjects_per_leaf_node_mean": round(statistics.mean(subj_per_lln), 2) if subj_per_lln else None,
            "subjects_per_leaf_node_median": statistics.median(subj_per_lln) if subj_per_lln else None,
            "children_per_internal_mean": round(statistics.mean(children_per_internal), 2),
            "n_internal": len(internal),
            "multi_category_nodes": multi_cat,
            "multi_category_rate": round(multi_cat/len(internal), 3),
            "single_category_rate": round(1 - multi_cat/len(internal), 3),
            "summary_tokens_mean": round(statistics.mean(summ_tokens), 1) if summ_tokens else None,
            "summary_tokens_median": statistics.median(summ_tokens) if summ_tokens else None}


def sample_internal(nodes, root, leaves_under, cats_under, k=6):
    internal = [n for n in nodes.values() if n["kind"] in ("internal", "root")]
    internal.sort(key=lambda n: (-n["level"], n["id"]))          # root first, then down the levels
    if len(internal) <= k:
        picks = internal
    else:
        step = len(internal) / k
        picks = [internal[int(i*step)] for i in range(k)]
    rows = []
    for n in picks:
        rows.append({"id": n["id"][:14], "level": n["level"], "kind": n["kind"],
                     "children": len(n["children"]), "leaves": leaves_under[n["id"]],
                     "cats": _cats_str(cats_under[n["id"]], 6),
                     "summary": " ".join((n.get("summary") or "").split())[:320]})
    return rows


# ---------- Part 3: traversal trace ----------
def trace_route(idx, question, cfg):
    """Instrumented cosine traversal: log each expansion's scored children (top ones) and the reached
    leaves, reusing the real answer.Router + answer.traverse."""
    router = _answer.Router(question, cfg, {})
    exps = []
    orig = router.score_nodes

    def traced(kids):
        sc = orig(kids)
        exps.append([(k["id"], k["kind"], " ".join(str(_answer._routing_text(k)).split())[:110], float(s))
                     for k, s in zip(kids, sc)])
        return sc
    router.score_nodes = traced
    tr = _answer.traverse(idx, router, cfg)
    return exps, tr["reached_leaves"], tr


def fmt_trace(exps, reached):
    reached_ids = {r["id"] for r in reached}
    lines = []
    for i, exp in enumerate(exps):
        ranked = sorted(exp, key=lambda e: -e[3])
        chosen = ranked[0]
        lines.append(f"  expansion {i+1}: scored {len(exp)} children; top pick score={chosen[3]:.3f}")
        for nid, kind, txt, s in ranked[:4]:
            mark = " <== CHOSE" if (nid, kind, txt, s) == chosen else ""
            leafmark = " [reached-leaf]" if nid in reached_ids else ""
            lines.append(f"      {s:.3f} [{kind}] {txt}{mark}{leafmark}")
    lines.append("  reached leaves: " + " | ".join(" ".join(r["content"].split())[:80] for r in reached))
    return lines


# ---------- figure ----------
def figure(movie, s_pc, s_ba, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle(f"Tree structure: per_category vs balanced ({movie})", fontweight="bold", fontsize=13)
    # panel 1: nodes per level
    ax = axes[0]
    levels = sorted(set(s_pc["per_level"]) | set(s_ba["per_level"]))
    x = range(len(levels)); w = 0.38
    ax.bar([i-w/2 for i in x], [s_pc["per_level"].get(l, 0) for l in levels], w, color=COL["per_category"], label="per_category")
    ax.bar([i+w/2 for i in x], [s_ba["per_level"].get(l, 0) for l in levels], w, color=COL["balanced"], label="balanced")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"L{l}\n(0=leaf)" if l == 0 else f"L{l}" for l in levels])
    ax.set_ylabel("number of nodes"); ax.set_title("Nodes per level", fontweight="bold")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # panel 2: mean subtree size (subjects) per level, internal nodes
    ax = axes[1]
    lv = sorted(set(s_pc["size_by_level"]) | set(s_ba["size_by_level"]))
    x = range(len(lv))
    ax.bar([i-w/2 for i in x], [s_pc["size_by_level"].get(l, (0, 0, 0))[0] for l in lv], w, color=COL["per_category"], label="per_category")
    ax.bar([i+w/2 for i in x], [s_ba["size_by_level"].get(l, (0, 0, 0))[0] for l in lv], w, color=COL["balanced"], label="balanced")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"L{l}" for l in lv])
    ax.set_ylabel("mean subjects under node"); ax.set_title("Node size per level (subjects/node)", fontweight="bold")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="a_dry_white_season")
    a = ap.parse_args()
    cfg = _cfg.load()
    movie = a.movie
    trees = load_trees(movie, cfg)
    der = {m: derive(trees[m]["nodes"], trees[m]["root"]) for m in trees}
    st = {m: stats(trees[m]["nodes"], trees[m]["root"], der[m][0]) for m in trees}
    gr = {m: granularity(trees[m]["nodes"], der[m][1]) for m in trees}
    figure(movie, st["per_category"], st["balanced"], HERE/"results"/f"tree_structure_{movie}.png")

    L = [f"# Tree inspection: per_category vs balanced ({movie})\n",
         "Analysis only. per_category loaded from disk; balanced rebuilt from its cached routing "
         "summaries (identical tree, zero LLM). all-mpnet embedder reused for traces; saved judged "
         "verdicts reused. Vocabulary: root, category, subject, field, source document.\n"]

    # Part 1
    L.append("## Part 1: structure\n")
    for m in ("per_category", "balanced"):
        s = st[m]
        L.append(f"### {m}: depth {s['depth']}, nodes/level {s['per_level']}")
        L.append(f"fanout children/internal: min {s['fan_min']}, median {s['fan_med']}, max {s['fan_max']}; "
                 f"histogram {s['fanhist']}")
        L.append("subjects/node per level (mean, min, max): " +
                 "; ".join(f"L{lv}={mm}" for lv, mm in s["size_by_level"].items()))
        L.append("\n```")
        out = []
        indent_tree(trees[m]["nodes"], trees[m]["root"], der[m][0], der[m][1], out, max_lines=120)
        L.extend(out)
        L.append("```\n")
    L.append(f"Structure figure: results/tree_structure_{movie}.png\n")

    # Part 2
    L.append("## Part 2: node contents and granularity\n")
    for m in ("per_category", "balanced"):
        L.append(f"### sampled internal nodes ({m})")
        for r in sample_internal(trees[m]["nodes"], trees[m]["root"], der[m][0], der[m][1]):
            L.append(f"- {r['kind']} {r['id']} L{r['level']} children={r['children']} leaves={r['leaves']}")
            L.append(f"    covers: {r['cats']}")
            L.append(f"    summary: {r['summary'] or '(none)'}")
        L.append("")
    L.append("### granularity comparison")
    L.append(f"| metric | per_category | balanced |")
    L.append(f"|---|---|---|")
    def row(lbl, key):
        return f"| {lbl} | {gr['per_category'][key]} | {gr['balanced'][key]} |"
    L.append(row("leaf-level nodes", "n_leaf_level_nodes"))
    L.append(row("subjects/leaf-node mean", "subjects_per_leaf_node_mean"))
    L.append(row("subjects/leaf-node median", "subjects_per_leaf_node_median"))
    L.append(row("children/internal mean", "children_per_internal_mean"))
    L.append(row("single-category rate (internal nodes)", "single_category_rate"))
    L.append(row("multi-category rate (internal nodes)", "multi_category_rate"))
    L.append(row("summary length tokens (mean)", "summary_tokens_mean"))
    L.append(row("summary length tokens (median)", "summary_tokens_median"))
    L.append("")

    # Part 3
    L.append("## Part 3: divergence traces (right under per_category, wrong under balanced)\n")
    pc_j = {json.loads(l)["qid"]: json.loads(l) for l in
            (HERE/"results"/"index_per_category_judged.jsonl").read_text().splitlines() if l.strip()}
    ba_j = {json.loads(l)["qid"]: json.loads(l) for l in
            (HERE/"results"/"index_balanced_judged.jsonl").read_text().splitlines() if l.strip()}
    qs = {q["qid"]: q for q in
          [json.loads(l) for l in (_BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip()]
          if q["movie"] == movie}
    diverge = [qid for qid in qs if qid in pc_j and qid in ba_j
               and pc_j[qid]["verdict"] == "correct" and ba_j[qid]["verdict"] == "incorrect"]
    L.append(f"Found {len(diverge)} questions right under per_category and wrong under balanced "
             f"(saved verdicts). Tracing up to 3.\n")
    for qid in diverge[:3]:
        q = qs[qid]
        L.append(f"### {qid}: {q['question']}")
        L.append(f"gold: {' | '.join(q['gold_answers'])}")
        L.append(f"per_category prediction (correct): {pc_j[qid]['prediction'][:120]}")
        L.append(f"balanced prediction (wrong): {ba_j[qid]['prediction'][:120]}")
        for m in ("per_category", "balanced"):
            exps, reached, tr = trace_route(trees[m], q["question"], cfg)
            L.append(f"\n{m} route (nodes_scored={tr['nodes_scored']}, depth_reached={tr['depth_reached']}):")
            L.extend(fmt_trace(exps, reached))
        L.append("")

    (HERE/"results"/"tree_inspection.md").write_text("\n".join(L) + "\n")

    # console summary
    print("GRANULARITY (per_category vs balanced):")
    for key, lbl in [("subjects_per_leaf_node_mean", "subjects/leaf-node mean"),
                     ("subjects_per_leaf_node_median", "subjects/leaf-node median"),
                     ("children_per_internal_mean", "children/internal mean"),
                     ("single_category_rate", "single-category rate"),
                     ("summary_tokens_mean", "summary tokens mean")]:
        print(f"  {lbl:28s} pc={gr['per_category'][key]:>7}   ba={gr['balanced'][key]:>7}")
    print(f"\nright-pc/wrong-ba questions traced: {min(3, len(diverge))} of {len(diverge)}")
    print("wrote results/tree_inspection.md and results/tree_structure_%s.png" % movie)


if __name__ == "__main__":
    main()
