"""Physical node-link diagram of the per_category vs balanced trees for one movie. Analysis only,
reuses tree_inspect.load_trees (per_category from disk, balanced from its cached summaries, zero LLM).
Collapses the ~900 subject leaves into their leaf-level nodes, positioned as a dendrogram: x spans the
subjects (node width proportional to subject count), y is depth from the root. Leaf-level nodes are
colored single-category (green) vs multi-category (orange). Writes results/tree_diagram_<movie>.png.

Vocabulary: root, category, subject, field, source document.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, pathlib, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as _cfg  # noqa: E402
import tree_inspect as ti  # noqa: E402

SINGLE = "#1baf7a"   # single-category leaf-level node
MULTI = "#eb6834"    # multi-category leaf-level node
INNER = "#4C72B0"    # internal / root node
EDGE = "#b9bec8"


def layout(nodes, root, leaves_under, cats_under):
    """Dendrogram positions. A drawn-leaf is a node whose children are all subject leaves (a
    leaf-level node); it gets horizontal width = its subject count. Internal nodes center over their
    drawn descendants. Returns pos[nid] = (x_center, width, depth) and the parent->child edges."""
    def drawn_children(nid):
        return [c for c in nodes[nid]["children"] if nodes[c]["kind"] != "leaf"]

    pos, edges, cur = {}, [], [0.0]

    def assign(nid, depth):
        dch = drawn_children(nid)
        if not dch:                                            # leaf-level node (terminal in drawing)
            w = leaves_under[nid]
            xc = cur[0] + w/2.0; cur[0] += w
            pos[nid] = (xc, w, depth)
            return xc
        xs = []
        for c in dch:
            xs.append(assign(c, depth+1))
            edges.append((nid, c))
        xc = sum(xs)/len(xs)
        pos[nid] = (xc, sum(leaves_under[c] for c in dch), depth)
        return xc
    assign(root, 0)
    return pos, edges


def draw_tree(ax, nodes, root, der, title):
    leaves_under, cats_under = der
    pos, edges = layout(nodes, root, leaves_under, cats_under)
    maxd = max(d for _, _, d in pos.values())
    # edges (parent depth above child depth; root at top)
    segs = []
    for p, c in edges:
        xp, _, dp = pos[p]; xc, _, dc = pos[c]
        segs.append([(xp, -dp), (xc, -dc)])
    ax.add_collection(LineCollection(segs, colors=EDGE, linewidths=0.6, zorder=1))
    # nodes
    for nid, (x, w, d) in pos.items():
        is_terminal = nodes[nid]["kind"] != "leaf" and all(nodes[c]["kind"] == "leaf" for c in nodes[nid]["children"])
        if nid == root:
            ax.scatter([x], [-d], s=90, color=INNER, edgecolor="white", zorder=3, marker="s")
        elif is_terminal:
            col = SINGLE if len(cats_under[nid]) == 1 else MULTI
            ax.scatter([x], [-d], s=8 + 2.2*w, color=col, edgecolor="white", linewidths=0.4, zorder=3)
        else:
            ax.scatter([x], [-d], s=34, color=INNER, edgecolor="white", zorder=2)
    n_term = sum(1 for nid in pos if nodes[nid]["kind"] != "leaf"
                 and all(nodes[c]["kind"] == "leaf" for c in nodes[nid]["children"]))
    single = sum(1 for nid in pos if nodes[nid]["kind"] != "leaf"
                 and all(nodes[c]["kind"] == "leaf" for c in nodes[nid]["children"]) and len(cats_under[nid]) == 1)
    ax.set_title(f"{title}   depth {maxd}, {n_term} leaf-level nodes, {single} single-category",
                 fontweight="bold", fontsize=11)
    ax.set_ylim(-maxd-0.4, 0.4)
    ax.set_yticks([-d for d in range(maxd+1)]); ax.set_yticklabels([f"depth {d}" for d in range(maxd+1)])
    ax.set_xlabel("subjects (node width proportional to subject count)")
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="a_dry_white_season")
    a = ap.parse_args()
    cfg = _cfg.load()
    trees = ti.load_trees(a.movie, cfg)
    der = {m: ti.derive(trees[m]["nodes"], trees[m]["root"]) for m in trees}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6), sharex=True)
    fig.suptitle(f"Tree shape: per_category vs balanced ({a.movie})", fontweight="bold", fontsize=14, y=0.98)
    draw_tree(axes[0], trees["per_category"]["nodes"], trees["per_category"]["root"], der["per_category"], "per_category")
    draw_tree(axes[1], trees["balanced"]["nodes"], trees["balanced"]["root"], der["balanced"], "balanced")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor=INNER, markersize=9, label="root / internal node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SINGLE, markersize=9, label="leaf-level node, single category"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MULTI, markersize=9, label="leaf-level node, multi category"),
    ], loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE/"results"/f"tree_diagram_{a.movie}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
