"""One diagram that is the tree AND the contents: a partial tree per mode where the shown leaf-level
nodes are expanded into boxes with their category label and the subjects written inside, connected to
the root by edges. Analysis only, reuses tree_inspect.load_trees (zero LLM). Writes
results/tree_annotated_<movie>.png. Vocabulary: root, category, subject, field, source document.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, pathlib, argparse, textwrap
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as _cfg  # noqa: E402
import tree_inspect as ti, tree_contents as tc  # noqa: E402

PC = "#1baf7a"; BA = "#eb6834"; INK = "#0c0e12"; SUB = "#52514e"; EDGE = "#b9bec8"; ROOTC = "#4C72B0"


def box(ax, cx, top, w, h, header, summary, subjects, accent, tag):
    ax.add_patch(FancyBboxPatch((cx-w/2, top-h), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor="#fcfcfd", edgecolor=accent, linewidth=1.6, zorder=3,
                                mutation_aspect=0.5))
    ax.add_patch(plt.Rectangle((cx-w/2, top-0.012), w, 0.012, facecolor=accent, edgecolor="none", zorder=4))
    y = top - 0.03
    for ln in textwrap.wrap(header, 40)[:2]:
        ax.text(cx-w/2+0.008, y, ln, fontsize=8.6, fontweight="bold", color=INK, va="top", zorder=5)
        y -= 0.023
    if summary:
        for ln in textwrap.wrap(summary, 46)[:2]:
            ax.text(cx-w/2+0.008, y, ln, fontsize=6.6, style="italic", color=SUB, va="top", zorder=5)
            y -= 0.0185
    y -= 0.006
    for name, cat in subjects:
        t = f"- {name}" + (f"  [{cat}]" if tag else "")
        ax.text(cx-w/2+0.012, y, textwrap.shorten(t, 44, placeholder=" ..."),
                fontsize=7.2, color=INK, va="top", family="monospace", zorder=5)
        y -= 0.02


def edge(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=EDGE, lw=1.1, zorder=1)


def draw_mode(ax, nodes, mode, y_root, y_box_top, accent, note, via_cluster):
    picks = tc.pick(nodes, mode, 3)
    xs = [0.18, 0.5, 0.82]; w, h = 0.30, y_box_top - (y_box_top - 0.30 if mode == "per_category" else y_box_top - 0.28)
    h = 0.30
    ax.scatter([0.5], [y_root], s=160, marker="s", color=ROOTC, edgecolor="white", zorder=4)
    ax.text(0.5, y_root+0.018, "root", ha="center", fontsize=9, fontweight="bold", color=ROOTC)
    for cx, node in zip(xs, picks):
        cats = Counter(nodes[c]["category"] for c in node["children"])
        n = len(node["children"])
        if mode == "per_category":
            header = f"{cats.most_common(1)[0][0]}  ({n} subjects, 1 category)"
            summary = "routing label = the category name"
        else:
            comp = ", ".join(f"{c} x{m}" if m > 1 else c for c, m in cats.most_common(4))
            header = f"{comp} ...  ({n} subjects, {len(cats)} categories)"
            summary = "summary: " + " ".join((node.get("summary") or "").split())
        if via_cluster:                                        # per_category: show a depth-2 hop
            cyx = cx*0.5 + 0.25                                # a cluster dot between root and box
            ax.scatter([cyx], [(y_root+y_box_top)/2], s=32, color=ROOTC, edgecolor="white", zorder=4)
            edge(ax, 0.5, y_root-0.01, cyx, (y_root+y_box_top)/2)
            edge(ax, cyx, (y_root+y_box_top)/2, cx, y_box_top)
        else:
            edge(ax, 0.5, y_root-0.01, cx, y_box_top)
        box(ax, cx, y_box_top, w, h, header, summary, tc.subjects_of(nodes, node)[:5], accent, mode == "balanced")
    ax.text(0.01, y_root, note, fontsize=8.4, color=SUB, va="center", style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="a_dry_white_season")
    a = ap.parse_args()
    cfg = _cfg.load()
    trees = ti.load_trees(a.movie, cfg)

    fig = plt.figure(figsize=(15, 12.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle(f"The tree with its contents: per_category vs balanced  ({a.movie})",
                 fontweight="bold", fontsize=14, y=0.985)
    ax.text(0.5, 0.955, "per_category", ha="center", fontsize=12.5, fontweight="bold", color=PC)
    draw_mode(ax, trees["per_category"]["nodes"], "per_category", 0.93, 0.80, PC,
              "141 leaf-level\nnodes total\n(3 shown),\ndepth up to 3,\nevery node\nsingle-category", True)
    ax.plot([0.03, 0.97], [0.47, 0.47], color="#d8d8d2", lw=1)
    ax.text(0.5, 0.45, "balanced", ha="center", fontsize=12.5, fontweight="bold", color=BA)
    draw_mode(ax, trees["balanced"]["nodes"], "balanced", 0.42, 0.36, BA,
              "28 leaf-level\nnodes total\n(3 shown),\ndepth 1,\nevery node ~33\nsubjects,\nmulti-category", False)
    out = HERE/"results"/f"tree_annotated_{a.movie}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
