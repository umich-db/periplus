"""Show what categories the nodes carry and an example of what sits under them. Analysis only,
reuses tree_inspect.load_trees (zero LLM). Picks a few leaf-level nodes per mode and renders a card
for each: the node's category label (per_category) or routing summary (balanced), and the subject
names under it (with a category tag for balanced, to show the seams). Writes
results/tree_contents_<movie>.png. Vocabulary: root, category, subject, field, source document.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, pathlib, argparse, textwrap
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as _cfg  # noqa: E402
import tree_inspect as ti  # noqa: E402

PC_COL = "#1baf7a"; BA_COL = "#eb6834"; INK = "#0c0e12"; SUB = "#52514e"


def leaf_level(nodes):
    return [n for n in nodes.values() if n["kind"] == "internal" and n["children"]
            and all(nodes[c]["kind"] == "leaf" for c in n["children"])]


def subjects_of(nodes, node):
    out = []
    for c in node["children"]:
        lf = nodes[c]
        name = lf["content"].split("::")[0].strip()
        out.append((name, lf["category"]))
    return out


def card(ax, header, sub, subjects, accent, tag_cats):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                               facecolor="#fcfcfd", edgecolor=accent, linewidth=1.6, zorder=0))
    ax.plot([0.01, 0.99], [0.985, 0.985], transform=ax.transAxes, color=accent, lw=4, zorder=1)
    y = 0.90
    for line in textwrap.wrap(header, 52)[:2]:
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="top")
        y -= 0.075
    if sub:
        for line in textwrap.wrap(sub, 60)[:3]:
            ax.text(0.05, y, line, transform=ax.transAxes, fontsize=8.2, style="italic", color=SUB, va="top")
            y -= 0.058
    y -= 0.02
    ax.text(0.05, y, "subjects under it:", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=SUB, va="top")
    y -= 0.075
    for name, cat in subjects:
        label = f"- {name}"
        if tag_cats:
            label += f"   [{cat}]"
        for j, line in enumerate(textwrap.wrap(label, 58)[:1]):
            ax.text(0.06, y, line, transform=ax.transAxes, fontsize=8.4, color=INK, va="top",
                    fontfamily="monospace")
        y -= 0.066
        if y < 0.04:
            break


def pick(nodes, mode, k=3):
    lln = leaf_level(nodes)
    if mode == "per_category":
        cand = [n for n in lln if len({nodes[c]["category"] for c in n["children"]}) == 1]
        cand.sort(key=lambda n: -len(n["children"]))
    else:
        cand = sorted(lln, key=lambda n: (-len({nodes[c]["category"] for c in n["children"]}), -len(n["children"])))
    return cand[:k]


def header_and_sub(nodes, node, mode):
    cats = Counter(nodes[c]["category"] for c in node["children"])
    n = len(node["children"])
    if mode == "per_category":
        head = f"{cats.most_common(1)[0][0]}  ({n} subjects, 1 category)"
        sub = "routing label: the category name (leaf-level nodes carry no generated summary)"
    else:
        comp = ", ".join(f"{c} x{m}" if m > 1 else c for c, m in cats.most_common(5))
        head = f"{comp}  ({n} subjects, {len(cats)} categories)"
        sub = "routing summary: " + " ".join((node.get("summary") or "").split())
    return head, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", default="a_dry_white_season")
    a = ap.parse_args()
    cfg = _cfg.load()
    trees = ti.load_trees(a.movie, cfg)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(f"What the nodes hold: per_category (single-category) vs balanced (seam-crossing)  ({a.movie})",
                 fontweight="bold", fontsize=13.5, y=0.985)
    axes[0, 0].set_title("per_category", fontweight="bold", fontsize=12, color=PC_COL, pad=10)
    axes[0, 1].set_title("balanced", fontweight="bold", fontsize=12, color=BA_COL, pad=10)
    for col, mode, accent, tag in ((0, "per_category", PC_COL, False), (1, "balanced", BA_COL, True)):
        nodes = trees[mode]["nodes"]
        picks = pick(nodes, mode)
        for row, node in enumerate(picks):
            head, sub = header_and_sub(nodes, node, mode)
            card(axes[row, col], head, sub, subjects_of(nodes, node)[:7], accent, tag)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE/"results"/f"tree_contents_{a.movie}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
