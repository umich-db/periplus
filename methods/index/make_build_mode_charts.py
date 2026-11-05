"""Build-mode comparison charts: per_category vs global_sort vs balanced on the 295-question
benchmark. Matches the report style in shared/make_graphs.py (matplotlib, horizontal bars, value
labels). Writes report/build_modes.png (all metrics, one figure) and report/build_modes_accuracy.png
(accuracy alone). Numbers are the measured results; sources noted in the code below.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTD = _BENCH/"report"; OUTD.mkdir(parents=True, exist_ok=True)

MODES = ["per_category", "global_sort", "balanced"]
COLOR = {"per_category": "#4C72B0", "global_sort": "#DD8452", "balanced": "#C44E52"}

# Measured results. Accuracy + fallback: pooled 295-question diagnostics run (one consistent run,
# the run McNemar was computed on: all pairs p >= 0.47, statistically tied). Purity: leaf-node mean
# (per_category 1.00 by construction). nodes_scored: median per question (deterministic traversal,
# balanced_full run). Build summary nodes: internal nodes needing an LLM routing summary, 10 Things
# I Hate About You as a representative document (build cost).
#
# This chart compares the three tree BUILD modes only. The shipped method's fallback choice
# (doc_leaves vs sibling) is a query-time ablation, not a build mode, so it is presented on the
# per-method figures (shared/make_graphs.py), not here.
PANELS = [
    {"title": "Judge accuracy", "unit": "%", "dir": "higher is better",
     "vals": {"per_category": 38.0, "global_sort": 38.0, "balanced": 35.9},
     "note": "no pair differs (McNemar exact, p >= 0.47): a statistical tie"},
    {"title": "Category purity (leaf nodes)", "unit": "", "dir": "higher is better",
     "vals": {"per_category": 1.00, "global_sort": 0.19, "balanced": 0.49}, "note": None},
    {"title": "Fallback rate", "unit": "%", "dir": "lower is better",
     "vals": {"per_category": 49.8, "global_sort": 49.5, "balanced": 50.8}, "note": None},
    {"title": "Median nodes scored / question", "unit": "", "dir": "lower is better (query cost)",
     "vals": {"per_category": 47, "global_sort": 54, "balanced": 53}, "note": None},
    {"title": "Build summary nodes (10 Things)", "unit": "", "dir": "lower is better (build cost)",
     "vals": {"per_category": 98, "global_sort": 17, "balanced": 17}, "note": None},
]


def draw(ax, panel):
    vals = [panel["vals"][m] for m in MODES]
    y = range(len(MODES))
    bars = ax.barh(y, vals, color=[COLOR[m] for m in MODES], edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(MODES); ax.invert_yaxis()
    ax.set_title(panel["title"], fontweight="bold", fontsize=11)
    mx = max(vals) if vals else 1
    fmt = (lambda v: f"{v:.1f}{panel['unit']}") if panel["unit"] == "%" else \
          (lambda v: f"{v:.2f}" if max(vals) <= 1.5 else f"{v:g}")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + mx*0.02, b.get_y() + b.get_height()/2, fmt(v),
                va="center", fontsize=9.5)
    ax.set_xlim(0, mx*1.20)
    ax.tick_params(labelsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    sub = panel["dir"] + (("\n" + panel["note"]) if panel["note"] else "")
    ax.set_xlabel(sub, fontsize=8.2, style="italic", color="#444")


def combined():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    fig.suptitle("Index build modes on the 295-question benchmark: per_category vs global_sort vs balanced",
                 fontweight="bold", fontsize=13.5, y=0.99)
    flat = axes.flatten()
    for ax, panel in zip(flat, PANELS):
        draw(ax, panel)
    # last cell: text summary of tree shape
    ax = flat[5]; ax.axis("off")
    ax.text(0.0, 0.92, "Tree shape", fontweight="bold", fontsize=11, transform=ax.transAxes)
    lines = [("per_category", "depth 4, uneven"),
             ("global_sort", "depth 2, uniform"),
             ("balanced", "depth 2, uniform")]
    for i, (m, d) in enumerate(lines):
        ax.text(0.0, 0.72 - i*0.16, "●", color=COLOR[m], fontsize=12, transform=ax.transAxes)
        ax.text(0.07, 0.72 - i*0.16, f"{m}: {d}", fontsize=10, transform=ax.transAxes, va="center_baseline")
    ax.text(0.0, 0.14, "Accuracy is a tie; the modes differ on coherence and cost.\n"
                       "Answer-time tokens/question are ~equal (same digest budget),\n"
                       "so cost differs at build, not query.",
            fontsize=8.4, style="italic", color="#444", transform=ax.transAxes, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTD/"build_modes.png", dpi=150); plt.close(fig)
    print("wrote", OUTD/"build_modes.png")


def accuracy_only():
    panel = PANELS[0]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    draw(ax, panel)
    ax.set_title("Judge accuracy by build mode (NarrativeQA, 10 movies, 295 Qs)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTD/"build_modes_accuracy.png", dpi=150); plt.close(fig)
    print("wrote", OUTD/"build_modes_accuracy.png")


if __name__ == "__main__":
    combined()
    accuracy_only()
