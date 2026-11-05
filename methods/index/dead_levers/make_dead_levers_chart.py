"""Dead-levers diagnostic: retrieval, tree shape, and context width do not move accuracy; extraction
(what gets captured into a subject and its span) is the live lever. One figure, four panels.

Sources (all real, measured on the 295-question NarrativeQA set):
  retrieval      results/routing_diagnostics.json  (partA current/oracle, partB doc_leaves)
  tree shape     results/diagnostics.json          (accuracy + McNemar across build modes)
  context width  results/span_diag.json            (routed span vs whole-document span)
  extraction     failure taxonomy of the covered-but-failing oracle set (n=134)

Vocabulary: root, category, subject, field, source document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = _BENCH/"report"; OUT.mkdir(parents=True, exist_ok=True)

CEIL = 44.1                    # oracle-leaf retrieval ceiling (routing_diagnostics partA oracle_all)
DEAD = "#9AA0A6"              # gray = a lever that does not move accuracy
DEAD_EDGE = "#6B7075"
LIVE = "#C44E52"             # red = the live lever
LIVE_MUTE = "#D9B3B4"

# dead levers: (title, verdict, [(setting, accuracy%)], reading)
LEVERS = [
    ("Retrieval  (route to the answer leaf)",
     "DEAD: no headroom above the ceiling",
     [("current routing", 39.3), ("doc_leaves fallback", 44.4), ("oracle leaf (perfect)", 44.1)],
     "doc_leaves already reaches the oracle-leaf ceiling.\nPerfect retrieval still tops out at 44%."),
    ("Tree shape  (build mode)",
     "DEAD: three shapes, one accuracy",
     [("per_category", 38.0), ("global_sort", 38.0), ("balanced", 35.9)],
     "Every pair ties (McNemar exact, p >= 0.47)."),
    ("Context width  (answer span)",
     "DEAD: 19x more context, no gain",
     [("routed span ~6.4K", 39.7), ("whole document ~120K", 37.3)],
     "Widening the span 6.4K -> 120K chars does not\nraise accuracy (it dips slightly)."),
]

EXTRACT = [("false coverage", 73.9), ("multi-hop", 15.7), ("answer synthesis", 10.4)]


def dead_panel(ax, title, verdict, items, reading):
    labels = [l for l, _ in items]
    vals = [v for _, v in items]
    y = range(len(items))
    ax.barh(y, vals, color=DEAD, edgecolor=DEAD_EDGE, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.axvline(CEIL, ls="--", lw=1.3, color=LIVE, alpha=0.85)
    ax.set_xlim(0, 54)
    ax.set_ylim(len(items) - 0.4, -0.9)
    ax.set_title(title, fontweight="bold", fontsize=11.5, loc="left")
    ax.text(0.985, 0.90, verdict, transform=ax.transAxes, ha="right", va="center",
            fontsize=9.5, fontweight="bold", color=LIVE,
            bbox=dict(boxstyle="round,pad=0.28", fc="#FBE9E9", ec=LIVE, lw=0.9))
    ax.set_xlabel(reading, fontsize=8.6, style="italic", color="#444", labelpad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
fig.suptitle("Retrieval, tree shape, and context width are dead levers.  Extraction is the bottleneck.",
             fontweight="bold", fontsize=14.5, y=0.985)

for ax, (t, v, items, r) in zip(axes.flatten()[:3], LEVERS):
    dead_panel(ax, t, v, items, r)

# oracle-ceiling label, placed once on the first panel (low, clear of the verdict badge)
axes[0, 0].text(CEIL + 0.7, 2.35, "oracle-leaf\nceiling 44%", fontsize=8.2, color=LIVE, va="center")

# live lever: extraction / coverage
axL = axes[1, 1]
labels = [l for l, _ in EXTRACT]
vals = [v for _, v in EXTRACT]
cols = [LIVE, LIVE_MUTE, LIVE_MUTE]
y = range(len(EXTRACT))
axL.barh(y, vals, color=cols, edgecolor="white", height=0.62)
axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=10); axL.invert_yaxis()
for i, v in enumerate(vals):
    axL.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=10,
             fontweight="bold" if i == 0 else "normal")
axL.set_xlim(0, 100)
axL.set_title("LIVE lever:  extraction / coverage", fontweight="bold", fontsize=11.5, loc="left", color=LIVE)
axL.text(0.985, 0.46, "THIS moves accuracy", transform=axL.transAxes, ha="right", va="center",
         fontsize=9.5, fontweight="bold", color="white",
         bbox=dict(boxstyle="round,pad=0.28", fc=LIVE, ec=LIVE))
axL.set_xlabel("Covered-but-failing questions (oracle set, n=134). 74% are false coverage:\n"
               "the answer text was never captured into the subject's span. Fix extraction, not routing.",
               fontsize=8.6, style="italic", color="#444", labelpad=6)
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)
axL.tick_params(length=0)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(hspace=0.55, wspace=0.28)
p = OUT/"dead_levers.png"
fig.savefig(p, dpi=170, bbox_inches="tight")
print("wrote", p)
