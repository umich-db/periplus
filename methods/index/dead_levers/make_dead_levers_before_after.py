"""Mechanism figure: a before/after sweep of the seven candidate accuracy levers. Six are things the
index controls (retrieval, tree shape, rebalancing, context width, and two extraction variants); each
is a pair of bars, baseline vs the intervention, and each pair is flat, the intervention does not move
accuracy. The seventh, the answer-time fallback (sibling -> doc_leaves), is the one lever that moves,
and it reaches the oracle-leaf ceiling.

Chart only. Every number is a real logged value from an existing diagnostic file (cited per row in the
footnotes). No build change, no routing change, no new evals. Style matches make_build_mode_charts.py.
Vocabulary: root, category, subject, field, source document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = _BENCH/"report"; OUT.mkdir(parents=True, exist_ok=True)

CEIL = 44.1                 # oracle-leaf ceiling (routing_diagnostics.json partA oracle_all)
GRAY = "#9AA0A6"           # baseline
DEAD = "#8FA8C8"           # intervention that does not move accuracy
ORA = "#C9D6E5"            # oracle upper bound (diagnostic, not an achievable index change)
LIVE = "#55A868"           # the fallback: the one intervention that moves
CEILC = "#C44E52"

# name, baseline, intervention, kind, footnote (with source file)
LEVERS = [
    ("1. Retrieval (routing)", 39.3, 44.1, "oracle",
     "per_category sibling (39.3) vs oracle leaf, perfect routing (44.1). Oracle = the ceiling.  [routing_diagnostics.json]"),
    ("2. Tree shape (build mode)", 38.0, 35.9, "dead",
     "per_category (38.0) vs balanced B+ build (35.9). McNemar p=0.47, tied.  [diagnostics.json]"),
    ("3. Rebalancing", 37.3, 37.6, "dead",
     "per_category (37.3) vs per_category_rebalanced (37.6). McNemar p=1.0, accuracy-neutral, structurally worse.  [rebalance_gate.json]"),
    ("4. Context width (span)", 39.7, 37.3, "dead",
     "routed span (39.7) vs whole source document handed to the answer model (37.3). More text hurts.  [span_diag.json]"),
    ("5. Extraction: exhaustive node", 0.0, 0.0, "dead",
     "A1 subset answerability. Exhaustive node-centric prompt recovers 0/19 strict.  [extraction_diagnostic.md]"),
    ("6. Extraction: relational edges", 0.0, 0.0, "dead",
     "A1 subset answerability. Capture rose to 7/19 strict but 0/19 became answerable.  [relational_extraction_test.md]"),
    ("7. Fallback (doc_leaves)", 39.3, 44.4, "live",
     "per_category sibling (39.3) vs doc_leaves fallback (44.4). McNemar p=0.031, reaches the oracle ceiling.  [routing_diagnostics.json]"),
]

ICOLOR = {"dead": DEAD, "oracle": ORA, "live": LIVE}
centers = [0, 1, 2, 3, 4, 5, 6.9]   # extra gap before the fallback row
BH = 0.36

fig, ax = plt.subplots(figsize=(12.5, 7.6))

for (name, base, interv, kind, _), yc in zip(LEVERS, centers):
    ax.barh(yc - 0.2, base, height=BH, color=GRAY, edgecolor="white", zorder=3)
    ax.barh(yc + 0.2, interv, height=BH, color=ICOLOR[kind], edgecolor="white", zorder=3,
            hatch="//" if kind == "oracle" else None)
    # value labels
    if base == 0 and interv == 0:
        ax.text(0.6, yc - 0.2, "0 / 19 (A1)", va="center", fontsize=9, color="#333")
        ax.text(0.6, yc + 0.2, "0 / 19 (A1)", va="center", fontsize=9, color="#333")
    else:
        ax.text(base + 0.5, yc - 0.2, f"{base:.1f}", va="center", fontsize=9.5)
        ax.text(interv + 0.5, yc + 0.2, f"{interv:.1f}", va="center", fontsize=9.5,
                fontweight="bold" if kind == "live" else "normal")

# oracle-leaf ceiling
ax.axvline(CEIL, ls="--", lw=1.4, color=CEILC, alpha=0.9, zorder=2)
ax.text(CEIL - 0.4, 2.2, "oracle-leaf\nceiling 44.1", color=CEILC, fontsize=9, va="center", ha="right")

# divider between the six index-controlled levers and the fallback row
ax.axhline(6.0, ls=":", lw=1.1, color="#888")
ax.text(49.6, 2.5, "index-controlled levers\n(do not move accuracy)", rotation=90, va="center",
        ha="center", fontsize=9, color="#555", style="italic")
ax.text(49.6, 6.9, "fallback\n(moves)", rotation=90, va="center", ha="center",
        fontsize=9, color=LIVE, style="italic", fontweight="bold")

ax.set_yticks(centers)
ax.set_yticklabels([n for n, *_ in LEVERS], fontsize=10.5)
ax.invert_yaxis()
ax.set_xlim(0, 50)
ax.set_xlabel("Judge accuracy (%). Rows 1-4 and 7 on all 295 questions; rows 5-6 on the A1 subset (answerability)",
              fontsize=9.5)
ax.set_title("Dead-lever sweep: six index-controlled levers do not move accuracy; the ceiling is reached by the fallback",
             fontweight="bold", fontsize=13.5, pad=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)

ax.legend(handles=[Patch(facecolor=GRAY, label="baseline"),
                   Patch(facecolor=DEAD, label="intervention (index-controlled)"),
                   Patch(facecolor=ORA, hatch="//", label="oracle upper bound"),
                   Patch(facecolor=LIVE, label="fallback (the move)"),
                   plt.Line2D([0], [0], ls="--", color=CEILC, label="oracle-leaf ceiling")],
          loc="center", bbox_to_anchor=(0.62, 0.42), fontsize=9, frameon=True, ncol=2)

# per-row footnotes (source of every number)
foot = "Sources (all real logged numbers):\n" + "\n".join(f"{n.split('.')[0]}. {fn}" for n, *_, fn in
                                                          [(l[0], l[4]) for l in LEVERS])
fig.tight_layout(rect=[0, 0.20, 1, 1])
fig.text(0.012, 0.005, foot, fontsize=7.6, style="italic", color="#444", va="bottom")

p = OUT/"dead_levers_before_after.png"
fig.savefig(p, dpi=170, bbox_inches="tight")
print("wrote", p)
for n, b, i, k, fn in LEVERS:
    print(f"  {n:34s} baseline={b:.1f}  intervention={i:.1f}  [{k}]")
