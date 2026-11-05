"""Phase 3 centerpiece: per-query routing work vs document size.

Reads results/eval/packed_probe/wiki_v8.json and plots, per movie, the median
number of child summaries scored per query:
  - packed (v8) routing: bounded, about f times depth, flat as documents grow
  - flat index traversal: v7's manifest scanned to route to a page, which grows
    with document size (the O(N) baseline)
v7 as actually shipped scores ~0 (its consolidated overview overflows the 12k
budget, so it reads one page and never scans the manifest); noted in the caption.

Style matches shared/make_graphs.py. Writes report/nodes_scored_vs_size.png.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = _BENCH
PROBE = json.loads((BENCH / "results" / "eval" / "packed_probe" / "wiki_v8.json").read_text())
OUTD = BENCH / "report"; OUTD.mkdir(parents=True, exist_ok=True)

OURS = "#C44E52"      # matches make_graphs palette (our method)
FLAT = "#4C72B0"

pm = PROBE["per_movie"]
movies = sorted(pm, key=lambda s: pm[s]["leaf_count"] or 0)
x = [pm[s]["leaf_count"] for s in movies]
v8 = [pm[s]["v8_nodes_scored_med"] for s in movies]
flat = [pm[s]["v7_flat_nodes_scored_med"] for s in movies]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, flat, "-o", color=FLAT, markersize=8, markeredgecolor="white",
        linewidth=2, label="flat index traversal (scan to route)", zorder=3)
ax.plot(x, v8, "-*", color=OURS, markersize=15, markeredgecolor="white",
        linewidth=2, label="packed v8 routing  ◄ ours", zorder=4)

ax.set_yscale("log")
ax.set_xlabel("Leaves (subjects) per document")
ax.set_ylabel("Child summaries scored per query (median, log scale)")
ax.set_title("Per-Query Routing Work vs Document Size", fontweight="bold")

# annotate the two regimes
ax.annotate(f"flat scan: {min(flat):,.0f} – {max(flat):,.0f}\n(grows with document size)",
            (x[-1], flat[-1]), xytext=(-10, 20), textcoords="offset points",
            fontsize=9, color=FLAT, ha="right")
ax.annotate(f"packed: ~{sum(v8)/len(v8):.0f} scored, flat\n(about f × depth, f={PROBE.get('fanout', 33)})",
            (x[-1], v8[-1]), xytext=(-10, -34), textcoords="offset points",
            fontsize=9, color=OURS, ha="right", fontweight="bold")

ax.grid(True, which="both", ls=":", alpha=0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="center left", frameon=False, fontsize=10)

fig.text(0.01, 0.005,
         "v7 as shipped scores ~0: its consolidated overview (13–15k chars) overflows the "
         "12k context budget, so it reads that one page and never scans the manifest.\n"
         "The flat-index line is that same manifest scanned WITHOUT the overview shortcut — the "
         "O(N) routing cost a flat index incurs. Median over each movie's questions, 295 total.",
         fontsize=7.5, style="italic", color="#444", va="bottom")

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUTD / "nodes_scored_vs_size.png", dpi=150)
print("wrote", OUTD / "nodes_scored_vs_size.png")

# quick console summary
ratio = (sum(flat) / len(flat)) / (sum(v8) / len(v8))
print(f"mean flat={sum(flat)/len(flat):,.0f}  mean v8={sum(v8)/len(v8):.0f}  "
      f"routing work reduced ~{ratio:.0f}x")
