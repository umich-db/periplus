"""Phase 7 centerpiece: per-query routing work vs document size, index vs v7.

Reads results/eval/nodes_probe/index.json and plots, per screenplay, the median number of
routing summaries scored per question:
  - index (ours): bounded, about f times depth, flat or logarithmic as documents grow
  - v7 flat index: the manifest it scans to route to a page, which climbs with document size

x axis is the index method's leaf count per screenplay (document-size proxy); the 10
screenplays span the spread. v7's number is derived read-only from its own stored index
shape (its traverse loop replayed with no LLM calls), labelled as such.

Style matches shared/make_packed_chart.py. Writes report/index_nodes_scored_vs_size.png.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = _BENCH
PROBE = json.loads((BENCH/"results"/"eval"/"nodes_probe"/"index.json").read_text())
OUTD = BENCH/"report"; OUTD.mkdir(parents=True, exist_ok=True)

OURS = "#C44E52"      # our method (matches make_packed_chart palette)
FLAT = "#4C72B0"

pm = PROBE["per_movie"]
movies = sorted(pm, key=lambda s: pm[s]["leaf_count"] or 0)
x = [pm[s]["leaf_count"] for s in movies]
idx = [pm[s]["index_nodes_scored_med"] for s in movies]
flat = [pm[s]["v7_nodes_scored_med"] for s in movies]
f = PROBE.get("fanout", 33)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, flat, "-o", color=FLAT, markersize=8, markeredgecolor="white",
        linewidth=2, label="v7 flat index (scan to route)", zorder=3)
ax.plot(x, idx, "-*", color=OURS, markersize=15, markeredgecolor="white",
        linewidth=2, label="index bounded routing  ◄ ours", zorder=4)

ax.set_yscale("log")
ax.set_xlabel("Leaves (subjects) per document")
ax.set_ylabel("Routing summaries scored per query (median, log scale)")
ax.set_title("Per-Query Routing Work vs Document Size", fontweight="bold")

ax.annotate(f"v7 flat scan: {min(flat):,.0f} – {max(flat):,.0f}\n(climbs with document size)",
            (x[-1], flat[-1]), xytext=(-14, -46), textcoords="offset points",
            fontsize=9, color=FLAT, ha="right")
ax.annotate(f"index: ~{sum(idx)/len(idx):.0f} scored, flat\n(about f × depth, f={f})",
            (x[len(x)//2], idx[len(idx)//2]), xytext=(0, 26), textcoords="offset points",
            fontsize=9, color=OURS, ha="center", fontweight="bold")

ax.grid(True, which="both", ls=":", alpha=0.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="center left", frameon=False, fontsize=10)

fig.text(0.5, 0.015,
         "index scores at most f child summaries per descent, bounded to budget_a x depth + "
         "budget_b moves, so its work tracks tree depth (log of leaf count).\nv7's number is "
         "its manifest scanned to route, derived read-only from its stored index (no LLM calls). "
         "Median per screenplay, 295 questions.",
         fontsize=8, style="italic", color="#444", va="bottom", ha="center")

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUTD/"index_nodes_scored_vs_size.png", dpi=150)
print("wrote", OUTD/"index_nodes_scored_vs_size.png")

ratio = (sum(flat)/len(flat))/(sum(idx)/len(idx))
print(f"mean v7_flat={sum(flat)/len(flat):,.0f}  mean index={sum(idx)/len(idx):.0f}  "
      f"routing work reduced ~{ratio:.0f}x")
