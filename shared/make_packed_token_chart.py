"""Token cost: v8 packed (4 routing modes) vs v7.
  Panel 1: per-question answer tokens (input + output), recurring.
  Panel 2: one-time build tokens.
Reads results/eval/wiki_v8_<mode>_summary.json and wiki_v7_summary.json.
Style matches shared/make_cost_chart.py / make_graphs.py. Writes report/token_cost_v8.png.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH = _BENCH
EVAL = BENCH / "results" / "eval"
OUTD = BENCH / "report"; OUTD.mkdir(parents=True, exist_ok=True)

OURS = "#C44E52"; OURS_L = "#E89A9C"      # our method (v8 keyword)
OTH = "#4C72B0"; OTH_L = "#9DB8DC"        # everything else


def load(name):
    return json.loads((EVAL / name).read_text())


# All v8 modes share the SAME packed indexes, so build tokens are identical across
# modes; use the one canonical number from build_costs.json (per-mode summary
# build_tokens drifted because lazy overviews were generated during answering).
V8_BUILD = sum(v["build_tokens_total"] for v in
               json.loads((BENCH / "methods" / "wiki_v8" / "build_costs.json").read_text()).values())

# rows: (label, in/q, out/q, build_total, is_ours)
rows = []
for mode in ["keyword", "cosine", "hybrid", "llm"]:
    s = load(f"wiki_v8_{mode}_summary.json"); n = s["n_questions"]
    rows.append((f"v8 {mode}", s["total_input_tokens"] / n, s["total_output_tokens"] / n,
                 V8_BUILD, mode == "keyword"))
v7 = load("wiki_v7_summary.json"); n7 = v7["n_questions"]
rows.append(("v7", v7["total_input_tokens"] / n7, v7["total_output_tokens"] / n7,
             v7["build_tokens"], False))

rows.sort(key=lambda r: r[1] + r[2])   # by answer tokens/q
labels = [r[0] for r in rows]
ci = [r[1] for r in rows]; co = [r[2] for r in rows]
build = [r[3] / 1e6 for r in rows]
ours = [r[4] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
y = range(len(rows))

# --- Panel 1: per-question answer tokens (input + output) ---
ax1.barh(y, ci, color=[OURS if o else OTH for o in ours], edgecolor="white")
ax1.barh(y, co, left=ci, color=[OURS_L if o else OTH_L for o in ours],
         hatch="//", edgecolor="white")
ax1.set_yticks(y); ax1.set_yticklabels([f"{l}  ◄ours" if o else l for l, o in zip(labels, ours)])
ax1.invert_yaxis()
for i, r in enumerate(rows):
    t = ci[i] + co[i]
    ax1.text(t + 60, i, f"{t:,.0f}", va="center", fontsize=9,
             fontweight="bold" if r[4] else "normal")
ax1.set_xlabel("Answer tokens per question (input + output)")
ax1.set_xlim(0, max(ci[i] + co[i] for i in y) * 1.15)
ax1.set_title("Per-Question Answer Cost (recurring)", fontweight="bold")
ax1.legend(handles=[Patch(facecolor=OTH, label="input tokens"),
                    Patch(facecolor=OTH_L, hatch="//", label="output tokens"),
                    Patch(facecolor=OURS, label="v8 keyword (ships)")],
           loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

# --- Panel 2: one-time build tokens ---
ax2.barh(y, build, color=[OURS if o else OTH for o in ours], edgecolor="white")
ax2.set_yticks(y); ax2.set_yticklabels(labels); ax2.invert_yaxis()
for i, b in enumerate(build):
    ax2.text(b + max(build) * 0.01, i, f"{b:.2f}M", va="center", fontsize=9)
ax2.set_xlabel("Build tokens (one-time, whole corpus)")
ax2.set_xlim(0, max(build) * 1.18)
ax2.set_title("One-Time Build Cost", fontweight="bold")
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

fig.suptitle("Token Cost: v8 packed (4 routing modes) vs v7 — NarrativeQA, 295 Qs",
             fontsize=13, fontweight="bold")
fig.text(0.01, 0.005,
         "v8 keyword matches v7 on per-question answer tokens (~4.8k) and accuracy (54.6% vs 55.9%); "
         "llm mode costs +46% tokens for lower accuracy. v8's pack step makes build ~65% pricier "
         "(one-time). The routing-work win (nodes_scored) is scan operations, not tokens.",
         fontsize=7.5, style="italic", color="#444", va="bottom")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUTD / "token_cost_v8.png", dpi=150)
print("wrote", OUTD / "token_cost_v8.png")
for l, a, b in zip(labels, [ci[i] + co[i] for i in y], build):
    print(f"  {l:12s} answer/q={a:,.0f}  build={b:.2f}M")
