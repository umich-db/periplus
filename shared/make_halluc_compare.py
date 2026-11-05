"""Hallucination comparison: abstain vs. non-abstain (guess) answer policies, for
RAPTOR and index v7 side by side. The point: overall fabrication rate is misleading
for abstaining methods -- they look low-fab mostly because they DECLINE to answer.
Answered-only, all four fabricate at a similar rate. Writes report/halluc_compare.png.
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
OUT = BENCH/"report"; OUT.mkdir(exist_ok=True)

FILES = [
    ("raptor\nabstain",   "methods/raptor/results/raptor_judged.jsonl",           "abstain"),
    ("raptor\nnative",    "methods/raptor_perm/results/raptor_perm_judged.jsonl", "guess"),
    ("index_v7\n(strict)", "methods/wiki_v7/results/wiki_v7_judged.jsonl",         "abstain"),
    ("index_v7\nguess",   "methods/wiki_v7_guess/results/wiki_v7_guess_judged.jsonl", "guess"),
]
STRICT, GUESS = "#4C72B0", "#C44E52"   # policy colors


def is_miss(x):
    return x.get("source") in ("miss", "none") or str(x.get("halluc_reason", "")).startswith("abstained")


rows = []
for label, f, policy in FILES:
    r = [json.loads(l) for l in open(BENCH/f)]
    n = len(r); ans = [x for x in r if not is_miss(x)]
    miss = 100*(n-len(ans))/n
    fab_all = 100*sum(1 for x in r if x["halluc"] == "fabricated")/n
    fab_ans = 100*sum(1 for x in ans if x["halluc"] == "fabricated")/len(ans)
    rows.append((label, policy, miss, fab_all, fab_ans))

fig, ax = plt.subplots(figsize=(11, 6))
x = range(len(rows)); w = 0.38
cols = [GUESS if p == "guess" else STRICT for _, p, _, _, _ in rows]

# overall fab (hatched, lighter) vs answered-only fab (solid)
def lighten(c, f=0.5):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

b1 = ax.bar([i-w/2 for i in x], [r[3] for r in rows], w,
            color=[lighten(c) for c in cols], hatch="//", edgecolor="white", label="Fabrication — overall (all Qs)")
b2 = ax.bar([i+w/2 for i in x], [r[4] for r in rows], w,
            color=cols, edgecolor="white", label="Fabrication — answered-only (honest)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.7, f"{b.get_height():.1f}%",
                ha="center", fontsize=10, fontweight="bold")

# miss% annotation under each method
for i, r in enumerate(rows):
    ax.text(i, -6.5, f"abstains {r[2]:.0f}%", ha="center", fontsize=9, color="#555", style="italic")

ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("LLM-judge fabrication rate (%)")
ax.set_ylim(-9, 70)
ax.axhline(0, color="#ccc", lw=0.8)
ax.set_title("Fabrication: abstain vs. best-guess policies (RAPTOR & index v7)", fontweight="bold", fontsize=13)
ax.legend(handles=[Patch(facecolor="#999", hatch="//", label="Overall (diluted by abstentions)"),
                   Patch(facecolor="#999", label="Answered-only (true fabrication rate)"),
                   Patch(facecolor=STRICT, label="strict / abstain policy"),
                   Patch(facecolor=GUESS, label="best-guess policy")],
          loc="upper left", frameon=False, fontsize=9)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(0.01, 0.005,
         "Overall fabrication makes abstaining methods look safe, but that is mostly because they decline to "
         "answer. Answered-only, all four fabricate at ~45-55%: guessing doesn't fabricate more per answer, it "
         "just answers the hard (high-risk) questions instead of abstaining.", fontsize=8, style="italic", color="#444")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT/"halluc_compare.png", dpi=150)
print("wrote", OUT/"halluc_compare.png")
for r in rows:
    print(f"  {r[0].replace(chr(10),' '):20s} abstain={r[2]:5.1f}%  fab_all={r[3]:5.1f}%  fab_answered={r[4]:5.1f}%")
