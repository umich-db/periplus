"""make_bench_token_chart.py — build vs answer token cost for the index (v7_window)
method across the two benchmarks, NarrativeQA and CUAD. Same visual encoding as
shared/make_graphs.py's build_vs_answer.png: solid bar = answer (recurring), hatched
light bar = build (one-time), stacked, horizontal.

Baselines were never run on CUAD, so this is the index method's own cost profile on
each benchmark (not a per-method comparison).

Numbers (all real, index_v7_window):
  NarrativeQA: answer = in 1,151,211 + out 254,425 = 1,405,636 ; build = 2,112,028 (otuputs)
  CUAD:        answer = 2,784,799 (eval_cuad.log, single pass, digests identical on both builds)
               build  = sum of 10 per-contract compile logs = 4,240,766

Run from bench/:  envs/base/bin/python shared/make_bench_token_chart.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)

# benchmark, n_questions, n_docs, answer_tokens, build_tokens, color
BENCH = [
    ("NarrativeQA\n(295 Qs, 10 movies)",   295, 10, 1405636, 2112028, "#C44E52"),
    ("CUAD\n(410 Qs, 10 contracts)",       410, 10, 2784799, 4240766, "#4C72B0"),
]


def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


fig, ax = plt.subplots(figsize=(9, 3.4))
y = range(len(BENCH))
labels = [b[0] for b in BENCH]
answer = [b[3] / 1e6 for b in BENCH]
build = [b[4] / 1e6 for b in BENCH]
cols = [b[5] for b in BENCH]

ax.barh(y, answer, color=cols, edgecolor="white", height=0.6)
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white", height=0.6)
ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("Total Tokens (millions)")
ax.set_title("index (v7_window): Build (hatched, one-time) vs Answer (solid) Tokens",
             fontweight="bold")
mxt = max(a + b for a, b in zip(answer, build)) or 1
for i, (a, b, row) in enumerate(zip(answer, build, BENCH)):
    perq = (row[3]) / row[1]
    ax.text(a + b + mxt * 0.01, i,
            f"ans {a:.1f}M + build {b:.1f}M   ({perq:,.0f} answer tok/Q)", va="center", fontsize=9)
ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                   Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")],
          loc="upper right", frameon=False)
ax.set_xlim(0, mxt * 1.6)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD / "build_vs_answer_bench.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "build_vs_answer_bench.png")
