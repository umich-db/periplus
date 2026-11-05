"""Input-tokens stacked bar (answer + build), v7 numbers, matching the paper figure style.
Numbers from results/eval/otuputs (in_tok = answer, build_tok = build)."""
import pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

OUT = pathlib.Path("report"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.linewidth": 1.2,
})

# label, answer_tokens, build_tokens, base_color
DATA = [
    ("DocETL",    12_115_945,         0, "#4C72B0"),
    ("GPT-5 Raw", 12_044_706,         0, "#C44E52"),
    ("Our Index",  1_155_495, 2_127_028, "#DD8452"),
    ("GraphRAG",   2_651_541, 2_357_445, "#8172B3"),
    ("RAPTOR",       325_691, 1_375_305, "#55A868"),
    ("RAG",        1_468_835,   462_400, "#937860"),
]
# sort largest-to-smallest total (top = biggest), like the original figure
DATA.sort(key=lambda r: r[1] + r[2], reverse=True)


def lighten(c, f=0.5):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def fmt(v):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{round(v/1e3)}K"


fig, ax = plt.subplots(figsize=(6.4, 3.9))
y = range(len(DATA))
XMAX = 16.0  # millions

for i, (lbl, ans, bld, col) in enumerate(DATA):
    a, b = ans / 1e6, bld / 1e6
    ax.barh(i, a, color=col, edgecolor="none", height=0.62)
    if b > 0:
        ax.barh(i, b, left=a, color=lighten(col), edgecolor="none", height=0.62)
    # inner white labels on wide-enough segments
    if a > 0.13 * XMAX:
        ax.text(a / 2, i, fmt(ans), va="center", ha="center",
                color="white", fontweight="bold", fontsize=13)
    if b > 0.13 * XMAX:
        ax.text(a + b / 2, i, fmt(bld), va="center", ha="center",
                color="white", fontweight="bold", fontsize=13)
    # outer total label
    ax.text(a + b + 0.2, i, fmt(ans + bld), va="center", ha="left", fontsize=15)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in DATA])
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "Our Index":
        lbl.set_fontweight("bold")
ax.invert_yaxis()

ax.set_xlabel("Input Tokens (Millions)")
ax.set_xlim(0, XMAX)
ax.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=0)

# neutral gray legend, answer (dark) vs build (light)
ax.legend(handles=[Patch(facecolor="#555555", label="Answer (295 Qs)"),
                   Patch(facecolor="#BBBBBB", label="Build (one-time)")],
          loc="lower right", frameon=True, fontsize=13, handlelength=1.4)

fig.tight_layout()
p = OUT / "input_tokens_v7.png"
fig.savefig(p, dpi=200, bbox_inches="tight")
print("wrote", p.resolve())
