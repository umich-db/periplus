"""Accuracy vs inference-tokens tradeoff scatter, v7 numbers, matching the paper figure style.
x = in_tok (answer/inference tokens, millions), y = judge_acc.  Source: results/eval/otuputs."""
import pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("report"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.linewidth": 1.2,
})

# label, inference_tokens (M), judge_acc (%), color, (label_dx, label_dy, ha)
PTS = [
    ("DocETL",    12.116, 57.3, "#4C72B0", (0.0,  3.6, "center")),
    ("GPT-5 Raw", 12.045, 52.9, "#C44E52", (0.0, -4.2, "center")),
    ("GraphRAG",   2.652, 43.7, "#8172B3", (0.55, 0.0, "left")),
    ("RAPTOR",     0.326, 38.0, "#55A868", (0.55, 1.6, "left")),
    ("RAG",        1.469, 35.4, "#937860", (0.55, -1.4, "left")),
]
STAR = ("Our Index", 1.155, 55.9, "#DD8452", (0.55, 2.6, "left"))

fig, ax = plt.subplots(figsize=(6.4, 3.4))

# circles
for lbl, x, y, col, (dx, dy, ha) in PTS:
    ax.scatter(x, y, s=190, color=col, edgecolor="#333333",
               linewidth=1.3, zorder=3)
    ax.annotate(lbl, (x, y), (x + dx, y + dy), ha=ha, va="center", fontsize=15)

# our method: star
slbl, sx, sy, scol, (sdx, sdy, sha) = STAR
ax.scatter(sx, sy, s=520, marker="*", color=scol, edgecolor="#333333",
           linewidth=1.3, zorder=4)
ax.annotate(slbl, (sx, sy), (sx + sdx, sy + sdy), ha=sha, va="center",
            fontsize=15, fontweight="bold")

ax.set_xlabel("Inference Tokens (Millions)")
ax.set_ylabel("Judge Accuracy (%)")
ax.set_xlim(-0.6, 15.5)
ax.set_ylim(12, 64)
ax.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
ax.set_yticks([20, 40, 60])
ax.set_yticklabels(["20%", "40%", "60%"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="both", color="#DDDDDD", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0)

fig.tight_layout()
p = OUT / "tradeoff_v7.png"
fig.savefig(p, dpi=200, bbox_inches="tight")
print("wrote", p.resolve())
