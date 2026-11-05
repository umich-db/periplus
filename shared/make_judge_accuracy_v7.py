"""Judge-accuracy horizontal bar chart (v7 numbers), matching the paper figure style.
Numbers from results/eval/otuputs (judge_acc column)."""
import pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("report"); OUT.mkdir(exist_ok=True)

# Match the original figure: serif font, same sizes.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.linewidth": 1.2,
})

# label, value, color  (top -> bottom, as drawn)
rows = [
    ("DocETL",    57.3, "#4C72B0"),
    ("Our Index", 55.9, "#DD8452"),
    ("GPT-5 Raw", 52.9, "#C44E52"),
    ("GraphRAG",  43.7, "#8172B3"),
    ("RAPTOR",    38.0, "#55A868"),
    ("RAG",       35.4, "#937860"),
]

labels = [r[0] for r in rows]
vals   = [r[1] for r in rows]
cols   = [r[2] for r in rows]
y = range(len(rows))

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.barh(y, vals, color=cols, edgecolor="none", height=0.62)

ax.set_yticks(y)
ax.set_yticklabels(labels)
# Bold "Our Index" like the original.
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "Our Index":
        lbl.set_fontweight("bold")
ax.invert_yaxis()

# value labels at bar ends
for i, v in enumerate(vals):
    ax.text(v + 0.8, i, f"{v:.1f}%", va="center", ha="left", fontsize=15)

ax.set_xlabel("Judge Accuracy (%)")
ax.set_xlim(0, 65)
ax.set_xticks(range(0, 70, 10))
ax.set_xticklabels([f"{t}%" for t in range(0, 70, 10)])

# clean spines: keep left + bottom only
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=0)

fig.tight_layout()
p = OUT / "judge_accuracy_v7.png"
fig.savefig(p, dpi=200, bbox_inches="tight")
print("wrote", p.resolve())
