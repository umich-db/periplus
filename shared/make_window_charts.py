"""make_window_charts.py — the "window" comparison figures for a fixed 8-method set.

Three figures, one per metric, all restricted to the same 8 methods and using the
same styling/methodology as shared/make_graphs.py:
  report/accuracy_window.png       — judge accuracy (%), from results/eval/otuputs
  report/token_window.png          — build (hatched, one-time) + answer (solid) tokens (M)
  report/hallucination_window.png  — wrong & fabricated (%), from each method's judged.jsonl

"Ours" is index_v7_window. raptor_abstain is shown as "raptor"; docetl_standard as
"docetl"; naive_gpt as "naive".

Run from bench/:  envs/base/bin/python shared/make_window_charts.py
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)

# method rows: display label, acc%, in_tok, out_tok, build_tok, judged-file tag (dir/stem)
# values straight from results/eval/otuputs (the source of truth used by make_graphs.py)
DATA = [
    # display          acc     in_tok    out_tok    build_tok   judged tag
    ("index_v7_window",   55.8, 1151211,  254425,   2112028,   "wiki_v7_window"),
    ("raptor",            38.0,  325691,  214099,   1375305,   "raptor"),
    ("rag",               35.4, 1468835,  180822,    462400,   "rag"),
    ("doctopus",          52.0, 1749554,  106983,    462400,   "doctopus"),
    ("graphrag",          43.7, 2651541,    8972,   2357445,   "graphrag_local"),
    ("leann",             51.2, 5748813,    5199,         0,   "leann"),
    ("docetl",            57.3,12115945,    9735,         0,   "docetl_standard"),
    ("naive",             52.9,12044706,  137365,         0,   "naive_gpt"),
]
OURS = {"index_v7_window"}

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
COLOR = {row[0]: _PALETTE[i % len(_PALETTE)] for i, row in enumerate(sorted(DATA))}
OURS_COL = "#C44E52"


def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def hbar(labels, vals, title, xlabel, fname, unit="%"):
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(labels) + 1.2))
    y = range(len(labels))
    cols = [OURS_COL if m in OURS else COLOR[m] for m in labels]
    bars = ax.barh(y, vals, color=cols, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in labels]); ax.invert_yaxis()
    for lbl, m in zip(ax.get_yticklabels(), labels):
        if m in OURS:
            lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")
    mx = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + mx * 0.01, b.get_y() + b.get_height() / 2, f"{v:.1f}{unit}", va="center", fontsize=10)
    ax.set_xlim(0, mx * 1.15)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTD / fname, dpi=150); plt.close(fig); print("wrote", OUTD / fname)


# ---------- 1) accuracy window ----------
acc = sorted(DATA, key=lambda r: -r[1])
hbar([r[0] for r in acc], [r[1] for r in acc],
     "Accuracy Window (NarrativeQA, 10 movies, 295 Qs)", "Judge Accuracy (%)", "accuracy_window.png")

# ---------- 2) hallucination window (wrong + fabricated) ----------
def wrong_fab(tag):
    p = pathlib.Path("methods") / tag / "results" / f"{tag}_judged.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return 100 * sum(1 for x in rs if x["verdict"] == "incorrect" and x["halluc"] == "fabricated") / len(rs)

hal = sorted(((r[0], wrong_fab(r[5])) for r in DATA), key=lambda t: t[1])
hbar([m for m, v in hal], [v for m, v in hal],
     "Hallucination Window (wrong + fabricated)", "Wrong & Fabricated (%)", "hallucination_window.png")

# ---------- 3) token window (build hatched, one-time + answer solid) ----------
tok = sorted(DATA, key=lambda r: (r[4] + r[2] + r[3]))
methods = [r[0] for r in tok]
answer = [(r[2] + r[3]) / 1e6 for r in tok]
build = [r[4] / 1e6 for r in tok]
fig, ax = plt.subplots(figsize=(9, 0.62 * len(methods) + 1.4))
y = range(len(methods))
cols = [OURS_COL if m in OURS else COLOR[m] for m in methods]
ax.barh(y, answer, color=cols, edgecolor="white")
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white")
ax.set_yticks(y)
ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in methods]); ax.invert_yaxis()
for lbl, m in zip(ax.get_yticklabels(), methods):
    if m in OURS:
        lbl.set_fontweight("bold")
ax.set_xlabel("Total Tokens (millions)")
ax.set_title("Token Window: Build (hatched, one-time) vs Answer (solid)", fontweight="bold")
mxt = max([a + b for a, b in zip(answer, build)]) or 1
for i, (a, b) in enumerate(zip(answer, build)):
    ax.text(a + b + mxt * 0.01, i, f"ans {a:.1f}M + build {b:.1f}M", va="center", fontsize=8)
ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                   Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")],
          loc="lower right", frameon=False)
ax.set_xlim(0, mxt * 1.28)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD / "token_window.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "token_window.png")

# ---------- 4) tradeoff window (accuracy vs inference tokens) ----------
# Same style as shared/make_tradeoff_v7.py: x = answer/inference tokens (M), y = judge
# accuracy (%); ours (index_v7_window) drawn as a star. Upper-left = cheap & accurate.
plt.rcParams.update({"font.family": "serif", "font.size": 15, "axes.linewidth": 1.2})
fig, ax = plt.subplots(figsize=(6.8, 4.0))
for r in DATA:
    disp, acc_v, in_tok = r[0], r[1], r[2]
    x = in_tok / 1e6
    if disp in OURS:
        ax.scatter(x, acc_v, s=520, marker="*", color=OURS_COL, edgecolor="#333",
                   linewidth=1.3, zorder=4)
        ax.annotate(disp, (x, acc_v), (x + 0.35, acc_v + 2.4), ha="left", va="center",
                    fontsize=14, fontweight="bold")
    else:
        ax.scatter(x, acc_v, s=190, color=COLOR[disp], edgecolor="#333", linewidth=1.3, zorder=3)
        ax.annotate(disp, (x, acc_v), (x + 0.35, acc_v + 1.4), ha="left", va="center", fontsize=13)
ax.set_xlabel("Inference Tokens (Millions)")
ax.set_ylabel("Judge Accuracy (%)")
ax.grid(True, color="#DDD", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(OUTD / "tradeoff_window.png", dpi=200, bbox_inches="tight")
plt.close(fig); print("wrote", OUTD / "tradeoff_window.png")
