"""make_v7_balanced_chart.py — charts for the v7_balanced (bounded-fanout B+) pass.

Two figures, both from the real artifacts produced by the wiki_v7_balanced method:
  report/v7_balanced_fanout.png  — the structural win: per-doc max fanout, v7 vs
      v7_balanced, with the f=33 bound. v7 has one unbounded node (73); balanced
      caps every node at f.
  report/v7_balanced_gate.png    — the accuracy gate: v7 vs v7_balanced judge
      accuracy with the McNemar result, showing balancing is accuracy-neutral.

Run from bench/:  envs/base/bin/python shared/make_v7_balanced_chart.py
"""
import json, math, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH = pathlib.Path(__file__).resolve().parent.parent
OUTD = BENCH / "report"; OUTD.mkdir(parents=True, exist_ok=True)
BAL = BENCH / "methods" / "wiki_v7_balanced"

C, S = 4000, 120
F = C // S                 # 33
OCC_LO = math.ceil(F / 2)  # 17

V7_COL = "#8C8C8C"      # baseline grey
BAL_COL = "#C44E52"     # ours red
BOUND_COL = "#4C72B0"

# ---------- load artifacts ----------
reports = json.loads((BAL / "output" / "digests_v7_balanced" / "_balance_summary.json").read_text())
nav = json.loads((BAL / "results" / "nav_probe.json").read_text())


def _verdicts(path):
    v = {}
    for l in pathlib.Path(path).read_text().splitlines():
        if l.strip():
            r = json.loads(l); v[r["qid"]] = 1 if r.get("verdict") == "correct" else 0
    return v


v7v = _verdicts(BENCH / "methods" / "wiki_v7" / "results" / "wiki_v7_judged.jsonl")
balv = _verdicts(BAL / "results" / "wiki_v7_balanced_judged.jsonl")
qids = sorted(set(v7v) & set(balv))
n = len(qids)
v7_correct = sum(v7v[q] for q in qids)
bal_correct = sum(balv[q] for q in qids)
b = sum(1 for q in qids if v7v[q] == 1 and balv[q] == 0)
c = sum(1 for q in qids if v7v[q] == 0 and balv[q] == 1)


def mcnemar_exact(b, c):
    nn = b + c
    if nn == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(nn, i) for i in range(k + 1)) * 0.5 ** nn)


p = mcnemar_exact(b, c)

# ---------- figure 1: fanout (the structural win) ----------
def short(doc):
    return doc.replace("__", " (").replace("_", " ")[:24]


docs = sorted(reports, key=lambda r: -r["max_fanout_before"])
labels = [short(r["doc"]) for r in docs]
before = [r["max_fanout_before"] for r in docs]
after = [r["max_fanout_after"] for r in docs]

fig, ax = plt.subplots(figsize=(9, 0.52 * len(docs) + 1.8))
y = range(len(docs))
h = 0.38
ax.barh([i + h / 2 for i in y], before, height=h, color=V7_COL, edgecolor="white", label="v7 (unbounded)")
ax.barh([i - h / 2 for i in y], after, height=h, color=BAL_COL, edgecolor="white", label="v7_balanced (≤ f)")
ax.axvline(F, color=BOUND_COL, ls="--", lw=1.6)
ax.text(F + 0.6, len(docs) - 0.4, f"f = C/s = {F}", color=BOUND_COL, fontsize=9, fontweight="bold", va="top")
ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("Max fanout (widest node's child count)")
ax.set_title("Structural win: bounded fanout, v7 vs v7_balanced", fontweight="bold")
for i, (bf, af) in enumerate(zip(before, after)):
    ax.text(bf + 0.6, i + h / 2, str(bf), va="center", fontsize=8)
    ax.text(af + 0.6, i - h / 2, str(af), va="center", fontsize=8)
ax.set_xlim(0, max(before) * 1.16)
ax.legend(loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.text(0.01, 0.01,
         f"v7's list-subtree grouping already caps most nodes at MAX_CHILDREN=24; one dict-derived node "
         f"(2012 synopsis, 73) exceeded f.\nBalancing split it into 3 groups of [25, 24, 24], every split "
         f"node in the occupancy window [{OCC_LO}, {F}]. No subject content changed.",
         fontsize=8, style="italic", color="#444", va="bottom")
fig.tight_layout(); fig.subplots_adjust(bottom=0.16)
fig.savefig(OUTD / "v7_balanced_fanout.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "v7_balanced_fanout.png")

# ---------- figure 2: accuracy gate ----------
fig, ax = plt.subplots(figsize=(6.6, 5.2))
accs = [100 * v7_correct / n, 100 * bal_correct / n]
bars = ax.bar(["v7", "v7_balanced"], accs, color=[V7_COL, BAL_COL], edgecolor="white", width=0.55)
for bar, a, cor in zip(bars, accs, [v7_correct, bal_correct]):
    ax.text(bar.get_x() + bar.get_width() / 2, a + 0.6, f"{a:.1f}%\n({cor}/{n})", ha="center", fontsize=10)
ax.set_ylabel("Judge accuracy (%)")
ax.set_ylim(0, max(accs) * 1.25)
ax.set_title("Accuracy gate: balancing holds v7's accuracy", fontweight="bold")
sig = "NOT significant" if p > 0.05 else "significant"
ax.text(0.5, 0.90,
        f"McNemar exact:  b={b} (v7 right, bal wrong),  c={c} (v7 wrong, bal right)\n"
        f"discordant={b + c},  two-sided p = {p:.3f}  →  {sig} at α=0.05",
        transform=ax.transAxes, ha="center", fontsize=9,
        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#ccc"))
ax.text(0.5, 0.72,
        f"digest byte-identical on both trees for {nav['n_digest_identical']}/{nav['n_questions']} questions",
        transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#555")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD / "v7_balanced_gate.png", dpi=150); plt.close(fig)
print("wrote", OUTD / "v7_balanced_gate.png")
