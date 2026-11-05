"""Guess vs. strict (abstain) tradeoff for index v7 — the figure that decides which
answer policy to ship. Same retrieved context; only the QA node differs:
  strict  = wiki_v7  : answer from digest or abstain ("not found"), + raw-doc fallback
  guess   = wiki_v7_guess : always commit to a best guess, never abstain

Panel 1: overall judge-accuracy vs LLM-fabrication, strict vs guess.
Panel 2: WHY not always guess — split the 295 Qs by what strict did. On the questions
         strict ABSTAINS, guessing recovers some accuracy but fabricates most of the
         time; abstention is concentrated on the highest-fabrication-risk questions.
Writes report/guess_vs_strict.png.
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
ACC, FAB = "#4C72B0", "#C44E52"   # accuracy (blue), fabrication (red)


def judged(m):
    return {x["qid"]: x for x in (json.loads(l) for l in
            open(BENCH/"methods"/m/"results"/f"{m}_judged.jsonl"))}


def is_miss(x):
    return x.get("source") in ("miss", "none") or str(x.get("halluc_reason", "")).startswith("abstained")


def rates(xs):
    n = len(xs) or 1
    return (100*sum(1 for x in xs if x["verdict"] == "correct")/n,
            100*sum(1 for x in xs if x["halluc"] == "fabricated")/n, len(xs))


s, g = judged("wiki_v7"), judged("wiki_v7_guess")
abst = [q for q, x in s.items() if is_miss(x)]
ansd = [q for q, x in s.items() if not is_miss(x)]

s_acc, s_fab, _ = rates(list(s.values()))
g_acc, g_fab, _ = rates(list(g.values()))
gA_acc, gA_fab, nA = rates([g[q] for q in abst if q in g])   # guess on strict's abstentions
gN_acc, gN_fab, nN = rates([g[q] for q in ansd if q in g])   # guess on strict's answers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# --- Panel 1: overall accuracy vs fabrication, strict vs guess ---
groups = ["strict\n(wiki_v7)", "best guess\n(wiki_v7_guess)"]
accs = [s_acc, g_acc]; fabs = [s_fab, g_fab]
x = range(len(groups)); w = 0.38
b1 = ax1.bar([i-w/2 for i in x], accs, w, color=ACC, edgecolor="white", label="Judge accuracy")
b2 = ax1.bar([i+w/2 for i in x], fabs, w, color=FAB, edgecolor="white", label="LLM fabrication")
for bars in (b1, b2):
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.8, f"{b.get_height():.1f}%",
                 ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(list(x)); ax1.set_xticklabels(groups)
ax1.set_ylabel("%"); ax1.set_ylim(0, 80)
ax1.set_title("Guess buys +13.9 pts accuracy for +11.2 pts fabrication", fontweight="bold", fontsize=11)
ax1.legend(frameon=False, loc="upper left")
for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

# --- Panel 2: guess decomposed by what strict did ---
groups2 = [f"Qs strict ANSWERED\n(n={nN})", f"Qs strict ABSTAINED\n(n={nA})"]
accs2 = [gN_acc, gA_acc]; fabs2 = [gN_fab, gA_fab]
x2 = range(len(groups2))
b1 = ax2.bar([i-w/2 for i in x2], accs2, w, color=ACC, edgecolor="white", label="Guess accuracy")
b2 = ax2.bar([i+w/2 for i in x2], fabs2, w, color=FAB, edgecolor="white", label="Guess fabrication")
for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.8, f"{b.get_height():.1f}%",
                 ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(list(x2)); ax2.set_xticklabels(groups2)
ax2.set_ylabel("%"); ax2.set_ylim(0, 80)
ax2.set_title("On strict's abstentions, guessing fabricates 68% of the time",
              fontweight="bold", fontsize=11)
ax2.legend(frameon=False, loc="upper left")
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)

fig.suptitle("Best-guess vs. strict-abstain answer policy (index v7, NarrativeQA, 295 Qs)",
             fontsize=13, fontweight="bold")
fig.text(0.01, 0.005,
         "Same retrieved context; only the QA node differs. Abstention is selective: it suppresses the "
         "questions where the model is most likely to fabricate, so 'always guess' is not free accuracy.",
         fontsize=8, style="italic", color="#444")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(OUT/"guess_vs_strict.png", dpi=150)
print("wrote", OUT/"guess_vs_strict.png")
print(f"strict overall: acc={s_acc:.1f} fab={s_fab:.1f} | guess overall: acc={g_acc:.1f} fab={g_fab:.1f}")
print(f"guess on strict-answered: acc={gN_acc:.1f} fab={gN_fab:.1f} | on strict-abstained: acc={gA_acc:.1f} fab={gA_fab:.1f}")
