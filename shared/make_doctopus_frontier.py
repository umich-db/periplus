"""Cost-vs-accuracy plot with the Doctopus budget frontier overlaid on the
other methods' single operating points.

Other methods: one (answer-tokens, judge-accuracy) point each, parsed from
results/eval/otuputs (the bench's source of truth, 295 Qs).
Doctopus: the full budget sweep from methods/doctopus/output/curve.json. Its
accuracy is measured on the 177-question held-out test split; its answer-token
cost is scaled to a 295-Q equivalent (x295/177) so it sits on the same x-axis.

Writes report/cost_vs_accuracy_doctopus.png. Non-destructive (does not touch
the existing make_graphs.py outputs).
"""
import json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)
SRC = pathlib.Path("results/eval/otuputs")
CURVE = pathlib.Path("methods/doctopus/output/curve.json")

# --- single-point methods from otuputs ---
rows = []
for ln in SRC.read_text().splitlines():
    s = ln.strip()
    if not s or s.startswith(("method", "-")):
        continue
    toks = s.split()
    if len(toks) < 10:
        continue
    def num(x): return float(re.sub(r"[^0-9.\-]+", "", x) or 0)
    try:
        rows.append({"method": toks[0], "acc": num(toks[1]),
                     "in_tok": num(toks[5]), "out_tok": num(toks[6])})
    except Exception:
        continue

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
COLOR = {r["method"]: _PALETTE[i % len(_PALETTE)]
         for i, r in enumerate(sorted(rows, key=lambda r: r["method"]))}

# --- Doctopus frontier ---
curve = json.load(open(CURVE))
N_TEST = curve[0]["n_test"]
N_FULL = 295
scale = N_FULL / N_TEST
fx = [c["answer_tokens_used"] * scale / 1e6 for c in curve]   # 295-Q-equiv, millions
fy = [c["judge_accuracy"] * 100 for c in curve]

fig, ax = plt.subplots(figsize=(9.5, 6.5))

# other methods as points
for r in rows:
    x = (r["in_tok"] + r["out_tok"]) / 1e6
    ax.scatter(x, r["acc"], s=150, color=COLOR[r["method"]], edgecolor="white", zorder=3)
    ax.annotate(r["method"], (x, r["acc"]), xytext=(6, 5),
                textcoords="offset points", fontsize=9)

# doctopus frontier as a connected line
ax.plot(fx, fy, "-o", color="#111111", lw=2.2, ms=6, zorder=4,
        label="doctopus (budget frontier)")
ax.annotate("doctopus\nfrontier", (fx[-1], fy[-1]), xytext=(8, -4),
            textcoords="offset points", fontsize=9, fontweight="bold")
# mark the knee (cheapest within 1% of its own peak)
peak = max(fy)
knee_i = min((i for i, y in enumerate(fy) if y >= peak - 1.0), key=lambda i: fx[i])
ax.scatter([fx[knee_i]], [fy[knee_i]], s=180, facecolor="none",
           edgecolor="#C44E52", lw=2.2, zorder=5)

ax.set_xscale("log")
ax.set_xlabel("Answer Token Cost (millions, log scale)  -  doctopus scaled to 295-Q equiv")
ax.set_ylabel("Judge Accuracy (%)")
ax.set_title("Cost vs Accuracy: methods (points) vs Doctopus budget frontier (line)",
             fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.text(0.01, 0.005,
         f"doctopus accuracy measured on {N_TEST}-Q held-out test split; "
         f"cost x{scale:.2f} for 295-Q equivalence. Calibration is a one-time cost (not shown).",
         fontsize=7, color="#666")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUTD / "cost_vs_accuracy_doctopus.png", dpi=150)
plt.close(fig)
print("wrote", OUTD / "cost_vs_accuracy_doctopus.png")
print(f"frontier: {len(curve)} pts, {fx[0]:.2f}M->{fx[-1]:.2f}M tok, "
      f"{fy[0]:.1f}%->{fy[-1]:.1f}%; knee @ {fx[knee_i]:.2f}M / {fy[knee_i]:.1f}%")
