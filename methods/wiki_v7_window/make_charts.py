"""Charts for the v7_window gate: accuracy, hallucination-under-absence, and token cost.

Design follows the dataviz method: form chosen by the data's job, categorical hues assigned in a
fixed order from the Okabe-Ito colorblind-safe palette (never cycled), thin marks with a 2px surface
gap, direct value labels, recessive axes, one y-scale per panel, a legend whenever two series appear.
Writes three PNGs to results/.

Vocabulary: root, category, subject, field, source document.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = pathlib.Path(__file__).resolve().parent
R = HERE / "results"

# Okabe-Ito, CVD-safe by construction (assigned in fixed order, not cycled)
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8f98"
INK, MUTED, SURF, GRID = "#1a1d21", "#6b7280", "#ffffff", "#e6e8eb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
    "axes.linewidth": 1.0, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _bars(ax, xs, ys, color, width, offset=0.0, labels=True, fmt="{:.1f}%"):
    b = ax.bar([x + offset for x in xs], ys, width * 0.92, color=color, zorder=3,
               edgecolor=SURF, linewidth=1.5)  # ~2px surface gap between adjacent fills
    if labels:
        for x, y in zip(xs, ys):
            ax.text(x + offset, y + max(ys) * 0.015, fmt.format(y), ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
    return b


def accuracy_chart():
    nqa = json.load(open(R / "window_gate_nqa.json"))
    cuad = json.load(open(R / "window_gate_cuad.json"))
    dq = nqa["data_quality"]
    groups = [
        ("NQA new-20\n(clean)", dq["new20_excl_misaligned"], 62.3),
        ("CUAD\noverall", cuad["overall"]["balanced"], 53.2),
    ]
    xs = list(range(len(groups)))
    bal = [g[1] for g in groups]
    win = [g[2] for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    _style(ax)
    w = 0.34
    _bars(ax, xs, bal, BLUE, w, -w / 2)
    _bars(ax, xs, win, ORANGE, w, +w / 2)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.axvline(0.5, color=GRID, linewidth=1.2)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=BLUE), Patch(color=ORANGE)], ["v7_balanced", "v7_window"],
              frameon=False, loc="upper right", ncol=2)
    ax.set_title("Judge accuracy: v7_balanced vs v7_window", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(R / "chart_accuracy.png", dpi=150)
    plt.close(fig)


def hallucination_chart():
    cuad = json.load(open(R / "window_gate_cuad.json"))
    n = cuad["no_answer_abstention"]["n"]
    fab = cuad["fabrication_under_absence"]["balanced"]
    abst = n - fab
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    # panel 1: the absence bucket outcome (one stacked bar)
    _style(ax1)
    ax1.bar([0], [abst], 0.5, color=GREEN, zorder=3, edgecolor=SURF, linewidth=1.5, label="abstained (correct)")
    ax1.bar([0], [fab], 0.5, bottom=[abst], color=VERM, zorder=3, edgecolor=SURF, linewidth=1.5,
            label="fabricated clause text")
    ax1.text(0, abst / 2, f"{abst}\n{100*abst/n:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.text(0, abst + fab / 2, f"{fab}\n{100*fab/n:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax1.set_xticks([0]); ax1.set_xticklabels([f"{n} no-answer\nclauses"])
    ax1.set_ylabel("questions"); ax1.set_ylim(0, n * 1.05)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    ax1.set_title("Behaviour under absence", fontweight="bold", fontsize=11)
    # panel 2: per-contract fabrication count
    pc = cuad["per_contract"]
    recs = [json.loads(l) for l in (R / "cuad_eval_records.jsonl").read_text().splitlines() if l.strip()]
    from collections import defaultdict
    fabc, noac = defaultdict(int), defaultdict(int)
    title = {}
    for r in recs:
        if not r["has_answer"]:
            noac[r["movie"]] += 1
            if r["balanced"]["kind"] == "fabricated_under_absence":
                fabc[r["movie"]] += 1
        title[r["movie"]] = r["contract"]
    order = sorted(noac, key=lambda c: -fabc[c])
    _style(ax2)
    ys = list(range(len(order)))[::-1]
    ax2.barh(ys, [noac[c] for c in order], 0.72, color=GRID, zorder=2, label="no-answer clauses")
    ax2.barh(ys, [fabc[c] for c in order], 0.72, color=VERM, zorder=3, edgecolor=SURF, linewidth=1.2,
             label="fabricated")
    for y, c in zip(ys, order):
        ax2.text(noac[c] + 0.3, y, f"{fabc[c]}/{noac[c]}", va="center", fontsize=8.5, color=INK)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([title[c].split("_")[0][:16] for c in order], fontsize=8.5)
    ax2.set_xlabel("no-answer clauses per contract")
    ax2.legend(frameon=False, loc="lower right", fontsize=9)
    ax2.set_title("Fabrication under absence, per contract", fontweight="bold", fontsize=11)
    fig.suptitle(f"Hallucination under absence (CUAD): {fab}/{n} no-answer clauses got fabricated text "
                 f"({100*fab/n:.0f}%)", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(R / "chart_hallucination.png", dpi=150)
    plt.close(fig)


def token_chart():
    nqa = json.load(open(R / "window_gate_nqa.json"))
    cuad = json.load(open(R / "window_gate_cuad.json"))
    # build tokens measured from compile logs (new content only; old-10 reused)
    NQA_BUILD, CUAD_BUILD = 6_415_821, 4_240_766
    phases = ["build\n(new docs)", "balance+window\npass", "answer\n(both builds)", "judge"]
    nqa_v = [NQA_BUILD, 0, nqa["answer_tokens_total"], nqa["judge_tokens_total"]]
    cuad_v = [CUAD_BUILD, 0, cuad["answer_tokens_total"], cuad["judge_tokens_total"]]
    xs = list(range(len(phases)))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _style(ax)
    w = 0.4
    _bars(ax, xs, [v / 1e6 for v in nqa_v], BLUE, w, -w / 2, fmt="{:.2f}M")
    _bars(ax, xs, [v / 1e6 for v in cuad_v], ORANGE, w, +w / 2, fmt="{:.2f}M")
    ax.set_xticks(xs); ax.set_xticklabels(phases, fontsize=9.5)
    ax.set_ylabel("tokens (millions)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}M"))
    from matplotlib.patches import Patch
    ax.legend([Patch(color=BLUE), Patch(color=ORANGE)],
              [f"NarrativeQA (30 movies, 885 q) - {sum(nqa_v)/1e6:.1f}M total",
               f"CUAD (10 contracts, 410 q) - {sum(cuad_v)/1e6:.1f}M total"],
              frameon=False, loc="upper right")
    ax.set_title("Token cost by phase (gpt-5)", fontweight="bold", pad=12)
    fig.text(0.5, 0.005, "The balance and window passes cost 0 LLM tokens (local all-mpnet embeddings). "
             "Answer tokens cover BOTH builds; identical digests shared one call, so this is not doubled.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(R / "chart_tokens.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    accuracy_chart()
    hallucination_chart()
    token_chart()
    print("wrote chart_accuracy.png, chart_hallucination.png, chart_tokens.png ->", R)


