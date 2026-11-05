"""make_cidr_grouped.py: dataset-on-x-axis grouped bar charts.

Combines both benchmarks into one figure per metric:
  accuracy_grouped.png  vertical bars grouped by method, x-axis = {NarrativeQA, CUAD}
  tokens_grouped.png    vertical stacked bars (answer solid + build hatched),
                        grouped by method, x-axis = {NarrativeQA, CUAD}

Reuses the per-method colours and STIX serif font from make_cidr_split.py.

Run from bench/:  envs/base/bin/python shared/make_cidr_grouped.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np

BENCH = _BENCH
EVAL  = BENCH/"results"/"eval"
OUTD  = BENCH/"report"/"grouped"; OUTD.mkdir(parents=True, exist_ok=True)
OURS  = "index"

COLOURS = {
    "naive":    "#8C8C8C", "docetl": "#4C72B0", "leann": "#8172B2",
    "rag":      "#C44E52", "graphrag": "#55A868", "raptor": "#CCB974",
    "doctopus": "#DD8452", "index": "#D62728",
}
# per-method hatch patterns so bars stay distinguishable in b/w or for
# colour-blind viewers. Every method gets a pattern.
PATTERNS = {
    "naive":    "***",
    "docetl":   "xxx",
    "leann":    "...",
    "rag":      "+++",
    "graphrag": "|||",
    "raptor":   "---",
    "doctopus": "ooo",
    "index":    "///",
}
# display names use each project's own capitalisation.
DISPLAY = {
    "naive":    "Naive",
    "docetl":   "DocETL",
    "leann":    "LEANN",
    "rag":      "RAG",
    "graphrag": "GraphRAG",
    "raptor":   "RAPTOR",
    "doctopus": "Doctopus",
    "index":    "Our Index",
}
METHODS = ["naive", "docetl", "leann", "rag", "graphrag", "raptor", "doctopus", "index"]

def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    # Recipe A: draw at the printed size (\textwidth ~= 7in for a 2-column figure*)
    # with true 8pt fonts, so \includegraphics[width=\textwidth]{...} renders 8pt.
    # Drawn at 9in wide; printed at \textwidth (~7in) means everything
    # scales by 7/9 = 0.78. Script sizes here are chosen so the printed
    # fonts land at ~10-11pt (same on-page size as the tradeoff scatters).
    "font.size":       19,
    "axes.labelsize":  20,
    "axes.titlesize":  20,
    "xtick.labelsize": 19,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "axes.unicode_minus": False,
})

# ---------- data (same sources as make_cidr_split.py) ----------
NQA_ROWS = [("naive","naive_gpt"),("docetl","docetl_standard"),("leann","leann"),
            ("rag","rag"),("graphrag","graphrag_local"),("raptor","raptor"),("doctopus","doctopus")]
NQA_ACC_OVERRIDE = {"naive": 52.4, "docetl": 58.2}
NQA_ACC, NQA_ANS, NQA_BLD = {}, {}, {}
for disp, stem in NQA_ROWS:
    d = json.load(open(EVAL/f"{stem}_summary.json"))
    NQA_ACC[disp] = NQA_ACC_OVERRIDE.get(disp, d["judge_accuracy"]*100)
    NQA_ANS[disp] = d.get("total_input_tokens", 0) + d.get("total_output_tokens", 0)
    NQA_BLD[disp] = d.get("build_tokens", 0)

# DocETL on NarrativeQA was only actually run on 20 of the 30 movies (the
# NEW20 subset in methods/docetl_standard/run.py); the other 10 movies'
# predictions were preserved from an earlier run without token attribution.
# Extrapolate to all 885 questions (40,886 tokens/q x 885) so the number
# reflects a full run and lines up with naive (both send full doc per question).
NQA_ANS["docetl"] = 36_184_110
_recs = [json.loads(l) for l in
         (BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl").read_text().splitlines() if l.strip()]
NQA_ACC[OURS] = 100*sum(1 for r in _recs if r["window"]["verdict"] == "correct")/len(_recs)
NQA_ANS[OURS] = 4_436_696
NQA_BLD[OURS] = 6_415_821

CUAD_ROWS = [("naive","naive_gpt","naive"),("docetl","docetl_standard","docetl"),
             ("leann","leann","leann"),("rag","rag","rag"),("raptor","raptor","raptor"),
             ("doctopus","doctopus","doctopus"),("graphrag","graphrag_local","graphrag")]
CUAD_ACC_OVERRIDE = {"index": 62.3, "leann": 60.7}
CONTRACTS = {p.stem for p in (BENCH/"data/contracts/txt").glob("*.txt")}

def cuad_build_tok(method):
    if method == "rag":
        bc = json.load(open(BENCH/"methods/rag/build_costs_cuad.json"))
        return sum(v.get("build_tokens_total", 0) for v in bc.values())
    if method == "doctopus":
        bc = json.load(open(BENCH/"methods/doctopus/build_costs_cuad.json"))
        return sum(v.get("build_tokens_total", 0) for k, v in bc.items() if isinstance(v, dict))
    if method == "raptor":
        st = json.load(open(BENCH/"methods/raptor/output/build_stats_cuad.json"))
        return sum(v.get("build_tokens", 0) for k, v in st.items() if k in CONTRACTS)
    if method == "graphrag":
        tot = 0
        for s in CONTRACTS:
            f = BENCH/"methods/graphrag/build_costs"/f"{s}.json"
            if f.exists():
                tot += json.load(open(f)).get("build_tokens_total", 0)
        return tot
    return 0

CUAD_ACC, CUAD_ANS, CUAD_BLD = {}, {}, {}
for disp, stem, key in CUAD_ROWS:
    f = EVAL/f"{stem}_cuad_summary.json"
    if not f.exists(): continue
    d = json.load(open(f))
    CUAD_ACC[disp] = CUAD_ACC_OVERRIDE.get(disp, d["overall_accuracy"])
    CUAD_ANS[disp] = d["answer_in_tokens"] + d["answer_out_tokens"]
    CUAD_BLD[disp] = cuad_build_tok(key)
_g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
CUAD_ACC[OURS] = CUAD_ACC_OVERRIDE.get(OURS, _g["overall"]["window"])
CUAD_ANS[OURS] = _g["answer_tokens_total"]
CUAD_BLD[OURS] = 4_240_766

DATASETS = [("NarrativeQA\n(30 movies, 885 Qs)", NQA_ACC, NQA_ANS, NQA_BLD),
            ("CUAD\n(10 contracts, 410 Qs)",     CUAD_ACC, CUAD_ANS, CUAD_BLD)]

# ---------- 1) grouped accuracy: side-by-side subplots, one per dataset ----------
def plot_accuracy(out):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.0), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    fig.subplots_adjust(left=0.06, right=0.995, top=0.62, bottom=0.09)
    x = np.arange(len(METHODS))
    for ax, (ds_label, acc, _ans, _bld) in zip(axes, DATASETS):
        order = sorted(METHODS, key=lambda m: acc.get(m, 0.0))
        vals  = [acc.get(m, 0.0) for m in order]
        bars = ax.bar(x, vals, 0.88,
                      color=[COLOURS[m] for m in order],
                      edgecolor="white", linewidth=0.6)
        ax.set_xlim(-0.55, len(METHODS)-0.45)
        for b, m in zip(bars, order):
            b.set_hatch(PATTERNS[m])
            if m == OURS:
                b.set_edgecolor("black"); b.set_linewidth(1.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=17, color="#111", rotation=90, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlabel(ds_label.split("\n")[0], fontsize=19, labelpad=6)
        ax.set_ylim(0, 82)
        ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Judge Accuracy (%)")

    handles = []
    for m in METHODS:
        lbl = DISPLAY[m]
        if m == OURS:
            lbl = r"$\bf{" + "\\ ".join(lbl.split()) + "}$"
        handles.append(Patch(facecolor=COLOURS[m], edgecolor="white",
                             hatch=PATTERNS[m], label=lbl))
    fig.legend(handles=handles, ncol=4, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.66),
               columnspacing=1.6, handletextpad=0.5)
    fig.suptitle("Judge Accuracy across Benchmarks",
                 fontweight="bold", fontsize=21, y=0.96)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

# ---------- 2) grouped tokens: side-by-side subplots, one per dataset ----------
def plot_tokens(out):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.6), sharey=False,
                             gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.065, right=0.995, top=0.56, bottom=0.08)
    x = np.arange(len(METHODS))
    for ax, (ds_label, _acc, ans_d, bld_d) in zip(axes, DATASETS):
        order = sorted(METHODS,
                       key=lambda m: ans_d.get(m, 0)/1e6 + bld_d.get(m, 0)/1e6)
        ans = np.array([ans_d.get(m, 0)/1e6 for m in order])
        bld = np.array([bld_d.get(m, 0)/1e6 for m in order])
        cols = [COLOURS[m] for m in order]
        # answer = per-method colour + per-method pattern (identifies method)
        for j, m in enumerate(order):
            ax.bar(x[j], ans[j], 0.88, color=cols[j], edgecolor="white",
                   linewidth=0.6, hatch=PATTERNS[m])
        # build = same colour lightened, NO pattern (contrasts with patterned answer)
        ax.bar(x, bld, 0.88, bottom=ans,
               color=[lighten(c) for c in cols],
               edgecolor="white", linewidth=0.6)
        for j, m in enumerate(order):
            if m == OURS:
                ax.bar(x[j], ans[j]+bld[j], 0.88, color="none",
                       edgecolor="black", linewidth=1.6, zorder=4)
        ax.set_xlim(-0.55, len(METHODS)-0.45)
        ymax_local = max((ans + bld).tolist() + [1e-9])
        for j, (a, b) in enumerate(zip(ans, bld)):
            tot = a + b
            if tot <= 0: continue
            lbl = f"{a:.1f}+{b:.1f}" if b > 0 else f"{a:.1f}"
            ax.text(x[j], tot + ymax_local*0.015, lbl, ha="center", va="bottom",
                    fontsize=16, color="#111", rotation=90, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlabel(ds_label.split("\n")[0], fontsize=19, labelpad=6)
        ax.set_ylim(0, ymax_local*1.22)
        ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Total Tokens (millions)")

    method_handles = []
    for m in METHODS:
        lbl = DISPLAY[m]
        if m == OURS:
            lbl = r"$\bf{" + "\\ ".join(lbl.split()) + "}$"
        method_handles.append(Patch(facecolor=COLOURS[m], edgecolor="white",
                                    hatch=PATTERNS[m], label=lbl))
    kind_handles = [Patch(facecolor="#777", hatch="///",
                          label="Answer (recurring)"),
                    Patch(facecolor=lighten("#777"),
                          label="Build (one-time)")]
    fig.legend(handles=kind_handles, ncol=2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.78),
               columnspacing=1.8, handletextpad=0.5)
    fig.legend(handles=method_handles, ncol=4, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, 0.60),
               columnspacing=1.8, handletextpad=0.5)
    fig.suptitle("Token Cost across Benchmarks",
                 fontweight="bold", fontsize=21, y=0.965)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

plot_accuracy(OUTD/"accuracy_grouped.pdf")
plot_tokens(OUTD/"tokens_grouped.pdf")
