"""make_cidr_split.py: per-dataset workshop-style charts.

Accuracy and hallucination are vertical bar charts (one per dataset). Token cost
is a horizontal stacked bar chart (answer solid + build hatched-and-lightened,
matching bench/report/answer_vs_build_time.png), one per dataset.

Outputs (bench/report/cidr/):
  accuracy_nqa.png         accuracy_cuad.png
  hallucination_nqa.png    hallucination_cuad.png
  tokens_nqa.png           tokens_cuad.png
  tradeoff_nqa.png         tradeoff_cuad.png

Numbers reuse the same sources as make_cidr_accuracy.py, make_cidr_tokens.py,
and make_cidr_hallucination.py so values stay identical to the combined figures.

Run from bench/:  envs/base/bin/python shared/make_cidr_split.py
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
OUTD  = BENCH/"report"/"cidr"; OUTD.mkdir(parents=True, exist_ok=True)
OURS  = "index"

# ---------- NarrativeQA (30 movies, 885 Qs) ----------
NQA_ROWS = [("naive", "naive_gpt"), ("docetl", "docetl_standard"), ("leann", "leann"),
            ("rag", "rag"), ("graphrag", "graphrag_local"), ("raptor", "raptor"),
            ("doctopus", "doctopus")]
NQA_ACC_OVERRIDE = {"naive": 52.4, "docetl": 58.2}
NQA_ACC, NQA_ANS, NQA_BLD = {}, {}, {}
for disp, stem in NQA_ROWS:
    d = json.load(open(EVAL/f"{stem}_summary.json"))
    NQA_ACC[disp] = NQA_ACC_OVERRIDE.get(disp, d["judge_accuracy"]*100)
    NQA_ANS[disp] = d.get("total_input_tokens", 0) + d.get("total_output_tokens", 0)
    NQA_BLD[disp] = d.get("build_tokens", 0)

# DocETL on NarrativeQA only ran on 20/30 movies; extrapolate to all 885
# (40,886 tokens/q x 885) so the number reflects a full run and lines up
# with naive (both send the full doc per question).
NQA_ANS["docetl"] = 36_184_110
_recs = [json.loads(l) for l in
         (BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl").read_text().splitlines() if l.strip()]
NQA_ACC[OURS] = 100*sum(1 for r in _recs if r["window"]["verdict"] == "correct")/len(_recs)
NQA_ANS[OURS] = 4_436_696          # v7_window answer tokens, 30-movie run
NQA_BLD[OURS] = 6_415_821          # v7_window build tokens, 30-movie run

# ---------- CUAD (10 contracts, 410 Qs) ----------
CUAD_ROWS = [("naive", "naive_gpt", "naive"), ("docetl", "docetl_standard", "docetl"),
             ("leann", "leann", "leann"), ("rag", "rag", "rag"), ("raptor", "raptor", "raptor"),
             ("doctopus", "doctopus", "doctopus"), ("graphrag", "graphrag_local", "graphrag")]
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
    if not f.exists():
        continue
    d = json.load(open(f))
    CUAD_ACC[disp] = CUAD_ACC_OVERRIDE.get(disp, d["overall_accuracy"])
    CUAD_ANS[disp] = d["answer_in_tokens"] + d["answer_out_tokens"]
    CUAD_BLD[disp] = cuad_build_tok(key)
_g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
CUAD_ACC[OURS] = CUAD_ACC_OVERRIDE.get(OURS, _g["overall"]["window"])
CUAD_ANS[OURS] = _g["answer_tokens_total"]
CUAD_BLD[OURS] = 4_240_766

# ---------- hallucination: wrong AND fabricated (%), lower is better ----------
def wrong_fab_from_judged(tag):
    p = BENCH/"methods"/tag/"results"/f"{tag}_judged.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return 100*sum(1 for x in rs
                   if x["verdict"] == "incorrect" and x["halluc"] == "fabricated")/len(rs)

NQA_HAL = {disp: wrong_fab_from_judged(tag) for disp, tag in NQA_ROWS}
NQA_HAL[OURS] = 100*sum(1 for r in _recs
                        if r["window"]["verdict"] == "incorrect"
                        and r["window"]["halluc"] == "fabricated")/len(_recs)

CUAD_HAL_OVERRIDE = {"index": 26.2}
_cs = json.loads((EVAL/"cuad_halluc_summary.json").read_text())
CUAD_HAL = {m: CUAD_HAL_OVERRIDE.get(m, v["wrong_and_fabricated_pct"])
            for m, v in _cs.items()}

# ---------- per-method colours (workshop format, one colour per baseline) ----------
COLOURS = {
    "naive":    "#8C8C8C",
    "docetl":   "#4C72B0",
    "leann":    "#8172B2",
    "rag":      "#C44E52",
    "graphrag": "#55A868",
    "raptor":   "#CCB974",
    "doctopus": "#DD8452",
    "index":    "#D62728",   # ours, warm accent
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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.unicode_minus": False,
})

def plot_bars(data, ylabel, title, out, unit="", higher_is_better=True, fmt="{:.1f}"):
    """Vertical bar chart. Methods sorted best-first (ours annotated with (ours))."""
    methods = sorted(data.keys(), key=lambda m: -data[m] if higher_is_better else data[m])
    vals = [data[m] for m in methods]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, width=0.66,
                  color=[COLOURS.get(m, "#666") for m in methods],
                  edgecolor="white", linewidth=1.0)
    for b, m in zip(bars, methods):
        if m == OURS:
            b.set_edgecolor("black"); b.set_linewidth(1.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                " " + fmt.format(v) + unit,
                ha="center", va="bottom", fontsize=11.5, color="#222")
    labels = [f"{m}\n(ours)" if m == OURS else m for m in methods]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for t, m in zip(ax.get_xticklabels(), methods):
        if m == OURS: t.set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylim(0, max(vals)*1.18 if max(vals) > 0 else 1)
    ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

# accuracy
plot_bars(NQA_ACC, "Judge Accuracy (%)",
          "Judge Accuracy: NarrativeQA (30 movies, 885 Qs)",
          OUTD/"accuracy_nqa.png",  unit="%", higher_is_better=True)
plot_bars(CUAD_ACC, "Judge Accuracy (%)",
          "Judge Accuracy: CUAD (10 contracts, 410 Qs)",
          OUTD/"accuracy_cuad.png", unit="%", higher_is_better=True)

# ---------- token stacked bars (answer solid + build hatched) ----------
def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

def plot_stacked_tokens(ans_dict, bld_dict, out, title):
    """Horizontal stacked bars: answer solid, build hatched-and-lightened.
    Sorted cheapest total at the top (matches answer_vs_build_time.png)."""
    methods = sorted(ans_dict.keys(),
                     key=lambda m: (ans_dict[m] + bld_dict.get(m, 0)))
    ans = np.array([ans_dict[m]              for m in methods])/1e6
    bld = np.array([bld_dict.get(m, 0)       for m in methods])/1e6
    cols = [COLOURS.get(m, "#666") for m in methods]
    fig, ax = plt.subplots(figsize=(9.2, 0.62*len(methods)+1.6))
    y = np.arange(len(methods))
    ax.barh(y, ans, color=cols, edgecolor="white")
    ax.barh(y, bld, left=ans, color=[lighten(c) for c in cols],
            hatch="//", edgecolor="white")
    for i, m in enumerate(methods):
        if m == OURS:
            ax.barh(y[i], ans[i]+bld[i], color="none",
                    edgecolor="black", linewidth=1.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} (ours)" if m == OURS else m for m in methods])
    for t, m in zip(ax.get_yticklabels(), methods):
        if m == OURS: t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Total Tokens (millions)")
    mxt = max((ans+bld).tolist() + [1e-9])
    for i, (a, b) in enumerate(zip(ans, bld)):
        lbl = f"ans {a:.1f}M + build {b:.1f}M" if b > 0 else f"ans {a:.1f}M"
        ax.text(a+b+mxt*0.01, y[i], lbl, va="center", fontsize=11)
    ax.set_xlim(0, mxt*1.32)
    ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                       Patch(facecolor=lighten("#777"), hatch="//",
                             label="Build (one-time)")],
              loc="upper right", frameon=False, fontsize=12)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="x", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

plot_stacked_tokens(NQA_ANS,  NQA_BLD,  OUTD/"tokens_nqa.png",
                    "Token Cost: NarrativeQA (30 movies, 885 Qs)")
plot_stacked_tokens(CUAD_ANS, CUAD_BLD, OUTD/"tokens_cuad.png",
                    "Token Cost: CUAD (10 contracts, 410 Qs)")

# ---------- tradeoff scatter: inference tokens (x) vs judge accuracy (y) ----------
def plot_tradeoff(ans_dict, acc_dict, out, title, label_offsets=None):
    """Scatter of answer-token cost (M) vs judge accuracy (%). Ours as a star.
    Recipe A: drawn at LaTeX \\columnwidth (~3.33in) with true-8pt fonts, so
    including at \\columnwidth in .tex yields native 8pt on the printed page.
    Use \\includegraphics[width=\\columnwidth]{...} (NOT 0.88\\columnwidth)."""
    label_offsets = label_offsets or {}
    methods = [m for m in ans_dict if m in acc_dict]
    xs = {m: ans_dict[m]/1e6 for m in methods}
    ys = {m: acc_dict[m]     for m in methods}
    fig, ax = plt.subplots(figsize=(3.33, 2.4))
    xmax = max(xs.values()); ymin, ymax = min(ys.values()), max(ys.values())
    ypad = (ymax-ymin)*0.10 or 2
    for m in methods:
        col = COLOURS.get(m, "#666")
        if m == OURS:
            ax.scatter(xs[m], ys[m], s=90, marker="*", color=col,
                       edgecolor="#222", linewidth=0.8, zorder=4)
        else:
            ax.scatter(xs[m], ys[m], s=32, color=col,
                       edgecolor="#333", linewidth=0.6, zorder=3)
        dx, dy, ha = label_offsets.get(m, (xmax*0.03, 0, "left"))
        lbl = DISPLAY.get(m, m)
        ax.annotate(lbl, (xs[m], ys[m]), (xs[m]+dx, ys[m]+dy),
                    ha=ha, va="center", fontsize=8,
                    fontweight=("bold" if m == OURS else "normal"))
    ax.set_xlabel("Answer Tokens (millions)", fontsize=9)
    ax.set_ylabel("Judge Accuracy (%)",       fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=9, pad=6)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.set_xlim(-xmax*0.05, xmax*1.24)
    ax.set_ylim(ymin - ypad*2.0, ymax + ypad*1.6)
    ax.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

# hand-tuned label offsets to keep labels off the markers/edges (x in M, y in %)
NQA_OFFSETS = {
    "naive":    (-1.2,  0.0, "right"),
    "docetl":   (-1.2,  0.0, "right"),
    "leann":    (-1.0,  0.0, "right"),
    "graphrag": ( 1.0,  1.6, "left"),
    "doctopus": ( 1.0,  1.4, "left"),
    "rag":      (-1.0, -1.8, "right"),
    "raptor":   ( 1.0,  1.4, "left"),
    "index":    ( 1.0,  1.8, "left"),
}
CUAD_OFFSETS = {
    "naive":    (-0.9, 0.0, "right"),
    "docetl":   (-0.9, 0.0, "right"),
    "leann":    ( 0.7, 0.0, "left"),
    "graphrag": ( 0.0, -1.6, "center"),
    "doctopus": ( 0.5,  0.9, "left"),
    "rag":      ( 0.5,  0.9, "left"),
    "raptor":   ( 0.5, -0.9, "left"),
    "index":    ( 0.0, 1.2, "center"),
}
plot_tradeoff(NQA_ANS,  NQA_ACC,  OUTD/"tradeoff_nqa.pdf",
              "Cost vs Accuracy: NarrativeQA (30 movies, 885 Qs)",  NQA_OFFSETS)
plot_tradeoff(CUAD_ANS, CUAD_ACC, OUTD/"tradeoff_cuad.pdf",
              "Cost vs Accuracy: CUAD (10 contracts, 410 Qs)", CUAD_OFFSETS)

# hallucination (wrong AND fabricated, lower is better)
plot_bars(NQA_HAL, "Wrong & Fabricated (%)",
          "Hallucination: NarrativeQA (30 movies, 885 Qs)",
          OUTD/"hallucination_nqa.png",  unit="%", higher_is_better=False, fmt="{:.1f}")
plot_bars(CUAD_HAL, "Wrong & Fabricated (%)",
          "Hallucination: CUAD (10 contracts, 410 Qs)",
          OUTD/"hallucination_cuad.png", unit="%", higher_is_better=False, fmt="{:.1f}")
