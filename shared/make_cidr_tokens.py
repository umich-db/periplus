"""make_cidr_tokens.py — one figure condensing the two token-cost charts:
  report/movies/build_vs_answer.png  NarrativeQA, 30 movies, 885 Qs
  report/cuad/cuad_tokens.png        CUAD, 10 contracts, 410 Qs
into report/cidr/tokens_all.png — two stacked bars per method (answer solid +
build hatched), one per benchmark.

Ours is labelled "index" (not index_v7_window); v7_early_start is dropped.
Sources are unchanged from the originals so the numbers stay identical.

Run from bench/:  envs/base/bin/python shared/make_cidr_tokens.py
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
EVAL = BENCH/"results"/"eval"
OUTD = BENCH/"report"/"cidr"; OUTD.mkdir(parents=True, exist_ok=True)
OURS = "index"

# ---------- 1) NarrativeQA, 30 movies / 885 Qs (from make_window_charts_30.py) ----------
NQA_ROWS = [("naive", "naive_gpt"), ("docetl", "docetl_standard"), ("leann", "leann"),
            ("rag", "rag"), ("graphrag", "graphrag_local"), ("raptor", "raptor"),
            ("doctopus", "doctopus")]
NQA = {}
for disp, stem in NQA_ROWS:
    d = json.load(open(EVAL/f"{stem}_summary.json"))
    NQA[disp] = (d.get("total_input_tokens", 0) + d.get("total_output_tokens", 0),
                 d.get("build_tokens", 0))
NQA[OURS] = (4_436_696, 6_415_821)          # v7_window answer / build, 30-movie run

# ---------- 2) CUAD, 10 contracts / 410 Qs (from make_cuad_token_chart.py) ----------
CONTRACTS = {p.stem for p in (BENCH/"data/contracts/txt").glob("*.txt")}

def cuad_answer_tok(stem):
    d = json.load(open(EVAL/f"{stem}_cuad_summary.json"))
    return d["answer_in_tokens"] + d["answer_out_tokens"]

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
    return 0                                 # leann (GPU), naive, docetl: no LLM build

CUAD_ROWS = [("naive", "naive_gpt", "naive"), ("docetl", "docetl_standard", "docetl"),
             ("leann", "leann", "leann"), ("rag", "rag", "rag"), ("raptor", "raptor", "raptor"),
             ("doctopus", "doctopus", "doctopus"), ("graphrag", "graphrag_local", "graphrag")]
CUAD = {disp: (cuad_answer_tok(stem), cuad_build_tok(key)) for disp, stem, key in CUAD_ROWS}
_g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
CUAD[OURS] = (_g["answer_tokens_total"], 4_240_766)

SERIES = [("NarrativeQA (30 movies, 885 Qs)", NQA,  "#4C72B0"),
          ("CUAD (10 contracts, 410 Qs)",     CUAD, "#DD8452")]

def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

# cheapest total (summed over both benchmarks) at the top
methods = sorted(set().union(*[set(d) for _, d, _ in SERIES]),
                 key=lambda m: sum(sum(d[m]) for _, d, _ in SERIES if m in d))
for m in methods:
    print(f"{m:10s} " + "  ".join(
        f"{lbl.split(' (')[0]}: ans {d[m][0]/1e6:5.1f}M + build {d[m][1]/1e6:5.1f}M"
        for lbl, d, _ in SERIES if m in d))

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 0.78*len(methods)+2.2))
h = 0.34
y = np.arange(len(methods))
totals = []
for k, (label, data, col) in enumerate(SERIES):
    off = (k - (len(SERIES)-1)/2) * h
    ans = np.array([data.get(m, (0, 0))[0] for m in methods])/1e6
    bld = np.array([data.get(m, (0, 0))[1] for m in methods])/1e6
    ax.barh(y+off, ans, height=h, color=col, edgecolor="white", label=label)
    ax.barh(y+off, bld, height=h, left=ans, color=lighten(col), hatch="//", edgecolor="white")
    for i, m in enumerate(methods):
        if m in data:
            ax.text(ans[i]+bld[i]+0.4, y[i]+off, f"{ans[i]:.1f}+{bld[i]:.1f}M",
                    va="center", fontsize=7.5, color="#333")
    totals.extend((ans+bld).tolist())

ax.set_yticks(y)
ax.set_yticklabels([f"{m}  ◄ ours" if m == OURS else m for m in methods])
for lbl, m in zip(ax.get_yticklabels(), methods):
    if m == OURS:
        lbl.set_fontweight("bold")
ax.invert_yaxis()
ax.axhspan(list(methods).index(OURS)-0.5, list(methods).index(OURS)+0.5,
           color="#C44E52", alpha=0.07, zorder=0)
ax.set_xlabel("Total Tokens (millions)  —  lower is better")
ax.set_title("Build (hatched, one-time) vs Answer (solid) Token Cost across Benchmarks",
             fontweight="bold", pad=48)
ax.set_xlim(0, max(totals)*1.18)
handles = [Patch(facecolor=c, label=lbl) for lbl, _, c in SERIES]
handles += [Patch(facecolor="#777", label="Answer (recurring)"),
            Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")]
ax.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
          loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(axis="x", ls=":", alpha=0.35)
ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD/"tokens_all.png", dpi=150)
plt.close(fig)
print("wrote", OUTD/"tokens_all.png")
