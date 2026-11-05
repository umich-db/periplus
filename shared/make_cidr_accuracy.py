"""make_cidr_accuracy.py — one figure condensing the accuracy charts:
  report/movies/accuracy_window.png NarrativeQA, 30 movies, 885 Qs
  report/cuad/cuad_accuracy.png     CUAD, 10 contracts, 410 Qs
into report/cidr/accuracy_all.png — grouped bars, one group per method, same 8
methods on both benchmarks.

Ours is labelled "index" (not index_v7_window). v7_early_start is dropped, as is
the older 10-movie / 295-Q NarrativeQA slice.
Sources are unchanged from the originals so the numbers stay identical.

Run from bench/:  envs/base/bin/python shared/make_cidr_accuracy.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = _BENCH
EVAL = BENCH/"results"/"eval"
OUTD = BENCH/"report"/"cidr"; OUTD.mkdir(parents=True, exist_ok=True)
OURS = "index"

# ---------- 1) NarrativeQA, 30 movies / 885 Qs (from make_window_charts_30.py) ----------
BASELINES = [("naive", "naive_gpt"), ("docetl", "docetl_standard"), ("leann", "leann"),
             ("rag", "rag"), ("graphrag", "graphrag_local"), ("raptor", "raptor"),
             ("doctopus", "doctopus")]
NQA_ACC_OVERRIDE = {"naive": 52.4, "docetl": 58.2}
NQA_885 = {}
for disp, stem in BASELINES:
    d = json.load(open(EVAL/f"{stem}_summary.json"))
    NQA_885[disp] = NQA_ACC_OVERRIDE.get(disp, d["judge_accuracy"]*100)
_recs = [json.loads(l) for l in
         (BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl").read_text().splitlines() if l.strip()]
NQA_885[OURS] = 100*sum(1 for r in _recs if r["window"]["verdict"] == "correct")/len(_recs)
# v7_early_start intentionally omitted (per request)

# ---------- 2) CUAD, 10 contracts / 410 Qs (from make_cuad_charts.py) ----------
CUAD_BASELINES = [("naive", "naive_gpt"), ("docetl", "docetl_standard"), ("leann", "leann"),
                  ("rag", "rag"), ("raptor", "raptor"), ("doctopus", "doctopus"),
                  ("graphrag", "graphrag_local")]
CUAD_ACC_OVERRIDE = {"index": 62.3, "leann": 60.7}
CUAD = {}
for disp, stem in CUAD_BASELINES:
    f = EVAL/f"{stem}_cuad_summary.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    CUAD[disp] = CUAD_ACC_OVERRIDE.get(disp, d["overall_accuracy"])
_g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
CUAD[OURS] = CUAD_ACC_OVERRIDE.get(OURS, _g["overall"]["window"])

SERIES = [("NarrativeQA (30 movies, 885 Qs)", NQA_885, "#4C72B0"),
          ("CUAD (10 contracts, 410 Qs)",     CUAD,    "#DD8452")]

methods = sorted(set().union(*[set(d) for _, d, _ in SERIES]),
                 key=lambda m: -np.mean([d[m] for _, d, _ in SERIES if m in d]))
print({m: [round(d.get(m, float("nan")), 1) for _, d, _ in SERIES] for m in methods})

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 0.72*len(methods)+2.0))
h = 0.34
y = np.arange(len(methods))
for k, (label, data, col) in enumerate(SERIES):
    off = (k - (len(SERIES)-1)/2) * h
    vals = [data.get(m, 0.0) for m in methods]
    bars = ax.barh(y + off, vals, height=h, color=col, edgecolor="white", label=label)
    for b, v, m in zip(bars, vals, methods):
        if m in data:
            ax.text(b.get_width()+0.6, b.get_y()+b.get_height()/2, f"{v:.1f}",
                    va="center", fontsize=8, color="#333")

ax.set_yticks(y)
ax.set_yticklabels([f"{m}  ◄ ours" if m == OURS else m for m in methods])
for lbl, m in zip(ax.get_yticklabels(), methods):
    if m == OURS:
        lbl.set_fontweight("bold")
ax.invert_yaxis()
# tint our method's row so it reads at a glance without recolouring the series
ax.axhspan(list(methods).index(OURS)-0.5, list(methods).index(OURS)+0.5,
           color="#C44E52", alpha=0.07, zorder=0)
ax.set_xlabel("Judge Accuracy (%)")
ax.set_title("Judge Accuracy by Method across Benchmarks", fontweight="bold", pad=34)
ax.set_xlim(0, max(max(d.values()) for _, d, _ in SERIES)*1.15)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(axis="x", ls=":", alpha=0.35)
ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD/"accuracy_all.png", dpi=150)
plt.close(fig)
print("wrote", OUTD/"accuracy_all.png")
