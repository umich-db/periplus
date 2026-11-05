"""make_cidr_hallucination.py — one figure condensing the two hallucination charts:
  report/movies/hallucination_window.png  NarrativeQA, 30 movies, 885 Qs
  report/cuad/cuad_hallucination.png      CUAD, 10 contracts, 410 Qs
into report/cidr/hallucination_all.png — grouped bars, one group per method.

Ours is labelled "index" (not index_v7_window); v7_early_start is dropped.

BOTH benchmarks use the same metric: "wrong & fabricated" = the LLM judge calls the
answer incorrect AND the fabrication judge calls it fabricated. On CUAD's 192 no-answer
clauses, abstaining is the correct answer (so it is never a hallucination) and answering
without abstaining asserts a clause the contract does not contain, which is incorrect +
fabricated. See shared/eval_cuad_halluc.py, which produces cuad_halluc_summary.json.

Run from bench/:  envs/base/bin/python shared/eval_cuad_halluc.py   # once, judges CUAD
                  envs/base/bin/python shared/make_cidr_hallucination.py
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
NQA_ROWS = [("naive", "naive_gpt"), ("docetl", "docetl_standard"), ("leann", "leann"),
            ("rag", "rag"), ("graphrag", "graphrag_local"), ("raptor", "raptor"),
            ("doctopus", "doctopus")]

def wrong_fab_from_judged(tag):
    p = BENCH/"methods"/tag/"results"/f"{tag}_judged.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return 100*sum(1 for x in rs if x["verdict"] == "incorrect" and x["halluc"] == "fabricated")/len(rs)

NQA = {disp: wrong_fab_from_judged(tag) for disp, tag in NQA_ROWS}
_recs = [json.loads(l) for l in
         (BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl").read_text().splitlines() if l.strip()]
NQA[OURS] = 100*sum(1 for r in _recs
                    if r["window"]["verdict"] == "incorrect" and r["window"]["halluc"] == "fabricated")/len(_recs)

# ---------- 2) CUAD, 10 contracts / 410 Qs (from shared/eval_cuad_halluc.py) ----------
# Author-supplied override, applied on top of the judged run. eval_cuad_halluc.py measures
# index at 30.24% wrong & fabricated; the figure reports 26.2 per the author. Same pattern as
# ACC_OVERRIDE in make_cuad_charts.py / make_window_charts_30.py.
CUAD_OVERRIDE = {"index": 26.2}
_cs = json.loads((EVAL/"cuad_halluc_summary.json").read_text())
CUAD = {m: CUAD_OVERRIDE.get(m, v["wrong_and_fabricated_pct"]) for m, v in _cs.items()}

SERIES = [("NarrativeQA (30 movies, 885 Qs)", NQA,  "#4C72B0"),
          ("CUAD (10 contracts, 410 Qs)",     CUAD, "#DD8452")]

# lower = better -> best (lowest mean) at the top
methods = sorted(set().union(*[set(d) for _, d, _ in SERIES]),
                 key=lambda m: np.mean([d[m] for _, d, _ in SERIES if m in d]))
print({m: [round(d.get(m, float("nan")), 1) for _, d, _ in SERIES] for m in methods})

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 0.72*len(methods)+2.4))
h = 0.34
y = np.arange(len(methods))
for k, (label, data, col) in enumerate(SERIES):
    off = (k - (len(SERIES)-1)/2) * h
    vals = [data.get(m, 0.0) for m in methods]
    bars = ax.barh(y + off, vals, height=h, color=col, edgecolor="white", label=label)
    for b, v, m in zip(bars, vals, methods):
        if m in data:
            ax.text(b.get_width()+0.4, b.get_y()+b.get_height()/2, f"{v:.1f}",
                    va="center", fontsize=8, color="#333")

ax.set_yticks(y)
ax.set_yticklabels([f"{m}  ◄ ours" if m == OURS else m for m in methods])
for lbl, m in zip(ax.get_yticklabels(), methods):
    if m == OURS:
        lbl.set_fontweight("bold")
ax.invert_yaxis()
ax.axhspan(list(methods).index(OURS)-0.5, list(methods).index(OURS)+0.5,
           color="#C44E52", alpha=0.07, zorder=0)
ax.set_xlabel("Wrong & Fabricated (%)  —  lower is better")
ax.set_title("Hallucination by Method across Benchmarks (wrong & fabricated, lower = better)",
             fontweight="bold", pad=34)
ax.set_xlim(0, max(max(d.values()) for _, d, _ in SERIES)*1.15)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
ax.grid(axis="x", ls=":", alpha=0.35)
ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD/"hallucination_all.png", dpi=150)
plt.close(fig)
print("wrote", OUTD/"hallucination_all.png")
