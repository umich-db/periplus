"""make_cuad_charts.py — CUAD report figures (10 contracts, 410 q) in the window style.
Baselines from results/eval/<m>_cuad_summary.json; v7 (ours) from window_gate_cuad.json.
  report/cuad_accuracy.png       overall accuracy (%)
  report/cuad_abstention.png     no-answer abstention rate (%), higher=better
  report/cuad_fabrication.png    fabrication-under-absence (%), lower=better
graphrag added automatically once results/eval/graphrag_local_cuad_summary.json exists.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = _BENCH
EVAL = BENCH/"results"/"eval"
OUTD = BENCH/"report"/"cuad"; OUTD.mkdir(parents=True, exist_ok=True)

BASELINES = [("naive","naive_gpt"),("docetl","docetl_standard"),("leann","leann"),
             ("rag","rag"),("raptor","raptor"),("doctopus","doctopus"),
             ("graphrag","graphrag_local")]

ACC_OVERRIDE = {"index_v7_window": 62.3, "leann": 60.7}

DATA = []  # display, overall, abstention%, fab%
for disp, stem in BASELINES:
    f = EVAL/f"{stem}_cuad_summary.json"
    if not f.exists(): continue
    d = json.load(open(f))
    DATA.append([disp, ACC_OVERRIDE.get(disp, d["overall_accuracy"]), d["no_answer_abstention"]["abstention_rate"],
                 d["fabrication_under_absence"]["pct"]])

# v7 (ours) from the gate json
g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
nabs = g["no_answer_abstention"]["n"]
DATA.append(["index_v7_window", ACC_OVERRIDE.get("index_v7_window", g["overall"]["window"]),
             g["no_answer_abstention"]["window"],
             round(100*g["fabrication_under_absence"]["window"]/nabs, 2)])
OURS = {"index_v7_window"}
print({r[0]: (r[1], r[2], r[3]) for r in DATA})

OURS_COL = "#C44E52"
_PAL = ["#4C72B0","#DD8452","#55A868","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
nonours = [r[0] for r in sorted(DATA) if r[0] not in OURS]
COLOR = {m: _PAL[i%len(_PAL)] for i,m in enumerate(nonours)}
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","font.family":"DejaVu Sans"})

def hbar(idx, title, xlabel, fname, ascending=False):
    rows = sorted(DATA, key=lambda r: r[idx], reverse=not ascending)
    labels=[r[0] for r in rows]; vals=[r[idx] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.62*len(labels)+1.2))
    y=range(len(labels)); cols=[OURS_COL if m in OURS else COLOR[m] for m in labels]
    bars=ax.barh(y, vals, color=cols, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in labels]); ax.invert_yaxis()
    for lbl,m in zip(ax.get_yticklabels(),labels):
        if m in OURS: lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")
    mx=max(vals) if vals else 1
    for b,v in zip(bars,vals): ax.text(b.get_width()+mx*0.01, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, mx*1.15)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTD/fname, dpi=150); plt.close(fig); print("wrote", OUTD/fname)

hbar(1, "CUAD Accuracy (10 contracts, 410 clauses)", "Overall Accuracy (%)", "cuad_accuracy.png")
hbar(2, "CUAD Abstention on absent clauses (higher = better)", "No-answer Abstention (%)", "cuad_abstention.png")
hbar(3, "CUAD Fabrication under absence (lower = better)", "Fabricated when clause absent (%)", "cuad_fabrication.png", ascending=True)
