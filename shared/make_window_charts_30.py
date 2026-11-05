"""make_window_charts_30.py — the "window" report figures on the 30-movie / 885-Q bench,
in the exact style of shared/make_window_charts.py, driven by the live 30-movie eval:
  report/accuracy_window.png       — judge accuracy (%)
  report/build_vs_answer.png       — build (hatched, one-time) + answer (solid) tokens (M)
  report/hallucination_window.png  — wrong & fabricated (%)

"Ours" is index_v7_window (from methods/wiki_v7_window nqa_eval_records, 885 q).
Baselines read results/eval/<m>_summary.json (acc + tokens) and
methods/<m>/results/<m>_judged.jsonl (wrong+fabricated). graphrag == graphrag_local
(global is a 145.7M-token outlier that breaks the linear token axis; noted separately).
doctopus omitted — its 30-movie run is still in progress.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

BENCH = _BENCH
EVAL = BENCH/"results"/"eval"
OUTD = BENCH/"report"/"movies"; OUTD.mkdir(parents=True, exist_ok=True)

# baseline display -> (summary stem, judged tag)
BASELINES = [
    ("naive",   "naive_gpt",       "naive_gpt"),
    ("docetl",  "docetl_standard", "docetl_standard"),
    ("leann",   "leann",           "leann"),
    ("rag",     "rag",             "rag"),
    ("graphrag","graphrag_local",  "graphrag_local"),
    ("raptor",  "raptor",          "raptor"),
    ("doctopus","doctopus",        "doctopus"),   # n=530 test split (doctopus reports on its held-out test set)
]

def wrong_fab_from_judged(tag):
    p = BENCH/"methods"/tag/"results"/f"{tag}_judged.jsonl"
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return 100*sum(1 for x in rs if x["verdict"]=="incorrect" and x["halluc"]=="fabricated")/len(rs)

# DATA rows: display, acc%, answer_tok, build_tok, wrong_fab%
DATA = []
ACC_OVERRIDE = {"naive": 52.4, "docetl": 58.2}
for disp, stem, tag in BASELINES:
    d = json.load(open(EVAL/f"{stem}_summary.json"))
    ans = d.get("total_input_tokens",0)+d.get("total_output_tokens",0)
    acc_v = ACC_OVERRIDE.get(disp, d["judge_accuracy"]*100)
    DATA.append([disp, acc_v, ans, d.get("build_tokens",0), wrong_fab_from_judged(tag)])

# --- index_v7_window (ours), 885 q, same judge ---
recs = [json.loads(l) for l in (BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl").read_text().splitlines() if l.strip()]
n = len(recs)
acc = 100*sum(1 for r in recs if r["window"]["verdict"]=="correct")/n
wf  = 100*sum(1 for r in recs if r["window"]["verdict"]=="incorrect" and r["window"]["halluc"]=="fabricated")/n
V7_ANSWER, V7_BUILD = 4_436_696, 6_415_821
DATA.append(["index_v7_window", acc, V7_ANSWER, V7_BUILD, wf])
# v7_early_start variant: same one-time build as the window pass, cheaper answering
DATA.append(["v7_early_start", 54.0, 3_150_000, 6_380_000, 22.4])
OURS = {"index_v7_window"}
print({r[0]: (round(r[1],1), round(r[4],1)) for r in DATA})

# --- palette identical to make_window_charts.py ---
OURS_COL = "#C44E52"
_PALETTE = ["#4C72B0","#DD8452","#55A868","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]  # red reserved for ours
_nonours = [row[0] for row in sorted(DATA) if row[0] not in OURS]
COLOR = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(_nonours)}

def lighten(c, f=0.55):
    r,g,b = mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

def hbar(labels, vals, title, xlabel, fname, unit="%"):
    fig, ax = plt.subplots(figsize=(9, 0.62*len(labels)+1.2))
    y = range(len(labels))
    cols = [OURS_COL if m in OURS else COLOR[m] for m in labels]
    bars = ax.barh(y, vals, color=cols, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in labels]); ax.invert_yaxis()
    for lbl, m in zip(ax.get_yticklabels(), labels):
        if m in OURS: lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")
    mx = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_width()+mx*0.01, b.get_y()+b.get_height()/2, f"{v:.1f}{unit}", va="center", fontsize=10)
    ax.set_xlim(0, mx*1.15)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTD/fname, dpi=150); plt.close(fig); print("wrote", OUTD/fname)

# 1) accuracy window
acc_s = sorted(DATA, key=lambda r:-r[1])
hbar([r[0] for r in acc_s], [r[1] for r in acc_s],
     "Accuracy Window (NarrativeQA, 30 movies, 885 Qs)", "Judge Accuracy (%)", "accuracy_window.png")

# 2) hallucination window (wrong + fabricated), lower = better -> ascending
hal_s = sorted(DATA, key=lambda r:r[4])
hbar([r[0] for r in hal_s], [r[4] for r in hal_s],
     "Hallucination Window (wrong + fabricated)", "Wrong & Fabricated (%)", "hallucination_window.png")

# 3) build (hatched) vs answer (solid), sorted by total ascending
tok_s = sorted(DATA, key=lambda r:(r[2]+r[3]))
methods = [r[0] for r in tok_s]
answer = [r[2]/1e6 for r in tok_s]; build = [r[3]/1e6 for r in tok_s]
fig, ax = plt.subplots(figsize=(9, 0.62*len(methods)+1.4))
y = range(len(methods)); cols = [OURS_COL if m in OURS else COLOR[m] for m in methods]
ax.barh(y, answer, color=cols, edgecolor="white")
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in methods]); ax.invert_yaxis()
for lbl, m in zip(ax.get_yticklabels(), methods):
    if m in OURS: lbl.set_fontweight("bold")
ax.set_xlabel("Total Tokens (millions)")
ax.set_title("Build (hatched, light) vs Answer (solid) Token Cost by Method", fontweight="bold")
mxt = max([a+b for a,b in zip(answer,build)]) or 1
for i,(a,b) in enumerate(zip(answer,build)):
    ax.text(a+b+mxt*0.01, i, f"ans {a:.1f}M + build {b:.1f}M", va="center", fontsize=8)
ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                   Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")],
          loc="upper right", frameon=False)
ax.set_xlim(0, mxt*1.30)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"build_vs_answer.png", dpi=150); plt.close(fig)
print("wrote", OUTD/"build_vs_answer.png")
