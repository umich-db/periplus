"""CUAD token-cost chart (build hatched + answer solid), report window style.
report/cuad_tokens.png. Answer tokens from results/eval/<m>_cuad_summary.json;
build tokens from each method's CUAD build-cost file. v7 (ours) from window_gate_cuad.json.
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
OUTD = BENCH/"report"/"cuad"; OUTD.mkdir(parents=True, exist_ok=True)

CONTRACTS = {p.stem for p in (BENCH/"data/contracts/txt").glob("*.txt")}

def answer_tok(stem):
    d = json.load(open(EVAL/f"{stem}_cuad_summary.json"))
    return d["answer_in_tokens"] + d["answer_out_tokens"]

def build_tok(method):
    if method == "rag":
        bc = json.load(open(BENCH/"methods/rag/build_costs_cuad.json"))
        return sum(v.get("build_tokens_total",0) for v in bc.values())
    if method == "doctopus":
        bc = json.load(open(BENCH/"methods/doctopus/build_costs_cuad.json"))
        return sum(v.get("build_tokens_total",0) for k,v in bc.items() if isinstance(v,dict))
    if method == "raptor":
        st = json.load(open(BENCH/"methods/raptor/output/build_stats_cuad.json"))
        return sum(v.get("build_tokens",0) for k,v in st.items() if k in CONTRACTS)
    if method == "graphrag":
        tot = 0
        for s in CONTRACTS:
            f = BENCH/"methods/graphrag/build_costs"/f"{s}.json"
            if f.exists(): tot += json.load(open(f)).get("build_tokens_total",0)
        return tot
    return 0  # leann (GPU), naive, docetl: no LLM build

# display, summary stem, build-method-key
ROWS = [("naive","naive_gpt","naive"),("docetl","docetl_standard","docetl"),
        ("leann","leann","leann"),("rag","rag","rag"),("raptor","raptor","raptor"),
        ("doctopus","doctopus","doctopus"),("graphrag","graphrag_local","graphrag")]
DATA = []  # display, answer, build
for disp, stem, key in ROWS:
    DATA.append([disp, answer_tok(stem), build_tok(key)])

g = json.load(open(BENCH/"methods/wiki_v7_window/results/window_gate_cuad.json"))
CUAD_BUILD = 4_240_766
DATA.append(["index_v7_window", g["answer_tokens_total"], CUAD_BUILD])
OURS = {"index_v7_window"}
for r in DATA: print(f"{r[0]:16s} ans={r[1]/1e6:5.1f}M build={r[2]/1e6:5.1f}M total={(r[1]+r[2])/1e6:5.1f}M")

OURS_COL="#C44E52"
_PAL=["#4C72B0","#DD8452","#55A868","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
nonours=[r[0] for r in sorted(DATA) if r[0] not in OURS]
COLOR={m:_PAL[i%len(_PAL)] for i,m in enumerate(nonours)}
def lighten(c,f=0.55):
    r,g,b=mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","font.family":"DejaVu Sans"})

tok=sorted(DATA, key=lambda r:(r[1]+r[2]))
methods=[r[0] for r in tok]; answer=[r[1]/1e6 for r in tok]; build=[r[2]/1e6 for r in tok]
fig,ax=plt.subplots(figsize=(9,0.62*len(methods)+1.4))
y=range(len(methods)); cols=[OURS_COL if m in OURS else COLOR[m] for m in methods]
ax.barh(y, answer, color=cols, edgecolor="white")
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in methods]); ax.invert_yaxis()
for lbl,m in zip(ax.get_yticklabels(),methods):
    if m in OURS: lbl.set_fontweight("bold")
ax.set_xlabel("Total Tokens (millions)"); ax.set_title("CUAD: Build (hatched) vs Answer (solid) Token Cost", fontweight="bold")
mxt=max(a+b for a,b in zip(answer,build)) or 1
for i,(a,b) in enumerate(zip(answer,build)): ax.text(a+b+mxt*0.01, i, f"ans {a:.1f}M + build {b:.1f}M", va="center", fontsize=8)
ax.legend(handles=[Patch(facecolor="#777",label="Answer (recurring)"),Patch(facecolor=lighten("#777"),hatch="//",label="Build (one-time)")], loc="upper right", frameon=False)
ax.set_xlim(0, mxt*1.32)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"cuad_tokens.png", dpi=150); plt.close(fig); print("wrote", OUTD/"cuad_tokens.png")
