"""CUAD hallucination chart — same definition as the movies hallucination metric:
answered (not abstained) but token-F1 == 0 vs gold, over all 410 clauses. For no-answer
clauses (gold empty) any non-abstention counts (fabrication under absence); for answerable
clauses it counts answers with zero lexical overlap. Purely lexical (no LLM re-judging).
Writes report/cuad/cuad_hallucination.png in the report window style (v7 gold).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = _BENCH
OUTD = BENCH/"report"/"cuad"; OUTD.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("ev", BENCH/"shared"/"eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
_f1 = ev._f1

_MISS = ("", "not found", "none", "n/a", "not applicable")
_ABSENT = ("not found","not present","not contain","does not contain","no such","not specified",
           "not mentioned","not stated","not included","absent","no clause","not addressed",
           "not in the contract","not in this contract","silent on","no provision")
def is_abstention(pred, source):
    p = (pred or "").strip().lower().rstrip(" .")
    if source == "miss" or p in _MISS: return True
    return any(x in p for x in _ABSENT)

Q = {q["qid"]: q for q in (json.loads(l) for l in (BENCH/"data/questions/cuad_questions.jsonl").read_text().splitlines() if l.strip())}

def halluc_pct(rows):
    n = h = 0
    for r in rows:
        q = Q.get(r["qid"])
        if q is None: continue
        n += 1
        pred = str(r.get("prediction","") or "")
        if is_abstention(pred, r.get("source")): continue         # abstained -> not a hallucination
        gold = q["gold_answers"]
        f1 = max((_f1(pred, g) for g in gold), default=0.0) if gold else 0.0
        if f1 == 0: h += 1                                         # answered, zero grounding
    return 100*h/n if n else 0

ROWS = [("naive","naive_gpt"),("docetl","docetl_standard"),("leann","leann"),
        ("rag","rag"),("raptor","raptor"),("doctopus","doctopus"),("graphrag","graphrag_local")]
DATA = []
for disp, stem in ROWS:
    tag = stem.replace("graphrag_local","graphrag")
    p = BENCH/"methods"/tag/"results"/f"{stem}_cuad.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    DATA.append([disp, halluc_pct(rows)])

# v7 (ours): predictions from cuad_eval_records.jsonl (window)
recs = [json.loads(l) for l in (BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl").read_text().splitlines() if l.strip()]
v7rows = [{"qid": r["qid"], "prediction": r["window"]["prediction"], "source": r["window"].get("source")} for r in recs]
DATA.append(["index_v7_window", halluc_pct(v7rows)])
OURS = {"index_v7_window"}
print({r[0]: round(r[1],1) for r in DATA})

OURS_COL="#C44E52"
_PAL=["#4C72B0","#DD8452","#55A868","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
nonours=[r[0] for r in sorted(DATA) if r[0] not in OURS]
COLOR={m:_PAL[i%len(_PAL)] for i,m in enumerate(nonours)}
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","font.family":"DejaVu Sans"})

rows=sorted(DATA, key=lambda r:r[1])   # lower = better -> ascending
labels=[r[0] for r in rows]; vals=[r[1] for r in rows]
fig,ax=plt.subplots(figsize=(9,0.62*len(labels)+1.2))
y=range(len(labels)); cols=[OURS_COL if m in OURS else COLOR[m] for m in labels]
bars=ax.barh(y, vals, color=cols, edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels([f"{m}  ◄ ours" if m in OURS else m for m in labels]); ax.invert_yaxis()
for lbl,m in zip(ax.get_yticklabels(),labels):
    if m in OURS: lbl.set_fontweight("bold")
ax.set_xlabel("Hallucinated (answered, token-F1=0) (%)"); ax.set_title("CUAD Hallucination (answered with zero grounding, lower=better)", fontweight="bold")
mx=max(vals) if vals else 1
for b,v in zip(bars,vals): ax.text(b.get_width()+mx*0.01, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10)
ax.set_xlim(0, mx*1.15)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"cuad_hallucination.png", dpi=150); plt.close(fig); print("wrote", OUTD/"cuad_hallucination.png")
