"""Per-question answer latency (median + p95) and build wall-time charts.
DATA SOURCE: results/eval/latency.json (median/p95 per question, single-thread).
  - 7 methods: median/p95 from recorded per-question latency_s.
  - docetl_standard/moar/doctopus: re-measured LIVE (N=25, single-thread) because
    their original runs never recorded per-question latency.
Build minutes: measured/recovered; coverage flagged (leann unmeasured, GPU env absent).
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)
D = json.loads(pathlib.Path("results/eval/latency.json").read_text())
D.pop("moar", None)  # MOAR dropped from the comparison

# Overlay freshly-measured build wall-time (buildtimer.py) over the hand-set
# numbers in latency.json. Each method's runner writes build_times/<m>.json
# during its build phase; prefer those when present.
_BTD = pathlib.Path("results/eval/build_times")
if _BTD.is_dir():
    for _p in _BTD.glob("*.json"):
        _bt = json.loads(_p.read_text())
        if _bt["method"] in D:
            D[_bt["method"]]["build_min"] = _bt["build_min"]
            D[_bt["method"]]["build_coverage"] = _bt.get("build_coverage", "full")

RE = {"docetl_standard","moar","doctopus"}  # re-measured live († unless a probe overrides below)

# Overlay genuine single-thread answer latency (latencyprobe.py) for the batched/
# concurrent methods (doctopus/moar/docetl_standard), replacing their
# "re-measured live" placeholders. The other 7 keep their recorded latency_s.
_LPD = pathlib.Path("results/eval/latency_probe")
if _LPD.is_dir():
    for _p in _LPD.glob("*.json"):
        _lp = json.loads(_p.read_text())
        if _lp["method"] in D and _lp.get("answer_n"):
            for _k in ("answer_median_s", "answer_p95_s", "answer_n", "answer_src"):
                if _k in _lp:
                    D[_lp["method"]][_k] = _lp[_k]
            RE.discard(_lp["method"])  # genuine single-thread now: drop the † re-measure flag
_PAL = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
COLOR = {m: _PAL[i % len(_PAL)] for i, m in enumerate(sorted(D))}

# ---- 1) per-question answer latency: median bar + p95 whisker, sorted by median ----
ms = sorted(D, key=lambda m: D[m]["answer_median_s"])
med = [D[m]["answer_median_s"] for m in ms]
p95 = [D[m]["answer_p95_s"] for m in ms]
labels = list(ms)
fig, ax = plt.subplots(figsize=(9, 0.62*len(ms)+1.7))
y = range(len(ms))
ax.barh(y, med, color=[COLOR[m] for m in ms], edgecolor="white", zorder=3)
# p95 whisker from median to p95
ax.errorbar(med, list(y), xerr=[[0]*len(ms), [p-m for p, m in zip(p95, med)]],
            fmt="none", ecolor="#333", elinewidth=1.3, capsize=4, zorder=4)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("Per-question answer latency (seconds), single-thread")
ax.set_title("Answer Latency by Method — median bar, whisker = p95", fontweight="bold")
mx = max(p95)
for i, (m, p) in enumerate(zip(med, p95)):
    ax.text(p+mx*0.01, i, f"med {m:.1f}s · p95 {p:.0f}s", va="center", fontsize=8)
ax.set_xlim(0, mx*1.20)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD/"answer_latency.png", dpi=150); plt.close(fig); print("wrote", OUTD/"answer_latency.png")

# ---- 2) build wall-time (one-time), coverage flagged ----
bm = [(m, D[m].get("build_min"), D[m].get("build_coverage")) for m in D]
known = sorted([x for x in bm if x[1] is not None], key=lambda x: x[1])
unk = [x for x in bm if x[1] is None]
order = known + unk
labels = [f"{m}" + (f" ({cov})" if (v is not None and cov not in ("none", "full")) else "")
          for m, v, cov in order]
vals = [(v if v is not None else 0) for _, v, _ in order]
fig, ax = plt.subplots(figsize=(9, 0.62*len(order)+1.6))
y = range(len(order))
bars = ax.barh(y, vals, color=[COLOR[m] for m, _, _ in order], edgecolor="white",
               hatch=["" if v is not None else "xx" for _, v, _ in order])
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("One-time build wall-time (minutes)")
ax.set_title("Build Wall-Time by Method (one-time, amortized over queries)", fontweight="bold")
mx = max(vals) or 1
for i, (m, v, cov) in enumerate(order):
    txt = "not measured (GPU env absent)" if v is None else (f"{v:.2f} min" if v < 1 else f"{v:.0f} min")
    ax.text(mx*0.01 if v is None else v+mx*0.01, i, txt, va="center", fontsize=8,
            color="#888" if v is None else "black")
ax.set_xlim(0, mx*1.22)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUTD/"build_time.png", dpi=150); plt.close(fig); print("wrote", OUTD/"build_time.png")
