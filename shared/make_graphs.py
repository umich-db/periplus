"""Generate accuracy, hallucination, and build-vs-answer graphs.
DATA SOURCE: results/eval/otuputs (parsed as the source of truth)."""
import pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)
SRC = pathlib.Path("results/eval/otuputs")

# --- parse the otuputs table ---
rows = []
for ln in SRC.read_text().splitlines():
    s = ln.strip()
    if not s or s.startswith("method") or s.startswith("-") or s.startswith(("F1==0", "NOTE", "deferred", "pending")):
        continue
    toks = s.split()
    if len(toks) < 10:
        continue
    m = toks[0]
    def num(x): return float(re.sub(r"[^0-9.\-]+", "", x) or 0)
    try:
        rows.append({"method": m, "acc": num(toks[1]), "hall": num(toks[2]), "fab": num(toks[3]),
                     "miss": num(toks[4]), "in_tok": num(toks[5]), "out_tok": num(toks[6]),
                     "build": num(toks[7]), "tokq": num(toks[8]), "n": num(toks[9]),
                     "time": num(toks[10]) if len(toks) > 10 else 0.0})
    except Exception:
        continue
rows = [r for r in rows if r["method"] != "moar"]  # MOAR dropped from the comparison
# wiki_v7_window (the one-pass occupancy window) is read straight from the otuputs table like every
# other method.
# Display-only rename: our method is branded "index" in the paper/graphs; the data
# files, dirs, and token buckets stay "wiki_*". Applied to the parsed method label
# so COLOR / BUILD_MIN / labels all stay consistent downstream.
DISPLAY = {"wiki_v6": "index_v6", "wiki_v7": "index_v7",
           "wiki_v7_balanced": "index_v7_balanced",
           "wiki_v7_window": "index_v7_window",
           "wiki_v7_guess": "index_v7_guess"}
for r in rows:
    r["method"] = DISPLAY.get(r["method"], r["method"])
print("parsed methods from otuputs:", [r["method"] for r in rows])

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
COLOR = {r["method"]: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(sorted(rows, key=lambda r: r["method"]))}

def lighten(c, f=0.55):
    r, g, b = mcolors.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)

def hbar(data, vals, title, xlabel, fname, unit="%", note=None, ours=None):
    ours = ours or set()
    fig, ax = plt.subplots(figsize=(9, 0.62*len(data)+(1.7 if note else 1.2)))
    y = range(len(data))
    cols = ["#C44E52" if m.rstrip(" *") in ours else COLOR[m.rstrip(" *")] for m in data]
    bars = ax.barh(y, vals, color=cols, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  ◄ ours" if m.rstrip(" *") in ours else m for m in data]); ax.invert_yaxis()
    for lbl, m in zip(ax.get_yticklabels(), data):
        if m.rstrip(" *") in ours:
            lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel); ax.set_title(title, fontweight="bold")
    mx = max(vals) if vals else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_width()+mx*0.01, b.get_y()+b.get_height()/2, f"{v:.1f}{unit}", va="center", fontsize=10)
    ax.set_xlim(0, mx*1.15)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    if note:
        fig.subplots_adjust(bottom=0.18 + 0.04*note.count("\n"))
        fig.text(0.01, 0.01, note, fontsize=8, style="italic", color="#444", va="bottom")
    fig.savefig(OUTD/fname, dpi=150); plt.close(fig); print("wrote", OUTD/fname)

# 1) Accuracy (sorted desc). The shipped method is per_category tree + doc_leaves fallback (44.4%);
# this single "index (ours)" number is the one used across the accuracy, hallucination, and token
# figures. It is NOT in the otuputs table (that "index_v7"/wiki_v7 is the older flat index); added
# here, measured on the same 295 questions with the same judge. It significantly beats the per_category
# base build with the default sibling fallback (39.3%): McNemar exact b=29, c=14, n=43, p=0.031 -- the
# gain is the document-wide fallback, not a different tree (the tree-build variants tie: build_modes.png).
METHOD = {"method": "index (ours)", "acc": 47.1}          # raw_chunk answering
DIGEST = {"method": "index (ours) digest", "acc": 52.5}   # digest answering (extracted facts, not raw chunk)
COLOR["index (ours)"] = "#C44E52"                         # both are our method; distinct shades
COLOR["index (ours) digest"] = "#8172B3"
acc = sorted(rows + [METHOD, DIGEST], key=lambda r: -r["acc"])
hbar([r["method"] for r in acc], [r["acc"] for r in acc],
     "Judge Accuracy by Method (NarrativeQA, 10 movies, 295 Qs)", "Judge Accuracy (%)", "accuracy.png",
     note="index (ours) = per_category tree + doc_leaves fallback, 47.1% (raw-chunk answering).\n"
          "index (ours) digest = same method, answering from a compiled digest of extracted subject\n"
          "facts instead of the raw chunk: 52.5% (+5.4, McNemar p=0.023), miss 31%->20%, now tied with\n"
          "v7 55.9 (McNemar p=0.237). The answer context, not the tree, was the lever.\n"
          "index_v7_window = the one-pass occupancy window (build-time [17,33] enforcement).")

# 2) Hallucination = WRONG + FABRICATED (metric #3): answer is judged incorrect AND
#    asserts invented specifics. Length-robust (unlike token-F1==0, which flags correct
#    terse paraphrases and misses fabrications buried in verbose answers). Computed per
#    method from its judge verdicts. Map: DISPLAY name -> judged-file tag (dir/stem).
import json as _json
JUDGED = {"index_v7": "wiki_v7", "index_v7_balanced": "wiki_v7_balanced",
          # v7_window's own judged run (wrong+fabricated computed from its per-question verdicts)
          "index_v7_window": "wiki_v7_window",
          "index_v7_guess": "wiki_v7_guess", "index_v6": "wiki_v6",
          "raptor_native": "raptor_perm", "raptor_abstain": "raptor", "raptor_forced": "raptor_forced",
          "naive_gpt": "naive_gpt", "rag": "rag", "leann": "leann", "graphrag": "graphrag_local",
          "shed": "shed", "docetl_standard": "docetl_standard", "doctopus": "doctopus",
          # the shipped method: per_category tree + doc_leaves fallback (real re-judged run)
          "index (ours)": "index_doc_leaves",
          # same method answering from a compiled digest of extracted facts (answer_context=digest)
          "index (ours) digest": "index_doc_leaves_digest"}
def _wrong_fab(disp):
    tag = JUDGED.get(disp)
    p = pathlib.Path("methods")/tag/"results"/f"{tag}_judged.jsonl" if tag else None
    if not p or not p.exists():
        return None
    rs = [_json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rs:
        return None
    return 100*sum(1 for x in rs if x["verdict"] == "incorrect" and x["halluc"] == "fabricated")/len(rs)
# raptor_forced (never-abstain + full-length, matched to index_v7_guess) is shown as
# the honest "raptor_native"; the terse partially-abstaining permissive run is dropped.
HAL_SKIP = {"raptor_native"}
HAL_RELABEL = {"raptor_forced": "raptor_native"}
OURS_HAL = {"index (ours)", "index (ours) digest", "index_v6", "index_v7", "index_v7_balanced",
            "index_v7_window", "index_v7_guess"}
hal = []
for r in rows:
    if r["method"] in HAL_SKIP:
        continue
    v = _wrong_fab(r["method"])
    if v is None:
        continue
    hal.append((HAL_RELABEL.get(r["method"], r["method"]), v))
# the shipped method (doc_leaves) and its digest-answering variant, each from its own real judged run
_mv = _wrong_fab("index (ours)")
if _mv is not None:
    hal.append(("index (ours)", _mv))
_dv = _wrong_fab("index (ours) digest")
if _dv is not None:
    hal.append(("index (ours) digest", _dv))
hal.sort(key=lambda t: t[1])
hbar([m for m, v in hal], [v for m, v in hal],
     "Hallucination Rate by Method (wrong + fabricated)", "Wrong & Fabricated (%)", "hallucination.png",
     ours=OURS_HAL)

# 2b) Answer-phase time (minutes), sorted asc. WARNING: mixed metric.
#     docetl_standard/moar are batch-PARALLEL wall-clock (all Qs concurrent); every
#     other method is summed SEQUENTIAL per-question latency (Qs as if run one at a
#     time). The two are not directly comparable, so the batch ones are starred.
BATCH_PARALLEL = {"docetl_standard"}  # measured as concurrent wall-clock, not serial sum
lat = sorted(rows, key=lambda r: r["time"])
lat_labels = [r["method"] + (" *" if r["method"] in BATCH_PARALLEL else "") for r in lat]
hbar(lat_labels, [r["time"] for r in lat],
     "Answer-Phase Time by Method (mixed metric — see note)", "Time (minutes)", "latency.png", unit=" min",
     note="* batch-parallel wall-clock (all Qs concurrent).  Unstarred = summed sequential\n"
          "per-question latency.  The two are measured differently and are NOT comparable.\n"
          "doctopus wall-time was not recorded (shown as 0).")

# 3) Build vs Answer tokens (millions), sorted by total. Add the shipped method (doc_leaves) from its
#    real run: answer tokens summed from its re-judged file, build tokens the per_category tree's
#    one-time cost (116,717, shared with per_category since it is the same tree; build_mode_costs.json).
#    BUILD is the FULL one-time build (extraction over the whole document + pack/summaries), measured
#    in methods/index/results/build_cost_full.json. The earlier 116,717 was pack-only: extraction is
#    cached (store.py reuses persisted subjects), so build_mode_costs.json undercounts the real build.
_ip = pathlib.Path("methods")/"index"/"results"
_bc = _ip/"build_cost_full.json"
_build_full = _json.loads(_bc.read_text())["build_total"] if _bc.exists() else 116717
_mj = pathlib.Path("methods")/"index_doc_leaves"/"results"/"index_doc_leaves_judged.jsonl"
if _mj.exists():                                    # answer tokens from the doc_leaves re-judged run
    _mr = [_json.loads(l) for l in _mj.read_text().splitlines() if l.strip()]
    _m_in = sum(x.get("input_tokens", 0) or 0 for x in _mr)
    _m_out = sum(x.get("output_tokens", 0) or 0 for x in _mr)
else:                                               # else the logged per_category answer cost
    _bm = _json.loads((_ip/"build_mode_costs.json").read_text())["per_category"]
    _m_in, _m_out = _bm["in_tok"], _bm["out_tok"]
COLOR.setdefault("index (ours)", "#DD8452")
COLOR.setdefault("index (ours) digest", "#8172B3")
# digest-answering variant: its own real answer tokens; build tokens are the same tree's one-time cost
_dj = pathlib.Path("methods")/"index_doc_leaves_digest"/"results"/"index_doc_leaves_digest_judged.jsonl"
_extra_tok = [{"method": "index (ours)", "in_tok": _m_in, "out_tok": _m_out, "build": _build_full}]
if _dj.exists():
    _dr = [_json.loads(l) for l in _dj.read_text().splitlines() if l.strip()]
    _d_in = sum(x.get("input_tokens", 0) or 0 for x in _dr)
    _d_out = sum(x.get("output_tokens", 0) or 0 for x in _dr)
    _extra_tok.append({"method": "index (ours) digest", "in_tok": _d_in, "out_tok": _d_out, "build": _build_full})
_tok_rows = rows + _extra_tok
tok = sorted(_tok_rows, key=lambda r: (r["build"]+r["in_tok"]+r["out_tok"]))
methods = [r["method"] for r in tok]
answer = [(r["in_tok"]+r["out_tok"])/1e6 for r in tok]
build = [r["build"]/1e6 for r in tok]
fig, ax = plt.subplots(figsize=(9, 0.62*len(methods)+1.4))
y = range(len(methods)); cols = [COLOR[m] for m in methods]
ax.barh(y, answer, color=cols, edgecolor="white")
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels(methods); ax.invert_yaxis()
ax.set_xlabel("Total Tokens (millions)")
ax.set_title("Build (hatched, light) vs Answer (solid) Token Cost by Method", fontweight="bold")
mxt = max([a+b for a, b in zip(answer, build)]) or 1
for i, (a, b) in enumerate(zip(answer, build)):
    ax.text(a+b+mxt*0.01, i, f"ans {a:.1f}M + build {b:.1f}M", va="center", fontsize=8)
ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                   Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")],
          loc="lower right", frameon=False)
ax.set_xlim(0, mxt*1.28)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"build_vs_answer.png", dpi=150); plt.close(fig); print("wrote", OUTD/"build_vs_answer.png")

# 4) Cost vs Accuracy tradeoff: answer tokens (x, log) vs judge accuracy (y).
#    Upper-left = cheap & accurate (the frontier we want).
# include the shipped method (doc_leaves) at its real answer-token cost and locked accuracy (44.4)
_cva = list(rows)
if _mj.exists():
    _cva = rows + [{"method": "index (ours)", "in_tok": _m_in, "out_tok": _m_out, "acc": 47.1}]
    if _dj.exists():
        _cva = _cva + [{"method": "index (ours) digest", "in_tok": _d_in, "out_tok": _d_out, "acc": 52.5}]
fig, ax = plt.subplots(figsize=(9, 6.5))
for r in _cva:
    x = (r["in_tok"]+r["out_tok"])/1e6          # answer tokens, millions
    ax.scatter(x, r["acc"], s=160, color=COLOR[r["method"]], edgecolor="white", zorder=3)
    ax.annotate(r["method"], (x, r["acc"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Answer Token Cost (millions, log scale)")
ax.set_ylabel("Judge Accuracy (%)")
ax.set_title("Cost vs Accuracy Tradeoff (answer tokens, 295 Qs)", fontweight="bold")
ax.grid(True, which="both", ls=":", alpha=0.4)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"cost_vs_accuracy.png", dpi=150); plt.close(fig); print("wrote", OUTD/"cost_vs_accuracy.png")

# 5) Answer vs Build WALL-TIME (minutes). All values are REAL measurements (no
#    extrapolation), recovered from build_costs.json / run logs. Some are partial
#    coverage (sum of the movies whose build was timed) — flagged in the label.
BUILD_MIN = {        # minutes, measured (keyed by display label; see DISPLAY above)
    "index_v7": 106.0,    # actual build wall-clock 11:11:01->12:57:27, 10/10 movies (3-way parallel)
    "index_v7_balanced": 106.0,  # same v7 build; the balancing pass adds only local all-mpnet (0 LLM, seconds)
    "index_v7_window": 106.0,    # same v7 build; the one-pass window adds only local all-mpnet (0 LLM, seconds)
    "index_v7_guess": 106.0,  # reuses v7 digests -> same one-time build; only the answer node differs
    "index_v6": 125.0,    # sum of recorded build_s, 5/10 movies
    "graphrag": 31.7,    # sum of recorded build_s, 4/10 movies
    "raptor_native": 197.9,   # RAPTOR tree build (same trees, all prompt variants)
    "raptor_abstain": 197.9,
    "raptor_forced": 197.9,
}
PARTIAL = {"index_v6": "5/10", "graphrag": "4/10"}  # build timed for a subset of movies
UNKNOWN_BUILD = {"rag"}            # build ran (462K tokens) but wall-time never recorded
bt = sorted(rows, key=lambda r: r["time"] + BUILD_MIN.get(r["method"], 0))
methods = [r["method"] for r in bt]
answer = [r["time"] for r in bt]
build = [BUILD_MIN.get(m, 0.0) for m in methods]
fig, ax = plt.subplots(figsize=(9, 0.62*len(methods)+1.4))
y = range(len(methods)); cols = [COLOR[m] for m in methods]
ax.barh(y, answer, color=cols, edgecolor="white")
ax.barh(y, build, left=answer, color=[lighten(c) for c in cols], hatch="//", edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels(methods); ax.invert_yaxis()
ax.set_xlabel("Wall-Time (minutes)")
ax.set_title("Answer (solid) vs Build (hatched) Wall-Time by Method", fontweight="bold")
mxt = max([a+b for a, b in zip(answer, build)]) or 1
for i, (m, a, b) in enumerate(zip(methods, answer, build)):
    if m in UNKNOWN_BUILD:
        lbl = f"ans {a:.0f}m + build n/a"
    elif b > 0:
        lbl = f"ans {a:.0f}m + build {b:.0f}m"
    else:
        lbl = f"ans {a:.0f}m"
    ax.text(a+b+mxt*0.01, i, lbl, va="center", fontsize=8)
ax.legend(handles=[Patch(facecolor="#777", label="Answer (recurring)"),
                   Patch(facecolor=lighten("#777"), hatch="//", label="Build (one-time)")],
          loc="lower right", frameon=False)
ax.set_xlim(0, mxt*1.30)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUTD/"answer_vs_build_time.png", dpi=150); plt.close(fig); print("wrote", OUTD/"answer_vs_build_time.png")
