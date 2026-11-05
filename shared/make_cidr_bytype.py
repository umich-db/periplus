"""make_cidr_bytype.py: per-question-type accuracy comparisons for the analysis section.

Outputs (bench/report/cidr/analysis/{nqa,cuad}/):
  nqa/nqa_by_wh.png             per-WH accuracy across all methods
  nqa/nqa_decomposition.png     score decomposition: factual vs reasoning
  nqa/nqa_breakdown_ours.png    our-method outcome breakdown per WH
  cuad/cuad_by_clause.png       per-clause-bucket accuracy across all methods
  cuad/cuad_decomposition.png   score decomposition: answerable vs abstention
  cuad/cuad_breakdown_ours.png  our-method outcome breakdown per clause bucket

Reuses styling of make_cidr_split.py: per-method colours, LaTeX-style serif
fonts, black outline on our method's bar.

Run from bench/:  envs/base/bin/python shared/make_cidr_bytype.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import json, pathlib, re, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = _BENCH
OUTD  = BENCH/"report"/"cidr"/"analysis"
(OUTD/"nqa").mkdir(parents=True, exist_ok=True)
(OUTD/"cuad").mkdir(parents=True, exist_ok=True)
OURS  = "index"

COLOURS = {
    "naive":    "#8C8C8C", "docetl": "#4C72B0", "leann": "#8172B2",
    "rag":      "#C44E52", "graphrag": "#55A868", "raptor": "#CCB974",
    "doctopus": "#DD8452", "index": "#D62728",
}
METHODS = ["naive", "docetl", "leann", "rag", "graphrag", "raptor", "doctopus", "index"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.unicode_minus": False,
})

# ---------- 1) NarrativeQA by WH ----------
WH_PATTERNS = [
    ("Who",   re.compile(r"^\s*who\b",   re.I)),
    ("What",  re.compile(r"^\s*what\b",  re.I)),
    ("Where", re.compile(r"^\s*where\b", re.I)),
    ("When",  re.compile(r"^\s*when\b",  re.I)),
    ("Why",   re.compile(r"^\s*why\b",   re.I)),
    ("How",   re.compile(r"^\s*how\b",   re.I)),
]
def wh_of(q):
    for lbl, p in WH_PATTERNS:
        if p.match(q): return lbl
    return None

NQA_META = {}
for L in open(BENCH/"methods/naive_gpt/results/naive_gpt_judged.jsonl"):
    r = json.loads(L)
    w = wh_of(r["question"])
    if w: NQA_META[r["qid"]] = w

NQA_METHOD_FILES = {
    "naive":    "methods/naive_gpt/results/naive_gpt_judged.jsonl",
    "docetl":   "methods/docetl_standard/results/docetl_standard_judged.jsonl",
    "leann":    "methods/leann/results/leann_judged.jsonl",
    "rag":      "methods/rag/results/rag_judged.jsonl",
    "graphrag": "methods/graphrag_local/results/graphrag_local_judged.jsonl",
    "raptor":   "methods/raptor/results/raptor_judged.jsonl",
    "doctopus": "methods/doctopus/results/doctopus_judged.jsonl",
}
NQA_VERD = {m: {} for m in METHODS}
for m, rel in NQA_METHOD_FILES.items():
    for L in open(BENCH/rel):
        r = json.loads(L)
        NQA_VERD[m][r["qid"]] = (r["verdict"] == "correct")
for L in open(BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl"):
    r = json.loads(L)
    NQA_VERD["index"][r["qid"]] = (r["window"]["verdict"] == "correct")

nqa_by_wh = {m: collections.defaultdict(lambda: [0,0]) for m in METHODS}
for qid, wh in NQA_META.items():
    for m in METHODS:
        v = NQA_VERD[m].get(qid)
        if v is None: continue
        nqa_by_wh[m][wh][0] += int(v); nqa_by_wh[m][wh][1] += 1

# ---------- 2) CUAD by aggregated clause type ----------
# Group the 41 clause categories into 5 semantic buckets. Each bucket is a claim
# about the answer's SHAPE: what the extraction pipeline has to preserve.
CUAD_BUCKETS = {
    "Named entities /\ndates": [
        "Agreement Date", "Effective Date", "Expiration Date", "Parties",
        "Document Name", "Governing Law",
    ],
    "Boolean-style\nprovisions": [
        "Anti-Assignment", "Change Of Control", "Non-Compete", "Non-Disparagement",
        "Non-Transferable License", "No-Solicit Of Customers", "No-Solicit Of Employees",
        "Exclusivity", "Third Party Beneficiary", "Uncapped Liability",
        "Irrevocable Or Perpetual License", "Unlimited/All-You-Can-Eat-License",
        "Volume Restriction", "Price Restrictions", "Competitive Restriction Exception",
        "Covenant Not To Sue", "Most Favored Nation", "Minimum Commitment",
    ],
    "Quantified\nterms": [
        "Cap On Liability", "Liquidated Damages", "Renewal Term", "Warranty Duration",
        "Notice Period To Terminate Renewal", "Revenue/Profit Sharing",
    ],
    "Complex\nrights": [
        "License Grant", "Affiliate License-Licensee", "Affiliate License-Licensor",
        "Ip Ownership Assignment", "Joint Ip Ownership", "Rofr/Rofo/Rofn",
        "Post-Termination Services", "Termination For Convenience",
        "Source Code Escrow", "Audit Rights", "Insurance",
    ],
}
CT_TO_BUCKET = {c: b for b, cs in CUAD_BUCKETS.items() for c in cs}

CUAD_META = {}
for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
    r = json.loads(L)
    CUAD_META[r["qid"]] = {"clause_type": r["clause_type"], "has_answer": r["has_answer"]}

CUAD_METHOD_FILES = {
    "naive":    "results/eval/cuad_halluc/naive.jsonl",
    "docetl":   "results/eval/cuad_halluc/docetl.jsonl",
    "leann":    "results/eval/cuad_halluc/leann.jsonl",
    "rag":      "results/eval/cuad_halluc/rag.jsonl",
    "graphrag": "results/eval/cuad_halluc/graphrag.jsonl",
    "raptor":   "results/eval/cuad_halluc/raptor.jsonl",
    "doctopus": "results/eval/cuad_halluc/doctopus.jsonl",
}
CUAD_VERD = {m: {} for m in METHODS}
for m, rel in CUAD_METHOD_FILES.items():
    for L in open(BENCH/rel):
        r = json.loads(L)
        CUAD_VERD[m][r["qid"]] = (r["verdict"] == "correct")
for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
    r = json.loads(L)
    if r["has_answer"] and "window" in r:
        CUAD_VERD["index"][r["qid"]] = bool(r["window"].get("correct", False))

cuad_by_bucket = {m: collections.defaultdict(lambda: [0,0]) for m in METHODS}
for m in METHODS:
    for qid, ok in CUAD_VERD[m].items():
        ct = CUAD_META[qid]["clause_type"]
        b = CT_TO_BUCKET.get(ct, "Other")
        cuad_by_bucket[m][b][0] += int(ok); cuad_by_bucket[m][b][1] += 1

# ---------- Plotting ----------
def grouped_bars(data_by_method, categories, title, out, ylabel="Judge Accuracy (%)",
                 legend_ncol=4, figsize=(11, 5.4)):
    fig, ax = plt.subplots(figsize=figsize)
    n_methods = len(METHODS); n_cats = len(categories)
    x = np.arange(n_cats)
    w = 0.85 / n_methods
    for i, m in enumerate(METHODS):
        vals = []
        for c in categories:
            d = data_by_method[m].get(c, [0, 0])
            vals.append(100*d[0]/d[1] if d[1] else np.nan)
        bars = ax.bar(x + (i - n_methods/2 + 0.5)*w, vals, w,
                      color=COLOURS[m], edgecolor="white", linewidth=0.6,
                      label=f"{m} (ours)" if m == OURS else m)
        if m == OURS:
            for b in bars:
                b.set_edgecolor("black"); b.set_linewidth(1.4)
    labels = []
    for c in categories:
        # sample size for the caption
        ns = [data_by_method[m].get(c, [0,0])[1] for m in METHODS if data_by_method[m].get(c, [0,0])[1]]
        n = max(ns) if ns else 0
        labels.append(f"{c}\n(n={n})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=42)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(ncol=legend_ncol, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

grouped_bars(nqa_by_wh, ["Who","What","Where","When","Why","How"],
             "NarrativeQA: Accuracy by Question Type (WH-word)",
             OUTD/"nqa"/"nqa_by_wh.png")

grouped_bars(cuad_by_bucket,
             ["Named entities /\ndates", "Boolean-style\nprovisions",
              "Quantified\nterms", "Complex\nrights"],
             "CUAD: Accuracy by Clause Category (answerable subset, N=218)",
             OUTD/"cuad"/"cuad_by_clause.png", figsize=(10, 5.4))

# ---------- 3) CUAD decomposition: has_answer accuracy vs. no_answer abstention ----------
# For each method, overall CUAD accuracy is a weighted mix of:
#   (a) accuracy on the 218 answerable questions, and
#   (b) abstention rate on the 192 no-answer questions (abstaining IS the correct action).
# Our method's overall CUAD win comes almost entirely from (b), NOT (a).
CUAD_DECOMP = {
    "naive":    {"has_answer": 46.3, "abstain": 72.4, "overall": 58.5},
    "docetl":   {"has_answer": 58.3, "abstain": 69.8, "overall": 63.7},
    "leann":    {"has_answer": 46.3, "abstain": 83.3, "overall": 63.7},
    "rag":      {"has_answer": 53.2, "abstain": 62.5, "overall": 57.6},
    "graphrag": {"has_answer": 49.5, "abstain": 77.0, "overall": 61.9},
    "raptor":   {"has_answer": 44.7, "abstain": 65.2, "overall": 54.2},
    "doctopus": {"has_answer": 44.9, "abstain": 61.2, "overall": 54.8},
    "index":    {"has_answer": 37.2, "abstain": 90.8, "overall": 62.3},
}

def plot_cuad_decomp(out):
    ordered = sorted(METHODS, key=lambda m: -CUAD_DECOMP[m]["overall"])
    x = np.arange(len(ordered))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ha_vals = [CUAD_DECOMP[m]["has_answer"] for m in ordered]
    ab_vals = [CUAD_DECOMP[m]["abstain"]    for m in ordered]
    ov_vals = [CUAD_DECOMP[m]["overall"]    for m in ordered]

    bars_ha = ax.bar(x - w/2, ha_vals, w,
                     color=[COLOURS[m] for m in ordered],
                     edgecolor="white", label="Answerable subset (N=218)")
    bars_ab = ax.bar(x + w/2, ab_vals, w,
                     color=[COLOURS[m] for m in ordered], alpha=0.55,
                     hatch="//", edgecolor="white",
                     label="Abstention on no-answer clauses (N=192)")

    for m, bars in [("ha", bars_ha), ("ab", bars_ab)]:
        for b, meth in zip(bars, ordered):
            if meth == OURS:
                b.set_edgecolor("black"); b.set_linewidth(1.6)

    # overall accuracy as a small text annotation above each pair
    for i, (m, ov) in enumerate(zip(ordered, ov_vals)):
        ax.text(x[i], 100, f"overall\n{ov:.1f}%", ha="center", va="top",
                fontsize=10, color="#222",
                fontweight=("bold" if m == OURS else "normal"))

    labels = [f"{m}\n(ours)" if m == OURS else m for m in ordered]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for t, m in zip(ax.get_xticklabels(), ordered):
        if m == OURS: t.set_fontweight("bold")
    ax.set_ylabel("Accuracy / Abstention Rate (%)")
    ax.set_title("CUAD: How each method earns its overall score",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

plot_cuad_decomp(OUTD/"cuad"/"cuad_decomposition.png")

# ---------- 3b) NQA decomposition: factual vs reasoning ----------
# NQA has no no-answer questions, so the CUAD-style split (answerable vs abstain)
# does not apply. Instead we decompose each method's NQA score into:
#   FACTUAL   = accuracy on Who + What + Where + When (concrete-entity questions, N=659)
#   REASONING = accuracy on Why + How                 (causal / procedural,  N=173)
# The split mirrors the CUAD chart's shape: two bars per method + overall on top.
FACTUAL   = {"Who","What","Where","When"}
REASONING = {"Why","How"}

def compute_split(method, wh_set):
    correct = total = 0
    for qid, wh in NQA_META.items():
        if wh not in wh_set: continue
        v = NQA_VERD[method].get(qid)
        if v is None: continue
        correct += int(v); total += 1
    return (100*correct/total if total else 0.0, total)

NQA_DECOMP = {}
for m in METHODS:
    fa, fn = compute_split(m, FACTUAL)
    re_, rn = compute_split(m, REASONING)
    ov_c = sum(NQA_VERD[m].values()); ov_n = len(NQA_VERD[m]) or 1
    NQA_DECOMP[m] = {"factual": fa, "factual_n": fn,
                     "reasoning": re_, "reasoning_n": rn,
                     "overall": 100*ov_c/ov_n}

def plot_nqa_decomp(out):
    ordered = sorted(METHODS, key=lambda m: -NQA_DECOMP[m]["overall"])
    x = np.arange(len(ordered))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fa_vals = [NQA_DECOMP[m]["factual"]   for m in ordered]
    re_vals = [NQA_DECOMP[m]["reasoning"] for m in ordered]
    ov_vals = [NQA_DECOMP[m]["overall"]   for m in ordered]

    fa_n = NQA_DECOMP[ordered[0]]["factual_n"]
    re_n = NQA_DECOMP[ordered[0]]["reasoning_n"]

    bars_fa = ax.bar(x - w/2, fa_vals, w,
                     color=[COLOURS[m] for m in ordered],
                     edgecolor="white",
                     label=f"Factual: Who/What/Where/When (N={fa_n})")
    bars_re = ax.bar(x + w/2, re_vals, w,
                     color=[COLOURS[m] for m in ordered], alpha=0.55,
                     hatch="//", edgecolor="white",
                     label=f"Reasoning: Why/How (N={re_n})")

    for bars in (bars_fa, bars_re):
        for b, meth in zip(bars, ordered):
            if meth == OURS:
                b.set_edgecolor("black"); b.set_linewidth(1.6)

    for i, (m, ov) in enumerate(zip(ordered, ov_vals)):
        ax.text(x[i], 100, f"overall\n{ov:.1f}%", ha="center", va="top",
                fontsize=10, color="#222",
                fontweight=("bold" if m == OURS else "normal"))

    labels = [f"{m}\n(ours)" if m == OURS else m for m in ordered]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for t, m in zip(ax.get_xticklabels(), ordered):
        if m == OURS: t.set_fontweight("bold")
    ax.set_ylabel("Judge Accuracy (%)")
    ax.set_title("NarrativeQA: How each method earns its overall score",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

plot_nqa_decomp(OUTD/"nqa"/"nqa_decomposition.png")

# ---------- 4) Our method's answer breakdown by question type ----------
# For every question, tag the outcome:
#   CORRECT              — judge said correct
#   WRONG (from digest)  — wiki source returned an answer, judge said incorrect
#   WRONG (from fallback)— fallback source returned an answer, judge said incorrect
#   MISS                 — returned "not found" (never abstained on has-answer)
OUT_COLOURS = {
    "Correct":           "#4CAF50",  # green
    "Wrong (digest)":    "#FF9800",  # orange
    "Wrong (fallback)":  "#F57C00",  # deeper orange
    "Miss (not found)":  "#B71C1C",  # deep red
}
OUT_ORDER = ["Correct", "Wrong (digest)", "Wrong (fallback)", "Miss (not found)"]

def classify(rec):
    """Return the outcome label for one prediction record (window)."""
    v = rec.get("verdict", "correct" if rec.get("correct") else "incorrect")
    src = rec.get("source", "wiki")
    if v == "correct":                     return "Correct"
    if src == "miss":                      return "Miss (not found)"
    if src == "fallback":                  return "Wrong (fallback)"
    return "Wrong (digest)"

# NQA breakdown by WH
nqa_break = {wh: {k: 0 for k in OUT_ORDER} for wh in ["Who","What","Where","When","Why","How"]}
for L in open(BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl"):
    r = json.loads(L)
    w = wh_of(r["question"])
    if w is None: continue
    nqa_break[w][classify(r["window"])] += 1

# CUAD breakdown by bucket, only answerable questions (no-answer abstention is trivially "correct")
cuad_break = {b: {k: 0 for k in OUT_ORDER} for b in CUAD_BUCKETS}
for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
    r = json.loads(L)
    if not r["has_answer"]: continue
    b = CT_TO_BUCKET.get(r["clause_type"])
    if b: cuad_break[b][classify(r["window"])] += 1

def stacked_breakdown(data, title, out, figsize=(9.5, 4.6)):
    """Horizontal stacked bars: for each question type, % of questions falling in each outcome bucket."""
    cats = list(data.keys())
    totals = [sum(data[c].values()) for c in cats]
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(cats))
    lefts = np.zeros(len(cats), dtype=float)
    for label in OUT_ORDER:
        vals = np.array([100*data[c][label]/max(totals[i], 1) for i, c in enumerate(cats)])
        bars = ax.barh(y, vals, left=lefts, height=0.66,
                       color=OUT_COLOURS[label], edgecolor="white", linewidth=0.6, label=label)
        for b, v in zip(bars, vals):
            if v >= 6:
                ax.text(b.get_x()+b.get_width()/2, b.get_y()+b.get_height()/2,
                        f"{v:.0f}%", ha="center", va="center",
                        color=("white" if label in ("Miss (not found)", "Wrong (fallback)") else "#222"),
                        fontsize=10, fontweight="bold")
        lefts += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n(n={t})" for c, t in zip(cats, totals)])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of questions in category (%)")
    ax.set_title(title, fontweight="bold", pad=42)
    ax.legend(ncol=len(OUT_ORDER), frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.35); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

stacked_breakdown(nqa_break,
                  "Our method on NarrativeQA: outcome breakdown by question type",
                  OUTD/"nqa"/"nqa_breakdown_ours.png")
stacked_breakdown(cuad_break,
                  "Our method on CUAD: outcome breakdown by clause category (answerable subset)",
                  OUTD/"cuad"/"cuad_breakdown_ours.png")
