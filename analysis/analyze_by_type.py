"""Per-question-type accuracy across all methods for both CUAD and NarrativeQA.

Writes:
  scratchpad_analysis/cuad_by_clause.json     — {method: {clause_type: (n_correct, n_total)}}
  scratchpad_analysis/nqa_by_wh.json          — {method: {wh_word: (n_correct, n_total)}}
  scratchpad_analysis/summary_tables.txt      — human-readable tables
"""
import json, pathlib, re, collections

BENCH = pathlib.Path("/home/phdonn/index/bench")
OUT = pathlib.Path(__file__).parent
OUT.mkdir(exist_ok=True)

# ---------- 1) CUAD: clause_type per qid ----------
CUAD_META = {}
for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
    r = json.loads(L)
    CUAD_META[r["qid"]] = {
        "clause_type": r["clause_type"],
        "has_answer":  r["has_answer"],
        "contract":    r["movie"],
        "question":    r["question"],
    }
print(f"CUAD questions loaded: {len(CUAD_META)}")

# per-method judged has_answer verdicts
CUAD_METHOD_FILES = {
    "naive":    "results/eval/cuad_halluc/naive.jsonl",
    "docetl":   "results/eval/cuad_halluc/docetl.jsonl",
    "leann":    "results/eval/cuad_halluc/leann.jsonl",
    "rag":      "results/eval/cuad_halluc/rag.jsonl",
    "graphrag": "results/eval/cuad_halluc/graphrag.jsonl",
    "raptor":   "results/eval/cuad_halluc/raptor.jsonl",
    "doctopus": "results/eval/cuad_halluc/doctopus.jsonl",
    # index (ours) has a special record file with window verdicts
}

# per-method per-question verdict on has_answer questions (correct/incorrect)
CUAD_VERD = {m: {} for m in list(CUAD_METHOD_FILES.keys()) + ["index"]}
for m, rel in CUAD_METHOD_FILES.items():
    for L in open(BENCH/rel):
        r = json.loads(L)
        CUAD_VERD[m][r["qid"]] = (r["verdict"] == "correct")

# our method: verdicts from cuad_eval_records.jsonl
for L in open(BENCH/"methods/wiki_v7_window/results/cuad_eval_records.jsonl"):
    r = json.loads(L)
    if r["has_answer"] and "window" in r:
        # window/balanced records use "correct":bool
        CUAD_VERD["index"][r["qid"]] = bool(r["window"].get("correct", False))

# per-clause accuracy on has_answer subset
cuad_by_clause = {}
for m, verd in CUAD_VERD.items():
    by = collections.defaultdict(lambda: [0, 0])  # [correct, total]
    for qid, ok in verd.items():
        ct = CUAD_META[qid]["clause_type"]
        by[ct][0] += int(ok); by[ct][1] += 1
    cuad_by_clause[m] = {k: v for k, v in by.items()}

# ---------- 2) NarrativeQA: WH categorization + per-method judged ----------
WH_PATTERNS = [
    ("Who",     re.compile(r"^\s*who\b",     re.I)),
    ("What",    re.compile(r"^\s*what\b",    re.I)),
    ("Where",   re.compile(r"^\s*where\b",   re.I)),
    ("When",    re.compile(r"^\s*when\b",    re.I)),
    ("Why",     re.compile(r"^\s*why\b",     re.I)),
    ("How",     re.compile(r"^\s*how\b",     re.I)),
    ("Which",   re.compile(r"^\s*which\b",   re.I)),
    ("Yes/No",  re.compile(r"^\s*(does|do|did|is|are|was|were|has|have|had|can|could|will|would|should)\b", re.I)),
    ("Name",    re.compile(r"^\s*name\b",    re.I)),
]

def wh_of(q):
    for label, pat in WH_PATTERNS:
        if pat.match(q): return label
    return "Other"

# tag per qid using naive's judged file (has question text)
NQA_META = {}
for L in open(BENCH/"methods/naive_gpt/results/naive_gpt_judged.jsonl"):
    r = json.loads(L)
    NQA_META[r["qid"]] = {"wh": wh_of(r["question"]), "movie": r["movie"], "question": r["question"]}
print(f"NarrativeQA questions loaded: {len(NQA_META)}")

# per-method judged records
NQA_METHOD_FILES = {
    "naive":    "methods/naive_gpt/results/naive_gpt_judged.jsonl",
    "docetl":   "methods/docetl_standard/results/docetl_standard_judged.jsonl",
    "leann":    "methods/leann/results/leann_judged.jsonl",
    "rag":      "methods/rag/results/rag_judged.jsonl",
    "graphrag": "methods/graphrag_local/results/graphrag_local_judged.jsonl",
    "raptor":   "methods/raptor/results/raptor_judged.jsonl",
    "doctopus": "methods/doctopus/results/doctopus_judged.jsonl",
}

NQA_VERD = {m: {} for m in list(NQA_METHOD_FILES.keys()) + ["index"]}
for m, rel in NQA_METHOD_FILES.items():
    p = BENCH/rel
    if not p.exists(): continue
    for L in open(p):
        r = json.loads(L)
        NQA_VERD[m][r["qid"]] = (r["verdict"] == "correct")

# ours
for L in open(BENCH/"methods/wiki_v7_window/results/nqa_eval_records.jsonl"):
    r = json.loads(L)
    NQA_VERD["index"][r["qid"]] = (r["window"]["verdict"] == "correct")

nqa_by_wh = {}
for m, verd in NQA_VERD.items():
    by = collections.defaultdict(lambda: [0, 0])
    for qid, ok in verd.items():
        wh = NQA_META[qid]["wh"] if qid in NQA_META else "Other"
        by[wh][0] += int(ok); by[wh][1] += 1
    nqa_by_wh[m] = {k: v for k, v in by.items()}

# ---------- 3) Report ----------
json.dump(cuad_by_clause, open(OUT/"cuad_by_clause.json", "w"), indent=2)
json.dump(nqa_by_wh,       open(OUT/"nqa_by_wh.json",       "w"), indent=2)

METHODS = ["naive", "docetl", "leann", "rag", "graphrag", "raptor", "doctopus", "index"]

with open(OUT/"summary_tables.txt", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("NARRATIVEQA: Accuracy by WH-word (all 885 questions)\n")
    f.write("=" * 80 + "\n")
    wh_types = ["Who", "What", "Where", "When", "Why", "How", "Which", "Yes/No", "Name", "Other"]
    f.write(f"{'WH':10s} {'N':>4s} " + " ".join(f"{m:>9s}" for m in METHODS) + "\n")
    totals = collections.Counter(m["wh"] for m in NQA_META.values())
    for wh in wh_types:
        n = totals.get(wh, 0)
        if n == 0: continue
        row = [f"{wh:10s} {n:>4d} "]
        for m in METHODS:
            d = nqa_by_wh.get(m, {}).get(wh)
            if not d: row.append(f"{'':>9s}"); continue
            row.append(f"{100*d[0]/d[1]:8.1f}%")
        f.write(" ".join(row) + "\n")

    # per-method overall on NQA
    f.write("\n")
    row = [f"{'OVERALL':10s} {len(NQA_META):>4d} "]
    for m in METHODS:
        v = NQA_VERD.get(m, {})
        if not v: row.append(f"{'':>9s}"); continue
        row.append(f"{100*sum(v.values())/len(v):8.1f}%")
    f.write(" ".join(row) + "\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("CUAD: Accuracy by clause type (has_answer subset, N=218)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'clause':45s} {'N':>3s} " + " ".join(f"{m:>9s}" for m in METHODS) + "\n")
    all_clauses = sorted({c for m in cuad_by_clause.values() for c in m})
    # sort by "our method's" accuracy descending (interesting: where do we win/lose?)
    def acc(m, c):
        d = cuad_by_clause.get(m, {}).get(c)
        return -1 if not d else 100*d[0]/d[1]
    all_clauses = sorted(all_clauses, key=lambda c: -acc("index", c))
    for c in all_clauses:
        n = max((cuad_by_clause[m].get(c, [0,0])[1] for m in METHODS), default=0)
        row = [f"{c:45s} {n:>3d} "]
        for m in METHODS:
            d = cuad_by_clause.get(m, {}).get(c)
            if not d: row.append(f"{'':>9s}"); continue
            row.append(f"{100*d[0]/d[1]:8.1f}%")
        f.write(" ".join(row) + "\n")
    # overall has_answer accuracy per method
    f.write("\n")
    row = [f"{'OVERALL has_answer':45s} {sum(v for k,v in [(c, len(m)) for m in cuad_by_clause.values() for c in m]) // len(cuad_by_clause):>3d} "]
    for m in METHODS:
        by = cuad_by_clause.get(m, {})
        tot = sum(x[1] for x in by.values())
        cor = sum(x[0] for x in by.values())
        if not tot: row.append(f"{'':>9s}"); continue
        row.append(f"{100*cor/tot:8.1f}%")
    f.write(" ".join(row) + "\n")

print("\n".join(open(OUT/"summary_tables.txt").read().splitlines()))
