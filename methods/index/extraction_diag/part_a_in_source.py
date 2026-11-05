"""Extraction diagnostic, Part A (no LLM): of the bucket-3 covered-but-failing questions, how many
are actually fixable (the gold answer is present in the source document, extract just missed it) vs a
benchmark artifact (the answer is not in the screenplay at all, it lives only in the plot summary).

Sample: all bucket-3 questions from 2 movies (10_things_i_hate_about_you, a_dry_white_season), taken
from the failure-taxonomy checkpoint. For each we check whether the gold answer appears in the whole
source document, and separately in the answer leaf's own chunk span. String + fuzzy (token) matching
plus the index's own char spans. No LLM, no build change.

Vocabulary: root, category, subject, field, source document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import json, re, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
IDX = HERE.parent
sys.path.insert(0, str(IDX))
import chunk as _chunk, store as _store   # noqa: E402
from _match import presence               # noqa: E402  (shared coverage check, Part A + Part B)

BENCH = _BENCH
FTAX = IDX/"results"/"checkpoints"/"ckpt_per_category__fb-sibling__route-cosine__k5__failtax.jsonl"
MOVIES = ["10_things_i_hate_about_you", "a_dry_white_season"]
OUTJSON = IDX/"results"/"extraction_diag_partA.json"


# --- load questions + failure taxonomy sample ---
qmap = {q["qid"]: q for q in (json.loads(l) for l in
        (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip())}
ftax = [json.loads(l) for l in FTAX.read_text().splitlines() if l.strip()]
sample = [r for r in ftax if r["bucket"] == 3 and r["slug"] in MOVIES]

# --- per movie: source text, chunk span map, original subjects per chunk (from the built leaves) ---
src, span_of, subs_of = {}, {}, {}
for slug in MOVIES:
    src[slug] = _chunk.load_source(slug)
    idx = _store.load_index(slug)
    span_of[slug], subs_of[slug] = {}, {}
    for n in idx["nodes"].values():
        if n["kind"] != "leaf":
            continue
        s = n.get("source") or {}
        cid = s.get("chunk_id")
        if cid is None:
            continue
        if cid not in span_of[slug] and s.get("start_char") is not None:
            span_of[slug][cid] = (s["start_char"], s["end_char"])
        subs_of[slug].setdefault(cid, []).append(n.get("content", ""))

results = []
for r in sample:
    slug, qid, cid = r["slug"], r["qid"], r["chunk1_id"]
    gold = qmap[qid]["gold_answers"]
    st, en = span_of[slug].get(cid, (None, None))
    chunk_text = src[slug][st:en] if st is not None else ""
    in_src = presence(gold, src[slug])
    in_chunk = presence(gold, chunk_text)
    results.append({"slug": slug, "qid": qid, "chunk_id": cid,
                    "question": qmap[qid]["question"], "gold": gold,
                    "in_source": in_src, "in_chunk": in_chunk,
                    "klass": "A1" if in_src["present"] else "A2"})

A1 = [r for r in results if r["klass"] == "A1"]
A2 = [r for r in results if r["klass"] == "A2"]
A1_chunk = [r for r in A1 if r["in_chunk"]["present"]]
A1_elsewhere = [r for r in A1 if not r["in_chunk"]["present"]]

OUTJSON.write_text(json.dumps({"sample_n": len(results), "movies": MOVIES,
                               "A1": A1, "A2": A2,
                               "A1_in_answer_chunk": len(A1_chunk),
                               "A1_elsewhere_in_source": len(A1_elsewhere)}, indent=2))

print(f"bucket-3 sample (2 movies): n={len(results)}")
print(f"  A1 fixable  (gold present in source document): {len(A1)}")
print(f"       of which in the answer leaf's own chunk : {len(A1_chunk)}")
print(f"       elsewhere in the source document        : {len(A1_elsewhere)}")
print(f"  A2 artifact (gold NOT in source at all)      : {len(A2)}")
print(f"  written {OUTJSON}")
for tag, grp in (("A1", A1), ("A2", A2)):
    print(f"\n  --- {tag} examples ---")
    for r in grp[:3]:
        print(f"   {r['qid']}  cov_src={r['in_source']['cov']} substr={r['in_source']['substr']} ent={r['in_source']['entity']}")
        print(f"     Q: {r['question']}")
        print(f"     gold: {r['gold']}")
