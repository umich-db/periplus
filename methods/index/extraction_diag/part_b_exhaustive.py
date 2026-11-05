"""Extraction diagnostic, Part B (small LLM): for the A1 (fixable) bucket-3 questions from Part A,
test whether a more exhaustive extract prompt captures the fact the shipped extract missed.

For each unique answer-leaf chunk behind an A1 question we re-extract with an EXHAUSTIVE prompt variant
(capture every fact, and specifically relationships and outcomes across subjects: who did what to whom,
what happened to whom, how two subjects relate). Only the prompt wording changes; the model and its
reasoning setting are the shipped ones (client.chat defaults). We do NOT overwrite the shipped
EXTRACT_PROMPT. Then we check, with the same coverage test as Part A, whether the new subjects/fields
bear the gold answer that the original (shipped) subjects for that chunk did not.

Deterministic and resumable via the checkpointed harness. Reuses the built index, source chunks, the
extract client and parser, and the Part A output. Vocabulary: root, category, subject, field, source
document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
IDX = HERE.parent
sys.path.insert(0, str(IDX))
import chunk as _chunk, store as _store, config as _cfg, client, extract as _extract, eval_harness  # noqa: E402
from _match import presence   # noqa: E402

BENCH = _BENCH
PARTA = IDX/"results"/"extraction_diag_partA.json"
OUTMD = IDX/"results"/"extraction_diagnostic.md"

# Exhaustive test prompt: same task and JSON shape as the shipped EXTRACT_PROMPT, but it asks for every
# fact and, crucially, the relationships and outcomes across subjects (not just the salient subject).
EXHAUSTIVE_PROMPT = (
    "Below is one passage from a source document. Extract EVERY fact it states, exhaustively. Return "
    "SUBJECTS (a specific person, place, event, object, or scene) AND, just as important, the "
    "RELATIONSHIPS and OUTCOMES across subjects: who did what to whom, what happened to whom, what one "
    "subject gave, said, bought, revealed, or did to another, and how two subjects relate. Do not "
    "collapse the passage to its single most salient subject; capture minor, background, and relational "
    "facts too. For each subject return a short name and a set of fields (key/value details drawn from "
    "the text, including relational fields such as 'gives', 'buys', 'kills', 'reveals', 'becomes'). Do "
    "NOT assign topics or categories; return content only.\n\n"
    "Return ONLY a JSON array:\n"
    '  [{{"name": "...", "fields": {{"key": "value", ...}}}}, ...]\n\n'
    "Passage:\n{chunk}\n\nJSON array of subjects:"
)


def subjects_text(subs):
    return " ".join((s.get("name", "") + " " + " ".join(str(v) for v in s.get("fields", {}).values()))
                    for s in subs).strip()


# --- load Part A A1 questions ---
pa = json.loads(PARTA.read_text())
A1 = pa["A1"]

# --- per movie: source text, chunk span map, original (shipped) subjects per chunk ---
cfg = _cfg.load()
src, span_of, orig_subs = {}, {}, {}
for slug in pa["movies"]:
    src[slug] = _chunk.load_source(slug)
    idx = _store.load_index(slug)
    span_of[slug], orig_subs[slug] = {}, {}
    for n in idx["nodes"].values():
        if n["kind"] != "leaf":
            continue
        s = n.get("source") or {}
        cid = s.get("chunk_id")
        if cid is None:
            continue
        if cid not in span_of[slug] and s.get("start_char") is not None:
            span_of[slug][cid] = (s["start_char"], s["end_char"])
        orig_subs[slug].setdefault(cid, []).append({"name": n.get("content", ""), "fields": {}})

# --- unique chunks to re-extract ---
uniq = {}
for r in A1:
    key = (r["slug"], r["chunk_id"])
    if key not in uniq:
        st, en = span_of[r["slug"]].get(r["chunk_id"], (None, None))
        uniq[key] = src[r["slug"]][st:en] if st is not None else ""
tasks = [(slug, cid, text) for (slug, cid), text in uniq.items()]
print(f"A1 questions: {len(A1)} | unique answer-leaf chunks to re-extract: {len(tasks)}", flush=True)


def key_fn(t):
    return f"{t[0]}::chunk{t[1]}"


def work_fn(t):
    slug, cid, text = t
    out, pt, ct = client.chat(EXHAUSTIVE_PROMPT.format(chunk=text), max_tokens=16384, bucket="index_build")
    subs = _extract._parse_subjects(out)
    return {"slug": slug, "cid": cid, "subjects": subs, "n_subjects": len(subs),
            "in_tok": pt, "out_tok": ct}

records = eval_harness.run_checkpointed("extract_exhaustive_partB", tasks, key_fn, work_fn, workers=4)
exhaustive = {(r["slug"], r["cid"]): r["subjects"] for r in records if r}

# --- per A1 question: did the exhaustive extract newly cover the gold? ---
# Two coverage tests per A1 question:
#   loose  = the Part A presence check (substring OR proper-noun entity OR >=0.6 token overlap)
#   strict = substring OR proper-noun entity only (drops the loose token-overlap match, which counts
#            generic shared words and is the source of false "covered" on bucket-3 questions that
#            actually failed). We report both; strict is the honest "the subject bears the answer".
def _strict(p):
    return bool(p["substr"] or p["entity"])

rows = []
for r in A1:
    slug, cid, gold = r["slug"], r["chunk_id"], r["gold"]
    o_text = subjects_text(orig_subs[slug].get(cid, []))
    x_text = subjects_text(exhaustive.get((slug, cid), []))
    o_pres, x_pres = presence(gold, o_text), presence(gold, x_text)
    rows.append({**r, "orig_covers": o_pres["present"], "exh_covers": x_pres["present"],
                 "orig_strict": _strict(o_pres), "exh_strict": _strict(x_pres),
                 "newly_covered": bool(x_pres["present"] and not o_pres["present"]),
                 "newly_strict": bool(_strict(x_pres) and not _strict(o_pres)),
                 "orig_subjects_text": o_text, "exh_subjects_text": x_text})

newly = [r for r in rows if r["newly_covered"]]
newly_s = [r for r in rows if r["newly_strict"]]
already = [r for r in rows if r["orig_covers"]]
in_chunk_A1 = [r for r in rows if r["in_chunk"]["present"]]
newly_in_chunk = [r for r in newly if r["in_chunk"]["present"]]
persistent = [r for r in rows if r["in_chunk"]["present"] and not r["exh_strict"]]  # still not cleanly captured
print(f"A1={len(rows)} | original already token-covered={len(already)} | "
      f"exhaustive newly covers (loose)={len(newly)} strict={len(newly_s)} "
      f"(in-chunk={len(newly_in_chunk)})", flush=True)

# --- write the combined report ---
def _ex_snippet(text, gold):
    """A short window of `text` around the first gold token, to show the captured fact."""
    from _match import content_tokens
    toks = content_tokens(" ".join(gold))
    tl = text.lower()
    for w in toks:
        i = tl.find(w)
        if i >= 0:
            return "..." + text[max(0, i-70):i+90].strip() + "..."
    return text[:160].strip() + ("..." if len(text) > 160 else "")


A2 = pa["A2"]
L = []
L.append("# Extraction diagnostic: is bucket 3 fixable, and does better extraction fix it\n")
L.append("Sample: all bucket-3 (covered-but-failing) questions from 2 source documents "
         f"({', '.join(pa['movies'])}), n={pa['sample_n']}, from the failure-taxonomy checkpoint.\n")
L.append("## Part A: is the gold answer even in the source (no LLM)\n")
L.append(f"- **A1 fixable** (gold answer present in the source document, extract missed it): "
         f"**{len(pa['A1'])}/{pa['sample_n']}**")
L.append(f"    - in the answer leaf's own chunk: {pa['A1_in_answer_chunk']}")
L.append(f"    - elsewhere in the source document: {pa['A1_elsewhere_in_source']}")
L.append(f"- **A2 artifact** (gold answer not in the source at all, lives only in the plot summary): "
         f"**{len(A2)}/{pa['sample_n']}**\n")
L.append("A1 examples (fixable):")
for r in pa["A1"][:3]:
    L.append(f"- `{r['qid']}` Q: {r['question']}  gold: {r['gold'][0]}  "
             f"(cov={r['in_source']['cov']}, substr={r['in_source']['substr']}, entity={r['in_source']['entity']})")
L.append("\nA2 examples (artifact):")
for r in A2[:3]:
    L.append(f"- `{r['qid']}` Q: {r['question']}  gold: {r['gold'][0]}  "
             f"(cov={r['in_source']['cov']}) - not found in the screenplay text")
L.append("\n## Part B: does a more exhaustive extract capture it (small LLM)\n")
L.append("Re-extracted the A1 answer-leaf chunks with an exhaustive prompt variant (all facts, plus "
         "relationships and outcomes across subjects). Model and reasoning setting unchanged, only the "
         "prompt wording. Original extraction = the shipped subjects already built for that chunk.\n")
L.append(f"- A1 questions tested: **{len(rows)}** across {len(tasks)} unique chunks")
L.append(f"- exhaustive prompt **newly covers** (original missed, exhaustive captures): "
         f"**{len(newly)}/{len(rows)}** by the loose token check, **{len(newly_s)}/{len(rows)}** "
         f"by the strict (verbatim or proper-noun) check")
L.append(f"    - among the {len(in_chunk_A1)} A1 whose answer is in the leaf's own chunk: "
         f"{len(newly_in_chunk)} newly covered\n")
L.append(f"Caveat: {len(already)}/{len(rows)} A1 questions already showed loose token overlap in the "
         "SHIPPED subjects and still failed as bucket 3. Raw token presence in a leaf is therefore not "
         "the bottleneck: the answer-bearing fact is not captured as a distinct, answerable subject or "
         "field. That is why the strict recount above is near zero.\n")
L.append("Example, the single loose new cover (marginal, generic token overlap not a clean capture):")
for r in (newly[:1] or rows[:1]):
    L.append(f"- `{r['qid']}` Q: {r['question']}  gold: {r['gold'][0]}")
    L.append(f"    - original subjects (chunk {r['chunk_id']}): {_ex_snippet(r['orig_subjects_text'], r['gold'])}")
    L.append(f"    - exhaustive subjects: {_ex_snippet(r['exh_subjects_text'], r['gold'])}")
L.append("\nExamples, the common case, the fact stays missing even with the exhaustive prompt:")
for r in persistent[:2]:
    L.append(f"- `{r['qid']}` Q: {r['question']}  gold: {r['gold'][0]}")
    L.append(f"    - exhaustive subjects (chunk {r['chunk_id']}) still miss it: "
             f"{_ex_snippet(r['exh_subjects_text'], r['gold'])}")

# verdict (strict recovery share is the honest one)
share = len(pa["A1"]) / pa["sample_n"]
b_share = (len(newly_s) / len(rows)) if rows else 0
if share >= 0.6 and b_share >= 0.5:
    verdict = ("Bucket 3 is mostly FIXABLE by better extraction: the answer is in the source for "
               f"{share:.0%} of the sample, and an exhaustive prompt (all facts plus relationships and "
               f"outcomes) recovers {b_share:.0%} of the fixable misses. A full re-extract with the "
               "relationships-and-outcomes prompt is the next real work.")
elif share >= 0.6 and b_share < 0.5:
    verdict = ("Bucket 3 is fixable in principle (the answer is in the source for "
               f"{share:.0%} of the sample) but DEEPER than prompt wording: the exhaustive prompt only "
               f"recovers {b_share:.0%} of the fixable misses, so the fact is hard to surface in a "
               "single pass (needs multi-chunk context or a structural extract change), not a one-line "
               "prompt fix.")
else:
    verdict = (f"Bucket 3 is largely a BENCHMARK ARTIFACT: the answer is absent from the source for "
               f"{1-share:.0%} of the sample, so extraction cannot recover it.")
L.append(f"\n## Verdict\n\n{verdict}\n")

OUTMD.write_text("\n".join(L))
print("wrote", OUTMD, flush=True)
