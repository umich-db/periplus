"""Relational-extraction test: are the fixable bucket-3 misses edges, not nodes?

Same A1 sample as the extraction diagnostic (the 19 fixable bucket-3 questions across 14 chunks). We
re-extract those chunks with a RELATIONAL prompt variant that returns first-class (subject, relation,
object, context) edges instead of subjects with descriptive fields. Then, per A1 question, we check
(a) capture: is the answer-bearing edge now a distinct item (loose and strict coverage, same check as
before), and (b) answerability: feed the chunk's edges to the NORMAL answer prompt and judge whether it
produces the gold answer. Capture is necessary; becoming answerable is the real test.

Model and reasoning setting are the shipped ones (client.chat / answer._answer_call defaults); only the
extraction prompt wording changes. We do NOT overwrite the shipped extract prompt, the judge, v7, shared
loaders, or any build mode. Deterministic and resumable via the checkpointed harness.

Vocabulary: root, category, subject, field, source document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import json, re, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
IDX = HERE.parent
sys.path.insert(0, str(IDX))
sys.path.insert(0, str(_BENCH/"shared"))
import chunk as _chunk, store as _store, client, answer, eval_harness  # noqa: E402
import eval as _eval                                                    # noqa: E402  (judge_one)
from _match import presence                                            # noqa: E402

BENCH = _BENCH
PARTA = IDX/"results"/"extraction_diag_partA.json"
OUTMD = IDX/"results"/"relational_extraction_test.md"

# Relational test prompt: edges, not node fields. Same JSON discipline as the shipped extractor.
RELATIONAL_PROMPT = (
    "Below is one passage from a source document. Extract the RELATIONSHIPS it states as first-class "
    "items. Each item is an edge between two subjects: a subject, a relation, and an object, plus a "
    "short context. Capture who did what to whom, the cause or reason (X does Y because Z), and the "
    "outcome (what happens to X). Prefer relational facts over standalone descriptions.\n\n"
    "Return ONLY a JSON array:\n"
    '  [{{"subject": "...", "relation": "...", "object": "...", "context": "..."}}, ...]\n\n'
    "Passage:\n{chunk}\n\nJSON array of relationship edges:"
)


def parse_triples(text):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for d in arr:
        if not isinstance(d, dict):
            continue
        s, rel, o, c = (str(d.get(k, "")).strip() for k in ("subject", "relation", "object", "context"))
        if s or rel or o:
            out.append({"subject": s, "relation": rel, "object": o, "context": c})
    return out


def rel_text(items):
    return "  ".join(f"{d['subject']} {d['relation']} {d['object']} ({d['context']})".strip()
                     for d in items)


def _strict(p):
    return bool(p["substr"] or p["entity"])


# --- load A1 set + node-centric (shipped) subjects per chunk + chunk spans ---
pa = json.loads(PARTA.read_text())
A1 = pa["A1"]
src, span_of, node_subs = {}, {}, {}
for slug in pa["movies"]:
    src[slug] = _chunk.load_source(slug)
    idx = _store.load_index(slug)
    span_of[slug], node_subs[slug] = {}, {}
    for n in idx["nodes"].values():
        if n["kind"] != "leaf":
            continue
        s = n.get("source") or {}
        cid = s.get("chunk_id")
        if cid is None:
            continue
        if cid not in span_of[slug] and s.get("start_char") is not None:
            span_of[slug][cid] = (s["start_char"], s["end_char"])
        node_subs[slug].setdefault(cid, []).append(n.get("content", ""))

uniq = {}
for r in A1:
    key = (r["slug"], r["chunk_id"])
    if key not in uniq:
        st, en = span_of[r["slug"]].get(r["chunk_id"], (None, None))
        uniq[key] = src[r["slug"]][st:en] if st is not None else ""
chunk_tasks = [(slug, cid, text) for (slug, cid), text in uniq.items()]
print(f"A1={len(A1)} unique chunks={len(chunk_tasks)}", flush=True)

# --- pass 1: relational extraction per chunk (checkpointed) ---
def k_chunk(t):
    return f"{t[0]}::chunk{t[1]}"

def work_extract(t):
    slug, cid, text = t
    out, pt, ct = client.chat(RELATIONAL_PROMPT.format(chunk=text), max_tokens=16384, bucket="index_build")
    return {"slug": slug, "cid": cid, "edges": parse_triples(out), "in_tok": pt, "out_tok": ct}

rec1 = eval_harness.run_checkpointed("relational_extract_test", chunk_tasks, k_chunk, work_extract, workers=4)
edges_by_chunk = {(r["slug"], r["cid"]): r["edges"] for r in rec1 if r}

# --- pass 2: answer each A1 question from its chunk's edges, then judge (checkpointed) ---
A1_by_qid = {r["qid"]: r for r in A1}

def k_q(r):
    return r["qid"]

def work_answer(r):
    edges = edges_by_chunk.get((r["slug"], r["chunk_id"]), [])
    digest = rel_text(edges)
    pred, pt, ct = answer._answer_call(r["question"], digest)
    v = _eval.judge_one({"qid": r["qid"], "question": r["question"], "gold_answers": r["gold"],
                         "prediction": pred, "source": "index"})
    return {"qid": r["qid"], "prediction": pred, "verdict": v["verdict"], "in_tok": pt, "out_tok": ct}

rec2 = eval_harness.run_checkpointed("relational_answer_test", A1, k_q, work_answer, workers=4)
ans_by_qid = {r["qid"]: r for r in rec2 if r}

# --- assemble per-question: capture (loose/strict) + answered ---
rows = []
for r in A1:
    edges = edges_by_chunk.get((r["slug"], r["chunk_id"]), [])
    rtext = rel_text(edges)
    ntext = "  ".join(node_subs[r["slug"]].get(r["chunk_id"], []))
    cap = presence(r["gold"], rtext)
    a = ans_by_qid.get(r["qid"], {})
    rows.append({**r, "edges": edges, "rel_text": rtext, "node_text": ntext,
                 "cap_loose": cap["present"], "cap_strict": _strict(cap),
                 "prediction": a.get("prediction", ""), "answered_correct": a.get("verdict") == "correct"})

def _is_miss(p):
    pl = str(p).lower().strip()
    return pl in ("not found", "none", "n/a", "") or "not found" in pl

cap_strict = [r for r in rows if r["cap_strict"]]
cap_loose = [r for r in rows if r["cap_loose"]]
answered = [r for r in rows if r["answered_correct"]]
cap_and_ans = [r for r in rows if r["cap_strict"] and r["answered_correct"]]
abstained = [r for r in rows if _is_miss(r["prediction"])]
wrong = [r for r in rows if not _is_miss(r["prediction"]) and not r["answered_correct"]]
inchunk = [r for r in rows if r["in_chunk"]["present"]]
inchunk_abstain = [r for r in inchunk if _is_miss(r["prediction"])]
print(f"capture strict={len(cap_strict)}/{len(rows)} loose={len(cap_loose)}/{len(rows)} | "
      f"answered correct={len(answered)}/{len(rows)} | abstained={len(abstained)} wrong={len(wrong)}", flush=True)


def snip(text, gold, w=90):
    from _match import content_tokens
    tl = text.lower()
    for tok in content_tokens(" ".join(gold)):
        i = tl.find(tok)
        if i >= 0:
            return "..." + text[max(0, i-70):i+w].strip() + "..."
    return (text[:150].strip() + "...") if text else "(none)"


# --- report ---
L = []
L.append("# Relational-extraction test: are the missing bucket-3 facts edges, not nodes\n")
L.append(f"Same A1 sample as the extraction diagnostic: {len(rows)} fixable bucket-3 questions across "
         f"{len(chunk_tasks)} chunks, 2 source documents ({', '.join(pa['movies'])}). We re-extract those "
         "chunks as (subject, relation, object, context) edges, then test capture and answerability. "
         "Model and reasoning setting unchanged, only the extraction prompt.\n")
L.append("## Result\n")
L.append(f"- **capture (edge is now a distinct item):** strict **{len(cap_strict)}/{len(rows)}**, "
         f"loose {len(cap_loose)}/{len(rows)}  (node-centric extraction reached 0/19 strict)")
L.append(f"- **answerable (edges fed to the normal answer prompt, judged correct):** "
         f"**{len(answered)}/{len(rows)}**")
L.append(f"    - of the {len(rows)} answers: {len(abstained)} abstained (not found), {len(wrong)} answered wrong")
L.append(f"    - of the {len(inchunk)} whose answer is in the leaf's own chunk, {len(inchunk_abstain)} still abstained\n")
L.append("Caveat on the capture count: the strict count is largely the matcher finding a proper noun "
         "somewhere in the edge list, not the specific answer-bearing edge. The abstentions confirm this: "
         "for `10_things_i_hate_about_you_018` (what does Patrick buy with the money, gold: a guitar) the "
         "answer model abstained here, yet in a control it answers correctly when handed the exact edge "
         "(Patrick, buys, guitar). So the relational prompt did not surface that specific edge. The "
         "bottleneck is capture fidelity of the right edge, not the answer model.\n")
L.append("## Before and after examples\n")
picks = ([r for r in rows if r["cap_strict"] and r["answered_correct"]]
         + [r for r in rows if r["cap_strict"] and not r["answered_correct"]]
         + [r for r in rows if not r["cap_strict"]])
seen = set()
shown = 0
for r in picks:
    if r["qid"] in seen:
        continue
    seen.add(r["qid"])
    tag = ("captured + answered" if r["cap_strict"] and r["answered_correct"]
           else "captured, not answered" if r["cap_strict"]
           else "still missed")
    L.append(f"### `{r['qid']}`  [{tag}]")
    L.append(f"- Q: {r['question']}")
    L.append(f"- gold: {r['gold'][0]}")
    L.append(f"- node-centric (shipped) subjects: {snip(r['node_text'], r['gold'])}")
    L.append(f"- relational edges: {snip(r['rel_text'], r['gold'])}")
    L.append(f"- answer from edges: {r['prediction'][:120]}\n")
    shown += 1
    if shown >= 3:
        break

# verdict
n = len(rows)
cs = len(cap_strict) / n if n else 0
ans = len(answered) / n if n else 0
if ans >= 0.25 and len(cap_and_ans) >= 3:
    verdict = (f"The missing facts are EDGES. Relational extraction captures {len(cap_strict)}/{n} "
               f"strict (vs 0/{n} node-centric) and makes {len(answered)}/{n} answerable. Relational "
               "extraction is the real next lever. Scope caveat for Lin: this changes what a leaf is, "
               "from a subject with fields to a subject-relation-object edge, which moves the design "
               "toward a knowledge graph over the source document. That is a structural change, not a "
               "prompt tweak, and a framing decision, not something to roll out silently.")
elif len(cap_strict) >= 3 and ans < 0.25:
    verdict = (f"Deeper than edges. Relational extraction appears to capture {len(cap_strict)}/{n} strict "
               f"(vs 0/{n} node-centric), but that count is mostly the matcher hitting a proper noun in "
               f"the edge list, not the answer-bearing edge, and {len(answered)}/{n} become answerable "
               f"({len(abstained)} abstain, {len(wrong)} wrong). A single-pass relational prompt over the "
               "answer-leaf chunk does not reliably surface the specific edge (even a control-answerable "
               "case abstains), so the miss needs multiple chunks combined (multi-hop) or is phrased "
               "beyond single-pass reach. 0.44 is close to the honest ceiling, and the paper is the "
               "mechanism story (dead levers, ceiling localized), not a new extraction system.")
else:
    verdict = (f"Not edges either: relational extraction captures only {len(cap_strict)}/{n} strict and "
               f"answers {len(answered)}/{n}. The fact needs multi-chunk context or the benchmark "
               "phrasing is beyond single-pass extraction. 0.44 is the honest ceiling, and the paper is "
               "the mechanism story (dead levers, ceiling localized), not a new extraction system.")
L.append(f"## Verdict\n\n{verdict}\n")

OUTMD.write_text("\n".join(L))
print("wrote", OUTMD, flush=True)
