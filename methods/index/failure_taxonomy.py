"""Failure taxonomy on the covered-but-failing oracle set. Analysis only. The covered oracle questions
(answer leaf clears the calibrated bar) that FAILED under the current-chunk condition of the span
diagnostic are the set to explain. Each is sorted, in a deterministic order, into one of three causes
that have different fixes:

  bucket 1 answer-synthesis floor: the gold answer IS in the answer leaf's chunk, the model still
           missed it. (At the model floor.)
  bucket 2 multi-hop: the gold is NOT in this one chunk, but IS derivable once the next-best leaf's
           chunk is added. (Needs multi-leaf retrieval + reasoning.)
  bucket 3 false coverage: not in this chunk and not recoverable with a second chunk either. The leaf
           embedded close to (question + gold) but the source text does not contain the answer, so the
           0.80 coverage proxy over-counted. (Extraction is the lever.)

Decision order: bucket 1 first (is the gold in this chunk), then bucket 2 (in this chunk plus one
more), else bucket 3. The "is the gold derivable from this text" test is a strict LLM classifier
(source text only, no outside knowledge). Reuses the built per_category index, the source documents,
the answer model + config, and eval_harness for a resumable, checkpointed run.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, pathlib, threading
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_BENCH/"shared"))
import client, config as _cfg, store as _store, pack_global as pg, eval_harness as EH  # noqa: E402

SLUGS = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
         "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
         "a_dry_white_season", "a_few_good_men"]
CFG = _cfg.load()
BAR = 0.5269
SPAN_CKPT = HERE/"results"/"checkpoints"/"ckpt_per_category__fb-sibling__route-cosine__k5__span-current.jsonl"

CLASSIFY_PROMPT = (
    "You are checking whether a SOURCE TEXT contains the information needed to answer a QUESTION with "
    "a specific GOLD ANSWER. Use ONLY the source text, not outside knowledge. Answer strictly one word:\n"
    "- 'yes' if the source text states or directly implies the gold answer (a reader of only this text "
    "could produce it).\n"
    "- 'no' otherwise.\n\n"
    "QUESTION: {q}\nGOLD ANSWER: {gold}\n\nSOURCE TEXT:\n{text}\n\nAnswer (yes or no):"
)

_lock = threading.Lock()
_orig = client.embed
def _le(*a, **k):
    with _lock:
        return _orig(*a, **k)
client.embed = _le


def supported(q, gold, text):
    out, _, _ = client.chat(CLASSIFY_PROMPT.format(q=q, gold=gold, text=text[:20000]),
                            max_tokens=2048, bucket="index_diag")
    return (out or "").strip().lower().startswith("y")


def failing_set():
    recs = [json.loads(l) for l in SPAN_CKPT.read_text().splitlines() if l.strip()]
    return [r for r in recs if r["maxcos"] >= BAR and r["verdict"] == "incorrect"]


def build_context(fail):
    """For each failing question, its best leaf's chunk (chunk1) and next-best leaf's chunk (chunk2).
    Deterministic top-2 by all-mpnet cosine of (question + gold) to leaf content."""
    by_slug = {}
    for r in fail:
        by_slug.setdefault(r["slug"], []).append(r["qid"])
    ctx = {}
    for slug, qids in by_slug.items():
        idx = _store.load_index(slug)
        leaves = sorted((n for n in idx["nodes"].values() if n["kind"] == "leaf"), key=lambda n: n["id"])
        lvecs = client.embed([lf["content"] for lf in leaves], model_name=CFG.embedding_model)
        qmap = {q["qid"]: q for q in pg._questions(slug)}
        for qid in qids:
            q = qmap[qid]; gold = " | ".join(q["gold_answers"])
            gv = client.embed([q["question"] + " " + gold], model_name=CFG.embedding_model)[0]
            order = np.argsort(-(lvecs @ gv))
            def chunk_of(i):
                cid = leaves[int(i)]["source"]["chunk_id"]
                ck = idx["chunks"].get(cid) or idx["chunks"].get(str(cid))
                return cid, (ck["text"] if ck else "")
            c1id, c1 = chunk_of(order[0]); c2id, c2 = chunk_of(order[1])
            ctx[f"{slug}::{qid}"] = {"question": q["question"], "gold": gold,
                                     "chunk1_id": c1id, "chunk1": c1, "chunk2_id": c2id, "chunk2": c2}
        print(f"context built for {slug} ({len(qids)} questions)", flush=True)
    return ctx


def main():
    fail = failing_set()
    print(f"covered-but-failing set: {len(fail)} questions", flush=True)
    ctx = build_context(fail)
    tag = EH.config_tag(CFG, extra="failtax")

    def key(r):
        return f"{r['slug']}::{r['qid']}"

    def work(r):
        c = ctx[key(r)]
        sup1 = supported(c["question"], c["gold"], c["chunk1"])
        if sup1:
            bucket, sup12 = 1, None
        else:
            sup12 = supported(c["question"], c["gold"], c["chunk1"] + "\n\n---\n\n" + c["chunk2"])
            bucket = 2 if sup12 else 3
        return {"slug": r["slug"], "qid": r["qid"], "question": c["question"], "gold": c["gold"],
                "bucket": bucket, "sup1": sup1, "sup12": sup12,
                "chunk1_id": c["chunk1_id"], "chunk2_id": c["chunk2_id"],
                "chunk1_preview": " ".join(c["chunk1"].split())[:700],
                "chunk2_preview": " ".join(c["chunk2"].split())[:500]}

    recs = [r for r in EH.run_checkpointed(tag, fail, key, work, workers=10) if r]
    n = len(recs)
    counts = {b: sum(1 for r in recs if r["bucket"] == b) for b in (1, 2, 3)}
    names = {1: "answer-synthesis floor", 2: "multi-hop", 3: "false coverage"}
    readings = {
        1: "Bucket 1 dominates: you are at the answer-model synthesis floor. ~0.44 is close to the "
           "honest ceiling for this benchmark and model. Write up and stop tuning.",
        2: "Bucket 2 dominates: the failures are multi-hop. Single-leaf retrieval cannot solve them; "
           "the next real contribution is retrieving and reasoning over multiple leaves. A design "
           "direction, not a tuning knob.",
        3: "Bucket 3 dominates: real coverage is below the 0.80 proxy. Extraction is the lever -- what "
           "got captured into the subject and its span."}
    dom = max(counts, key=counts.get)

    L = ["# Failure taxonomy: covered-but-failing oracle questions\n",
         "Analysis only. The set is the covered oracle questions (answer leaf cosine >= "
         f"{BAR}) that failed under the current-chunk condition of the span diagnostic. Each is sorted, "
         "in the order bucket 1 (gold in this chunk) then bucket 2 (gold in this chunk plus the "
         "next-best leaf's chunk) else bucket 3, using a strict LLM classifier over the source text "
         "only. Reuses the built per_category index, the source documents, the answer model, and the "
         "checkpointed harness. Vocabulary: root, category, subject, field, source document.\n",
         f"Set size: {n} questions.\n", "## Buckets\n"]
    for b in (1, 2, 3):
        L.append(f"- bucket {b} ({names[b]}): {counts[b]} ({round(100*counts[b]/n, 1)}%)")
    L.append(f"\n## Reading\n\n{readings[dom]}\n")
    L.append("## Examples (3 per bucket)\n")
    for b in (1, 2, 3):
        L.append(f"### bucket {b}: {names[b]}")
        exs = [r for r in recs if r["bucket"] == b][:3]
        if not exs:
            L.append("(none)")
        for r in exs:
            L.append(f"- [{r['slug']}] {r['qid']}: {r['question']}")
            L.append(f"    gold: {r['gold']}")
            L.append(f"    chunk read (leaf's chunk {r['chunk1_id']}): {r['chunk1_preview']}")
            if b == 2:
                L.append(f"    second chunk that completes it ({r['chunk2_id']}): {r['chunk2_preview']}")
        L.append("")
    (HERE/"results"/"failure_taxonomy.md").write_text("\n".join(L) + "\n")
    (HERE/"results"/"failure_taxonomy.json").write_text(json.dumps(
        {"n": n, "counts": counts, "dominant": dom, "records": recs}, indent=2, ensure_ascii=False))

    print("\n=== FAILURE TAXONOMY (covered-but-failing oracle set) ===")
    for b in (1, 2, 3):
        print(f"  bucket {b} {names[b]:24s}: {counts[b]:3d} ({round(100*counts[b]/n,1)}%)")
    print("  READING:", readings[dom])


if __name__ == "__main__":
    main()
