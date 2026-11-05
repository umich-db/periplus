"""Fact-selection test, Stage 1 (oracle set, embedding selection, no LLM for selection).

The one untested lever: narrow the retrieved chunk to the sentences that answer the question, then
answer. We isolate selection from routing by using the oracle answer-leaf set (results/oracle_leaves.json),
so every question starts from the same ~6.5k-char span the oracle-leaf test used. For each question we
answer four ways and judge each:
  full  = the whole oracle chunk (the ~0.44 oracle baseline)
  k3/k5/k8 = the top 3 / 5 / 8 sentences by all-mpnet cosine to the question, in original order

Selection is embedding-only and transient (nothing persisted as a query index). The answer model, its
config, and the judge prompt are the shipped ones. We reuse the judge's exact verdict prompt for a
verdict-only call (accuracy is the metric here); we do not modify eval.py, the answer path, v7, shared
loaders, or any build mode. Deterministic and resumable via the checkpointed harness.

Vocabulary: root, category, subject, field, source document. No em dashes.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[3])

import json, re, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
IDX = HERE.parent
sys.path.insert(0, str(IDX))
sys.path.insert(0, str(_BENCH/"shared"))
import chunk as _chunk, config as _cfg, client, answer, eval_harness  # noqa: E402
import eval as _eval                                                   # noqa: E402

BENCH = _BENCH
ORACLE = IDX/"results"/"oracle_leaves.json"
OUTMD = IDX/"results"/"fact_selection_test.md"
KS = [3, 5, 8]
CONDS = ["full", "k3", "k5", "k8"]

cfg = _cfg.load()
qmap = {q["qid"]: q for q in (json.loads(l) for l in
        (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip())}
oracle = json.loads(ORACLE.read_text())

_MISS = ("", "not found", "none", "n/a")


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if len(s) >= 12:
            out.append(s)
    return out


def judge_verdict(question, gold_answers, pred):
    """Verdict only, using the shipped judge's exact prompt and client (eval.JUDGE_PROMPT / eval.ac).
    Mirrors eval.judge_one's miss shortcut and correct/incorrect parse; no LLM call on a miss."""
    pl = str(pred or "").strip().lower()
    if pl in _MISS or pl.startswith("[error") or "not found" in pl:
        return "incorrect"
    gold = " | ".join(gold_answers)
    jt, _, _ = _eval.ac.chat([{"role": "user", "content":
                               _eval.JUDGE_PROMPT.format(q=question, gold=gold, pred=pred)}],
                             max_tokens=2048, bucket="judge")
    line0 = jt.strip().split("\n")[0].lower()
    return "correct" if ("incorrect" not in line0 and "correct" in line0) else "incorrect"


# --- precompute per-question digests (full + top-k), embedding selection is transient ---
print("precomputing sentence selections (local embeddings)...", flush=True)
src_cache = {}
DIGESTS = {}
for i, (key, o) in enumerate(oracle.items()):
    slug, qid = o["slug"], o["qid"]
    if qid not in qmap:
        continue
    if slug not in src_cache:
        src_cache[slug] = _chunk.load_source(slug)
    chunk_text = src_cache[slug][o["start_char"]:o["end_char"]]
    sents = split_sentences(chunk_text)
    d = {"full": chunk_text}
    if len(sents) <= min(KS):
        for k in KS:
            d[f"k{k}"] = " ".join(sents)
    else:
        q = qmap[qid]["question"]
        vecs = np.array(client.embed(sents, model_name=cfg.embedding_model))
        qv = np.array(client.embed([q], model_name=cfg.embedding_model)[0])
        scores = vecs @ qv
        order = np.argsort(-scores)
        for k in KS:
            idx = sorted(int(j) for j in order[:k])
            d[f"k{k}"] = " ".join(sents[j] for j in idx)
    DIGESTS[qid] = d
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(oracle)} selections done", flush=True)

qids = [o["qid"] for o in oracle.values() if o["qid"] in qmap]
tasks = [(qid, cond) for qid in qids for cond in CONDS]
print(f"questions={len(qids)} conditions={CONDS} tasks={len(tasks)}", flush=True)


def key_fn(t):
    return f"{t[0]}::{t[1]}"


def work_fn(t):
    qid, cond = t
    q = qmap[qid]
    digest = DIGESTS[qid][cond]
    pred, pt, ct = answer._answer_call(q["question"], digest)
    v = judge_verdict(q["question"], q["gold_answers"], pred)
    return {"qid": qid, "cond": cond, "prediction": pred, "verdict": v}


records = eval_harness.run_checkpointed("fact_select_stage1", tasks, key_fn, work_fn, workers=8)
by = {}
for r in records:
    if r:
        by[(r["qid"], r["cond"])] = r

# --- aggregate ---
def correct(qid, cond):
    r = by.get((qid, cond))
    return r is not None and r["verdict"] == "correct"

n = len(qids)
acc = {c: sum(1 for q in qids if correct(q, c)) / n for c in CONDS}
fixed, broken = {}, {}
for k in KS:
    c = f"k{k}"
    fixed[k] = [q for q in qids if not correct(q, "full") and correct(q, c)]
    broken[k] = [q for q in qids if correct(q, "full") and not correct(q, c)]
best_k = max(KS, key=lambda k: acc[f"k{k}"])
beats = acc[f"k{best_k}"] > acc["full"]

print("\nACCURACY  full={:.4f}  k3={:.4f}  k5={:.4f}  k8={:.4f}".format(
    acc["full"], acc["k3"], acc["k5"], acc["k8"]), flush=True)
print(f"best k={best_k} ({acc[f'k{best_k}']:.4f})  beats full baseline: {beats}", flush=True)
for k in KS:
    print(f"  k{k}: fixed={len(fixed[k])} broken={len(broken[k])} net={len(fixed[k])-len(broken[k])}", flush=True)

# --- report ---
def snip(s, n=220):
    s = re.sub(r"\s+", " ", s or "").strip()
    return (s[:n] + "...") if len(s) > n else s

L = []
L.append("# Fact-selection test: narrow the chunk before answering\n")
L.append("Stage 1, oracle answer-leaf set (routing held out), n={}. Embedding selection (all-mpnet, "
         "transient), the shipped answer model and config, verdict from the shipped judge prompt.\n".format(n))
L.append("## Stage 1 result (the decisive number)\n")
L.append(f"- full chunk (oracle baseline): **{acc['full']:.3f}**")
for k in KS:
    L.append(f"- top-{k} sentences: **{acc[f'k{k}']:.3f}**  (fixed {len(fixed[k])}, broke {len(broken[k])}, "
             f"net {len(fixed[k])-len(broken[k]):+d})")
L.append(f"- best k = **{best_k}** at {acc[f'k{best_k}']:.3f}; beats full-chunk baseline: "
         f"**{'YES' if beats else 'NO'}**\n")

L.append("## Examples where narrowing fixed a previously failing question\n")
shown = 0
for q in fixed[best_k]:
    o = next(v for v in oracle.values() if v["qid"] == q)
    L.append(f"### `{q}`")
    L.append(f"- Q: {qmap[q]['question']}")
    L.append(f"- gold: {qmap[q]['gold_answers'][0]}")
    L.append(f"- full chunk (missed): {snip(DIGESTS[q]['full'])}")
    L.append(f"- top-{best_k} sentences (answered): {snip(DIGESTS[q][f'k{best_k}'])}")
    L.append(f"- answer from narrowed: {by[(q, f'k{best_k}')]['prediction'][:100]}\n")
    shown += 1
    if shown >= 3:
        break
if shown == 0:
    L.append("(none: no question was fixed by narrowing)\n")

if beats:
    verdict = (f"Narrowing BEATS the ceiling. Top-{best_k} sentence selection reaches "
               f"{acc[f'k{best_k}']:.3f} vs the full-chunk oracle baseline {acc['full']:.3f}. Fact "
               "selection is the accuracy lever and the first thing to move past 0.44. Next: the LLM "
               "selection variant (Stage 2) and the end-to-end method wiring (Stage 3).")
else:
    verdict = (f"Narrowing does NOT beat the baseline. Best top-k ({best_k}) is {acc[f'k{best_k}']:.3f} "
               f"vs full-chunk {acc['full']:.3f}. The failure is not the blob: the model needs the "
               "surrounding context, or the miss is synthesis or multi-hop. 0.44 is the honest ceiling "
               "and the paper is the mechanism story.")
L.append(f"## Verdict\n\n{verdict}\n")
OUTMD.write_text("\n".join(L))
print("wrote", OUTMD, flush=True)
