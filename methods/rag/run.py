"""rag baseline: vanilla RAG over each screenplay.

Build (one-time/doc): chunk (~1k tokens, 100 overlap), embed chunks with the
Azure embedding deployment -> build_tokens = embedding cost.
Answer (per question): embed query, top-k=5 cosine retrieve, feed chunks+question
to gpt-5 for a 1-15 word answer.

Resumable; embeddings cached to output/emb/<slug>.npz.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, time, pathlib, importlib.util
import numpy as np

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
import sys; sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import time_build
import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")

import os
_CUAD = bool(os.environ.get("CUAD"))
_DOCDIR = "contracts/txt" if _CUAD else "movies/txt"
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
METHOD = "rag"
MROOT = BENCH/"methods"/METHOD
EMB = MROOT/"output"/"emb"; EMB.mkdir(parents=True, exist_ok=True)
OUT = MROOT/"results"/f"{METHOD}{_SFX}.jsonl"; OUT.parent.mkdir(parents=True, exist_ok=True)
BUILD_COSTS = MROOT/f"build_costs{_SFX}.json"
K = 5

def chunk_tokens(text, size=1000, overlap=100):
    toks = ENC.encode(text)
    chunks, i = [], 0
    while i < len(toks):
        chunks.append(ENC.decode(toks[i:i+size]))
        if i+size >= len(toks): break
        i += size-overlap
    return chunks

def embed_batch(texts, bucket):
    vecs, tot = [], 0
    for i in range(0, len(texts), 64):
        v, t = ac.embed(texts[i:i+64], bucket=bucket)
        vecs.extend(v); tot += t
    return np.array(vecs, dtype=np.float32), tot

def build_movie(slug, text, build_costs):
    cache = EMB/f"{slug}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["chunks"]), d["vecs"]
    chunks = chunk_tokens(text)
    vecs, tot = embed_batch(chunks, bucket="rag_build")
    np.savez(cache, chunks=np.array(chunks, dtype=object), vecs=vecs)
    build_costs[slug] = {"build_tokens_total": tot, "n_chunks": len(chunks)}
    BUILD_COSTS.write_text(json.dumps(build_costs, indent=2))
    print(f"  built {slug}: {len(chunks)} chunks, {tot:,} embed tokens", flush=True)
    return chunks, vecs

SYS = "You answer a question about a movie using ONLY the screenplay excerpts provided. Answer in 10 words or fewer. If the answer is not in the excerpts, return exactly: not found"

def main():
    movies = {p.stem: p.read_text(encoding="utf-8") for p in (BENCH/"data"/_DOCDIR).glob("*.txt")}
    questions = [json.loads(l) for l in (BENCH/"data/questions"/_QFILE).read_text().splitlines() if l.strip()]
    build_costs = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}

    # build indices
    idx = {}
    with time_build("rag", n_total=len(movies)) as _bt:
        for slug, text in movies.items():
            idx[slug] = build_movie(slug, text, build_costs)
            _bt.built()

    qcount = {}
    for q in questions: qcount[q["movie"]] = qcount.get(q["movie"],0)+1

    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["qid"])
    f = open(OUT, "a", encoding="utf-8")
    n = len(done)
    for q in questions:
        if q["qid"] in done: continue
        chunks, vecs = idx[q["movie"]]
        t0 = time.time()
        qv, _ = ac.embed([q["question"]], bucket="rag_answer")
        qv = np.array(qv[0], dtype=np.float32)
        sims = vecs @ qv / (np.linalg.norm(vecs,axis=1)*np.linalg.norm(qv)+1e-9)
        top = np.argsort(-sims)[:K]
        ctx = "\n\n---\n\n".join(chunks[i] for i in top)
        try:
            pred, pt, ct = ac.chat([{"role":"system","content":SYS},
                {"role":"user","content":f"EXCERPTS:\n{ctx}\n\nQUESTION: {q['question']}\nANSWER:"}],
                max_tokens=2048, bucket="rag_answer")  # default reasoning, 2048 cap
        except Exception as e:
            pred, pt, ct = f"[ERROR: {repr(e)[:120]}]", 0, 0
        lat = round(time.time()-t0,2)
        bt = build_costs.get(q["movie"],{}).get("build_tokens_total",0)
        amort = round(bt/max(1,qcount[q["movie"]]))
        rec = {"method":METHOD,"movie":q["movie"],"qid":q["qid"],"question":q["question"],
               "gold_answers":q["gold_answers"],"prediction":pred.strip(),
               "input_tokens":pt,"output_tokens":ct,"total_tokens":pt+ct,
               "latency_s":lat,"build_tokens":amort}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
        n+=1
        if n%30==0: print(f"  [{n}/{len(questions)}] {q['qid']} :: {pred[:50]}", flush=True)
    f.close()
    print(f"DONE {METHOD}: {n} answered -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
