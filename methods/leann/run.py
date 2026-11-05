"""leann — LEANN low-storage vector index (arXiv 2506.08276), StarTrail-org/LEANN.

LEANN recomputes embeddings on the fly (storage-efficient) -> requires a LOCAL
embedder; we use a GPU sentence-transformer (reported), NOT Azure embeddings
(spec allows local fallback when the repo can't use Azure embeddings). QA
generation uses Azure gpt-5 via the shared client.

build_tokens = 0 in Azure terms (embeddings are local/GPU). answer tokens = gpt-5.

Usage: run.py build --movie <slug>   (GPU, endpoint-free)
       run.py answer --movie <slug>  (gpt-5 QA)
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse, importlib.util
import tiktoken
from leann import LeannBuilder, LeannSearcher

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build
ENC = tiktoken.get_encoding("cl100k_base")
import os
_CUAD = bool(os.environ.get("CUAD"))
_DOCDIR = "contracts/txt" if _CUAD else "movies/txt"
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
N_MOVIES = len(list((BENCH/"data"/_DOCDIR).glob("*.txt")))

METHOD = "leann"
MROOT = BENCH/"methods"/METHOD
IDX = MROOT/"output"/("indexes_cuad" if _CUAD else "indexes"); IDX.mkdir(parents=True, exist_ok=True)
RES = MROOT/"results"; RES.mkdir(parents=True, exist_ok=True)
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"   # local GPU embedder (reported)
K = 20   # LEANN's own canonical default (apps/base_rag_example.py --top-k default=20)

def chunk_tokens(text, size=1000, overlap=100):
    toks = ENC.encode(text); out=[]; i=0
    while i < len(toks):
        out.append(ENC.decode(toks[i:i+size]))
        if i+size >= len(toks): break
        i += size-overlap
    return out

def cmd_build(slug):
    idx_path = str(IDX/f"{slug}.leann")
    if pathlib.Path(idx_path + ".meta.json").exists() or pathlib.Path(idx_path).exists():
        print(f"[build] {slug}: exists, skip"); return
    text = (BENCH/"data"/_DOCDIR/f"{slug}.txt").read_text(encoding="utf-8")
    chunks = chunk_tokens(text)
    b = LeannBuilder(backend_name="hnsw", embedding_model=EMB_MODEL, embedding_mode="sentence-transformers")
    for c in chunks:
        b.add_text(c)
    t0 = time.time()
    b.build_index(idx_path)
    dt = time.time()-t0
    add_build("leann", dt/60, n_total=N_MOVIES)
    print(f"[build] {slug}: {len(chunks)} chunks -> {idx_path} ({round(dt,1)}s, GPU local embed)", flush=True)

SYS = "You answer a question about a movie using ONLY the screenplay excerpts provided. Answer in 10 words or fewer. If the answer is not in the excerpts, return exactly: not found"

def cmd_answer(slug):
    idx_path = str(IDX/f"{slug}.leann")
    searcher = LeannSearcher(idx_path)
    questions = [json.loads(l) for l in (BENCH/"data/questions"/_QFILE).read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]
    out = RES/f"{METHOD}{_SFX}_{slug}.jsonl"
    done=set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["qid"])
    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done: continue
        t0=time.time()
        try:
            results = searcher.search(q["question"], top_k=K)
            ctx = "\n\n---\n\n".join(r.text for r in results)
            pred, pt, ct = ac.chat([{"role":"system","content":SYS},
                {"role":"user","content":f"EXCERPTS:\n{ctx}\n\nQUESTION: {q['question']}\nANSWER:"}],
                max_tokens=2048, bucket="leann_answer", reasoning_effort="minimal")
        except Exception as e:
            pred, pt, ct = f"[ERROR: {repr(e)[:120]}]", 0, 0
        lat=round(time.time()-t0,2)
        f.write(json.dumps({"method":METHOD,"movie":slug,"qid":q["qid"],"question":q["question"],
            "gold_answers":q["gold_answers"],"prediction":pred.strip(),"source":"leann",
            "input_tokens":pt,"output_tokens":ct,"total_tokens":pt+ct,"latency_s":lat,"build_tokens":0},
            ensure_ascii=False)+"\n"); f.flush()
    f.close()
    print(f"DONE answer {slug}: {len(qs)}", flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["build","answer"]); ap.add_argument("--movie",required=True)
    a=ap.parse_args()
    cmd_build(a.movie) if a.cmd=="build" else cmd_answer(a.movie)

if __name__=="__main__":
    main()
