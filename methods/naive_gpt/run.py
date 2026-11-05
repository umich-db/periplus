"""naive_gpt baseline: feed the full cleaned screenplay + question to Azure,
ask for a 1-15 word answer. Stateless (build_tokens=0). Resumable: skips qids
already present in the output JSONL."""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, time, pathlib, importlib.util, os

BENCH = _BENCH
# CUAD=1 switches the corpus/questions/output to the CUAD contract set
_CUAD = bool(os.environ.get("CUAD"))
_DOCDIR = "contracts/txt" if _CUAD else "movies/txt"
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
import sys; sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import write as _write_build
_write_build("naive_gpt", 0.0, coverage="none")  # stateless: no index build

METHOD = "naive_gpt"
OUT_DIR = BENCH/"methods"/METHOD/"results"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR/f"{METHOD}{_SFX}.jsonl"

# load questions + document texts (movies, or CUAD contracts when CUAD=1)
questions = [json.loads(l) for l in (BENCH/"data"/"questions"/_QFILE).read_text().splitlines() if l.strip()]
movies = {}
for p in (BENCH/"data"/_DOCDIR).glob("*.txt"):
    movies[p.stem] = p.read_text(encoding="utf-8")

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip():
            done.add(json.loads(l)["qid"])

SYS = "You answer a question about a movie using ONLY the screenplay provided. Answer in 10 words or fewer. If the answer is not in the screenplay, return exactly: not found"

f = open(OUT, "a", encoding="utf-8")
n_done = len(done)
for i, q in enumerate(questions):
    if q["qid"] in done:
        continue
    doc = movies.get(q["movie"], "")
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"SCREENPLAY:\n{doc}\n\nQUESTION: {q['question']}\nANSWER:"},
    ]
    t0 = time.time()
    try:
        text, pt, ct = ac.chat(msgs, max_tokens=1024, bucket="method")  # default reasoning, 1024 cap (user original)
    except Exception as e:
        text, pt, ct = f"[ERROR: {repr(e)[:120]}]", 0, 0
    lat = round(time.time() - t0, 2)
    rec = {"method": METHOD, "movie": q["movie"], "qid": q["qid"], "question": q["question"],
           "gold_answers": q["gold_answers"], "prediction": text.strip(),
           "input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct,
           "latency_s": lat, "build_tokens": 0}
    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    n_done += 1
    print(f"[{n_done}/{len(questions)}] {q['qid']} in={pt} out={ct} {lat}s", flush=True)
f.close()
print(f"DONE {METHOD}: {n_done}/{len(questions)} answered -> {OUT}", flush=True)
