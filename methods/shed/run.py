"""SHED (adapted): Structured-Heading-Tree agentic QA over screenplays.

Faithful adaptation of Ruiying-Ma/SHED to this benchmark:
  * SHT is recovered structurally from screenplay scene headings (sht.py),
    "local-first, no external APIs" -> build_tokens = 0.
  * The agent is handed the numbered Table of Contents (scene headings) and a
    read_section tool; it reads sections on demand, then answers. SHED uses
    LangChain native tool-calls; our shared azure_client has no tool-calling,
    so we mirror the exact protocol with a text ReAct loop (READ <id> / ANSWER).

Output format matches the other methods (naive_gpt-compatible) so shared/eval.py
and the accuracy leaderboard treat it identically. Resumable via the done set.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, re, time, pathlib, importlib.util

BENCH = _BENCH
import sys as _sys; _sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS as _DOCS  # noqa: E402
_CORPUS_KEY = "cuad" if __import__("os").environ.get("CUAD") else "nqa"

spec = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# local SHT segmenter
spec2 = importlib.util.spec_from_file_location("sht", BENCH / "methods" / "shed" / "sht.py")
sht = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sht)

import sys; sys.path.insert(0, str(BENCH / "shared"))
from buildtimer import time_build

METHOD = "shed"
OUT_DIR = BENCH / "methods" / METHOD / "results"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{METHOD}.jsonl"

MAX_STEPS = 6          # agent reasoning/read turns before a forced answer
MAX_READS_PER_TURN = 5
SECTION_CHAR_CAP = 6000  # truncate a single section's content fed back to agent

# --- load questions + movie texts, build one SHT per movie -----------------
questions = list(_DOCS[_CORPUS_KEY].questions)
movies, tocs = {}, {}
_txts = list((BENCH / "data" / "movies" / "txt").glob("*.txt"))
with time_build("shed", n_total=len(_txts)) as _bt:
    for p in _txts:
        secs = sht.build_sht(p.read_text(encoding="utf-8"))
        movies[p.stem] = {s["id"]: s for s in secs}
        tocs[p.stem] = sht.numbered_toc(secs)
        _bt.built()

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        if l.strip():
            done.add(json.loads(l)["qid"])

SYS = (
    "You answer a question about a movie using ONLY its screenplay. "
    "You are given the TABLE OF CONTENTS: a numbered list of scene headings, one per line as `id | heading`. "
    "You may retrieve the full text of any scene with a read command.\n\n"
    "Each turn, respond in EXACTLY one of these two forms:\n"
    "  READ <id> [<id> ...]   - to fetch the content of one or more scenes (use the numeric id before `|`)\n"
    "  ANSWER: <answer>       - your final answer, 10 words or fewer\n\n"
    "Use the table of contents to find the scenes most likely to contain the answer, read them, then answer. "
    "Read only the scenes you need. If the screenplay does not contain the answer, respond `ANSWER: not found`."
)

_READ_RE = re.compile(r"\bREAD\s+([0-9 ]+)", re.I)


def parse_reads(text):
    ids = []
    for m in _READ_RE.finditer(text):
        ids.extend(tok for tok in m.group(1).split() if tok.isdigit())
    return ids


def run_question(q):
    sections = movies.get(q["movie"], {})
    toc = tocs.get(q["movie"], "")
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"TABLE OF CONTENTS:\n{toc}\n\nQUESTION: {q['question']}"},
    ]
    in_tok = out_tok = 0
    answer = None
    for step in range(MAX_STEPS):
        force = step == MAX_STEPS - 1
        if force:
            msgs.append({"role": "user", "content": "You must answer now. Respond with `ANSWER: <answer>` (10 words or fewer)."})
        text, pt, ct = ac.chat(msgs, max_tokens=2048, bucket="method")
        in_tok += pt; out_tok += ct
        msgs.append({"role": "assistant", "content": text})

        if "ANSWER:" in text.upper():
            answer = text[text.upper().index("ANSWER:") + len("ANSWER:"):].strip()
            break
        ids = parse_reads(text)[:MAX_READS_PER_TURN]
        if not ids:
            answer = text.strip()  # no tool call, no ANSWER tag -> treat as final
            break
        # fulfil read_section calls
        chunks = []
        for sid in ids:
            sec = sections.get(sid)
            if sec is None:
                chunks.append(f"[section {sid} not found]")
            else:
                body = sec["content"][:SECTION_CHAR_CAP]
                chunks.append(f"### {sid} | {sec['heading']}\n{body}")
        msgs.append({"role": "user", "content": "\n\n".join(chunks)})

    if answer is None:
        answer = "not found"
    return answer, in_tok, out_tok


def main():
    f = open(OUT, "a", encoding="utf-8")
    n_done = len(done)
    for q in questions:
        if q["qid"] in done:
            continue
        t0 = time.time()
        try:
            text, pt, ct = run_question(q)
        except Exception as e:
            text, pt, ct = f"[ERROR: {repr(e)[:120]}]", 0, 0
        lat = round(time.time() - t0, 2)
        rec = {"method": METHOD, "movie": q["movie"], "qid": q["qid"], "question": q["question"],
               "gold_answers": q["gold_answers"], "prediction": text.strip(),
               "input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct,
               "latency_s": lat, "build_tokens": 0}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        n_done += 1
        print(f"[{n_done}/{len(questions)}] {q['qid']} in={pt} out={ct} {lat}s :: {text[:60]}", flush=True)
    f.close()
    print(f"DONE {METHOD}: {n_done}/{len(questions)} answered -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
