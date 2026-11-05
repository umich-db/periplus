"""llm_index answer pass -> benchmark JSONL.

Uses pipeline.query_with_traversal with per-question retrieval traces via the
cli.py setup (paths overridden, LLM routed to shared Azure client). Snapshots
the wiki_answer token bucket per question for clean per-question attribution.
build_tokens per row = amortized share of the movie's one-time build cost.

Usage: answer.py --movie <slug>     (one movie; writes results/llm_index_<slug>.jsonl)
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os, sys, json, time, pathlib, argparse, importlib.util

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Benchmark errors must remain resumable unanswered questions, not scored misses.
_os.environ.setdefault("LLM_INDEX_STRICT_API", "1")
import cli  # triggers setup: ac, cold_start, pipeline, run; patches llm; overrides paths

BENCH = _BENCH
ac, pipeline, run = cli.ac, cli.pipeline, cli.run

# Which corpus this run answers. CUAD=1 switches documents, questions and the
# output suffix, exactly as every baseline runner does.
_CUAD = bool(os.environ.get("CUAD"))
_CORPUS = "cuad" if _CUAD else "nqa"
_SFX = "_cuad" if _CUAD else ""
_QUESTION_FILE = _os.environ.get("BENCH_QUESTION_FILE", "").strip()
_INDEX_ROOT = _os.environ.get("LLM_INDEX_ROOT", "").strip()
_RUN_NAME = _os.environ.get("LLM_INDEX_RUN_NAME", "llm_index").strip()
_MAX_QUESTIONS = int(_os.environ.get("BENCH_MAX_QUESTIONS", "0"))
_QUESTION_ID = _os.environ.get("BENCH_QUESTION_ID", "").strip()
_QUESTION_ALLOWLIST = json.loads(_os.environ.get("BENCH_QID_ALLOWLIST", "null"))
_SHARD_ID = _os.environ.get("BENCH_SHARD_ID", "").strip()

sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS, wiki_docids  # noqa: E402

# slug -> the document id llm_index built its digest under. This was a ten-entry
# dict written by hand, which is the whole reason our index reported on ten
# movies while every baseline reported on thirty, and had no CUAD numbers at
# all. It is derived from the corpus now, and wiki_docids raises rather than
# silently skipping a document it cannot resolve.
SLUG2DOCID = wiki_docids(_CORPUS)

def wiki_answer_tot(counters):
    c = counters.get("wiki_answer", {})
    return c.get("prompt", 0), c.get("completion", 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--movie", required=True)
    slug = ap.parse_args().movie
    if slug not in SLUG2DOCID:
        print(f"{slug} is not in the {_CORPUS} corpus"); sys.exit(1)
    docid = SLUG2DOCID[slug]
    # Answer from the BALANCED tree. Section 3.4's split/borrow/merge is part of
    # the method, so answering from the raw compiled tree would be answering
    # from a different index than the paper describes. Falls back to the raw
    # tree only if the balancing pass has not run, and says which it used.
    safe = run.safe_filename(docid)
    balanced_root = (pathlib.Path(_INDEX_ROOT) if _INDEX_ROOT else
                     BENCH/"methods"/"llm_index"/"output"/"digests_balanced")
    balanced = balanced_root/safe
    raw = run.find_digest_dir(docid) or (run.DIGESTS / safe)
    if pathlib.Path(balanced/"_manifest.json").exists():
        wiki_dir, which = balanced, "balanced"
    elif _INDEX_ROOT:
        print(f"NO DIGEST in explicitly configured LLM_INDEX_ROOT for {slug}: {balanced}")
        sys.exit(1)
    elif pathlib.Path(raw).exists():
        wiki_dir, which = raw, "UNBALANCED (3.4 pass has not run)"
    else:
        print(f"NO DIGEST for {slug} ({docid})"); sys.exit(1)
    print(f"[llm_index] {slug}: {which} tree at {wiki_dir}", flush=True)
    raw_doc = DOCS[_CORPUS].doc_path(slug)

    reused_unknown_build = os.environ.get("LLM_INDEX_BUILD_COST_MODE") == "reused_gpt54_unknown"
    build_costs = {} if reused_unknown_build else json.loads((HERE/"build_costs.json").read_text())
    digest_key = wiki_dir.name
    build_total = build_costs.get(digest_key, {}).get("build_tokens_total", 0)

    # The 295-question file was the other half of the coverage gap: our index
    # answered a subset of one corpus while the baselines answered all 885.
    if _QUESTION_FILE:
        question_path = pathlib.Path(_QUESTION_FILE)
        if not question_path.is_absolute():
            question_path = BENCH/question_path
        supplied = [json.loads(line) for line in question_path.read_text().splitlines()
                    if line.strip()]
        qs = []
        for q in supplied:
            movie = q.get("movie", q.get("document"))
            if movie == slug:
                qs.append({**q, "movie": movie})
    else:
        qs = [q for q in DOCS[_CORPUS].questions if q["movie"] == slug]
    if _MAX_QUESTIONS:
        qs = qs[:_MAX_QUESTIONS]
    if _QUESTION_ID:
        qs = [q for q in qs if q["qid"] == _QUESTION_ID]
        if not qs:
            print(f"NO QUESTION {_QUESTION_ID} for {slug}"); sys.exit(1)
    if _QUESTION_ALLOWLIST is not None:
        wanted = set(_QUESTION_ALLOWLIST)
        qs = [q for q in qs if q["qid"] in wanted]
        if {q["qid"] for q in qs} != wanted:
            raise RuntimeError("Question shard contains an unknown question")
    amort = None if reused_unknown_build else round(build_total / max(1, len(qs)))

    result_dir = BENCH/"methods"/_RUN_NAME/"results"
    result_dir.mkdir(parents=True, exist_ok=True)
    shard = f"_{_SHARD_ID}" if _SHARD_ID else f"_{_QUESTION_ID}" if _QUESTION_ID else ""
    out = result_dir/f"{_RUN_NAME}{_SFX}_{slug}{shard}.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["qid"])

    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done: continue
        os.environ["BENCH_ACTIVE_QID"] = q["qid"]
        os.environ["BENCH_ACTIVE_MOVIE"] = slug
        bp, bc = wiki_answer_tot(ac.get_counters())
        call_start = len(ac.get_call_records())
        t0 = time.perf_counter()
        try:
            pred, source, trace = pipeline.query_with_traversal(q["question"], str(wiki_dir), str(raw_doc))
        except Exception as e:  # noqa: BLE001
            # An exhausted key must STOP the pass. Recorded as an "[ERROR" row it
            # becomes a zero-token miss: 676 of them were written, scored as 0.0%
            # accuracy and published to the results files before anyone looked.
            # No row is written for this question, so a resume picks it up.
            if isinstance(e, ac.QuotaExhausted) or ("412" in str(e) and "limit" in str(e).lower()):
                f.close()
                print(f"STOPPING {slug}: API key out of quota; {q['qid']} and later left unanswered", flush=True)
                sys.exit(3)
            if cli._STRICT_API:
                f.close()
                print(f"STOPPING {slug}: {q['qid']} failed ({type(e).__name__}); "
                      "question and later rows left unanswered for resume", flush=True)
                raise
            pred, source = f"[ERROR: {repr(e)[:120]}]", "error"
            trace = {"error": repr(e), "traversal": pipeline._LAST_TRAVERSAL_TRACE,
                     "evidence": pipeline._LAST_EVIDENCE_TRACE}
        lat = round(time.perf_counter()-t0, 3)
        ap2, ac2 = wiki_answer_tot(ac.get_counters())
        in_tok, out_tok = ap2-bp, ac2-bc
        call_records = ac.get_call_records()[call_start:]
        successful = [r for r in call_records if r["status"] == "ok"]
        if in_tok != sum(r["input_tokens"] for r in successful) or out_tok != sum(r["output_tokens"] for r in successful):
            raise RuntimeError("Per-question usage does not match the API ledger")
        rec = {"method":_RUN_NAME,"movie":slug,"qid":q["qid"],"question":q["question"],
               "gold_answers":q["gold_answers"],"prediction":pred,"source":source,
               "input_tokens":in_tok,"output_tokens":out_tok,"total_tokens":in_tok+out_tok,
               "latency_s":lat,"build_tokens":amort,"retrieval_trace":trace,
               "api_calls":call_records,"thinking_requested":cli._FORCE_THINK,
               "build_cost_scope":"saved GPT-5.4 extraction reused; actual historical build usage unavailable"
                   if reused_unknown_build else "legacy build_costs.json amortized estimate"}
        f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
        print(f"  {q['qid']} src={source} in={in_tok} out={out_tok} {lat}s :: {pred[:55]}", flush=True)
    f.close()
    print(f"DONE {slug}: {len(qs)} questions -> {out}", flush=True)

if __name__ == "__main__":
    main()
