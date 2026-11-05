"""Driver for the user's wiki_index/run.py (v7 CLI), adapted to this server
WITHOUT editing their code:
  - module-global paths (BASE/CORPUS/DIGESTS/OUTPUT) overridden after import
  - every LLM call (cold_start/pipeline/run) routed through shared azure_client
    (gpt-5, minimal reasoning) with token buckets: wiki_build / wiki_answer / judge

Subcommands mirror run.py: compile [slug...] | evaluate | all
After evaluate, also emits the benchmark JSONL results/llm_index.jsonl.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, importlib.util

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
MROOT = BENCH/"methods"/"llm_index"

# --- shared azure client ---
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# --- import user's v7 code ---
sys.path.insert(0, str(WIKI_SRC))
import cold_start, pipeline, run  # noqa: E402
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build

# --- per-call LLM log (for token accounting) ---
CALL_LOG = []  # list of {phase, in, out}
BUILD_PHASES = ("discover", "proposer", "reconciler", "compile_p1", "compile_p2", "consolidate")
_FORCE_THINK = _os.environ.get("LLM_INDEX_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}
_NO_THINK = (not _FORCE_THINK and
             any(name in ac.LLM_MODEL.lower() for name in ("nemotron", "qwen")))
_THINKING_MAX_TOKENS = int(_os.environ.get("LLM_INDEX_THINKING_MAX_TOKENS", "4096"))
_STRICT_API = _os.environ.get("LLM_INDEX_STRICT_API", "0").lower() in {"1", "true", "yes"}
_MODEL_EXTRA = ({"chat_template_kwargs": {"enable_thinking": True}}
                if _FORCE_THINK else
                {"chat_template_kwargs": {"enable_thinking": False}}
                if _NO_THINK else None)

def _bucket(phase):
    if phase == "judge":
        return "judge"
    if any(phase.startswith(p) for p in BUILD_PHASES):
        return "wiki_build"
    return "wiki_answer"

def _llm(prompt, max_tokens=16384, phase="unknown"):
    bucket = _bucket(phase)
    if _FORCE_THINK:
        # The allowance includes reasoning: short routing requests also need
        # enough room to finish thinking and return a visible choice.
        max_tokens = max(max_tokens, _THINKING_MAX_TOKENS)
    try:
        for generation_attempt in range(2 if _FORCE_THINK else 1):
            before = len(ac.get_call_records())
            text, pt, ct = ac.chat([{"role": "user", "content": prompt}],
                                   max_tokens=max_tokens, bucket=bucket,
                                   reasoning_effort="minimal" if _NO_THINK else None,
                                   extra_body=_MODEL_EXTRA,
                                   usage_context={"phase": phase,
                                       "qid": _os.environ.get("BENCH_ACTIVE_QID"),
                                       "movie": _os.environ.get("BENCH_ACTIVE_MOVIE"),
                                       "run": _os.environ.get("LLM_INDEX_RUN_NAME"),
                                       "generation_attempt": generation_attempt + 1})
            CALL_LOG.append({"phase": phase, "bucket": bucket, "in": pt, "out": ct})
            received = [r for r in ac.get_call_records()[before:] if r["status"] == "ok"]
            truncated = bool(received and received[-1].get("finish_reason") == "length")
            if not truncated:
                break
            if _FORCE_THINK and generation_attempt == 0:
                max_tokens *= 2
                continue
            if _STRICT_API:
                raise RuntimeError(f"LLM exhausted its output allowance during {phase}")
    except Exception as e:  # noqa: BLE001
        # An exhausted key must stop the build. Degrading it to "" makes discovery
        # "fail" quietly and the document is skipped with zero tokens, which the
        # build then reports as done.
        if (_STRICT_API or isinstance(e, ac.QuotaExhausted)
                or ("412" in str(e) and "limit" in str(e).lower())):
            raise
        cold_start.log.warning(f"  llm error ({phase}): {repr(e)[:120]}")
        text, pt, ct = "", 0, 0
    if _STRICT_API and not (text or "").strip():
        raise RuntimeError(f"LLM returned empty content during {phase}; question remains unanswered")
    return (text or "").strip()

cold_start.llm = _llm
pipeline.llm = _llm
run.llm = _llm

def _embed(texts):
    """Embedder for the fact->chunk provenance fallback.

    Used only when a fact's exact signature was lost in the extraction merge and
    it has to be placed back into the source by similarity. Billed to wiki_build;
    on any failure the pipeline logs it and falls back to lexical matching.
    """
    vecs, tok = ac.embed(list(texts), bucket="wiki_build")
    CALL_LOG.append({"phase": "provenance_embed", "bucket": "wiki_build", "in": tok, "out": 0})
    return vecs

pipeline.set_embedder(_embed)

# Semantic navigation (paper §3.1, §3.5, Figure 1b). The index routes by an LLM
# reading plain-text summaries rather than by a key comparison, so the router is
# the method, not an optimisation: without it traverse scores summaries by
# literal word overlap with the question, which is a search key by another name.
# Routing calls land in the wiki_answer bucket via _bucket's default, so their
# cost is reported as answer cost, which is what they are.
pipeline.set_router(pipeline.llm_route_children)

# --- override hardcoded paths (module globals; functions read them at call time) ---
run.BASE = BENCH/"data"/"narrativeqa"          # holds qaps_with_titles.csv
run.CORPUS = MROOT/"corpus"                      # MOVIES_10-named symlinks -> cleaned txt
run.OUTPUT = MROOT/"output"
run.DIGESTS = run.OUTPUT/"digests"
run.OUTPUT.mkdir(parents=True, exist_ok=True)
run.DIGESTS.mkdir(parents=True, exist_ok=True)


def _corpus_docids():
    """Document ids for the corpus being built (30 screenplays, or 10 contracts
    under CUAD=1), so the build timer's coverage reads 30/30 rather than 40/10."""
    import sys as _s; _s.path.insert(0, str(BENCH / "shared"))
    from corpus import wiki_docids
    return list(wiki_docids("cuad" if _os.environ.get("CUAD") else "nqa").values())


def compile_needed(doc_ids):
    """Only compile docs whose digest lacks a manifest (resumable)."""
    todo = []
    for d in doc_ids:
        wd = run.DIGESTS/run.safe_filename(d)
        if (wd/"_manifest.json").exists():
            cold_start.log.info(f"[compile] {d}: already built, skip")
        else:
            todo.append(d)
    return todo


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "compile":
        ids = sys.argv[2:] or run.MOVIES_10
        ids = compile_needed(ids)
        if ids:
            _t0 = time.time()
            run.cmd_compile(ids)
            add_build("llm_index", (time.time()-_t0)/60, n_total=len(_corpus_docids()), inc=len(ids))
        # dump build token cost
        bt = sum(c["in"] + c["out"] for c in CALL_LOG if c["bucket"] == "wiki_build")
        print(f"\n[tokens] wiki_build total = {bt:,}")
    elif cmd == "evaluate":
        run.cmd_evaluate()
    elif cmd == "all":
        ids = compile_needed(run.MOVIES_10)
        if ids:
            _t0 = time.time()
            run.cmd_compile(ids)
            add_build("llm_index", (time.time()-_t0)/60, n_total=len(_corpus_docids()), inc=len(ids))
        run.cmd_evaluate()
    else:
        print(f"unknown cmd: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    main()
