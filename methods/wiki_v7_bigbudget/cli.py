"""Driver for wiki_v7 with a RAISED context budget (the 'more context' control).

Baseline for the embedding-retrieval experiment. Everything is the ORIGINAL wiki_v7
(same build, same keyword traverse, same strict QA + raw-doc fallback) EXCEPT the
retrieval char budget is raised from 12k -> 24k. On these digests the 12k budget is
entirely consumed by the (truncated) consolidated summary, so v7 never reads detail
pages; at 24k it reads the FULL consolidated overview AND keyword-traverses into
detail pages. This isolates "does more context / reading detail pages help" from
"does embedding-based page selection help" (wiki_v7_collapse / wiki_v7_sem).

Digests are REUSED from wiki_v7 (no recompile).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, importlib.util

BIG_BUDGET = 24_000  # 2x the default CONTEXT_BUDGET; full consolidated + keyword detail pages

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
MROOT = BENCH/"methods"/"wiki_v7_bigbudget"

# --- shared azure client ---
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# --- import user's v7 code ---
sys.path.insert(0, str(WIKI_SRC))
import cold_start, pipeline, run  # noqa: E402
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build  # noqa: E402,F401

CALL_LOG = []
BUILD_PHASES = ("discover", "proposer", "reconciler", "compile_p1", "compile_p2", "consolidate")

def _bucket(phase):
    if phase == "judge":
        return "judge"
    if any(phase.startswith(p) for p in BUILD_PHASES):
        return "wiki_build"
    return "wiki_answer"

def _llm(prompt, max_tokens=16384, phase="unknown"):
    bucket = _bucket(phase)
    try:
        text, pt, ct = ac.chat([{"role": "user", "content": prompt}],
                               max_tokens=max_tokens, bucket=bucket)
    except Exception as e:  # noqa: BLE001
        cold_start.log.warning(f"  llm error ({phase}): {repr(e)[:120]}")
        text, pt, ct = "", 0, 0
    CALL_LOG.append({"phase": phase, "bucket": bucket, "in": pt, "out": ct})
    return (text or "").strip()

cold_start.llm = _llm
pipeline.llm = _llm
run.llm = _llm

# --- override hardcoded paths; REUSE wiki_v7's already-built digests ---
run.BASE = BENCH/"data"/"narrativeqa"
run.CORPUS = BENCH/"methods"/"wiki_v7"/"corpus"
run.OUTPUT = MROOT/"output"
run.DIGESTS = BENCH/"methods"/"wiki_v7"/"output"/"digests_v7"
run.OUTPUT.mkdir(parents=True, exist_ok=True)

# --- THE change: raise the traversal budget. CONTEXT_BUDGET is baked into traverse's
# default arg at definition time, so reassigning the constant won't take; wrap instead.
_orig_traverse = pipeline.traverse
def _big_traverse(question, wiki_dir, max_chars=BIG_BUDGET):
    return _orig_traverse(question, wiki_dir, max_chars=BIG_BUDGET)
pipeline.traverse = _big_traverse


def main():
    print(f"wiki_v7_bigbudget is answer-only (budget={BIG_BUDGET}, digests reused from wiki_v7).")
    print("Run: python answer.py --movie <slug>")


if __name__ == "__main__":
    main()
