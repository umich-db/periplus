"""Driver for wiki_v7 with RAPTOR-style AUGMENT (summary + embedded detail pages) retrieval.

Same build/answer as the ORIGINAL wiki_v7 (strict digest QA prompt + raw-document
fallback). The ONLY change is the retrieval step: instead of pipeline.traverse
(which on these digests just reads the truncated consolidated summary because it
exceeds CONTEXT_BUDGET), we retrieve the query-relevant detail PAGES by blended
cosine+keyword score over their embedded summaries (shared/sem_retrieval.py,
augment_retrieve). Digests are REUSED from wiki_v7 (no recompile).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, importlib.util

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
MROOT = BENCH/"methods"/"wiki_v7_aug"

# --- shared azure client ---
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# --- import user's v7 code ---
sys.path.insert(0, str(WIKI_SRC))
import cold_start, pipeline, run  # noqa: E402
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build  # noqa: E402,F401
import sem_retrieval as sr  # noqa: E402

# --- per-call LLM log (for token accounting) ---
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

# --- THE change: swap retrieval only; keep the original strict QA + fallback ---
# query_with_traversal calls the module-global `traverse`; rebinding it here reroutes
# retrieval to embedding-based collapsed-tree while leaving the answer node untouched.
pipeline.traverse = sr.augment_retrieve


def main():
    print("wiki_v7_aug is answer-only (digests reused from wiki_v7).")
    print("Run: python answer.py --movie <slug>")


if __name__ == "__main__":
    main()
