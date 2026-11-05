"""Driver for wiki_v7 in BEST-GUESS mode (the second "node").

Identical setup to wiki_v7/cli.py, with ONE behavioral change: the answer node.
- wiki_v7 (strict/fallback): digest QA prompt says "if the answer is not in the
  digest, return exactly: not found" -> misses fall through to a raw-document read.
- wiki_v7_guess (best guess): same v7 traversal + digest context, but the answer
  node is instructed to ALWAYS give its best guess from the available context and
  never emit "not found". No raw-document fallback.

This mirrors the raptor two-node experiment: one strict node, one best-guess node,
same retrieved context feeding both.

Digests are REUSED from the already-built wiki_v7 output (no recompile). Only the
answer pass differs, so this driver exposes just `query_bestguess`, which is
monkeypatched over pipeline.query_with_traversal so answer.py stays symmetric.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, importlib.util

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
MROOT = BENCH/"methods"/"wiki_v7_guess"

# --- shared azure client ---
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# --- import user's v7 code ---
sys.path.insert(0, str(WIKI_SRC))
import cold_start, pipeline, run  # noqa: E402
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build  # noqa: E402,F401

# --- per-call LLM log (for token accounting) ---
CALL_LOG = []  # list of {phase, in, out}
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
                               max_tokens=max_tokens, bucket=bucket)  # default reasoning (matches paper)
    except Exception as e:  # noqa: BLE001
        cold_start.log.warning(f"  llm error ({phase}): {repr(e)[:120]}")
        text, pt, ct = "", 0, 0
    CALL_LOG.append({"phase": phase, "bucket": bucket, "in": pt, "out": ct})
    return (text or "").strip()

cold_start.llm = _llm
pipeline.llm = _llm
run.llm = _llm

# --- override hardcoded paths (module globals; functions read them at call time) ---
run.BASE = BENCH/"data"/"narrativeqa"
run.CORPUS = BENCH/"methods"/"wiki_v7"/"corpus"
run.OUTPUT = MROOT/"output"
# REUSE wiki_v7's already-built digests -- only the answer node changes here.
run.DIGESTS = BENCH/"methods"/"wiki_v7"/"output"/"digests_v7"
run.OUTPUT.mkdir(parents=True, exist_ok=True)


# --- the best-guess answer node (the only real difference from wiki_v7) ---
BESTGUESS_QA_PROMPT = (
    "You are answering a question about a document using the compiled wiki digest "
    "below.\nAnswer in 10 words or fewer. Even if the digest does not fully contain "
    "the answer, make your best guess from the available context. Never say "
    "'not found' -- always commit to an answer.\n\n"
    "Question: {question}\n\nWiki Digest:\n{digest}\n\nAnswer:"
)


def query_bestguess(question, wiki_dir, raw_doc_path=None):
    """Same v7 traversal/context, but always answer -- no 'not found', no raw fallback."""
    wiki_dir = pipeline.Path(wiki_dir)
    digest = pipeline.traverse(question, wiki_dir)
    prompt = BESTGUESS_QA_PROMPT.format(question=question, digest=digest or "(empty)")
    predicted = pipeline.llm(prompt, max_tokens=2048, phase="query_bestguess")
    predicted = (
        (predicted or "").strip().splitlines()[0].strip(" .\"'")
        if predicted else ""
    )
    # Guard against a model that ignores the instruction and still refuses.
    if predicted.lower() in ("not found", "", "none", "n/a"):
        predicted = predicted or "unknown"
    return predicted, "wiki_guess", {}


# make answer.py symmetric with wiki_v7's (it calls pipeline.query_with_traversal)
pipeline.query_with_traversal = query_bestguess


def main():
    print("wiki_v7_guess is answer-only (digests reused from wiki_v7).")
    print("Run: python answer.py --movie <slug>")


if __name__ == "__main__":
    main()
