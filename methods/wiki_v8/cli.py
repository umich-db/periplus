"""Driver for the user's wiki_index/packed.py (v8 packed index), adapted to this
server WITHOUT editing their code:
  - every LLM call (cold_start / pipeline / packed) routed through the shared
    azure_client with token buckets: packed_build / packed_answer / judge
  - the cluster embedder is injected as azure_client.embed (text-embedding-3-large)
  - source documents read from bench/data/movies/txt/<slug>.txt

Subcommands:
  compile [slug...]   build the packed index for one or more movies (default: all 10)

Phase 2 adds `answer` / `evaluate` here; Phase 1 is build-only.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, time, json, pathlib, importlib.util

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
MROOT = BENCH / "methods" / "wiki_v8"
TXT = BENCH / "data" / "movies" / "txt"
OUTPUT = MROOT / "output" / "packed_v8"

MOVIES_10 = [
    "10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
    "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
    "a_dry_white_season", "a_few_good_men",
]

# --- shared azure client (imported by path, like the other methods) ----------
spec = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

# --- import the user's v8 code ------------------------------------------------
sys.path.insert(0, str(WIKI_SRC))
import cold_start, pipeline, packed  # noqa: E402
sys.path.insert(0, str(BENCH / "shared"))
from buildtimer import add_build  # noqa: E402

# --- per-call LLM log + bucket routing ---------------------------------------
CALL_LOG = []
BUILD_PHASES = ("discover", "proposer", "reconciler", "compile_p1", "compile_p2",
                "consolidate", "pack_summary")


def _bucket(phase):
    if phase == "judge":
        return "judge"
    if any(phase.startswith(p) for p in BUILD_PHASES):
        return "packed_build"
    return "packed_answer"


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
packed.llm = _llm

# --- inject the cluster embedder (Azure text-embedding-3-large) ---------------
packed.set_cluster_embedder(lambda texts: ac.embed(texts, bucket="packed_build"))

# --- config: load overrides from methods/wiki_v8/config.json if present -------
packed.configure(packed.PackedConfig.from_file(MROOT / "config.json"))


def _build_tokens():
    return sum(c["in"] + c["out"] for c in CALL_LOG if c["bucket"] == "packed_build")


def compile_needed(slugs):
    """Only build movies whose packed store lacks packed.json (resumable)."""
    todo = []
    for s in slugs:
        if (OUTPUT / s / "packed.json").exists():
            cold_start.log.info(f"[compile] {s}: already built, skip")
        else:
            todo.append(s)
    return todo


def cmd_compile(slugs):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stats_all = {}
    t0 = time.time()
    for s in slugs:
        doc_path = TXT / f"{s}.txt"
        if not doc_path.exists():
            cold_start.log.warning(f"  no source text for {s}: {doc_path}")
            continue
        cold_start.log.info("=" * 60)
        cold_start.log.info(f"WIKI v8 - PACK BUILD: {s}  (C={packed.CFG.node_budget_tokens}, "
                            f"s={packed.CFG.child_summary_tokens}, f={packed.CFG.fanout})")
        before = _build_tokens()
        stats = packed.build_packed_index(str(doc_path), str(OUTPUT), s, s.replace("_", " "))
        if stats:
            stats["build_tokens"] = _build_tokens() - before
            stats_all[s] = stats
    if slugs:
        add_build("wiki_v8", (time.time() - t0) / 60, n_total=len(MOVIES_10), inc=len(slugs))

    (MROOT / "output" / "compile_stats.json").write_text(json.dumps(stats_all, indent=2))
    print("\n" + "=" * 60)
    for s, st in stats_all.items():
        print(f"  {s:34s}  leaves={st['doc_leaves']:4d}  depth={st['depth']}  "
              f"f_max={st['fanout_max']:2d}  nodes={st['n_nodes']:4d}  "
              f"build_tok={st['build_tokens']:,}")
    print(f"\n[tokens] packed_build total = {_build_tokens():,}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compile"
    if cmd == "compile":
        slugs = sys.argv[2:] or MOVIES_10
        slugs = compile_needed(slugs)
        if slugs:
            cmd_compile(slugs)
        else:
            print("nothing to build")
    else:
        print(f"unknown cmd: {cmd} (Phase 1 supports: compile)")
        sys.exit(1)


if __name__ == "__main__":
    main()
