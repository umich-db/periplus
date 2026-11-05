"""wiki_v6 runner: the user's cold_start.py 3-phase pipeline, unchanged logic,
LLM routed through the shared Azure client (default reasoning, matches paper).

  BUILD: bootstrap master template (sample->propose->reconcile, corpus-level)
         + 2-pass extraction per movie -> v3 wiki (index/facts/relationships/
         timeline/characters/events).
  ANSWER: query_with_refinement (flat read + raw-doc fallback + backfill).

build_tokens = bootstrap + compile (per movie + corpus-level bootstrap share);
answer tokens = query.

Usage:
  run.py build                  # bootstrap + compile all 10 (single process)
  run.py answer --movie <slug>  # answer one movie's questions
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse, importlib.util

BENCH = _BENCH
import sys as _sys; _sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS as _DOCS  # noqa: E402
_CORPUS_KEY = "cuad" if __import__("os").environ.get("CUAD") else "nqa"

WIKI_SRC = _BENCH/"src"/"cold_start"     # cold_start.py lives here
COLD = _BENCH/"src"/"cold_start"
METHOD = "wiki_v6"
MROOT = BENCH/"methods"/METHOD
WIKIS = MROOT/"output"/"wikis"; WIKIS.mkdir(parents=True, exist_ok=True)
OUTDIR = MROOT/"results"; OUTDIR.mkdir(parents=True, exist_ok=True)
BUILD_COSTS = MROOT/"build_costs.json"
CORPUS = BENCH/"data"/"movies"/"txt"

# shared azure client FIRST (loads .env into os.environ so cold_start's client constructs)
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
sys.path.insert(0, str(COLD))
import cold_start  # noqa: E402
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build

BUILD_PHASES = ("proposer", "reconciler", "compile_p1", "compile_p2")

def _llm(prompt, max_tokens=16384, phase="unknown"):
    bucket = "v6_build" if any(phase.startswith(p) for p in BUILD_PHASES) else "v6_answer"
    try:
        text, _pt, _ct = ac.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, bucket=bucket)
        return (text or "").strip()
    except Exception as e:  # noqa: BLE001
        cold_start.log.warning(f"  llm error ({phase}): {repr(e)[:120]}")
        return ""
cold_start.llm = _llm

SLUGS = ["10_things_i_hate_about_you","12_and_holding","1492_conquest_of_paradise","17_again_film",
         "2012_film","50_50_2011_film","8mm_film","9_2005_film","a_dry_white_season","a_few_good_men"]

def bucket_tot(name):
    c = ac.get_counters().get(name, {})
    return c.get("prompt", 0)+c.get("completion", 0)

TPL_PATH = MROOT/"output"/"_master_template.json"

def cmd_bootstrap():
    # Phase 1: bootstrap master template (corpus-level, one-time).
    # n_samples=10 (all movies) -> broader template -> denser extraction, fewer misses.
    build_costs = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}
    b0 = bucket_tot("v6_build"); _t0 = time.time()
    template, outlines, sampled = cold_start.bootstrap_template(str(CORPUS), n_samples=10, seed=42)
    add_build("wiki_v6", (time.time()-_t0)/60, n_total=len(SLUGS), count=False)
    if not template:
        print("BOOTSTRAP FAILED"); sys.exit(2)
    TPL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TPL_PATH.write_text(json.dumps(template, indent=2))
    build_costs["_bootstrap"] = {"build_tokens_total": bucket_tot("v6_build")-b0}
    BUILD_COSTS.write_text(json.dumps(build_costs, indent=2))
    print(f"V6_BOOTSTRAP_DONE tokens={build_costs['_bootstrap']['build_tokens_total']} sections={len(template.get('sections',[]))}")

def cmd_compile(slug):
    # Phase 2: compile ONE movie (2-pass extraction -> v3 wiki). Template must exist.
    build_costs = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}
    template = json.loads(TPL_PATH.read_text())
    wd = WIKIS/slug
    if (wd/"index.md").exists():
        print(f"[compile] {slug}: exists, skip"); return
    b0 = bucket_tot("v6_build"); t0 = time.time()
    compiled = cold_start.compile_document_2pass(str(CORPUS/f"{slug}.txt"), template)
    if compiled:
        cold_start.write_v3_wiki(str(WIKIS), slug, slug.replace("_", " "), compiled)
        build_costs[slug] = {"build_tokens_total": bucket_tot("v6_build")-b0, "build_s": round(time.time()-t0,1)}
        BUILD_COSTS.write_text(json.dumps(build_costs, indent=2))
        add_build("wiki_v6", (time.time()-t0)/60, n_total=len(SLUGS))
        print(f"[compile] {slug}: {build_costs[slug]['build_tokens_total']} tokens")
    else:
        print(f"[compile] {slug}: FAILED")

def cmd_answer(slug):
    build_costs = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}
    # build_tokens per question = (this movie's compile + amortized bootstrap) / n_questions
    boot = build_costs.get("_bootstrap", {}).get("build_tokens_total", 0)
    questions = list(_DOCS[_CORPUS_KEY].questions)
    qs = [q for q in questions if q["movie"] == slug]
    movie_build = build_costs.get(slug, {}).get("build_tokens_total", 0)
    amort = round((movie_build + boot/len(SLUGS)) / max(1, len(qs)))

    wd = WIKIS/slug; raw = CORPUS/f"{slug}.txt"
    out = OUTDIR/f"{METHOD}_{slug}.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["qid"])
    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done: continue
        b0p = ac.get_counters().get("v6_answer", {}); bp, bc = b0p.get("prompt",0), b0p.get("completion",0)
        t0 = time.time()
        try:
            pred, source, _ = cold_start.query_with_refinement(q["question"], str(wd), str(raw))
        except Exception as e:  # noqa: BLE001
            pred, source = f"[ERROR: {repr(e)[:120]}]", "error"
        lat = round(time.time()-t0, 2)
        a = ac.get_counters().get("v6_answer", {})
        in_tok, out_tok = a.get("prompt",0)-bp, a.get("completion",0)-bc
        rec = {"method":METHOD,"movie":slug,"qid":q["qid"],"question":q["question"],
               "gold_answers":q["gold_answers"],"prediction":pred,"source":source,
               "input_tokens":in_tok,"output_tokens":out_tok,"total_tokens":in_tok+out_tok,
               "latency_s":lat,"build_tokens":amort}
        f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
        print(f"  {q['qid']} src={source} in={in_tok} out={out_tok} {lat}s :: {pred[:50]}", flush=True)
    f.close()
    print(f"DONE {slug}: {len(qs)} -> {out}", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["bootstrap","compile","answer"]); ap.add_argument("--movie")
    a = ap.parse_args()
    if a.cmd == "bootstrap": cmd_bootstrap()
    elif a.cmd == "compile": cmd_compile(a.movie)
    else: cmd_answer(a.movie)

if __name__ == "__main__":
    main()
