"""rerun_all.py — run every method over both corpora, timing and scoring each.

Drives scripts/run_method.sh, which already knows each method's venv and its
build/answer split, and adds the three things a full sweep needs:

  * build wall-clock recorded PER CORPUS, not once per method. The old
    build_times/<m>.json held a single number covering whichever corpus happened
    to run last, so a build-time chart could not separate 30 screenplays from 10
    contracts.
  * uniform scoring. One scorer per corpus for every method: shared/eval.py on
    NarrativeQA, shared/eval_cuad_baselines.py on CUAD. No method brings its own.
  * resumability across the whole sweep, not just within a phase. State lives in
    results/eval/rerun_state.json, so killing this and re-issuing it continues.

Every phase underneath is already resumable, so a re-run costs nothing for work
already done.

    envs/base/bin/python scripts/rerun_all.py                  # everything
    envs/base/bin/python scripts/rerun_all.py --only naive_gpt --corpus cuad
    envs/base/bin/python scripts/rerun_all.py --dry-run
    envs/base/bin/python scripts/rerun_all.py --status
"""

import os, sys, json, time, argparse, subprocess, pathlib, datetime

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS, SCORED  # noqa: E402

STATE = BENCH / "results" / "eval" / "rerun_state.json"
LOGS = BENCH / "logs" / "rerun"
BASE = "envs/base/bin/python"

# Cheapest first, so the chain is proven end to end before the long builds start.
# (method, needs_build). Embedding-only builds are listed as not needing a
# rebuild: rag, leann and doctopus index with text-embedding-3-large, which this
# rerun does not change, so their indexes carry over untouched and only their
# answering is redone. Every method whose BUILD calls the chat model is rebuilt.
PLAN = [
    ("naive_gpt", False),
    ("docetl_standard", False),
    ("rag", False),
    ("doctopus", False),
    ("leann", False),
    ("raptor", True),
    ("llm_index", True),
    ("graphrag", True),
]

# Which scored names each method produces, for the scoring step.
FANOUT = {"graphrag": ["graphrag_local", "graphrag_global"]}


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except ValueError:
            pass
    return {"done": {}, "failed": {}, "started": None}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def log_path(method, corpus, phase):
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"{method}_{corpus}_{phase}.log"


def run(cmd, env, logfile, dry=False):
    """Run one phase, streaming to its own log. Returns (ok, seconds)."""
    if dry:
        print(f"      would run: {' '.join(cmd)}")
        return True, 0.0
    t0 = time.time()
    with open(logfile, "a") as f:
        f.write(f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} "
                f"{' '.join(cmd)} =====\n")
        f.flush()
        p = subprocess.run(cmd, cwd=BENCH, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode == 0, time.time() - t0


def rows_present(method, corpus):
    d = {"graphrag_local": "graphrag", "graphrag_global": "graphrag"}.get(method, method)
    f = BENCH / "methods" / d / "results" / f"{method}{corpus.result_suffix}.jsonl"
    if not f.exists():
        return 0
    return sum(1 for l in f.read_text().splitlines() if l.strip())


def built_count(method):
    """How many documents this method has an index for.

    Our index writes one _manifest.json per document; other methods write one
    directory per document under output/. Either way, a count that rises across
    a build phase means the phase actually built something."""
    d = BENCH / "methods" / method / "output"
    if not d.exists():
        return 0
    # The balanced tree mirrors every manifest, so count the source tree only or
    # 3 documents read as 6 once the 3.4 pass has run.
    manifests = sum(1 for m in d.rglob("_manifest.json") if "digests_balanced" not in m.parts)
    if manifests:
        return manifests
    return sum(1 for p in d.rglob("*") if p.is_dir() and any(p.iterdir()))


def key_alive():
    """One tiny call. False on an exhausted key, so a sweep stops before it fills
    result files with zero-token misses that look complete."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ac", BENCH / "shared" / "azure_client.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
    try:
        ac.chat([{"role": "user", "content": "ok"}], max_tokens=8, bucket="probe", max_retries=3)
        return True, ""
    except Exception as e:                       # noqa: BLE001
        return False, str(e)[:200]


def write_build_time(method, corpus, minutes, n_docs):
    d = BENCH / "results" / "eval" / "build_times"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{method}{corpus.result_suffix}.json"
    (d / name).write_text(json.dumps({
        "method": method, "corpus": corpus.key, "build_min": round(minutes, 3),
        "build_coverage": "full", "n_built": n_docs, "n_total": n_docs,
    }, indent=2))


def score(method, corpus, env, dry=False):
    names = FANOUT.get(method, [method])
    if corpus.key == "cuad":
        cmd = [BASE, "shared/eval_cuad_baselines.py"] + names
    else:
        cmd = [BASE, "shared/eval.py"] + names
    return run(cmd, env, log_path(method, corpus.key, "score"), dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="limit to these methods")
    ap.add_argument("--corpus", choices=list(DOCS), help="limit to one corpus")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    st = load_state()
    if args.status:
        print(json.dumps(st, indent=2)); return
    st["started"] = st.get("started") or datetime.datetime.now().isoformat(timespec="seconds")

    corpora = [DOCS[args.corpus]] if args.corpus else list(DOCS.values())
    plan = [(m, b) for m, b in PLAN if not args.only or m in args.only]

    for corpus in corpora:
        n_expected = len(corpus.questions)
        print(f"\n########## {corpus.name}: {len(corpus.slugs)} docs, "
              f"{n_expected} questions ##########", flush=True)
        env = dict(os.environ, BENCH_ROOT=str(BENCH))
        if corpus.cuad_env:
            env["CUAD"] = corpus.cuad_env
        else:
            env.pop("CUAD", None)

        for method, needs_build in plan:
            key = f"{method}:{corpus.key}"
            print(f"\n--- {key}", flush=True)
            if not args.dry_run:
                alive, why = key_alive()
                if not alive:
                    print(f"\nSTOPPING: the API key is not usable ({why}).\n"
                          f"Nothing further will run; re-issue this command once the key "
                          f"has quota. Completed work is kept.", flush=True)
                    save_state(st); sys.exit(3)

            if needs_build and not args.skip_build:
                bkey = key + ":build"
                if st["done"].get(bkey):
                    print(f"    build: already done ({st['done'][bkey]['min']:.1f} min)")
                else:
                    print(f"    build: starting ({len(corpus.slugs)} docs)", flush=True)
                    before = built_count(method)
                    ok, secs = run(["bash", "scripts/run_method.sh", method, "build"],
                                   env, log_path(method, corpus.key, "build"), args.dry_run)
                    if ok:
                        # Only record wall-clock for a build that built something.
                        # Every build phase is resumable, so a method whose index
                        # was built outside this sweep finishes here in seconds,
                        # and writing those seconds as its build time would
                        # replace a real 8-hour number with a near-zero one. If
                        # nothing new was built, any existing record stands.
                        after = built_count(method)
                        existing = (BENCH / "results" / "eval" / "build_times" /
                                    f"{method}{corpus.result_suffix}.json").exists()
                        if after > before or not existing:
                            write_build_time(method, corpus, secs / 60, len(corpus.slugs))
                            print(f"    build: done in {secs/60:.1f} min", flush=True)
                        else:
                            print(f"    build: already built outside this sweep "
                                  f"({after} docs); keeping the recorded time", flush=True)
                        # A build phase is done only when every document has an index.
                        # A dead key made a phase "succeed" with 3/40 built, and the
                        # resumed sweep then skipped the build entirely.
                        if after >= len(corpus.slugs):
                            st["done"][bkey] = {"min": secs / 60, "built_new": after - before}
                        else:
                            st["failed"][bkey] = f"only {after}/{len(corpus.slugs)} documents built"
                            print(f"    build: INCOMPLETE ({after}/{len(corpus.slugs)}); "
                                  f"will retry on the next run", flush=True)
                    else:
                        st["failed"][bkey] = "build returned nonzero"
                        print(f"    build: FAILED after {secs/60:.1f} min "
                              f"(see {log_path(method, corpus.key, 'build')})", flush=True)
                        save_state(st)
                        continue
                    save_state(st)

            akey = key + ":answer"
            have = rows_present(FANOUT.get(method, [method])[0], corpus)
            if have >= n_expected:
                print(f"    answer: already complete ({have}/{n_expected})")
            else:
                print(f"    answer: starting (have {have}/{n_expected})", flush=True)
                ok, secs = run(["bash", "scripts/run_method.sh", method, "answer"],
                               env, log_path(method, corpus.key, "answer"), args.dry_run)
                have = rows_present(FANOUT.get(method, [method])[0], corpus)
                st["done"][akey] = {"min": secs / 60, "rows": have}
                if not ok:
                    st["failed"][akey] = f"answer returned nonzero ({have}/{n_expected} rows)"
                print(f"    answer: {have}/{n_expected} rows in {secs/60:.1f} min", flush=True)
                save_state(st)

            if have:
                ok, secs = score(method, corpus, env, args.dry_run)
                print(f"    score: {'ok' if ok else 'FAILED'} ({secs/60:.1f} min)", flush=True)
                save_state(st)

    if not args.dry_run:
        run([BASE, "shared/collect_metrics.py"], dict(os.environ, BENCH_ROOT=str(BENCH)),
            LOGS / "collect.log")
        run([BASE, "shared/make_rerun_charts.py"], dict(os.environ, BENCH_ROOT=str(BENCH)),
            LOGS / "charts.log")
    print("\nsweep finished. metrics: results/eval/rerun_metrics.json, "
          "charts: report/rerun/", flush=True)


if __name__ == "__main__":
    main()
