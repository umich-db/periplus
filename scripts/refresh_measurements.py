"""refresh_measurements.py — regenerate the two hand-off files from a run tree.

  /home/phdonn/index/output.txt   accuracy + build/answer tokens, per corpus,
                                  ascending by accuracy (the CIDR convention)
  /home/phdonn/index/latency.txt  answer minutes, build minutes, total, per
                                  method, descending by total

Both are derived from the run tree's own records, never typed in:
  - accuracy and tokens from results/eval/rerun_metrics.json, which
    shared/collect_metrics.py rebuilds from the per-question rows first
  - answer minutes = sum of per-question latency_s / 60; for the batched
    methods (DocETL, Doctopus) that record no inline latency, the sampled
    sequential probe's median x n / 60, which is how the file was first built
  - build minutes from results/eval/build_times/<m>.json, except RAPTOR, whose
    own per-tree accumulator survives resumes and is the honest total, and the
    index, whose record sums its stamped wall-clock segments

    BENCH_ROOT=/home/phdonn/index/gpt54 envs/base/bin/python scripts/refresh_measurements.py
"""

import os, re, sys, json, pathlib, datetime as dt, subprocess

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])
SRC_BENCH = pathlib.Path(__file__).resolve().parents[1]           # where the code lives
OUT_TXT = pathlib.Path("/home/phdonn/index/output.txt")
LAT_TXT = pathlib.Path("/home/phdonn/index/latency.txt")
sys.path.insert(0, str(SRC_BENCH / "shared"))
from corpus import SCORED, DISPLAY  # noqa: E402

DIRS = {m: m for m in SCORED}
DIRS.update({"graphrag_local": "graphrag", "graphrag_global": "graphrag"})


def collect():
    subprocess.run([str(SRC_BENCH / "envs/base/bin/python"), str(SRC_BENCH / "shared/collect_metrics.py")],
                   cwd=BENCH, env=dict(os.environ, BENCH_ROOT=str(BENCH)),
                   stdout=subprocess.DEVNULL, check=True)
    return json.loads((BENCH / "results/eval/rerun_metrics.json").read_text())


def write_output(d):
    env = pathlib.Path("/home/phdonn/index/.env").read_text()
    model = re.findall(r'^LLM_MODEL="?([^"\n]+)', env, re.M)[-1]
    L = [f"Benchmark results  |  run tree: {BENCH.name}  |  model: {model}  |  judge: LLM-as-judge  |  written {dt.datetime.now():%Y-%m-%d %H:%M}",
         "Tokens are input+output. build = one-time index construction; answer = recurring, summed over all questions.",
         "Rows ascend by accuracy (CIDR figure convention). 'partial' = method covers fewer questions than the corpus; 'not run' = no rows yet.", ""]
    # Rows already in output.txt that carry numbers, per corpus, keyed by method:
    # a hand-entered row survives until a complete, scored run replaces it.
    prior_out, cur = {}, None
    if OUT_TXT.exists():
        names = {v: k for k, v in DISPLAY.items()}
        for line in OUT_TXT.read_text().splitlines():
            if line.startswith("=== NarrativeQA"): cur = "nqa"
            elif line.startswith("=== CUAD"): cur = "cuad"
            else:
                mm = re.match(r"(.+?)\s{2,}([\d.]+)\s+[\d,]+", line)
                if cur and mm and mm.group(1).strip() in names:
                    prior_out.setdefault(cur, {})[names[mm.group(1).strip()]] = line
    for key, c in d["corpora"].items():
        L.append(f"=== {c['name']}  ({c['n_docs']} docs, {c['n_questions']} questions) ===")
        L.append(f"{'method':<11s} {'accuracy%':>9s} {'answer tok':>12s} {'build tok':>11s} {'TOTAL tok':>12s} {'n':>5s}  status")
        L.append("-" * 76)
        ms = sorted(c["methods"].items(),
                    key=lambda kv: (kv[1].get("accuracy_pct") is None, kv[1].get("accuracy_pct") or 0))
        for m, r in ms:
            if r.get("status") == "not run":
                if m in prior_out.get(key, {}):
                    L.append(prior_out[key][m]); continue
                L.append(f"{r['display']:<11s} {'-':>9s} {'-':>12s} {'-':>11s} {'-':>12s} {'-':>5s}  not run"); continue
            acc = r.get("accuracy_pct"); at = r.get("answer_tokens") or 0; bt = r.get("build_tokens") or 0
            if m in prior_out.get(key, {}) and (acc is None or str(r.get("status", "")).startswith("partial")) and m != "doctopus":
                L.append(prior_out[key][m]); continue             # keep the existing row until a full, scored run replaces it
            L.append(f"{r['display']:<11s} {(f'{acc:.1f}' if acc is not None else '-'):>9s} {at:>12,d} {bt:>11,d} "
                     f"{at + bt:>12,d} {r.get('n_rows', 0):>5d}  {r['status']}")
        L.append("")
    OUT_TXT.write_text("\n".join(L) + "\n")


def answer_minutes(m, corpus_sfx=""):
    f = BENCH / "methods" / DIRS[m] / "results" / f"{m}{corpus_sfx}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float)) and r["latency_s"] > 0]
    if lat:
        return sum(lat) / 60
    pr = BENCH / "results/eval/latency_probe" / f"{DIRS[m]}{corpus_sfx}.json"
    if pr.exists():
        p = json.loads(pr.read_text())
        if p.get("answer_n"):
            return p["answer_median_s"] * len(rows) / 60
    return None


def build_minutes(m, corpus_sfx=""):
    if m == "raptor":
        f = BENCH / "methods/raptor/output" / f"build_stats{corpus_sfx}.json"
        if f.exists():
            d = json.loads(f.read_text())
            return sum(v.get("build_min", 0) for v in d.values() if isinstance(v, dict))
    f = BENCH / "results/eval/build_times" / f"{DIRS[m]}{corpus_sfx}.json"
    if f.exists():
        return json.loads(f.read_text()).get("build_min")
    return None


def _complete_and_clean(m, corpus_sfx=""):
    """A method's numbers are publishable only if it answered every question with
    real answers and has been scored. A partial or API-failure run has a real
    wall-clock and token count, but they are not the method's cost, and written
    into a ranked table they read as if they were."""
    f = BENCH / "methods" / DIRS[m] / "results" / f"{m}{corpus_sfx}.jsonl"
    if not f.exists():
        return False
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if any(str(r.get("prediction", "")).startswith("[ERROR") for r in rows):
        return False
    return (BENCH / "results/eval" / f"{m}{corpus_sfx}_summary.json").exists()


def _existing_rows(path):
    """Rows already in latency.txt, keyed by method name, so a row this script
    cannot measure yet (a hand-entered placeholder) is carried forward instead of
    dropped. A measured row always replaces it."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines()[2:]:
            mm = re.match(r"(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if mm:
                out[mm.group(1).strip().rstrip(".")] = (float(mm.group(2)), float(mm.group(3)))
    return out


def write_latency():
    rows, prior = [], _existing_rows(LAT_TXT)
    for m in SCORED:
        name = DISPLAY[m]
        a, b = answer_minutes(m), build_minutes(m)
        # doctopus is published at its known partial coverage, as it always has been
        if a is not None and (_complete_and_clean(m) or m == "doctopus"):
            rows.append((name, a, b or 0.0))
        elif name in prior:
            rows.append((name, *prior[name]))            # carried forward, not measured here
    rows.sort(key=lambda t: -(t[1] + t[2]))
    out = [f"{'method':<18s} {'answer':>6s}  {'build':>6s}  {'TOTAL':>6s}", "-" * 44]
    out += [f"{n:<18s} {a:6.2f}  {b:6.2f}  {a + b:6.2f}" for n, a, b in rows]
    LAT_TXT.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    d = collect()
    write_output(d)
    write_latency()
    print(f"refreshed {OUT_TXT} and {LAT_TXT} from {BENCH}")
    print(LAT_TXT.read_text())
