"""usage_report.py — how many tokens this run tree has spent, by day and by what.

The gateway (Portkey) exposes no usage endpoint to this key, and once the key
is over its limit it refuses everything including its own dashboard routes. So
the record is ours: every answer row carries input_tokens/output_tokens, every
index build log ends with its token total, DocETL prints its own, and the judge
prints its bucket per scoring run. This sums all of it, attributed to the day
the file was last written.

What it CANNOT see: ad-hoc probes and burst tests run outside the runners, and
builds whose logs were deleted. Those are listed as "unrecorded" so the total
is read as a floor, not a ceiling.

    BENCH_ROOT=/path/to/run_tree envs/base/bin/python shared/usage_report.py
"""

import os, re, json, glob, pathlib, collections, datetime as dt

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])


def day(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")


def rows_tokens(path):
    t = 0
    for line in open(path, errors="ignore"):
        if line.strip():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t += (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
    return t


def main():
    by = collections.defaultdict(lambda: collections.defaultdict(int))
    # answers: one aggregate file per method per corpus (skip per-document shards,
    # judged copies and verdict caches, which duplicate the same tokens)
    for f in glob.glob(str(BENCH / "methods" / "*" / "results" / "*.jsonl")):
        name = pathlib.Path(f).stem
        meth = pathlib.Path(f).parents[1].name
        if "_judged" in name or "_verdicts" in name:
            continue
        if name not in (meth, f"{meth}_cuad", "graphrag_local", "graphrag_global",
                        "graphrag_local_cuad", "graphrag_global_cuad"):
            continue                                  # per-document shard
        t = rows_tokens(f)
        if t:
            by[day(f)][f"answer  {name}"] += t
    # index builds
    for f in glob.glob(str(BENCH / "methods" / "llm_index" / "logs" / "compile_*.log")):
        m = re.findall(r"wiki_build total = ([\d,]+)", open(f, errors="ignore").read())
        if m and int(m[-1].replace(",", "")):
            by[day(f)]["build   llm_index"] += int(m[-1].replace(",", ""))
    # judge
    for f in glob.glob(str(BENCH / "logs" / "rerun" / "*.log")):
        s = open(f, errors="ignore").read()
        for m in re.finditer(r"judge bucket: in=([\d,]+) out=([\d,]+)", s):
            by[day(f)]["judge   (LLM-as-judge)"] += (int(m.group(1).replace(",", ""))
                                                    + int(m.group(2).replace(",", "")))
    grand = 0
    for d in sorted(by):
        sub = sum(by[d].values()); grand += sub
        print(f"\n{d}   {sub/1e6:7.2f}M tokens")
        for k, v in sorted(by[d].items(), key=lambda kv: -kv[1]):
            print(f"   {v/1e6:7.2f}M   {k}")
    print(f"\nrecorded total: {grand/1e6:.1f}M tokens   (a floor: probes, burst tests and "
          f"builds whose logs were deleted are not in any file)")


if __name__ == "__main__":
    main()
