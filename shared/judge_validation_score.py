"""Score the manual judge-validation labels.

Reads bench/results/eval/judge_validation_sample.csv after the `human_verdict`
column is filled in (values `correct` or `incorrect`; anything else is skipped),
and reports:

  - overall % agreement (judge vs human)
  - Cohen's kappa
  - confusion matrix (judge x human)
  - per-dataset and per-method breakdowns

Usage (from bench/):  envs/base/bin/python shared/judge_validation_score.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import csv, pathlib, collections

BENCH = _BENCH
CSV_PATH = BENCH/"results"/"eval"/"judge_validation_sample.csv"

def cohens_kappa(pairs):
    """pairs: iterable of (judge, human), both in {'correct','incorrect'}."""
    n = len(pairs)
    if n == 0: return float("nan")
    a = sum(1 for j, h in pairs if j == h) / n           # observed agreement
    labels = ["correct", "incorrect"]
    pj = {L: sum(1 for j, _ in pairs if j == L)/n for L in labels}
    ph = {L: sum(1 for _, h in pairs if h == L)/n for L in labels}
    e  = sum(pj[L]*ph[L] for L in labels)                # expected agreement
    return (a - e) / (1 - e) if (1 - e) > 0 else float("nan")

rows = list(csv.DictReader(open(CSV_PATH)))
valid = [r for r in rows if r["human_verdict"].strip().lower() in ("correct", "incorrect")]
skipped = len(rows) - len(valid)
print(f"Loaded {len(rows)} rows ({len(valid)} labeled, {skipped} unlabeled/skipped)")
if not valid:
    print("No labels yet. Fill `human_verdict` in the CSV and re-run.")
    raise SystemExit

pairs = [(r["judge_verdict"].strip().lower(), r["human_verdict"].strip().lower())
         for r in valid]
n = len(pairs)
agree = sum(1 for j, h in pairs if j == h)
pct = 100 * agree / n
k = cohens_kappa(pairs)

print(f"\nOverall agreement: {agree}/{n} = {pct:.1f}%")
print(f"Cohen's kappa:     {k:.3f}")

# confusion matrix
labels = ["correct", "incorrect"]
cm = collections.Counter()
for j, h in pairs: cm[(j, h)] += 1
print("\nConfusion matrix (rows = judge, cols = human):")
print(f"                {'human=correct':>15s} {'human=incorrect':>17s}")
for j in labels:
    print(f"  judge={j:9s}  "
          + f"{cm[(j, 'correct')]:>15d} {cm[(j, 'incorrect')]:>17d}")

# per-dataset
print("\nBy dataset:")
by_ds = collections.defaultdict(list)
for r, p in zip(valid, pairs):
    by_ds[r["dataset"]].append(p)
for ds, ps in sorted(by_ds.items()):
    a = sum(1 for j, h in ps if j == h)
    print(f"  {ds:12s}  {a}/{len(ps)} = {100*a/len(ps):5.1f}%   kappa={cohens_kappa(ps):+.3f}")

# per-method
print("\nBy method:")
by_m = collections.defaultdict(list)
for r, p in zip(valid, pairs):
    by_m[r["method"]].append(p)
for m in sorted(by_m):
    ps = by_m[m]
    a = sum(1 for j, h in ps if j == h)
    print(f"  {m:10s}  {a}/{len(ps)} = {100*a/len(ps):5.1f}%   kappa={cohens_kappa(ps):+.3f}")
