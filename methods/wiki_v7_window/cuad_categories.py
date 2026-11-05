"""CUAD category-discovery gate: print the categories v7 discovered on each contract.

This runs BEFORE any eval tokens are spent. Subject discovery was designed and tuned on screenplays;
on contracts we expect things like parties, obligations, termination, and payment terms rather than
characters and scenes. If discovery is degenerate (one giant category swallowing the contract, or
per-chunk fragments that never merged into shared categories), we halt and report instead of paying
for a 410-question eval on a broken index.

Degeneracy criteria. These are calibrated against the 30-screenplay corpus the method was designed
on, so the gate measures "worse than the home distribution", not an arbitrary bar:
  - one giant category: a single top-level section holds >= 70% of all subject leaves
    (screenplays: 2/30 documents do this; it is the real failure mode)
  - fragmented: unmerged per-chunk fragments, detected as NESTED/DUPLICATE category names within one
    document (one category name containing another after normalisation), which is what failed merging
    actually looks like
  - too few: fewer than 3 top-level sections

NOTE on a rejected heuristic: "median category size <= 1 subject" was tried first and rejected. It
fires on 7/30 screenplays, i.e. on the method's own home distribution, because v7 renders any section
under PAGE_SIZE into a SINGLE page, so a one-page category means a compact section, not a fragment.
Counting manifest pages per category is not a fragmentation test and it is not used as one here.

Usage: python cuad_categories.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib, statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
V7 = BENCH / "methods" / "wiki_v7" / "output" / "digests_v7"
sys.path.insert(0, str(BENCH / "methods" / "wiki_v7"))
import cli  # noqa: E402
run = cli.run


def top(path):
    return str(pathlib.PurePosixPath(path).parts[0])


def main():
    sel = json.loads((HERE / "results" / "cuad_selection.json").read_text())
    out, degenerate = [], []
    for s in sel:
        d = V7 / run.safe_filename(s["wiki_key"])
        mp = d / "_manifest.json"
        if not mp.exists():
            print(f"  [missing build] {s['title'][:60]}")
            continue
        m = json.loads(mp.read_text())
        leaves = [e for e in m if e["level"] == "leaf"]
        per_cat = defaultdict(int)
        for e in leaves:
            per_cat[top(e["path"])] += 1
        cats = sorted(per_cat, key=lambda c: -per_cat[c])
        n_leaves = len(leaves)
        biggest = per_cat[cats[0]] / n_leaves if n_leaves and cats else 0
        med = statistics.median(per_cat.values()) if per_cat else 0
        import re as _re
        norm = [_re.sub(r"[^a-z]", "", c.lower()) for c in cats]
        dup_pairs = sum(1 for i in range(len(norm)) for j in range(i + 1, len(norm))
                        if norm[i] and norm[j] and (norm[i] in norm[j] or norm[j] in norm[i]))
        flags = []
        if biggest >= 0.70:
            flags.append(f"one-giant-category({biggest:.0%})")
        if dup_pairs > 0 or len(cats) > 40:
            flags.append(f"fragmented(dup_name_pairs={dup_pairs}, n={len(cats)})")
        if len(cats) < 3:
            flags.append(f"too-few-categories({len(cats)})")
        if flags:
            degenerate.append((s["title"], flags))
        out.append({"title": s["title"], "slug": s["slug"], "tokens": s["tokens"],
                    "n_subjects": n_leaves, "n_categories": len(cats),
                    "largest_share": round(biggest, 3), "median_category_size": med,
                    "duplicate_name_pairs": dup_pairs,
                    "categories": {c: per_cat[c] for c in cats}, "flags": flags})
        print(f"\n{s['title'][:70]}")
        print(f"  {n_leaves} subjects across {len(cats)} categories "
              f"(largest {biggest:.0%}, median {med}){'  FLAGS: ' + ','.join(flags) if flags else ''}")
        for c in cats[:14]:
            print(f"    {per_cat[c]:4d}  {c.replace('_', ' ')}")
        if len(cats) > 14:
            print(f"    ... and {len(cats)-14} more")

    (HERE / "results" / "cuad_categories.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 78)
    if degenerate:
        print(f"GATE: DEGENERATE on {len(degenerate)}/{len(out)} contracts. Halt and report before eval.")
        for t, f in degenerate:
            print(f"  {t[:60]}: {','.join(f)}")
    else:
        print(f"GATE: PASS. Discovery produced non-degenerate categories on all {len(out)} contracts.")
    print(f"wrote {HERE/'results'/'cuad_categories.json'}")


if __name__ == "__main__":
    main()
