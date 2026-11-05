"""Part 1 verification: v7 (raw) vs v7_balanced (cap-only) vs v7_window (full window), on the 10
built screenplays. Confirms (a) the structural diff is grouping-only, (b) every non-_summary.md page
is byte-identical between v7 and v7_window (subject pages + the consolidated.md root digest + index.md
unchanged), (c) v7_window satisfies the leaf-group window with only documented root-exempt residuals.

Usage: python compare_part1.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
V7 = BENCH / "methods" / "wiki_v7" / "output" / "digests_v7"
BAL = BENCH / "methods" / "wiki_v7_balanced" / "output" / "digests_v7_balanced"
WIN = HERE / "output" / "digests_v7_window"
sys.path.insert(0, str(HERE))
import window as W  # noqa: E402
F, OCC_LO = W.F, W.OCC_LO


def stats(manifest):
    by, ch = W._children_map(manifest)
    lg = W.leaf_group_stats(manifest)
    fo, fo_at = None, None
    from collections import Counter
    c = Counter(e.get("parent") for e in manifest if e.get("parent") is not None)
    fo = max(c.values()) if c else 0
    internal = sum(1 for p in ch if p is not None and by.get(p, {}).get("level") != "leaf" or (p in by))
    return {"nodes": len(manifest),
            "internal_nodes": sum(1 for e in manifest if e["level"] in ("branch", "root")),
            "leaf_entries": sum(1 for e in manifest if e["level"] == "leaf"),
            "max_fanout": fo, **lg}


def fs_byte_identity(doc):
    """Every file under v7/doc that is NOT a _summary.md must be byte-identical under v7_window/doc.
    Returns (identical_non_summary, changed_non_summary_paths, summary_changed, summary_added, summary_removed)."""
    src, dst = V7 / doc, WIN / doc
    src_files = {p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file()}
    dst_files = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
    IGNORE = {"_manifest.json", "_template.json", "_balance_report.json", "_window_report.json"}
    def is_summary(rel):
        return rel.rsplit("/", 1)[-1] == "_summary.md"
    changed_non_summary, summ_changed = [], 0
    for rel in sorted(src_files & dst_files):
        if rel in IGNORE:
            continue
        same = (src / rel).read_bytes() == (dst / rel).read_bytes()
        if not same:
            if is_summary(rel):
                summ_changed += 1
            else:
                changed_non_summary.append(rel)
    summ_added = sum(1 for r in (dst_files - src_files) if is_summary(r) and r not in IGNORE)
    summ_removed = sum(1 for r in (src_files - dst_files) if is_summary(r) and r not in IGNORE)
    non_summary_added = [r for r in (dst_files - src_files) if not is_summary(r) and r not in IGNORE]
    non_summary_removed = [r for r in (src_files - dst_files) if not is_summary(r) and r not in IGNORE]
    return changed_non_summary, summ_changed, summ_added, summ_removed, non_summary_added, non_summary_removed


def main():
    docs = sorted(p.name for p in V7.iterdir() if p.is_dir())
    rows = []
    all_grouping_only = True
    for doc in docs:
        v7m = json.loads((V7 / doc / "_manifest.json").read_text())
        balm = json.loads((BAL / doc / "_manifest.json").read_text()) if (BAL / doc / "_manifest.json").exists() else v7m
        winm = json.loads((WIN / doc / "_manifest.json").read_text())
        s7, sb, sw = stats(v7m), stats(balm), stats(winm)
        chg, sc, sa, sr, nsa, nsr = fs_byte_identity(doc)
        grouping_only = (not chg) and (not nsa) and (not nsr)
        all_grouping_only &= grouping_only
        rows.append({"doc": doc, "v7": s7, "v7_balanced": sb, "v7_window": sw,
                     "byte_identity": {"non_summary_pages_changed": chg,
                                       "summary_pages_changed": sc, "summary_pages_added": sa,
                                       "summary_pages_removed": sr,
                                       "non_summary_added": nsa, "non_summary_removed": nsr,
                                       "grouping_only": grouping_only}})

    (HERE / "results" / "part1_compare.json").parent.mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "part1_compare.json").write_text(json.dumps(rows, indent=2))

    print(f"PART 1 — v7 vs v7_balanced vs v7_window (f={F}, window [{OCC_LO},{F}])")
    print("=" * 108)
    h = f"{'doc':26s} | {'nodes (v7/bal/win)':22s} | {'maxfan (v7/bal/win)':20s} | {'leaf-grp size (win)':18s} | below | grp-only"
    print(h); print("-" * 108)
    for r in rows:
        n = f"{r['v7']['nodes']}/{r['v7_balanced']['nodes']}/{r['v7_window']['nodes']}"
        mf = f"{r['v7']['max_fanout']}/{r['v7_balanced']['max_fanout']}/{r['v7_window']['max_fanout']}"
        lgs = f"[{r['v7_window']['leaf_group_size_min']},{r['v7_window']['leaf_group_size_max']}]"
        bi = r["byte_identity"]
        print(f"{r['doc'][:26]:26s} | {n:22s} | {mf:20s} | {lgs:18s} | "
              f"{r['v7_window']['leaf_groups_below_floor_nonexempt']:5d} | "
              f"{'YES' if bi['grouping_only'] else 'NO '}  (Δ_summary={bi['summary_pages_changed']}+{bi['summary_pages_added']}-{bi['summary_pages_removed']})")
    print("-" * 108)
    tot_below = sum(r["v7_window"]["leaf_groups_below_floor_nonexempt"] for r in rows)
    tot_exempt = sum(r["v7_window"]["exempt_small_categories"] for r in rows)
    win_maxfan = max(r["v7_window"]["max_fanout"] for r in rows)
    v7_maxfan = max(r["v7"]["max_fanout"] for r in rows)
    print(f"max fanout: v7={v7_maxfan}  ->  v7_window={win_maxfan} (<= f={F}: {win_maxfan <= F})")
    print(f"leaf-group floor: {tot_below} residual below-floor leaf-groups corpus-wide "
          f"(root-exempt structural isolates in large categories); {tot_exempt} exempt small categories")
    print(f"byte-identity: every non-_summary page (subjects + consolidated.md root digest + index.md) "
          f"identical in ALL docs: {all_grouping_only}")
    print(f"single-category rate (leaf-group, v7_window): "
          f"{round(sum(r['v7_window']['single_category_rate_leaf_group'] for r in rows)/len(rows),3)}")


if __name__ == "__main__":
    main()
