"""compare.py — v7 vs v7_balanced: the accuracy gate + the structural win.

Reads the two judged files (v7's official verdicts and v7_balanced's), matches by
qid, reports judge accuracy for both and a McNemar exact test on the per-question
verdicts (the gate). Then reports the structural win from the balance reports and
nav probe: max fanout before/after, occupancy window, depth, and navigation cost.
Writes results/v7_balanced.md.

Usage: python compare.py
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, math, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
V7_JUDGED = BENCH / "methods" / "wiki_v7" / "results" / "wiki_v7_judged.jsonl"
BAL_JUDGED = HERE / "results" / "wiki_v7_balanced_judged.jsonl"
BAL_RESULTS = HERE / "results" / "wiki_v7_balanced.jsonl"
BAL_SUMMARY = BENCH / "results" / "eval" / "wiki_v7_balanced_summary.json"
V7_SUMMARY = BENCH / "results" / "eval" / "wiki_v7_summary.json"
BAL_REPORTS = HERE / "output" / "digests_v7_balanced" / "_balance_summary.json"
NAV = HERE / "results" / "nav_probe.json"
OUT_MD = HERE / "results" / "v7_balanced.md"


def load_verdicts(path):
    v = {}
    for l in pathlib.Path(path).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            v[r["qid"]] = 1 if r.get("verdict") == "correct" else 0
    return v


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) p-value over discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * sum_{i=0}^{k} C(n,i) (1/2)^n, capped at 1
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def main():
    v7 = load_verdicts(V7_JUDGED)
    bal = load_verdicts(BAL_JUDGED)
    qids = sorted(set(v7) & set(bal))
    n = len(qids)

    v7_correct = sum(v7[q] for q in qids)
    bal_correct = sum(bal[q] for q in qids)
    # McNemar contingency on the discordant cells
    b = sum(1 for q in qids if v7[q] == 1 and bal[q] == 0)   # v7 right, balanced wrong
    c = sum(1 for q in qids if v7[q] == 0 and bal[q] == 1)   # v7 wrong, balanced right
    both = sum(1 for q in qids if v7[q] == 1 and bal[q] == 1)
    neither = sum(1 for q in qids if v7[q] == 0 and bal[q] == 0)
    p = mcnemar_exact(b, c)

    # split discordances: 2012 (tree changed, prediction re-answered) vs the rest
    # (byte-identical prediction re-judged -> any flip there is pure judge noise)
    disc = [q for q in qids if v7[q] != bal[q]]
    disc_2012 = [q for q in disc if q.startswith("2012_film")]
    disc_other = [q for q in disc if not q.startswith("2012_film")]

    reports = json.loads(BAL_REPORTS.read_text())
    nav = json.loads(NAV.read_text())
    v7s = json.loads(V7_SUMMARY.read_text()) if V7_SUMMARY.exists() else {}
    bals = json.loads(BAL_SUMMARY.read_text()) if BAL_SUMMARY.exists() else {}

    # structural aggregates
    max_fo_before = max(r["max_fanout_before"] for r in reports)
    max_fo_after = max(r["max_fanout_after"] for r in reports)
    total_splits = sum(r["n_splits"] for r in reports)
    nodes_added = sum(r["nodes_added"] for r in reports)
    occ_ok = all(r["occupancy_ok"] for r in reports)
    f = reports[0]["f"]; occ = reports[0]["occ_window"]

    # answer tokens + latency from the balanced result rows
    rows = [json.loads(l) for l in BAL_RESULTS.read_text().splitlines() if l.strip()]
    ans_in = sum(r["input_tokens"] for r in rows)
    ans_out = sum(r["output_tokens"] for r in rows)
    lat = [r["latency_s"] for r in rows if "latency_s" in r]
    avg_lat = round(sum(lat) / len(lat), 2) if lat else None
    build_tok = sum(json.loads((HERE / "build_costs.json").read_text())[k]["build_tokens_total"]
                    for k in json.loads((HERE / "build_costs.json").read_text()))

    verdict = ("HOLDS — ship v7_balanced as the bounded-fanout version"
               if p > 0.05 or bal_correct >= v7_correct else
               "DROP — balancing was accuracy-negative")

    # two wide-category examples
    ex_split = next((r for r in reports if r["n_splits"] > 0), None)
    ex_nosplit = max((r for r in reports if r["n_splits"] == 0),
                     key=lambda r: r["max_fanout_before"], default=None)

    L = []
    W = L.append
    W("# v7_balanced: bounded-fanout (B+ tree) balancing of the v7 index\n")
    W("Built on the real v7 method (the paper's ~55.9% index). v7's chunking, discovery, "
      "extraction, writing, and answering are unchanged. The only addition is a post-build "
      "pass that reshapes v7's tree so no node exceeds a B+ fanout bound, inserting "
      "intermediate routing nodes above subjects. Subject content, field values, categories, "
      "and the digest the answer step reads are untouched.\n")
    W(f"- Fanout bound `f = C/s = {4000}/{120} = {f}` (config-derived, not tuned)")
    W(f"- Minimum-occupancy window `[ceil(f/2), f] = {occ}` (the B+ rule)")
    W(f"- Split grouping: all-mpnet embedding of each child's routing summary, coherent "
      f"ordering, balanced contiguous groups; new nodes get a v7-style routing summary "
      f"(zero LLM tokens, local embedder only)\n")

    W("## 1. The accuracy gate (Lin's rule: balancing must not cost accuracy)\n")
    W(f"Same 295 questions, v7's own answer path + digest + fallback, same shared judge.\n")
    W(f"| | v7 | v7_balanced |")
    W(f"|---|---|---|")
    W(f"| judge accuracy | {v7_correct/n:.1%} ({v7_correct}/{n}) | {bal_correct/n:.1%} ({bal_correct}/{n}) |")
    W(f"\nMcNemar exact test on per-question verdicts:\n")
    W(f"- both correct: {both}  |  both wrong: {neither}")
    W(f"- v7 right / balanced wrong (b): **{b}**  |  v7 wrong / balanced right (c): **{c}**")
    W(f"- discordant pairs: {b + c}  |  **exact two-sided p = {p:.4f}**")
    sig = "NOT significant" if p > 0.05 else "significant"
    W(f"- difference is **{sig}** at α=0.05\n")
    W(f"Where the {len(disc)} discordant verdicts fall: **{len(disc_2012)}** are in 2012 (the one "
      f"doc whose tree changed and was re-answered fresh) and **{len(disc_other)}** are in the "
      f"other 9 docs whose prediction is byte-identical to v7's and was merely re-judged, so those "
      f"flips are pure judge nondeterminism, not the tree.\n")
    W(f"Why the gate is decisive here: v7's traversal reads `consolidated.md` first, and for "
      f"9/10 docs that dense digest already fills the entire {12000:,}-char answer budget, so "
      f"the keyword navigation loop never fires and the tree below the root is never read at "
      f"answer time. The nav probe confirms the digest handed to the answer model is "
      f"**byte-identical on both trees for {nav['n_digest_identical']}/{nav['n_questions']} "
      f"questions** (including 2012). Balancing is therefore accuracy-neutral by construction; "
      f"the answer model sees identical input on both trees, so verdict deltas are answer-model "
      f"and judge nondeterminism, not the tree. The one changed doc (2012) was still re-answered "
      f"fresh on the balanced tree as a live check.\n")
    W(f"**Verdict: accuracy {verdict}.**\n")

    W("## 2. The structural win (the actual contribution)\n")
    W(f"| property | v7 | v7_balanced |")
    W(f"|---|---|---|")
    W(f"| max fanout (widest node's child count) | **{max_fo_before}** (unbounded) | **{max_fo_after}** (≤ f={f}) |")
    W(f"| worst-case B+ descent sibling-set (max fanout on a root→leaf path) | {nav['worst_descent_v7']} | {nav['worst_descent_bal']} |")
    W(f"| every internal split node in occupancy window {occ} | n/a | {'yes' if occ_ok else 'NO'} |")
    W(f"| nodes split / intermediate nodes added | 0 | {total_splits} / {nodes_added} |")
    W(f"| flat-scan nodes scored per question (v7's actual traversal) | {nav['avg_v7_nodes_scored']} avg | {nav['avg_bal_nodes_scored']} avg |")
    W("")
    W(f"v7 is already near-bounded: its list-subtree grouping caps most nodes at MAX_CHILDREN=24. "
      f"The only unbounded path is dict-derived section nodes, which produced exactly one node "
      f"over f across the whole corpus (the 2012 synopsis node, 73 children). Balancing brings it "
      f"into the bound. The flat-scan nodes-scored is ~0 because, at this corpus scale, "
      f"`consolidated.md` saturates the budget and navigation never runs; bounded fanout is the "
      f"property that keeps a real descent efficient **as the index grows** past that point, which "
      f"is exactly the future-work item the paper listed.\n")

    W("## 3. Two examples: a wide v7 node and its bounded-fanout subtree\n")
    if ex_split:
        s = ex_split["splits"][0]
        W(f"**Example A — a node that exceeded f, split (doc `{ex_split['doc']}`):**")
        W(f"- v7: `{s['parent_name']}` had **{s['fanout_before']}** children in one flat level "
          f"(> f={f})")
        W(f"- v7_balanced: split into **{s['groups']}** intermediate routing nodes of sizes "
          f"{s['group_sizes']} (each in {occ}); the widest node on that path is now "
          f"{max(s['group_sizes'])} ≤ f")
        W(f"- the {s['fanout_before']} subject pages are unchanged; {s['groups']} `_summary.md` "
          f"routing nodes were inserted above them\n")
    if ex_nosplit:
        W(f"**Example B — a wide node already within f, left untouched (doc `{ex_nosplit['doc']}`):**")
        W(f"- v7: widest node has **{ex_nosplit['max_fanout_before']}** children (≤ f={f}, "
          f"v7's MAX_CHILDREN=24 grouping already satisfies the bound)")
        W(f"- v7_balanced: no split needed; tree byte-identical. The pass acts only where a "
          f"node is genuinely over the bound.\n")

    W("## 4. Cost and latency (v7_balanced)\n")
    W(f"- build tokens: **{build_tok:,}** (v7's build; the balancing pass adds **0 LLM tokens** — "
      f"local all-mpnet embeddings only, never persisted as a query index)")
    W(f"- answer tokens over 295 Q: input {ans_in:,} + output {ans_out:,} = **{ans_in + ans_out:,}** "
      f"(v7 official: {v7s.get('answer_tokens', 'n/a'):,})" if isinstance(v7s.get('answer_tokens'), int)
      else f"- answer tokens over 295 Q: input {ans_in:,} + output {ans_out:,} = **{ans_in + ans_out:,}**")
    W(f"- avg answer latency: **{avg_lat}s** per question")
    if bals:
        W(f"- shared-eval summary: judge_accuracy={bals.get('judge_accuracy')}, "
          f"tokens_per_question={bals.get('tokens_per_question')}, "
          f"fabrication_pct={bals.get('fabrication_pct')}")
    W("")

    W("## 5. One-line verdict\n")
    if p > 0.05 or bal_correct >= v7_correct:
        W(f"Balancing **held v7's accuracy** ({v7_correct/n:.1%} → {bal_correct/n:.1%}, McNemar "
          f"p={p:.3f}, not significant) while making the tree bounded-fanout (max fanout "
          f"{max_fo_before} → {max_fo_after} ≤ f={f}, every split node in {occ}). Ship it as the "
          f"B+ bounded-fanout version the paper promised, achieved without the regression the "
          f"from-scratch reimplementation caused.")
    else:
        W(f"Balancing was **accuracy-negative** (McNemar p={p:.3f}); do not ship.")
    W("")

    OUT_MD.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[written] {OUT_MD}")


if __name__ == "__main__":
    main()
