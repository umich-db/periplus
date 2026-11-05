"""Compare methods on the question set they all actually answered.

The corpus grew from 10 screenplays (295 questions) to 30 (885) mid-project, and
not every method was re-run on the larger set. Comparing a method scored on 885
questions against one scored on 295 is meaningless, so this restricts every
method to the intersection of question ids and rescores from the cached judge
verdicts. No API calls; reads only committed files.

    envs/base/bin/python analysis/compare_on_shared_qids.py
    envs/base/bin/python analysis/compare_on_shared_qids.py --qids data/questions/questions_30.jsonl
"""
import argparse, json, pathlib, sys

BENCH = pathlib.Path(__file__).resolve().parents[1]


def load(method, qids):
    """(n, accuracy, answer_tokens) for `method` restricted to `qids`, or None."""
    verdicts = BENCH / "results" / "eval" / f"{method}_verdicts.jsonl"
    results = BENCH / "methods" / method / "results" / f"{method}.jsonl"
    if not (verdicts.exists() and results.exists()):
        return None
    v = {}
    for line in verdicts.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            v[d["qid"]] = d["verdict"]
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    sub = [r for r in rows if r["qid"] in qids and r["qid"] in v]
    if not sub:
        return None
    correct = sum(1 for r in sub if v[r["qid"]] == "correct")
    tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in sub)
    return len(sub), correct / len(sub), tokens


def build_tokens(method):
    s = BENCH / "results" / "eval" / f"{method}_summary.json"
    return json.load(open(s)).get("build_tokens", 0) if s.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="data/questions/questions.jsonl",
                    help="question file defining the comparison set (default: the original 295)")
    ap.add_argument("methods", nargs="*",
                    default=["naive_gpt", "wiki_v7", "wiki_v6", "wiki_v8", "leann", "doctopus",
                             "graphrag_local", "graphrag_global", "rag", "raptor", "shed", "moar",
                             "docetl_standard"])
    a = ap.parse_args()

    qf = BENCH / a.qids
    qids = {json.loads(l)["qid"] for l in qf.read_text().splitlines() if l.strip()}
    print(f"comparison set: {len(qids)} questions from {a.qids}\n")

    out = []
    for m in a.methods:
        got = load(m, qids)
        if got is None:
            print(f"  (skipped {m}: no verdicts or no overlap)", file=sys.stderr)
            continue
        n, acc, tok = got
        out.append((m, n, acc, tok, tok / n, build_tokens(m)))

    out.sort(key=lambda r: -r[2])
    print(f"{'method':18s} {'n':>5} {'accuracy':>9} {'answer tok':>13} {'per q':>9} {'build tok':>12}")
    for m, n, acc, tok, perq, bt in out:
        if tok == 0:
            # DocETL batches its map operation, so tokens are only attributable at
            # the pipeline level -- see results/eval/<m>_summary.json for the total.
            note, cells = "  <- tokens not per-question", f"{'n/a':>13} {'n/a':>9}"
        else:
            note, cells = "", f"{tok:13,} {perq:9,.0f}"
        if n != len(qids):
            note += f"  <- only {n} of {len(qids)}"
        print(f"{m:18s} {n:5d} {acc:9.1%} {cells} {bt:12,}{note}")


if __name__ == "__main__":
    main()
