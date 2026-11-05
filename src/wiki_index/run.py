"""
run.py — CLI for Wiki v7 traversal pipeline.

No bootstrap phase — template is hardcoded in pipeline.py.

Usage:
  python llm-wiki/traversal/run.py compile      # Compile + fan-out limits
  python llm-wiki/traversal/run.py evaluate     # Link-traversal QA
  python llm-wiki/traversal/run.py all          # Compile then evaluate
"""

import json
import re
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cold_start"))

from pipeline import (
    compile_all, query_with_traversal,
    get_llm_log, llm_cost_summary, safe_filename, log,
)
from cold_start import llm

BASE = Path("/Users/phdonn/narrativeqa")
CORPUS = BASE / "tmp"
OUTPUT = Path(__file__).parent / "output"
DIGESTS = OUTPUT / "digests_v7"

MOVIES_10 = [
    "10_Things_I_Hate_About_You",
    "12_and_Holding",
    "1492:_Conquest_of_Paradise",
    "17_Again_(film)",
    "2012_(film)",
    "50-50_(2011_film)",
    "8mm_(film)",
    "9_(2005_film)",
    "A_Dry_White_Season",
    "A_Few_Good_Men",
]

MOVIE_TITLES = [
    "10 Things I Hate About You",
    "12 and Holding",
    "1492: Conquest of Paradise",
    "17 Again (film)",
    "2012 (film)",
    "50/50 (2011 film)",
    "8mm (film)",
    "9 (2005 film)",
    "A Dry White Season",
    "A Few Good Men",
]

TITLE_TO_DOCID = dict(zip(MOVIE_TITLES, MOVIES_10))

JUDGE_PROMPT = """\
You are an answer equivalence judge. Given a question, the ground-truth answer(s), \
and a predicted answer, determine if the prediction is essentially correct.

Two answers are equivalent if they convey the same core fact, even if worded differently. \
For example: "An obstetrician" and "He is an OB/GYN doctor" are equivalent. \
"Stay close to home" and "Attend the University of Oregon" are NOT equivalent.

Question: {question}
Ground truth answer 1: {answer1}
Ground truth answer 2: {answer2}
Predicted answer: {predicted}

Is the predicted answer correct? Reply with ONLY "correct" or "incorrect"."""


def token_f1(pred, gt):
    p, g = pred.lower().split(), gt.lower().split()
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def normalize_title(title):
    return re.sub(r"[^a-z0-9]", "", title.lower())


def find_digest_dir(title):
    doc_id = TITLE_TO_DOCID.get(title)
    if doc_id:
        d = DIGESTS / safe_filename(doc_id)
        if d.exists():
            return d
    norm = normalize_title(title)
    for d in DIGESTS.iterdir():
        if d.is_dir() and normalize_title(d.name) == norm:
            return d
    return None


def find_raw_doc(title):
    doc_id = TITLE_TO_DOCID.get(title)
    if doc_id:
        for ext in (".txt", ".json"):
            p = CORPUS / (doc_id + ext)
            if p.exists():
                return p
    return None


def save_logs():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "llm_call_log.json").write_text(json.dumps(get_llm_log(), indent=2))
    (OUTPUT / "cost_summary.json").write_text(json.dumps(llm_cost_summary(), indent=2))


def _print_costs():
    costs = llm_cost_summary()
    if not costs:
        return
    print("\n  LLM Cost:")
    total_in = sum(v["input_tokens"] for v in costs.values())
    total_out = sum(v["output_tokens"] for v in costs.values())
    for phase, stats in sorted(costs.items()):
        print(f"    {phase:20s}  calls={stats['calls']}  "
              f"in={stats['input_tokens']:,}  out={stats['output_tokens']:,}")
    print(f"    {'TOTAL':20s}  in={total_in:,}  out={total_out:,}")


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_compile(doc_ids=None):
    if doc_ids is None:
        doc_ids = MOVIES_10

    log.info("=" * 60)
    log.info(f"WIKI v7 — COMPILE ({len(doc_ids)} docs, fixed template)")
    log.info("=" * 60)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = compile_all(CORPUS, DIGESTS, doc_ids=doc_ids)

    (OUTPUT / "compile_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    save_logs()

    print("\n" + "=" * 60)
    for doc_id, stats in sorted(results.items()):
        if stats:
            print(f"  {doc_id:40s}  pages={stats['pages']}  depth={stats['max_depth']}  "
                  f"sections={stats['sections']}")
        else:
            print(f"  {doc_id:40s}  FAILED")
    _print_costs()


def cmd_evaluate():
    log.info("=" * 60)
    log.info("WIKI v7 — EVALUATE (link traversal)")
    log.info("=" * 60)

    df = pd.read_csv(BASE / "qaps_with_titles.csv")
    qa_df = df[df["document_title"].isin(MOVIE_TITLES)].groupby("document_title").head(30)
    N = len(qa_df)
    log.info(f"Questions: {N}")

    for title in MOVIE_TITLES:
        d = find_digest_dir(title)
        if d:
            log.info(f"  Found v7 wiki: {title} -> {d.name}")
        else:
            log.warning(f"  NO v7 wiki for: {title}")

    results = []
    correct_total = 0

    for idx, (_, row) in enumerate(qa_df.iterrows()):
        title = row["document_title"]
        question = str(row["question"])
        answer1 = str(row["answer1"])
        answer2 = str(row["answer2"])

        wiki_dir = find_digest_dir(title)
        raw_path = find_raw_doc(title)

        if not wiki_dir:
            results.append({
                "movie_id": title, "question": question,
                "answer1": answer1, "answer2": answer2,
                "predicted": "not found", "source": "miss",
                "judge_verdict": "incorrect", "judge_score": 0,
                "f1": 0.0, "hallucination": 0, "total_miss": 1,
            })
            continue

        predicted, source, _ = query_with_traversal(question, wiki_dir, raw_path)

        is_miss = predicted.lower() in ("not found", "", "none", "n/a")
        if is_miss:
            verdict, score = "incorrect", 0
        else:
            jprompt = JUDGE_PROMPT.format(
                question=question, answer1=answer1,
                answer2=answer2, predicted=predicted,
            )
            raw_v = llm(jprompt, max_tokens=2048, phase="judge")
            v = (raw_v or "").strip().lower()
            is_correct = "correct" in v and "incorrect" not in v
            verdict = "correct" if is_correct else "incorrect"
            score = 1 if is_correct else 0

        f1 = max(token_f1(predicted, answer1), token_f1(predicted, answer2))
        is_halluc = (f1 == 0) and not is_miss
        correct_total += score
        total_done = len(results) + 1
        acc = correct_total / total_done

        results.append({
            "movie_id": title, "question": question,
            "answer1": answer1, "answer2": answer2,
            "predicted": predicted, "source": source,
            "judge_verdict": verdict, "judge_score": score,
            "f1": f1, "hallucination": int(is_halluc), "total_miss": int(is_miss),
        })

        log.info(
            f"[{total_done}/{N}] {title}\n"
            f"    Q: {question}\n"
            f"    GT1: {answer1[:80]}\n"
            f"    PRED: {predicted}  (source={source})\n"
            f"    Judge: {verdict} | F1: {f1:.3f} | Running Acc: {acc:.1%}"
        )

    # Save
    results_df = pd.DataFrame(results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT / "wiki_qa_v7_results.csv", index=False)

    n = len(results_df)
    judge_acc = results_df["judge_score"].mean()
    avg_f1 = results_df["f1"].mean()
    hall_rate = results_df["hallucination"].mean()
    miss_rate = results_df["total_miss"].mean()
    source_counts = results_df["source"].value_counts().to_dict()

    per_movie = {}
    for m, sub in results_df.groupby("movie_id"):
        per_movie[m] = {
            "n": int(len(sub)),
            "judge_accuracy": float(sub["judge_score"].mean()),
            "avg_f1": float(sub["f1"].mean()),
            "hallucination_rate": float(sub["hallucination"].mean()),
            "wiki_answers": int((sub["source"] == "wiki").sum()),
            "fallback_answers": int((sub["source"] == "fallback").sum()),
        }

    summary = {
        "method": "Wiki v7 (fixed template + traversal)",
        "run_date": datetime.now().isoformat(),
        "total_questions": n,
        "judge_accuracy": round(judge_acc, 4),
        "avg_f1": round(avg_f1, 4),
        "hallucination_rate": round(hall_rate, 4),
        "total_miss_rate": round(miss_rate, 4),
        "source_breakdown": source_counts,
        "per_movie": per_movie,
    }
    (OUTPUT / "wiki_qa_v7_summary.json").write_text(json.dumps(summary, indent=2))
    save_logs()

    print(f"\n{'='*60}")
    print("WIKI v7 RESULTS (fixed template + traversal)")
    print(f"{'='*60}")
    print(f"  Questions:     {n}")
    print(f"  Judge Acc:     {judge_acc:.1%}")
    print(f"  Token F1:      {avg_f1:.4f}")
    print(f"  Hallucination: {hall_rate:.1%}")
    print(f"  Total Miss:    {miss_rate:.1%}")
    print(f"  Source: wiki={source_counts.get('wiki',0)}  "
          f"fallback={source_counts.get('fallback',0)}  "
          f"miss={source_counts.get('miss',0)}")
    print(f"\n  Per movie:")
    for m in MOVIE_TITLES:
        if m in per_movie:
            p = per_movie[m]
            print(f"    {m:35s}  judge={p['judge_accuracy']:.0%}  f1={p['avg_f1']:.3f}  "
                  f"hall={p['hallucination_rate']:.0%}  "
                  f"wiki={p['wiki_answers']}  fb={p['fallback_answers']}")
    _print_costs()


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [compile|evaluate|all]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "compile":
        cmd_compile()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "all":
        cmd_compile()
        cmd_evaluate()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
