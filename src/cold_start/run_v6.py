"""
run_v6.py — CLI for Wiki v6 cold-start pipeline.

Usage:
  python llm-wiki/cold_start/run_v6.py bootstrap     # Phase 1: discover template
  python llm-wiki/cold_start/run_v6.py compile        # Phase 2: compile 10 movies
  python llm-wiki/cold_start/run_v6.py evaluate       # Phase 3: 295 Qs with refinement
  python llm-wiki/cold_start/run_v6.py all            # All phases
  python llm-wiki/cold_start/run_v6.py sample         # Bootstrap + compile 1 movie for inspection

Outputs: llm-wiki/cold_start/output/
Comparison baseline: v5.1 (run_wiki_qa_v5_1.py)
"""

import json
import os
import re
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cold_start import (
    bootstrap_template, compile_corpus, compile_document_2pass,
    write_v3_wiki, clean_document, get_llm_log, llm_cost_summary,
    query_with_refinement, _load_v3_digest, llm, parse_json,
    safe_filename, log,
)

BASE = Path("/Users/phdonn/narrativeqa")
CORPUS = BASE / "tmp"
OUTPUT = Path(__file__).parent / "output"
DIGESTS = OUTPUT / "digests_v6"

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

# Map document_title → doc_id for wiki lookup
TITLE_TO_DOCID = dict(zip(MOVIE_TITLES, MOVIES_10))

# Same QA + judge prompts as v3/v5.1 for fair comparison
QA_PROMPT = """\
You are answering a question about a movie/story using ONLY the compiled wiki digest below.
Answer in 1-15 words. If the answer is not in the digest, return exactly: not found

Question: {question}

Wiki Digest:
{digest}

Answer:"""

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
    """Find the v6 wiki directory for a movie title."""
    doc_id = TITLE_TO_DOCID.get(title)
    if doc_id:
        d = DIGESTS / safe_filename(doc_id)
        if d.exists():
            return d
    # Fallback: fuzzy match
    norm = normalize_title(title)
    for d in DIGESTS.iterdir():
        if d.is_dir() and normalize_title(d.name) == norm:
            return d
    return None


def find_raw_doc(title):
    """Find the raw source document for fallback."""
    doc_id = TITLE_TO_DOCID.get(title)
    if doc_id:
        for ext in (".txt", ".json"):
            p = CORPUS / (doc_id + ext)
            if p.exists():
                return p
    return None


# ─── Save helpers ────────────────────────────────────────────────────────────

def save_outputs(outlines=None, template=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if outlines:
        (OUTPUT / "candidate_outlines.json").write_text(
            json.dumps(outlines, indent=2, default=str))
    if template:
        (OUTPUT / "unified_template.json").write_text(
            json.dumps(template, indent=2))


def save_logs():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "llm_call_log.json").write_text(json.dumps(get_llm_log(), indent=2))
    (OUTPUT / "cost_summary.json").write_text(json.dumps(llm_cost_summary(), indent=2))


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_bootstrap():
    """Phase 1: sample, propose, reconcile."""
    log.info("=" * 60)
    log.info("WIKI v6 — PHASE 1: BOOTSTRAP")
    log.info("=" * 60)

    template, outlines, sampled_ids = bootstrap_template(
        CORPUS, n_samples=5, seed=42, threshold_pct=40
    )

    print("\n" + "=" * 60)
    print("CANDIDATE OUTLINES")
    print("=" * 60)
    for i, outline in enumerate(outlines):
        doc_id = outline.get("_doc_id", f"doc_{i}")
        doc_type = outline.get("document_type", "unknown")
        sections = outline.get("sections", [])
        print(f"\n--- Outline {i+1}: {doc_id} (type: {doc_type}) ---")
        for s in sections:
            subs = [sub["name"] for sub in s.get("subsections", [])]
            print(f"  {s['name']}: {subs}")

    if template:
        print("\n" + "=" * 60)
        print("UNIFIED TEMPLATE")
        print("=" * 60)
        print(f"Corpus type: {template.get('corpus_type', 'unknown')}")
        for s in template.get("sections", []):
            print(f"\n  {s['name']}: {s.get('description', '')}")
            for sub in s.get("subsections", []):
                print(f"    - {sub['name']}: {sub.get('description', '')}")

    save_outputs(outlines=outlines, template=template)
    save_logs()
    _print_costs()
    return template


def cmd_compile(template=None, doc_ids=None):
    """Phase 2: 2-pass compile for 10 movies."""
    if template is None:
        tpath = OUTPUT / "unified_template.json"
        if not tpath.exists():
            log.error("No template found. Run 'bootstrap' first.")
            return
        template = json.loads(tpath.read_text())

    if doc_ids is None:
        doc_ids = MOVIES_10

    log.info("=" * 60)
    log.info(f"WIKI v6 — PHASE 2: COMPILE ({len(doc_ids)} documents, 2-pass)")
    log.info("=" * 60)

    results = compile_corpus(CORPUS, template, DIGESTS, doc_ids=doc_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "compile_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    save_logs()

    print("\n" + "=" * 60)
    print("COMPILATION RESULTS")
    print("=" * 60)
    for doc_id, stats in sorted(results.items()):
        if stats:
            print(f"  {doc_id:40s}  pages={stats['pages']}  chars={stats['characters']}  "
                  f"events={stats['events']}  facts={stats['facts']}")
        else:
            print(f"  {doc_id:40s}  FAILED")

    _print_costs()


def cmd_evaluate():
    """Phase 3: run 295 questions with wiki + fallback + backfill.
    Uses same QA/judge prompts as v3/v5.1 for fair comparison.
    """
    log.info("=" * 60)
    log.info("WIKI v6 — PHASE 3: EVALUATE (295 Qs, with refinement)")
    log.info("=" * 60)

    df = pd.read_csv(BASE / "qaps_with_titles.csv")
    qa_df = df[df["document_title"].isin(MOVIE_TITLES)].groupby("document_title").head(30)
    N = len(qa_df)
    log.info(f"Questions: {N}")

    # Verify wikis exist
    for title in MOVIE_TITLES:
        d = find_digest_dir(title)
        if d:
            log.info(f"  Found v6 wiki: {title} -> {d.name}")
        else:
            log.warning(f"  NO v6 wiki for: {title}")

    results = []
    correct_total = 0
    backfill_totals = {"facts_added": 0, "entities_added": 0, "relationships_added": 0}

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
                "predicted": "not found", "wiki_source": "none",
                "source": "miss", "judge_verdict": "incorrect",
                "judge_score": 0, "f1": 0.0,
                "hallucination": 0, "total_miss": 1,
            })
            continue

        # Phase 3: wiki answer → fallback → backfill
        predicted, source, bf_stats = query_with_refinement(
            question, wiki_dir, raw_path
        )
        for k in backfill_totals:
            backfill_totals[k] += bf_stats.get(k, 0)

        # Judge
        is_miss = predicted.lower() in ("not found", "", "none", "n/a")
        if is_miss:
            verdict, score = "incorrect", 0
        else:
            jprompt = JUDGE_PROMPT.format(
                question=question, answer1=answer1, answer2=answer2, predicted=predicted,
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
            "predicted": predicted, "wiki_source": "v6",
            "source": source,
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

    # ── Save results ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT / "wiki_qa_v6_results.csv", index=False)

    n = len(results_df)
    judge_acc = results_df["judge_score"].mean()
    avg_f1 = results_df["f1"].mean()
    hall_rate = results_df["hallucination"].mean()
    miss_rate = results_df["total_miss"].mean()

    # Source breakdown
    source_counts = results_df["source"].value_counts().to_dict()

    per_movie = {}
    for m, sub in results_df.groupby("movie_id"):
        per_movie[m] = {
            "n": int(len(sub)),
            "judge_accuracy": float(sub["judge_score"].mean()),
            "avg_f1": float(sub["f1"].mean()),
            "hallucination_rate": float(sub["hallucination"].mean()),
            "total_miss_rate": float(sub["total_miss"].mean()),
            "wiki_answers": int((sub["source"] == "wiki").sum()),
            "fallback_answers": int((sub["source"] == "fallback").sum()),
            "miss_answers": int((sub["source"] == "miss").sum()),
        }

    summary = {
        "method": "LLM Wiki QA v6 (cold-start 3-phase)",
        "wiki_version": "v6",
        "run_date": datetime.now().isoformat(),
        "total_questions": n,
        "judge_accuracy": round(judge_acc, 4),
        "avg_f1": round(avg_f1, 4),
        "hallucination_rate": round(hall_rate, 4),
        "total_miss_rate": round(miss_rate, 4),
        "source_breakdown": source_counts,
        "backfill_totals": backfill_totals,
        "per_movie": per_movie,
    }
    (OUTPUT / "wiki_qa_v6_summary.json").write_text(json.dumps(summary, indent=2))
    save_logs()

    print(f"\n{'='*60}")
    print("WIKI v6 RESULTS — 10 Movies (cold-start 3-phase)")
    print(f"{'='*60}")
    print(f"  Questions:     {n}")
    print(f"  Judge Acc:     {judge_acc:.1%}")
    print(f"  Token F1:      {avg_f1:.4f}")
    print(f"  Hallucination: {hall_rate:.1%}")
    print(f"  Total Miss:    {miss_rate:.1%}")
    print(f"  Source: wiki={source_counts.get('wiki',0)}  "
          f"fallback={source_counts.get('fallback',0)}  "
          f"miss={source_counts.get('miss',0)}")
    print(f"  Backfill: {backfill_totals}")
    print(f"\n  Per movie:")
    for m in MOVIE_TITLES:
        if m in per_movie:
            p = per_movie[m]
            print(f"    {m:35s}  judge={p['judge_accuracy']:.0%}  f1={p['avg_f1']:.3f}  "
                  f"hall={p['hallucination_rate']:.0%}  miss={p['total_miss_rate']:.0%}  "
                  f"wiki={p['wiki_answers']}  fb={p['fallback_answers']}")
    print(f"{'='*60}")
    _print_costs()


def cmd_sample():
    """Bootstrap + compile 1 movie for inspection."""
    template = cmd_bootstrap()
    if not template:
        return
    cmd_compile(template=template, doc_ids=[MOVIES_10[0]])

    sample_dir = DIGESTS / safe_filename(MOVIES_10[0])
    if sample_dir.exists():
        print("\n" + "=" * 60)
        print(f"SAMPLE WIKI: {MOVIES_10[0]}")
        print("=" * 60)
        for f in sorted(sample_dir.rglob("*.md")):
            rel = f.relative_to(sample_dir)
            content = f.read_text()
            print(f"\n--- {rel} ({len(content)} chars) ---")
            print(content[:500])
            if len(content) > 500:
                print(f"  ... ({len(content) - 500} more chars)")


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_v6.py [bootstrap|compile|evaluate|all|sample]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "compile":
        cmd_compile()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "all":
        template = cmd_bootstrap()
        if template:
            cmd_compile(template=template)
            cmd_evaluate()
    elif cmd == "sample":
        cmd_sample()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
