"""Select 20 NEW NarrativeQA screenplays to add to the existing 10.

Rules (from the prompt): screenplays (documents.csv kind == "movie") not among the current 10,
with at least 25 questions in qaps.csv, ranked by document length (story_word_count, NarrativeQA's
own per-doc count, available for all docs with no fetch), sampled EVENLY ACROSS THE LENGTH RANGE
(2 per decile, not biased long). Prints titles, lengths, question counts, set. Writes the picked
set to results/nqa_selection.json for the build step (text is fetched later from HF by document_id).

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])
_NQA_META = _pl.Path(_os.environ.get("NQA_META_DIR") or _BENCH.parent)

import re, json, pathlib
import pandas as pd

ROOT = _NQA_META
HERE = pathlib.Path(__file__).resolve().parent

# the 10 already in the bench (document_id, from wiki_index/run.py + scrape/extract mappings)
EXISTING_IDS = {
    "78e0e28b686c7157d598082bfaa8aaaab821b78b", "bbea33d8cce51af36cf7aab15b83d1372f4523aa",
    "c41dba67fbd19688656d31ccffec145c12f15b94", "16a6a06abf4ff93fe3143b66c45dbeae6550816b",
    "6a14d1ba1aa1a6719840cb98a8feb6a711291aa4", "9292406d5193d3402195df7e5647fd168da2d15a",
    "6ed5d860df4edd3d32f1aca52ec369f0d21039a0", "2fe5752889751cbb4bbeab3deba4f6e315f90f0a",
    "08ac6c0dca91fee93d27c7caf85bb57e17d4fc46", "dd40f45256efa1ec710101bcb925c02e634532e2",
}
MIN_QUESTIONS = 25
N_SELECT = 20


def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def main():
    docs = pd.read_csv(ROOT / "documents.csv")
    qaps = pd.read_csv(ROOT / "qaps.csv")
    qcount = qaps.groupby("document_id").size()                       # rows per doc in qaps.csv

    cand = docs[(docs["kind"] == "movie") & (~docs["document_id"].isin(EXISTING_IDS))].copy()
    cand["n_questions"] = cand["document_id"].map(qcount).fillna(0).astype(int)
    cand = cand[cand["n_questions"] >= MIN_QUESTIONS].copy()
    cand["story_word_count"] = pd.to_numeric(cand["story_word_count"], errors="coerce")
    cand = cand[cand["story_word_count"] > 0].dropna(subset=["story_word_count"])
    cand = cand.sort_values(["story_word_count", "document_id"]).reset_index(drop=True)

    n = len(cand)
    # even sampling across the whole sorted-by-length range: 20 evenly-spaced picks = 2 per decile,
    # spanning the range so the set is not biased long. Deterministic (no RNG).
    idx = sorted({min(n - 1, round((i + 0.5) * n / N_SELECT)) for i in range(N_SELECT)})
    while len(idx) < N_SELECT:                                        # fill any collisions
        for j in range(n):
            if j not in idx:
                idx.append(j); idx = sorted(set(idx))
                if len(idx) == N_SELECT:
                    break
    picked = cand.iloc[idx].copy()

    rows = []
    for _, r in picked.iterrows():
        title = str(r["wiki_title"])
        rows.append({
            "document_id": r["document_id"], "wiki_title": title,
            "wiki_key": title.replace(" ", "_"), "slug": slugify(title),
            "set": r["set"], "story_word_count": int(r["story_word_count"]),
            "n_questions_raw": int(r["n_questions"]), "n_questions_capped": min(30, int(r["n_questions"])),
        })
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "nqa_selection.json").write_text(json.dumps(rows, indent=2))

    print(f"NarrativeQA candidate pool: {n} screenplays (kind=movie, not in current 10, "
          f">= {MIN_QUESTIONS} questions, word_count>0)")
    print(f"word_count range across pool: {int(cand['story_word_count'].min()):,} .. "
          f"{int(cand['story_word_count'].max()):,}\n")
    print(f"SELECTED 20 (evenly spaced across the length range, ~2 per decile):")
    print(f"{'#':>2}  {'words':>7}  {'~tok':>7}  {'q':>3}  {'set':>5}  title")
    print("-" * 78)
    tr = te = va = 0
    for i, r in enumerate(rows, 1):
        approx_tok = int(r["story_word_count"] * 1.4)
        print(f"{i:2d}  {r['story_word_count']:7,}  {approx_tok:7,}  {r['n_questions_capped']:3d}  "
              f"{r['set']:>5}  {r['wiki_title'][:44]}")
        tr += r["set"] == "train"; te += r["set"] == "test"; va += r["set"] == "valid"
    tot_q = sum(r["n_questions_capped"] for r in rows)
    print("-" * 78)
    print(f"new questions (capped at 30/movie): {tot_q}  |  set split: train={tr} test={te} valid={va}")
    print(f"total question set after adding: 295 (old) + {tot_q} (new) = {295 + tot_q}")
    print(f"\nwrote {HERE/'results'/'nqa_selection.json'}")


if __name__ == "__main__":
    main()
