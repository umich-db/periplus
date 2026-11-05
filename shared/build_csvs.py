"""Build qaps.csv and qaps_with_titles.csv under data/narrativeqa/ from the
deepmind/narrativeqa repo (data/nqa_repo/). Drops tokenized variants, keeps
`set`, joins wiki_title as document_title. Does NOT build the filtered file
(filter rule unknown -> ask user)."""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import csv, pathlib

REPO = _BENCH / "data" / "nqa_repo"
OUT = _BENCH / "data" / "narrativeqa"
OUT.mkdir(parents=True, exist_ok=True)

# --- load documents.csv: document_id -> wiki_title ---
id2title = {}
with open(REPO / "documents.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        id2title[row["document_id"]] = (row.get("wiki_title") or "").strip()

# --- read qaps.csv, drop tokenized cols, keep set ---
qaps_cols = ["document_id", "question", "answer1", "answer2", "set"]
qt_cols = ["document_id", "document_title", "question", "answer1", "answer2", "set"]

n = 0
missing_title = 0
with open(REPO / "qaps.csv", newline="", encoding="utf-8") as f, \
     open(OUT / "qaps.csv", "w", newline="", encoding="utf-8") as q_out, \
     open(OUT / "qaps_with_titles.csv", "w", newline="", encoding="utf-8") as qt_out:
    reader = csv.DictReader(f)
    qw = csv.DictWriter(q_out, fieldnames=qaps_cols); qw.writeheader()
    qtw = csv.DictWriter(qt_out, fieldnames=qt_cols); qtw.writeheader()
    for row in reader:
        did = row["document_id"]
        base = {
            "document_id": did,
            "question": row["question"],
            "answer1": row["answer1"],
            "answer2": row["answer2"],
            "set": row.get("set", ""),
        }
        qw.writerow(base)
        title = id2title.get(did, "")
        if not title:
            missing_title += 1
        qtw.writerow({**base, "document_title": title})
        n += 1

print(f"qaps.csv rows: {n}")
print(f"qaps_with_titles.csv rows: {n}")
print(f"rows with EMPTY title (unmatched): {missing_title}")
