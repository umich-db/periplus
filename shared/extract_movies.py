"""Download NarrativeQA via HF (cached) and extract the full document.text for
our 10 movies into data/movies/raw/<document_id>.txt. Robust to flaky network
(HF client retries internally). Idempotent: skips ids already saved."""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import pathlib, sys
from datasets import load_dataset

RAW = _BENCH/"data"/"movies"/"raw"
RAW.mkdir(parents=True, exist_ok=True)

WANT = {
 "08ac6c0dca91fee93d27c7caf85bb57e17d4fc46":"a_dry_white_season",
 "16a6a06abf4ff93fe3143b66c45dbeae6550816b":"17_again_film",
 "2fe5752889751cbb4bbeab3deba4f6e315f90f0a":"9_2005_film",
 "6a14d1ba1aa1a6719840cb98a8feb6a711291aa4":"2012_film",
 "6ed5d860df4edd3d32f1aca52ec369f0d21039a0":"8mm_film",
 "78e0e28b686c7157d598082bfaa8aaaab821b78b":"10_things_i_hate_about_you",
 "9292406d5193d3402195df7e5647fd168da2d15a":"50_50_2011_film",
 "bbea33d8cce51af36cf7aab15b83d1372f4523aa":"12_and_holding",
 "c41dba67fbd19688656d31ccffec145c12f15b94":"1492_conquest_of_paradise",
 "dd40f45256efa1ec710101bcb925c02e634532e2":"a_few_good_men",
}
found = set()
for split in ("test", "train"):
    if len(found) == len(WANT):
        break
    print(f"[{split}] loading (cached download)...", flush=True)
    ds = load_dataset("deepmind/narrativeqa", split=split)
    for rec in ds:
        doc = rec["document"]
        did = doc["id"]
        if did in WANT and did not in found:
            slug = WANT[did]
            text = doc.get("text", "") or ""
            (RAW/f"{slug}__{did}.txt").write_text(text, encoding="utf-8")
            found.add(did)
            print(f"  saved {slug} ({did}) len={len(text)}", flush=True)
            if len(found) == len(WANT):
                break
missing = [f"{WANT[i]}({i})" for i in WANT if i not in found]
print(f"DONE: {len(found)}/{len(WANT)} saved. missing={missing}", flush=True)
