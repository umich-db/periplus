"""Prepare the 20 NEW NarrativeQA screenplays for a v7 build.

Source text comes from NarrativeQA's own full-text archive (deepmind/narrativeqa,
data/narrativeqa_full_text.zip, one <document_id>.content per document). All 20 selected docs are
IMSDb HTML, exactly like the existing 10, so the text is cleaned with the repo's OWN scraper helpers
(bench/shared/scrape_movies.py extract_script + clean) and is therefore processed identically to the
current corpus. No network scraping: the raw HTML ships inside the archive.

Writes:
  bench/data/movies/txt/<slug>.txt                     cleaned screenplay text
  bench/methods/wiki_v7/corpus/<wiki_key>.txt          symlink the v7 build compiles from
  bench/data/questions/questions_30.jsonl              295 old rows verbatim + 590 new rows

Questions follow the existing convention exactly: every qaps.csv row for the document, capped at 30
per movie, emitted as {qid: "<slug>_NNN", movie: slug, question, gold_answers: [answer1, answer2]}.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])
_NQA_META = _pl.Path(_os.environ.get("NQA_META_DIR") or _BENCH.parent)

import sys, json, zipfile, pathlib, importlib.util
import pandas as pd

ROOT = _NQA_META
BENCH = ROOT / "bench"
HERE = pathlib.Path(__file__).resolve().parent
TXT = BENCH / "data" / "movies" / "txt"
CORPUS = BENCH / "methods" / "wiki_v7" / "corpus"
QDIR = BENCH / "data" / "questions"

# The imsdb cleaner is reproduced here VERBATIM from bench/shared/scrape_movies.py rather than
# imported: that module has no __main__ guard, so importing it re-scrapes the existing 10 movies
# from the network as a side effect. Same logic, no side effect.
def extract_script(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("td", class_="scrtext") or soup.find("pre")
    return node.get_text() if node is not None else ""


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    out, blanks = [], 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks <= 2:
                out.append("")
        else:
            blanks = 0
            out.append(ln)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


_ac_spec = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(_ac_spec); _ac_spec.loader.exec_module(ac)


def main():
    sel = json.loads((HERE / "results" / "nqa_selection.json").read_text())
    qaps = pd.read_csv(ROOT / "qaps.csv")

    from huggingface_hub import hf_hub_download
    zpath = hf_hub_download("deepmind/narrativeqa", "data/narrativeqa_full_text.zip", repo_type="dataset")
    z = zipfile.ZipFile(zpath)

    TXT.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    QDIR.mkdir(parents=True, exist_ok=True)

    rows, new_questions, failures = [], [], []
    for r in sel:
        did, slug, wkey = r["document_id"], r["slug"], r["wiki_key"]
        raw = z.read(f"{did}.content").decode("utf-8", "ignore")
        text = _sm.clean(_sm.extract_script(raw))
        n_tok = ac.count_tokens(text)
        n_words = len(text.split())
        if n_words < 3000:                                   # a real screenplay is far larger
            failures.append({"slug": slug, "title": r["wiki_title"], "words": n_words})
            continue
        (TXT / f"{slug}.txt").write_text(text, encoding="utf-8")
        link = CORPUS / f"{wkey}.txt"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pathlib.Path("../../../data/movies/txt") / f"{slug}.txt")

        qs = qaps[qaps["document_id"] == did].head(30)       # same cap the existing 295 used
        for i, (_, q) in enumerate(qs.iterrows()):
            new_questions.append({"qid": f"{slug}_{i:03d}", "movie": slug,
                                  "question": str(q["question"]),
                                  "gold_answers": [str(q["answer1"]), str(q["answer2"])]})
        rows.append({**r, "tokens": n_tok, "words": n_words, "n_questions": len(qs)})
        print(f"  {r['wiki_title'][:40]:42s} words={n_words:6,} tokens={n_tok:6,} q={len(qs)}")

    # combined question set: the original 295 verbatim + the new rows
    old = [json.loads(l) for l in (QDIR / "questions.jsonl").read_text().splitlines() if l.strip()]
    combined = old + new_questions
    with open(QDIR / "questions_30.jsonl", "w", encoding="utf-8") as f:
        for q in combined:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    (HERE / "results" / "nqa_prepared.json").write_text(json.dumps(rows, indent=2))
    toks = [r["tokens"] for r in rows]
    print(f"\nprepared {len(rows)}/20 new movies "
          f"(token range {min(toks):,} .. {max(toks):,}, total {sum(toks):,})")
    if failures:
        print(f"FAILED to extract ({len(failures)}): {failures}")
    print(f"questions: {len(old)} old + {len(new_questions)} new = {len(combined)} "
          f"-> {QDIR/'questions_30.jsonl'}")
    print(f"corpus symlinks -> {CORPUS}")


if __name__ == "__main__":
    main()
