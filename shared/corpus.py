"""corpus.py — the one list of documents every method builds over.

The 30-screenplay corpus grew out of the original 10, and each method absorbed
the extra 20 through its own `run_new20.sh`. That left the corpus defined in a
dozen places: `scripts/run_method.sh` still carries the old ten in a shell array,
every method has a second script with the other twenty, and nothing fails loudly
when a method quietly covers a different set than the one it is compared against.
Row counts in the results tables differ for exactly that reason.

So the corpus is defined once, here, from what is actually on disk, and the
runner drives every method from it.

    from corpus import DOCS
    DOCS["nqa"].slugs        # 30 screenplay slugs
    DOCS["cuad"].slugs       # 10 contract slugs
    DOCS["nqa"].questions    # [{qid, movie, question, gold_answers}, ...]
"""

import os, json, pathlib

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])


class Corpus:
    def __init__(self, key, name, txt_dir, qfile, result_suffix, cuad_env):
        self.key = key
        self.name = name
        self.txt_dir = BENCH / txt_dir
        self.qfile = BENCH / "data" / "questions" / qfile
        self.result_suffix = result_suffix      # "" or "_cuad"
        self.cuad_env = cuad_env                # what run_method.sh wants in CUAD

    @property
    def slugs(self):
        return sorted(p.stem for p in self.txt_dir.glob("*.txt"))

    @property
    def questions(self):
        return [json.loads(l) for l in self.qfile.read_text().splitlines() if l.strip()]

    def doc_path(self, slug):
        return self.txt_dir / f"{slug}.txt"

    def __repr__(self):
        return f"<Corpus {self.key}: {len(self.slugs)} docs, {len(self.questions)} questions>"


DOCS = {
    "nqa": Corpus("nqa", "NarrativeQA", "data/movies/txt", "questions_30.jsonl", "", ""),
    "cuad": Corpus("cuad", "CUAD", "data/contracts/txt", "cuad_questions.jsonl", "_cuad", "1"),
}

# Every method the benchmark compares, with the venv it needs. graphrag emits two
# result sets from one run, so it is listed once and scored twice.
METHODS = [
    "naive_gpt", "docetl_standard", "leann", "rag", "graphrag",
    "raptor", "doctopus", "llm_index",
]

# What the scoring and charting side calls each method. graphrag fans out.
# wiki_v6 and wiki_v8 are earlier versions of our index, kept in the tree as
# ablations. Only llm_index carries the Section 3 conformance work, so it is the
# only one of ours that is the method the paper describes, and the only one the
# sweep runs. Re-add them here to score an ablation.
# The eight the CIDR figures compare, in the paper's own legend order. SHED,
# MOAR and graphrag_global exist in this repo but are not in those figures, so
# they are not part of this sweep; add them back here to include them.
SCORED = [
    "naive_gpt", "docetl_standard", "leann", "rag", "graphrag_local",
    "raptor", "doctopus", "llm_index",
]

# Display names for the charts. "index" is ours; see the note in the README about
# the wiki_* / index_* split (same method, historical directory names).
DISPLAY = {
    "naive_gpt": "Naive", "rag": "RAG", "raptor": "RAPTOR", "leann": "LEANN",
    "graphrag_local": "GraphRAG", "graphrag_global": "GraphRAG-global",
    "docetl_standard": "DocETL", "moar": "MOAR", "shed": "SHED",
    "doctopus": "Doctopus", "llm_index": "Periplus",
}

OURS = "llm_index"


# --- our index's document ids ------------------------------------------------
#
# wiki_v* name their corpus files by the document's display id, not by the
# question-file slug: "17_Again_(film)" and "CUAD_ceres_inc_...". The answer
# scripts carried a hand-written ten-entry slug->title dict, which is why our
# index covered ten movies while the baselines covered thirty, and why it had no
# CUAD path at all. The mapping is derivable, so derive it and check it.

import re as _re


def _norm(doc_id):
    """A document id reduced to the form the question files use as a slug."""
    s = _re.sub(r"[^a-z0-9]+", "_", str(doc_id).lower())
    s = _re.sub(r"_+", "_", s).strip("_")
    return s[5:] if s.startswith("cuad_") else s      # contracts carry a CUAD_ prefix


def wiki_docids(corpus_key=None, corpus_dir=None):
    """{slug: document id} for the wiki_v* corpus, restricted to one corpus.

    Raises if a slug in the question set has no document, because a silently
    missing document is exactly the failure this function exists to prevent: it
    shows up as a method that "ran" on fewer questions than everything else.
    """
    d = pathlib.Path(corpus_dir or (BENCH / "methods" / "llm_index" / "corpus"))
    found = {}
    for f in sorted(d.glob("*.txt")):
        found.setdefault(_norm(f.stem), f.stem)
    if corpus_key is None:
        return found
    # Normalise the slug side too. One CUAD slug ends in an underscore, left by
    # the 60-character truncation that produced it, and comparing a stripped
    # document id against an unstripped slug drops that contract silently.
    wanted = DOCS[corpus_key].slugs
    found = {k.strip("_"): v for k, v in found.items()}
    missing = [s for s in wanted if s.strip("_") not in found]
    if missing:
        raise KeyError(f"{corpus_key}: no wiki document for {len(missing)} slug(s): "
                       f"{missing[:3]}")
    return {s: found[s.strip("_")] for s in wanted}


if __name__ == "__main__":
    for k, c in DOCS.items():
        print(c)
        print("   ", ", ".join(c.slugs[:3]), "...")
        try:
            m = wiki_docids(k)
            print(f"    wiki docids resolved: {len(m)}/{len(c.slugs)}")
        except KeyError as e:
            print("    wiki docids:", e)
