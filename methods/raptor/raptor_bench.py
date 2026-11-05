"""RAPTOR baseline (arXiv 2401.18059) wired into the benchmark.

Uses the official implementation (methods/raptor/upstream) for the faithful part:
100-token leaf chunks, recursive UMAP + GMM(BIC) soft clustering, and collapsed-
tree retrieval. We plug in:
  - summarization + QA via the shared Azure gpt-5 client (token buckets
    raptor_build / raptor_answer), so cost is comparable to the other methods
  - embeddings via RAPTOR's own SBERT model (multi-qa-mpnet-base-cos-v1, the
    paper's model), which is local and free

Subcommands:
  build [slug...]        build + save a RAPTOR tree per movie
  answer --movie <slug>  answer that movie's questions -> results/raptor_<slug>.jsonl
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse, importlib.util

BENCH = _BENCH
HERE = BENCH / "methods" / "raptor"
UPSTREAM = HERE / "upstream"
import os
_CUAD = bool(os.environ.get("CUAD"))
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
TREES = HERE / "output" / ("trees_cuad" if _CUAD else "trees")
TXT = BENCH / "data" / ("contracts/txt" if _CUAD else "movies/txt")

sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig  # noqa: E402
from raptor import BaseSummarizationModel, BaseQAModel  # noqa: E402
from raptor.EmbeddingModels import SBertEmbeddingModel  # noqa: E402

MOVIES_10 = [
    "10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
    "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
    "a_dry_white_season", "a_few_good_men",
]


class AzureSummarizationModel(BaseSummarizationModel):
    """RAPTOR node summaries via Azure gpt-5. Faithful prompt from the paper."""
    def summarize(self, context, max_tokens=150, stop_sequence=None):
        # gpt-5 is a reasoning model: give generous max_completion_tokens so the
        # visible summary is not eaten by hidden reasoning.
        try:
            text, _, _ = ac.chat(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content":
                  f"Write a summary of the following, including as many key details "
                  f"as possible: {context}:"}],
                max_tokens=1200, bucket="raptor_build")
            return (text or "").strip()
        except Exception as e:  # noqa: BLE001
            print(f"  summarize error: {repr(e)[:100]}", flush=True)
            return ""


class AzureQAModel(BaseQAModel):
    """Answer from RAPTOR's retrieved (collapsed-tree) context via Azure gpt-5.
    Retrieval-only: no raw-document fallback.

    permissive=False (strict): abstain with 'not found' when the answer is not in
      the retrieved context (matches the other methods' abstain behavior).
    permissive=True: RAPTOR's native style, always give the best answer, never
      abstain (trades misses for more guessing/fabrication)."""
    def __init__(self, permissive=False):
        self.permissive = permissive

    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        if self.permissive:
            user = (f"Using the context below, give the best full answer you can to the "
                    f"question, in 10 words or fewer. Always give your best answer.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        else:
            user = (f"Use ONLY the context below to answer.\nAnswer in 10 words or fewer. "
                    f"If the answer is not in the context, reply exactly: not found\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        try:
            text, _, _ = ac.chat(
                [{"role": "system", "content": "You are a question answering assistant."},
                 {"role": "user", "content": user}],
                max_tokens=2048, bucket="raptor_answer")
            return (text or "").strip().splitlines()[0].strip(" .\"'") if text else "not found"
        except Exception as e:  # noqa: BLE001
            print(f"  qa error: {repr(e)[:100]}", flush=True)
            return "not found"


def _config(permissive=False):
    return RetrievalAugmentationConfig(
        summarization_model=AzureSummarizationModel(),
        qa_model=AzureQAModel(permissive=permissive),
        embedding_model=SBertEmbeddingModel(),   # multi-qa-mpnet-base-cos-v1 (paper), also for retrieval
    )


def build(slugs):
    TREES.mkdir(parents=True, exist_ok=True)
    stats = {}
    for slug in slugs:
        tree_path = TREES / slug
        if tree_path.exists():
            print(f"[build] {slug}: exists, skip", flush=True); continue
        text = (TXT / f"{slug}.txt").read_text(encoding="utf-8", errors="ignore")
        before = ac.get_counters().get("raptor_build", {}).get("prompt", 0) + \
            ac.get_counters().get("raptor_build", {}).get("completion", 0)
        t0 = time.time()
        RA = RetrievalAugmentation(config=_config())
        RA.add_documents(text)
        RA.save(str(tree_path))
        after = ac.get_counters().get("raptor_build", {}).get("prompt", 0) + \
            ac.get_counters().get("raptor_build", {}).get("completion", 0)
        stats[slug] = {"build_tokens": after - before, "build_min": round((time.time()-t0)/60, 2)}
        print(f"[build] {slug}: {stats[slug]['build_tokens']:,} tok, "
              f"{stats[slug]['build_min']} min -> {tree_path}", flush=True)
    sp = HERE / "output" / f"build_stats{_SFX}.json"
    prev = json.loads(sp.read_text()) if sp.exists() else {}
    prev.update(stats); sp.write_text(json.dumps(prev, indent=2))


def _answer_bucket(c):
    b = c.get("raptor_answer", {})
    return b.get("prompt", 0), b.get("completion", 0)


def answer(slug, permissive=False):
    tree_path = TREES / slug
    if not tree_path.exists():
        print(f"NO tree for {slug}"); sys.exit(1)
    RA = RetrievalAugmentation(config=_config(permissive), tree=str(tree_path))
    tag = "raptor_perm" if permissive else "raptor"

    build_stats = json.loads((HERE / "output" / f"build_stats{_SFX}.json").read_text())
    build_tot = build_stats.get(slug, {}).get("build_tokens", 0)
    questions = [json.loads(l) for l in
                 (BENCH / "data" / "questions" / _QFILE).read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]
    amort = round(build_tot / max(1, len(qs)))

    out = HERE / "results" / f"{tag}{_SFX}_{slug}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["qid"])
    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done:
            continue
        bp, bc = _answer_bucket(ac.get_counters())
        t0 = time.time()
        try:
            pred = RA.answer_question(question=q["question"], collapse_tree=True,
                                      top_k=10, max_tokens=3500)
        except Exception as e:  # noqa: BLE001
            pred = f"[ERROR: {repr(e)[:100]}]"
        lat = round(time.time() - t0, 2)
        ap, acc = _answer_bucket(ac.get_counters())
        in_tok, out_tok = ap - bp, acc - bc
        rec = {"method": tag, "movie": slug, "qid": q["qid"], "question": q["question"],
               "gold_answers": q["gold_answers"], "prediction": pred, "source": "raptor",
               "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
               "latency_s": lat, "build_tokens": amort}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"  {q['qid']} in={in_tok} out={out_tok} {lat}s :: {pred[:55]}", flush=True)
    f.close()
    print(f"DONE {slug}: {len(qs)} questions -> {out}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("usage: raptor_bench.py build [slug...] | answer --movie <slug>"); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2:] or MOVIES_10)
    elif cmd == "answer":
        ap = argparse.ArgumentParser(); ap.add_argument("--movie", required=True)
        ap.add_argument("--permissive", action="store_true")
        a = ap.parse_args(sys.argv[2:]); answer(a.movie, a.permissive)
    else:
        print(f"unknown cmd: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    main()
