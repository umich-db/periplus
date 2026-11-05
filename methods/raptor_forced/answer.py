"""RAPTOR 'forced' QA policy -- matched to index_v7_guess for a fair fabrication test.

Same RAPTOR trees + collapsed-tree retrieval (top_k=10) as raptor_native, but the QA
node is changed on TWO axes we found made raptor_native look artificially clean:
  1. NEVER abstain: raptor_native still replies 'not found' on ~7% of Qs (its riskiest);
     here, if the model abstains we RE-PROMPT to force a definitive best guess.
  2. FULLER answers: raptor_native emits terse ~5-word spans; shorter answers assert
     fewer specifics -> fewer fabrications. Here we ask for a fuller ~10-15 word answer,
     matching index_v7_guess's length so length is not the confound.

Reuses methods/raptor/raptor_bench.py machinery (Azure client, RAPTOR lib, trees).
Writes results/raptor_forced_<slug>.jsonl.  Usage: answer.py --movie <slug>
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse, importlib.util, re

BENCH = _BENCH
RB_PATH = BENCH/"methods"/"raptor"/"raptor_bench.py"
spec = importlib.util.spec_from_file_location("raptor_bench", RB_PATH)
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

HERE = pathlib.Path(__file__).resolve().parent

_ABSTAIN = ("not found", "unknown", "n/a", "none", "i don't know", "i do not know",
            "cannot", "can't", "no information", "not mentioned", "not specified",
            "not stated", "insufficient", "unclear", "no answer")


def _clean(text):
    return (text or "").strip().splitlines()[0].strip(" .\"'") if text else ""


def _is_abstain(a):
    al = a.lower()
    return (not a) or any(k in al for k in _ABSTAIN)


class ForcedQAModel(rb.BaseQAModel):
    """Always answer; fuller ~10-15 word answers. Matches index_v7_guess policy."""
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        base = (
            "Using the context below, answer the question in a full phrase of about "
            "10-15 words that states the specific answer with its relevant details.\n"
            "You MUST commit to a definitive answer -- NEVER reply 'not found', "
            "'unknown', or 'I don't know'. If the context is insufficient, give your "
            "single best guess anyway.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        sysmsg = {"role": "system", "content": "You are a question answering assistant."}
        try:
            text, _, _ = rb.ac.chat([sysmsg, {"role": "user", "content": base}],
                                    max_tokens=2048, bucket="raptor_answer")
            ans = _clean(text)
            if _is_abstain(ans):  # it abstained despite instructions -> force a guess
                force = base + ("\n\n(You did not commit to an answer. Give your single "
                                "best specific guess now in ~10-15 words -- no disclaimers, "
                                "no 'not found'.)")
                text2, _, _ = rb.ac.chat([sysmsg, {"role": "user", "content": force}],
                                         max_tokens=2048, bucket="raptor_answer")
                ans2 = _clean(text2)
                if ans2 and not _is_abstain(ans2):
                    ans = ans2
                elif ans2:
                    ans = ans2  # keep whatever it produced rather than an empty abstain
            return ans or "unknown"
        except Exception as e:  # noqa: BLE001
            print(f"  qa error: {repr(e)[:100]}", flush=True)
            return "unknown"


def _config():
    return rb.RetrievalAugmentationConfig(
        summarization_model=rb.AzureSummarizationModel(),
        qa_model=ForcedQAModel(),
        embedding_model=rb.SBertEmbeddingModel(),
    )


def answer(slug):
    tree_path = rb.TREES / slug
    if not tree_path.exists():
        print(f"NO tree for {slug}"); sys.exit(1)
    RA = rb.RetrievalAugmentation(config=_config(), tree=str(tree_path))
    tag = "raptor_forced"

    build_stats = json.loads((rb.HERE/"output"/"build_stats.json").read_text())
    build_tot = build_stats.get(slug, {}).get("build_tokens", 0)
    questions = [json.loads(l) for l in
                 (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]
    amort = round(build_tot / max(1, len(qs)))

    out = HERE/"results"/f"{tag}_{slug}.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): done.add(json.loads(l)["qid"])
    f = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done: continue
        bp, bc = rb._answer_bucket(rb.ac.get_counters())
        t0 = time.time()
        try:
            pred = RA.answer_question(question=q["question"], collapse_tree=True,
                                      top_k=10, max_tokens=3500)
        except Exception as e:  # noqa: BLE001
            pred = f"[ERROR: {repr(e)[:100]}]"
        lat = round(time.time()-t0, 2)
        ap, acc = rb._answer_bucket(rb.ac.get_counters())
        in_tok, out_tok = ap-bp, acc-bc
        rec = {"method": tag, "movie": slug, "qid": q["qid"], "question": q["question"],
               "gold_answers": q["gold_answers"], "prediction": pred, "source": "raptor",
               "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok+out_tok,
               "latency_s": lat, "build_tokens": amort}
        f.write(json.dumps(rec, ensure_ascii=False)+"\n"); f.flush()
        print(f"  {q['qid']} in={in_tok} out={out_tok} {lat}s :: {pred[:55]}", flush=True)
    f.close()
    print(f"DONE {slug}: {len(qs)} questions -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--movie", required=True)
    answer(ap.parse_args().movie)
