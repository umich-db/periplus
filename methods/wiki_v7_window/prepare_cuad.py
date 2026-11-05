"""Prepare the 10 longest CUAD contracts (>= 50K tokens) for a v7 build, and build their 410 questions.

CUAD (theatticusproject/cuad-qa) is SQuAD format. In the HuggingFace conversion each contract has ONE
context (the full contract text) but emits SEVERAL rows per clause type, one per annotated answer span.
So rows are grouped by (title, clause type) and their spans merged, which restores CUAD's real shape:
one question per clause type, 41 types per contract. A clause type whose merged span list is empty means
the clause is ABSENT from the contract and the correct behaviour is to say so; has_answer records that.

Questions use the fixed hand-written templates in bench/data/cuad_question_templates.json verbatim
(one per clause type, no per-instance LLM rewriting), so the mapping stays auditable.

Writes:
  bench/data/contracts/txt/<slug>.txt              contract text
  bench/methods/wiki_v7/corpus/<key>.txt           symlink the v7 build compiles from
  bench/data/questions/cuad_questions.jsonl        10 x 41 = 410 questions
  results/cuad_selection.json                      the picked contracts

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])
_NQA_META = _pl.Path(_os.environ.get("NQA_META_DIR") or _BENCH.parent)

import re, json, pathlib, importlib.util
from collections import defaultdict

ROOT = _NQA_META
BENCH = ROOT / "bench"
HERE = pathlib.Path(__file__).resolve().parent
CTXT = BENCH / "data" / "contracts" / "txt"
CORPUS = BENCH / "methods" / "wiki_v7" / "corpus"
QDIR = BENCH / "data" / "questions"
MIN_TOKENS = 50_000
N_SELECT = 10

_s = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(_s); _s.loader.exec_module(ac)


def slugify(t):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", t.lower())).strip("_")[:60]


def main():
    from datasets import load_dataset
    templates = json.loads((BENCH / "data" / "cuad_question_templates.json").read_text())["templates"]

    rows = []
    for sp in ("train", "test"):
        rows += list(load_dataset("theatticusproject/cuad-qa", split=sp, revision="refs/convert/parquet"))

    ctx_of, spans = {}, defaultdict(list)
    for r in rows:
        t = r["title"]
        ctx_of.setdefault(t, r["context"])
        m = re.search(r'related to\s+"([^"]+)"', r["question"])
        if not m:
            continue
        # keep the char offset with each span so alignment stays verifiable after the merge
        spans[(t, m.group(1))] += list(zip(r["answers"]["text"], r["answers"]["answer_start"]))

    lengths = {t: ac.count_tokens(c) for t, c in ctx_of.items()}
    ranked = sorted(lengths, key=lambda t: -lengths[t])
    over = [t for t in ranked if lengths[t] >= MIN_TOKENS]
    picked = over[:N_SELECT] if len(over) >= N_SELECT else ranked[:N_SELECT]
    note = ("10 longest at or above 50K tokens" if len(over) >= N_SELECT
            else f"only {len(over)} cleared 50K; taking the 10 longest overall")

    CTXT.mkdir(parents=True, exist_ok=True); CORPUS.mkdir(parents=True, exist_ok=True)
    QDIR.mkdir(parents=True, exist_ok=True)

    sel, questions = [], []
    for t in picked:
        slug = slugify(t)
        key = "CUAD_" + slug
        (CTXT / f"{slug}.txt").write_text(ctx_of[t], encoding="utf-8")
        link = CORPUS / f"{key}.txt"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pathlib.Path("../../../data/contracts/txt") / f"{slug}.txt")

        n_has = 0
        for i, ctype in enumerate(sorted(templates)):
            pairs = list(dict.fromkeys(spans.get((t, ctype), [])))    # dedup (text, offset), keep order
            gold = list(dict.fromkeys(txt for txt, _ in pairs))       # judge reference: unique texts
            has = len(gold) > 0
            n_has += has
            questions.append({
                "qid": f"{slug}_{i:03d}", "movie": slug, "contract_title": t,
                "clause_type": ctype, "question": templates[ctype],
                "gold_answers": gold,
                "gold_spans": [{"text": txt, "answer_start": int(st)} for txt, st in pairs],
                "has_answer": has,
            })
        sel.append({"title": t, "slug": slug, "wiki_key": key, "tokens": lengths[t],
                    "chars": len(ctx_of[t]), "n_has_answer": n_has, "n_no_answer": 41 - n_has})
        print(f"  {lengths[t]:7,} tok  has_ans={n_has:2d}/41  {t[:64]}")

    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "cuad_selection.json").write_text(json.dumps(sel, indent=2))
    with open(QDIR / "cuad_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (HERE / "results" / "cuad_keys.txt").write_text("\n".join(s["wiki_key"] for s in sel) + "\n")

    ha = sum(s["n_has_answer"] for s in sel)
    print(f"\nselection rule: {note}")
    print(f"pool: {len(ctx_of)} contracts, {len(over)} at or above {MIN_TOKENS:,} tokens")
    print(f"selected {len(sel)}; token range {min(s['tokens'] for s in sel):,} .. "
          f"{max(s['tokens'] for s in sel):,}")
    print(f"questions: {len(questions)} (= {len(sel)} x 41)  has_answer={ha}  no_answer={len(questions)-ha}")
    print(f"wrote {QDIR/'cuad_questions.jsonl'}")


if __name__ == "__main__":
    main()
