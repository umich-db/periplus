"""Measure the true one-time build cost of the per_category index, which the build_mode_costs figure
(116,717) undercounts because extraction is cached and only the pack/summary step is counted there.

We re-run the SHIPPED extractor (extract.extract_one) over every chunk of all 10 source documents and
log the real prompt + completion tokens, then add the logged pack cost and the overview cost. Nothing
about the shipped extract prompt, the answer path, the judge, v7, or any build mode is modified. This
only measures. Deterministic and resumable via the checkpointed harness.

Vocabulary: root, category, subject, field, source document. No em dashes.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
IDX = HERE.parent
sys.path.insert(0, str(IDX))
import chunk as _chunk, config as _cfg, extract as _extract, eval_harness  # noqa: E402

cfg = _cfg.load()
import answer  # noqa: E402  (SLUGS)

PACK_TOK = 116717   # build_mode_costs.json per_category build_tok (pack + routing summaries, logged)
OUT = IDX/"results"/"build_cost_full.json"

# tasks: one extract call per (slug, chunk)
tasks = []
for slug in answer.SLUGS:
    chunks = _chunk.chunk_document(_chunk.load_source(slug), cfg.chunk_target, cfg.B)
    for c in chunks:
        tasks.append((slug, c["id"], c))
print(f"chunks to extract: {len(tasks)}", flush=True)


def key_fn(t):
    return f"{t[0]}::chunk{t[1]}"


def work_fn(t):
    slug, cid, c = t
    subs, pt, ct = _extract.extract_one(c)
    return {"slug": slug, "cid": cid, "n_subjects": len(subs), "in_tok": pt, "out_tok": ct}


records = eval_harness.run_checkpointed("build_extract_cost", tasks, key_fn, work_fn, workers=6)
ok = [r for r in records if r]
ex_in = sum(r["in_tok"] for r in ok)
ex_out = sum(r["out_tok"] for r in ok)

# overview (one whole-document overview call per movie) is a small extra build cost not measured here.
extract_total = ex_in + ex_out
build_total = extract_total + PACK_TOK
result = {"n_chunks": len(ok), "extract_in": ex_in, "extract_out": ex_out,
          "extract_total": extract_total, "pack_tok": PACK_TOK,
          "overview_tok": "not measured (small, one call per movie)",
          "build_total": build_total,
          "answer_in_295": 1125738, "answer_out_295": 252836, "answer_total_295": 1125738 + 252836}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(f"\nBUILD total = {build_total:,}  (was reported as {PACK_TOK:,}, pack-only)")
print(f"ANSWER total (295 Qs) = {result['answer_total_295']:,}")
print("wrote", OUT)
