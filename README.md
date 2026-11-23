# Periplus: Indexing Long Documents for LLM-Based Analysis

Periplus is a research implementation of a hierarchical, human-readable index for
long-document question answering. The model discovers a document's schema, extracts records in
two passes, writes a tree of bounded Markdown pages, and navigates summaries to
retrieve answers. Source spans support a bounded fallback to the original text.

This release contains the repaired renderer, tree balancer, link validator, and
retrieval implementation, plus benchmark and baseline source code. Datasets,
generated indexes, experimental answers, credentials, and Git history from the
research workspace are excluded.

## Install

Use Python 3.11 on Linux:

```bash
git clone https://github.com/umich-db/periplus.git
cd periplus
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` locally. `AZURE_API_BASE` accepts an Azure OpenAI endpoint or an
OpenAI-compatible `/v1` endpoint; `AZURE_DEPLOYMENT` selects the chat model.
`AZURE_EMBED_DEPLOYMENT` selects a supported embedding deployment. A separate chat
endpoint can be selected with `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL`.
Never commit `.env` or credentials. No credentials are included here.

Local balancing uses `sentence-transformers/all-mpnet-base-v2`, downloaded on
first use. A GPU is optional for the core index. Set `HF_HUB_OFFLINE=1` after
caching this model if network access must be disabled. Model calls incur costs.

## Build and try your own document

`examples/story.txt` is an original, tiny demonstration, not a benchmark dataset.
Use any UTF-8 document you have permission to process:

```bash
python reproduce.py build --source examples/story.txt --output output/demo \
  --id demo --title 'The Observatory'
python reproduce.py ask --source examples/story.txt \
  --index output/demo/balanced/demo --question 'Who repaired the telescope?'
```

A build discovers the schema, extracts records, renders raw pages, balances the
tree, and requires strict integrity validation. It writes validation and build
usage reports under `output/demo/`. Use a fresh output directory or document ID
for every rebuild. Query output includes an answer, retrieval trace, evidence,
and observed API usage. Empty or truncated responses and API errors fail visibly.

Keep the source text available at its indexed path: strict validation verifies
its checksum and source spans. Rebuild after changing it.

## Reproduce experiments

Obtain the original NarrativeQA metadata and source documents separately from the
[NarrativeQA project](https://github.com/google-deepmind/narrativeqa), respecting
its terms and each document's rights. The benchmark also includes CUAD preparation
and baseline runners; obtain CUAD separately for those experiments.

For a portable experiment, build each source document as above. Provide one JSON
object per question, with `qid`, `question`, and an `answers` list, in a JSONL file.
Keep question IDs and references fixed across methods. Evaluate each document:

```bash
python reproduce.py evaluate --source /path/to/document.txt \
  --index output/run/balanced/document-id \
  --questions /path/to/questions.jsonl --output predictions.jsonl
```

The output preserves question fields and adds `prediction`, `answer_source`,
`wall_seconds`, retrieval traces, and per-question call records. The output file
must be new. Failed requests stop the run instead of becoming scored misses.
Build reports record extraction API usage separately from answer usage; local
balancing runtime is included in build wall time. Usage marked unknown must not
be counted as zero. Reasoning tokens are part of output tokens, not additional
tokens. Save endpoint/model configuration, dependency versions, source checksums,
and commit SHA alongside each experiment; do not include credential values.

The original benchmark harness is included under `methods/`, `shared/`,
`scripts/`, and `analysis/`. Preparation scripts such as
`methods/wiki_v7_window/select_nqa.py` and `prepare_nqa.py` describe the historical
corpus selection; `shared/eval.py` and analysis scripts contain the scoring and
comparison logic. These harness scripts retain their historical data layout and
require externally prepared inputs; they are not the portable entrypoint.
`requirements/` records the separate benchmark environments; baselines can
require incompatible dependencies and additional setup. Compare methods on the
same question IDs with the same judge model, prompt, and settings. The historical
ten-document subset had 295 questions; the expanded 30-document set had 885.

**Historical scores are not claimed for this repaired release.** Generation and
judge results depend on the model and endpoint. A fresh, complete benchmark must
be run before attaching new accuracy or cost claims to the paper. This repository
contains no experimental datasets or precomputed index artifacts.

## Index integrity and repaired behavior

The renderer preserves extracted fields, including long records split across
continuation pages, and assigns collision-safe page paths. Manifest parent/child
connections, lateral links, and continuation links are validated. Balancing
preserves subject content and evidence while enforcing the documented fanout
window (17–33, with explicit small-category exceptions). Retrieval checks source
identity before using offsets and records which pages and evidence it reads.

```bash
python -m unittest discover -s methods/llm_index -p 'test_*.py' -q
python methods/llm_index/test_paper_conformance.py
python methods/llm_index/verify_tree.py output/demo/balanced/demo \
  --strict-integrity --json
```

Tests create synthetic documents and mock model calls; credentials and benchmark
datasets are unnecessary. They cover page preservation, malformed links,
connectivity, balancing, provenance, bounded routing, and stale-source rejection.
The optional historical gateway-cutoff extension is absent and explicitly skipped
by the conformance script.

The implementation retains the historical `methods/llm_index` directory,
`llm_index` benchmark identifier, and `LLM_INDEX_*` environment variables for
compatibility with existing experiment scripts and results. Its display name in
benchmark charts is Periplus.

## Source and paper

Prepared from research source commit
`513c635dcbcc3cca381fb753fbc8c40dccec9c88`, with release-specific portability fixes,
a standalone entrypoint, documentation, and CI. Core code is in
`src/wiki_index/pipeline.py`, `methods/llm_index/balance_window.py`, and
`methods/llm_index/verify_tree.py`.

Paper title: *Indexing Long Documents for LLM-Based Analysis*. A public paper URL
and finalized citation have not been supplied; add them when available. This
repository URL can be linked directly from the paper.
