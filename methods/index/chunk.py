"""Phase 1: chunk the source document with no LLM call.

One universal rule for every source document: walk the text, accumulate blocks (runs
of non-blank lines) into a running chunk, and close the chunk before it would exceed
`B` tokens. A single block with no blank line that already exceeds `B` is hard-split at
the token cap. No chunk ever exceeds `B`.

Each chunk keeps its character span (start_char, end_char) into the RAW source document,
because later steps and the answering fallback point back to the source.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, pathlib, importlib.util, statistics, argparse

BENCH = _BENCH
HERE = pathlib.Path(__file__).resolve().parent
TXT = BENCH/"data"/"movies"/"txt"

sys.path.insert(0, str(HERE))
import config as _cfg  # noqa: E402

_spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)


def _ntok(text):
    return ac.count_tokens(text)


def _blocks(text):
    """Character spans (start, end) of maximal runs of non-blank lines."""
    blocks, pos, start = [], 0, None
    for line in text.splitlines(keepends=True):
        ls = pos; pos += len(line)
        if line.strip() == "":
            if start is not None:
                blocks.append((start, ls)); start = None
        elif start is None:
            start = ls
    if start is not None:
        blocks.append((start, pos))
    return blocks


def _hard_split(text, start, end, B):
    """Split an oversized single block into <= B-token spans at line boundaries."""
    spans, cur_start, pos, seg = [], start, start, ""
    for line in text[start:end].splitlines(keepends=True):
        ls = pos; pos += len(line)
        if seg and _ntok(seg + line) > B:
            spans.append((cur_start, ls)); cur_start = ls; seg = ""
        seg += line
    if seg.strip():
        spans.append((cur_start, end))
    return spans


def chunk_document(text, chunk_target, B):
    """Return chunks: [{id, text, start_char, end_char, tokens}].

    Close a chunk at the first blank-line seam (block boundary) at or after
    `chunk_target` tokens, so cuts land on real scene/paragraph seams. `B` is only a
    hard cap: a single block with no blank line that runs past `B` is hard-split.
    No chunk exceeds `B`.
    """
    blocks = _blocks(text)
    block_tok = [_ntok(text[s:e]) for s, e in blocks]
    spans, cur, cur_tok = [], None, 0
    for (bs, be), bt in zip(blocks, block_tok):
        if bt > B:                                   # rare oversized block, no seam -> hard-split
            if cur:
                spans.append(cur); cur, cur_tok = None, 0
            spans.extend(_hard_split(text, bs, be, B))
            continue
        cur = (bs, be) if cur is None else (cur[0], be)
        cur_tok += bt
        if cur_tok >= chunk_target:                  # close at this blank-line seam
            spans.append(cur); cur, cur_tok = None, 0
    if cur:
        spans.append(cur)
    chunks = []
    for i, (s, e) in enumerate(spans):
        t = text[s:e]
        chunks.append({"id": i, "text": t, "start_char": s, "end_char": e, "tokens": _ntok(t)})
    # invariant (acceptance, enforced): no chunk exceeds B
    for c in chunks:
        assert c["tokens"] <= B, f"chunk {c['id']} has {c['tokens']} tokens > B={B}"
    return chunks


def load_source(slug):
    return (TXT/f"{slug}.txt").read_text(encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    a = ap.parse_args()
    cfg = _cfg.load()
    chunks = chunk_document(load_source(a.movie), cfg.chunk_target, cfg.B)
    toks = [c["tokens"] for c in chunks]
    print(json.dumps({
        "movie": a.movie, "chunk_target": cfg.chunk_target, "B": cfg.B, "n_chunks": len(chunks),
        "chunk_tokens": {"min": min(toks), "median": int(statistics.median(toks)), "max": max(toks)},
        "max_le_B": max(toks) <= cfg.B,
    }, indent=2))
