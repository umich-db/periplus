"""Embedding-based retrieval for the index/wiki v7 experiments (RAPTOR's idea).

WHY: on these digests v7's `traverse` is effectively query-independent -- every
movie's consolidated.md (~13-15k chars) exceeds CONTEXT_BUDGET (12k), so traverse
reads the truncated global summary and stops before the 300+ detail pages or any
page scoring ever run. The query only influences the QA prompt, not retrieval.

FIX (RAPTOR-style): embed each wiki PAGE summary once and retrieve the
query-relevant detail pages by a blended cosine+keyword score. Two flavors:
  - collapsed_retrieve : pure top-k over page summaries (RAPTOR collapsed tree).
  - semantic_traverse  : v7-style greedy walk + query expansion, but seeded with a
    CAPPED consolidated overview so detail pages actually get read.

Uses the same local SBERT model RAPTOR uses (multi-qa-mpnet-base-cos-v1; encoded
with normalize_embeddings=True so a dot product IS cosine). Free, no API.
Page-summary embeddings are cached per digest under shared/cache/sem_emb/ and
reused across variants and movies.
"""
import json, re, hashlib, pathlib
import numpy as np

CONTEXT_BUDGET = 12_000  # match wiki_index/pipeline.py
SEED_CHARS = 4_000       # semantic_traverse: chars of consolidated overview to seed with
W_COS, W_KW, BRANCH_BONUS = 0.65, 0.35, 0.05  # blended-score weights

CACHE = pathlib.Path(__file__).resolve().parent / "cache" / "sem_emb"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    return _MODEL


def embed(texts):
    v = _model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def _words(t):
    return set(re.findall(r"[a-zA-Z]{3,}", str(t).lower()))


def _read(path):
    return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")


def load_page_embeddings(wiki_dir):
    """(entries, matrix) for all leaf+branch pages; embed summaries once, cache."""
    wiki_dir = pathlib.Path(wiki_dir)
    manifest = json.loads((wiki_dir / "_manifest.json").read_text())
    entries = [e for e in manifest if e.get("level") in ("leaf", "branch")]
    key = hashlib.sha1(f"{wiki_dir.name}|{len(entries)}".encode()).hexdigest()[:12]
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{wiki_dir.name}.{key}.npz"
    if cf.exists():
        mat = np.load(cf)["vecs"]
        if mat.shape[0] == len(entries):
            return entries, mat
    mat = embed([e["summary"] for e in entries]) if entries else np.zeros((0, 768), np.float32)
    np.savez(cf, vecs=mat)
    return entries, mat


def _norm(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _blend(cos, kw, entries):
    """Blended, scale-robust score: min-max normalize each component, then weight."""
    score = W_COS * _norm(cos) + W_KW * _norm(kw)
    for i, e in enumerate(entries):
        if e["level"] == "branch":
            score[i] += BRANCH_BONUS
    return score


def collapsed_retrieve(question, wiki_dir, max_chars=CONTEXT_BUDGET, header_chars=0):
    """RAPTOR collapsed-tree: top pages by blended cosine+keyword, read full content."""
    wiki_dir = pathlib.Path(wiki_dir)
    entries, mat = load_page_embeddings(wiki_dir)
    if not entries:
        return ""
    q_vec = embed([question])[0]
    q_words = _words(question)
    cos = mat @ q_vec
    kw = np.array([len(q_words & _words(e["summary"])) for e in entries], dtype=np.float32)
    score = _blend(cos, kw, entries)
    order = np.argsort(-score)

    collected, total = [], 0
    if header_chars > 0:
        c = wiki_dir / "consolidated.md"
        if c.exists():
            t = _read(c)[:header_chars]
            collected.append(t); total += len(t)
    for i in order:
        if total >= max_chars:
            break
        if cos[i] <= 0 and kw[i] == 0:   # no semantic and no lexical signal -> stop
            break
        p = wiki_dir / entries[i]["path"]
        if not p.exists():
            continue
        t = _read(p)[:max_chars - total]
        collected.append(t); total += len(t)
    return "\n\n---\n\n".join(collected)


def augment_retrieve(question, wiki_dir, max_chars=24_000):
    """AUGMENT: keep the FULL consolidated summary as base context, then append the
    query-relevant detail PAGES (blended cosine+keyword) until the raised budget.
    Direct counterpart to wiki_v7_bigbudget (same budget, full summary + detail pages)
    but pages are selected by embedding score instead of keyword traversal."""
    wiki_dir = pathlib.Path(wiki_dir)
    collected, total = [], 0
    c = wiki_dir / "consolidated.md"
    if c.exists():                      # full summary as the base (untruncated within budget)
        t = _read(c)[:max_chars]
        collected.append(t); total += len(t)
    entries, mat = load_page_embeddings(wiki_dir)
    if entries and total < max_chars:
        q_vec = embed([question])[0]
        q_words = _words(question)
        cos = mat @ q_vec
        kw = np.array([len(q_words & _words(e["summary"])) for e in entries], dtype=np.float32)
        score = _blend(cos, kw, entries)
        for i in np.argsort(-score):
            if total >= max_chars:
                break
            if cos[i] <= 0 and kw[i] == 0:
                break
            p = wiki_dir / entries[i]["path"]
            if not p.exists():
                continue
            t = _read(p)[:max_chars - total]
            collected.append(t); total += len(t)
    return "\n\n---\n\n".join(collected)


def semantic_traverse(question, wiki_dir, max_chars=CONTEXT_BUDGET, seed_chars=SEED_CHARS):
    """v7 greedy walk + query expansion, but seed a CAPPED consolidated overview so
    the blended-score page selection actually runs and fills the rest of the budget."""
    wiki_dir = pathlib.Path(wiki_dir)
    entries, mat = load_page_embeddings(wiki_dir)
    q_vec = embed([question])[0]
    cos = mat @ q_vec
    expanded = _words(question)
    visited, collected, total = set(), [], 0

    c = wiki_dir / "consolidated.md"
    if c.exists() and seed_chars > 0:
        t = _read(c)[:seed_chars]
        collected.append(t); total += len(t); expanded.update(_words(t))

    while total < max_chars:
        kw = np.array([len(expanded & _words(e["summary"])) for e in entries], dtype=np.float32)
        score = _blend(cos, kw, entries)
        cand = [i for i in range(len(entries)) if entries[i]["path"] not in visited]
        if not cand:
            break
        i = max(cand, key=lambda j: score[j])
        if cos[i] <= 0 and kw[i] == 0:
            break
        visited.add(entries[i]["path"])
        p = wiki_dir / entries[i]["path"]
        if not p.exists():
            continue
        t = _read(p)[:max_chars - total]
        collected.append(t); total += len(t); expanded.update(_words(t))
    return "\n\n---\n\n".join(collected)
