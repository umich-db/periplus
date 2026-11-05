"""
pipeline.py — Wiki v7: LLM-discovered schema, derived limits, recursive tree.

BUILD (per document):
  1. DISCOVER — LLM reads the document, proposes information categories
     (1 call per doc, unique template per doc, domain-agnostic)
  2. EXTRACT  — 2-pass chunked extraction using discovered schema
  3. ORGANIZE — recursively build wiki tree; nodes split when content exceeds
     the page-size budget. Depth is not capped — tree grows as deep as needed.

QUERY:
  Manifest-driven traversal with descend / ascend / lateral hop.
  Pages don't need explicit [[links]] — the manifest is the routing table.

LIMITS — derived from one free parameter (CONTEXT_BUDGET):
  CONTEXT_BUDGET  = effective chars a QA LLM can use without degradation
  PAGES_PER_QUERY = root + branch + 2-3 leaves + 1 lateral ≈ 5
  PAGE_SIZE       = CONTEXT_BUDGET / PAGES_PER_QUERY
  MAX_CHILDREN    = PAGE_SIZE / avg summary line length
"""

import json
import re
import sys
import inspect
import logging
import hashlib
from collections import Counter
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cold_start"))
from cold_start import (
    llm, parse_json, parse_json_with_retry, clean_document, chunk_text,
    safe_filename, log, get_llm_log, llm_cost_summary,
    _merge_extraction_into,
    COMPILE_PASS1_PROMPT, COMPILE_PASS2_PROMPT,
    FALLBACK_QA_PROMPT,
)

# ═════════════════════════════════════════════════════════════════════════════
# DERIVED LIMITS
# ═════════════════════════════════════════════════════════════════════════════
#
# One free parameter: how many chars the QA LLM can effectively use.
# Extractive QA degrades past ~12K chars of context (needle-in-haystack
# studies on GPT-4/5 class models). This is the real constraint.

CONTEXT_BUDGET = 12_000          # chars — QA prompt size budget

# Everything else is derived:
# A typical traversal reads: consolidated summary + 2-3 detail pages.
# So pages should be sized to fit ~4-5 in the budget.
TYPICAL_PAGES = 5
PAGE_SIZE = CONTEXT_BUDGET // TYPICAL_PAGES           # 2400 chars per page
SUMMARY_LINE_SIZE = 100                               # ~1 line per child
MAX_CHILDREN = PAGE_SIZE // SUMMARY_LINE_SIZE          # 24 children per branch

# The root's consolidated summary is a node like any other, so it is subject to
# the same capacity limit (paper §3.1 invariants). Uncapped it swallows the whole
# answering budget: measured on the 40-document corpus the model returned 11.6K to
# 23.5K chars against a 12,000-char budget, so the digest saturated at the root on
# every question and the traversal never read one page of the tree below it. Half
# the budget is the documented target (_consolidate asks for it); this makes it a
# bound rather than a request, so the other half is always left for detail pages.
ROOT_SUMMARY_CAP = CONTEXT_BUDGET // 2                # 6000 chars

# Routing budget (paper §3.4). C is the most one routing step may read while
# choosing a direction; S is one child's routing descriptor. Both are CHARACTERS,
# the unit the pages are written and measured in. Fanout follows: f = C/S.
# These mirror methods/wiki_v7_balanced/balance.py, which enforces f on the tree.
ROUTING_BUDGET = 4_000                                # C
DESCRIPTOR_CHARS = 120                                # S
MAX_FANOUT = ROUTING_BUDGET // DESCRIPTOR_CHARS       # f = 33

# Lateral links per subject (paper §3.3, §3.5). Bounded so a subject page's
# "Related" block stays a routing aid rather than a second copy of the index.
MAX_LINKS = 6

# Joiners. Both count against the answering budget, because both are text the
# answer model reads.
_DIGEST_SEP = "\n\n---\n\n"             # between index pages in the digest
_PASSAGE_SEP = "\n\n[...]\n\n"          # between source passages in the fallback

# Consolidation reads every page. Pages that don't fit one prompt are summarized
# in batches of this size and the partials reduced — see _consolidate().
CONSOLIDATE_BATCH_CHARS = 80_000

# Extraction chunking. Boundaries are saved at build time (_chunking.json) and
# reused by insert_text() so an edit only re-extracts the chunk it landed in.
CHUNK_SIZE = 30_000
CHUNK_OVERLAP = 2_000

# On a "not found", query_with_traversal re-walks the tree with this much more
# budget before falling back to the raw document.
DEEP_RETRY_FACTOR = 3

# Optional embedder for the fact→chunk fallback: a callable taking a list of
# strings and returning a list of vectors. Left unset the fallback is lexical,
# so the pipeline has no embedding dependency. Drivers that have one (the bench
# routes through shared/azure_client.py) can install it with set_embedder().
_EMBED_FN = None


def set_embedder(fn):
    """Install the embedder used to place facts whose exact signature was lost."""
    global _EMBED_FN
    _EMBED_FN = fn


# Source offsets from the most recent extract(..., return_provenance=True).
# Carried out-of-band so extract()'s return shape stays what callers expect.
_LAST_SPANS = {}

# Pages the most recent traverse() actually read, in visit order. Carried
# out-of-band for the same reason: the fallback needs to know which subjects the
# descent reached so it can return to their evidence spans (§3.5), and traverse's
# return shape (the digest string) is what every caller and ablation expects.
_LAST_VISITED = []


# Paper §3.4: descent into the index is unbounded; what is bounded is the number
# of times traversal may level up to a higher node and descend a different branch.
LEVEL_UP_BUDGET = 3

# Semantic navigation (paper §3.1, §3.5, Figure 1b). The index routes by an LLM
# reading plain-text summaries, not by a key comparison, so the router is the
# method rather than an optimisation. Left unset traverse falls back to keyword
# scoring, which is a search key by another name -- so the drivers install
# llm_route_children and the fallback exists only to survive a failed call.
_ROUTER = None


def set_router(fn):
    """Install the per-level navigator, e.g. set_router(pipeline.llm_route_children).

    The router is called once per level of the descent with only that node's
    children, and returns them reordered, most relevant first. That is the
    paper's routing step (§3.5): a short call that reads the children's routing
    descriptors and judges which are relevant, bounded by the routing budget.
    """
    global _ROUTER
    _ROUTER = fn


# ═════════════════════════════════════════════════════════════════════════════
# CHUNK BOUNDARIES + FACT PROVENANCE
# ═════════════════════════════════════════════════════════════════════════════

def chunk_bounds(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Char offsets of each chunk, numbered from 1.

    Must step identically to cold_start.chunk_text — the boundaries are the
    contract between build and insert, so text[start:end] has to be exactly the
    chunk that was extracted.
    """
    bounds, i, n = [], 0, 0
    while i < len(text):
        n += 1
        bounds.append({"n": n, "start": i, "end": min(i + size, len(text))})
        if i + size >= len(text):
            break
        i += max(1, size - overlap)
    return bounds


def _sig(item):
    """Stable identity for an extracted item.

    Matches the signature _merge_extraction_into uses to deduplicate, so an item
    that survives the merge can be looked up by the same key it was recorded under.
    """
    if isinstance(item, (dict, list)):
        return json.dumps(item, sort_keys=True, default=str)
    return str(item)


_PLACEHOLDER = "[not present in this document]"


def _record_provenance(data, chunk_no, prov, chunk_text=None, chunk_start=0, spans=None):
    """Tag every fact in one chunk's extraction with where it came from.

    prov  — {signature: {chunk numbers}}, the coarse link the tree is built on.
    spans — {signature: [{chunk, start, end, how}]}, the source offsets the
            paper calls for, so any value on a subject traces back to the passage.
    """
    if isinstance(data, list):
        for x in data:
            _record_provenance(x, chunk_no, prov, chunk_text, chunk_start, spans)
        return

    if isinstance(data, dict):
        key = _sig(data)
    elif isinstance(data, str) and data.strip() and data != _PLACEHOLDER:
        key = _sig(data)
    else:
        return

    prov.setdefault(key, set()).add(chunk_no)
    if spans is not None and chunk_text is not None:
        s, e, how = _locate_in_chunk(data, chunk_text, chunk_start, _sent_cache(chunk_text))
        rec = {"chunk": chunk_no, "start": s, "end": e, "how": how}
        bucket = spans.setdefault(key, [])
        if rec not in bucket:
            bucket.append(rec)

    if isinstance(data, dict):
        for v in data.values():
            _record_provenance(v, chunk_no, prov, chunk_text, chunk_start, spans)


def _merge_paper(base, new_data, template):
    """Merge one chunk's extraction into the running record, per the paper (§3.3).

    "The model returns each field as either a list or a single value, and merging
    handles the two cases differently. List fields, such as a character's actions,
    accumulate entries with exact duplicates removed. Single-value fields, such as
    a character's role, keep the first non-empty value, on the assumption that a
    document defines a fact where it first states it; resolving conflicting values
    more comprehensively is left to future work."

    So a conflicting scalar keeps the first value and the later one is dropped.
    An earlier revision promoted the pair to a list, which quietly changes a
    single-value field into a list field and is the "more comprehensive"
    resolution §3.3 defers rather than the rule it states.
    """
    if not new_data or not isinstance(new_data, dict):
        return
    for key, value in new_data.items():
        if not value or value == _PLACEHOLDER or value == {}:
            continue
        if key not in base or not base[key] or base[key] == _PLACEHOLDER:
            base[key] = value
            continue

        cur = base[key]
        if isinstance(cur, list) and isinstance(value, list):
            seen = {_sig(x) for x in cur}
            for item in value:
                if _sig(item) not in seen:
                    cur.append(item)
                    seen.add(_sig(item))
        elif isinstance(cur, dict) and isinstance(value, dict):
            _merge_paper(cur, value, template)
        elif isinstance(cur, list):
            if _sig(value) not in {_sig(x) for x in cur}:
                cur.append(value)
        elif isinstance(value, list):
            base[key] = [cur] + [v for v in value if _sig(v) != _sig(cur)]
        # Conflicting scalars: the first non-empty value stands; the later one is
        # dropped. A document defines a fact where it first states it (§3.3).


_SENT_CACHE = {}


def _sent_cache(chunk_text):
    """Sentence spans for a chunk, computed once (thousands of facts share one chunk)."""
    key = (len(chunk_text), chunk_text[:64])
    if key not in _SENT_CACHE:
        _SENT_CACHE.clear()          # only ever need the chunk being processed
        _SENT_CACHE[key] = _sentence_spans(chunk_text)
    return _SENT_CACHE[key]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _sentence_spans(text, min_len=20):
    """(start, end, token set) for each sentence-ish span of a chunk."""
    spans, pos = [], 0
    for piece in _SENT_SPLIT.split(text):
        i = text.find(piece, pos)
        if i < 0:
            continue
        pos = i + len(piece)
        if len(piece) >= min_len:
            spans.append((i, pos, _tokens(piece)))
    return spans


def _locate_in_chunk(item, chunk_text, chunk_start, sent_spans=None):
    """Char offsets in the source document of the passage that produced this fact.

    The paper (§3.3) tags each extracted field with the source offset of the
    passage it came from. Exact match pins it when the model quoted the source.
    It usually does not — extraction paraphrases and normalizes — so the fallback
    is the best-overlapping sentence window rather than the whole chunk, which at
    30,000 chars is not a passage in any useful sense.

    Returns (start, end, how) where how is "verbatim", "window" or "chunk".
    """
    # Placeholders are the model saying the document is silent — locating them in
    # the source would invent provenance for a non-fact.
    def _usable(v):
        return isinstance(v, str) and len(v) > 12 and v != _PLACEHOLDER

    values = []
    if isinstance(item, dict):
        values = [v for v in item.values() if _usable(v)]
    elif _usable(item):
        values = [item]
    if not values:
        return chunk_start, chunk_start + len(chunk_text), "chunk"

    # 1. verbatim quote
    for v in sorted(values, key=len, reverse=True):
        i = chunk_text.find(v)
        if i >= 0:
            return chunk_start + i, chunk_start + i + len(v), "verbatim"

    # 2. best-overlapping sentence window
    if sent_spans is None:
        sent_spans = _sentence_spans(chunk_text)
    want = _tokens(" ".join(values))
    if want and sent_spans:
        best, best_score = None, 0.0
        for s, e, toks in sent_spans:
            if not toks:
                continue
            hit = len(want & toks)
            if not hit:
                continue
            score = hit / len(want)          # how much of the fact this sentence carries
            if score > best_score:
                best, best_score = (s, e), score
        # A single incidental word in common is noise, not a location.
        if best and best_score >= 0.34:
            return chunk_start + best[0], chunk_start + best[1], "window"

    # 3. nothing better than the chunk itself
    return chunk_start, chunk_start + len(chunk_text), "chunk"


def _fact_text(item):
    """The words of a fact, for matching it back against the source text."""
    if isinstance(item, dict):
        parts = []
        for v in item.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v if isinstance(x, (str, int, float)))
        return " ".join(parts)
    return str(item)


def _tokens(text):
    return set(re.findall(r"[a-zA-Z0-9]{3,}", str(text).lower()))


class ProvenanceResolver:
    """Links a fact to the chunk it came from, and records how the link was made.

    Exact signature lookup is the primary path, but it is not complete: pass 2
    and _merge_extraction_into reshape items as they merge, so a fact that ends
    up on a page may no longer carry the signature it was recorded under. Those
    facts would otherwise silently lose their chunk link.

    So unmatched facts fall back to locating themselves in the source document —
    against chunk embeddings when an embedder is supplied, else lexical overlap
    against the chunk text. Every resolution is counted by method, and the counts
    are written to _provenance.json, so a page's chunk list can be read knowing
    how much of it is exact and how much is inferred.
    """

    def __init__(self, prov=None, chunk_texts=None, embed_fn=None, min_overlap=2,
                 spans=None, bounds=None):
        self.prov = prov or {}
        self.chunk_texts = chunk_texts or []
        self.embed_fn = embed_fn
        self.min_overlap = min_overlap
        # spans: {signature: [{chunk, start, end, how}]}, the source offsets
        # recorded during extraction. bounds: the chunk boundary table, so a fact
        # resolved only to a chunk number still yields that chunk's offsets.
        self.spans = spans or {}
        self.bounds = {b["n"]: b for b in (bounds or [])}
        self.stats = {"exact": 0, "embedding": 0, "lexical": 0, "unresolved": 0}
        self.unresolved = []                       # sample of facts we could not place
        self._chunk_tokens = [_tokens(c) for c in self.chunk_texts]
        self._chunk_vecs = None
        if self.embed_fn and self.chunk_texts:
            try:
                self._chunk_vecs = self.embed_fn(self.chunk_texts)
            except Exception as e:                 # embedder optional — never fail a build
                log.warning(f"  provenance: chunk embedding unavailable ({e}); "
                            f"falling back to lexical matching")
                self._chunk_vecs = None

    # -- fallbacks ----------------------------------------------------------
    def _by_embedding(self, text):
        if self._chunk_vecs is None:
            return None
        try:
            v = self.embed_fn([text])[0]
            best, best_sim = None, 0.0
            for i, cv in enumerate(self._chunk_vecs):
                num = sum(a * b for a, b in zip(v, cv))
                den = (sum(a * a for a in v) ** 0.5) * (sum(b * b for b in cv) ** 0.5)
                sim = num / den if den else 0.0
                if sim > best_sim:
                    best, best_sim = i + 1, sim
            return best if best_sim >= 0.35 else None
        except Exception:
            return None

    def _by_lexical(self, text):
        t = _tokens(text)
        if not t:
            return None
        best, best_n = None, 0
        for i, ct in enumerate(self._chunk_tokens):
            n = len(t & ct)
            if n > best_n:
                best, best_n = i + 1, n
        return best if best_n >= self.min_overlap else None

    def _resolve(self, item, found, count=True):
        """Place one fact in the source. `count=False` for a second pass over
        content already counted, so spans_for does not double the link stats."""
        def tally(k):
            if count:
                self.stats[k] += 1

        hit = self.prov.get(_sig(item))
        if hit:
            tally("exact")
            found.update(hit)
            return
        text = _fact_text(item)
        if not text.strip():
            return
        n = self._by_embedding(text)
        if n:
            tally("embedding")
            found.add(n)
            return
        n = self._by_lexical(text)
        if n:
            tally("lexical")
            found.add(n)
            return
        tally("unresolved")
        if count and len(self.unresolved) < 25:
            self.unresolved.append(_one_line(text, 160))

    # -- public -------------------------------------------------------------
    def _walk(self, content, visit):
        def walk(d):
            if isinstance(d, list):
                for x in d:
                    walk(x)
            elif isinstance(d, dict):
                visit(d)
                for v in d.values():
                    if isinstance(v, (list, dict)):
                        walk(v)
            elif isinstance(d, str) and d.strip() and d != _PLACEHOLDER:
                visit(d)

        walk(content)

    def chunks_for(self, content):
        found = set()
        self._walk(content, lambda item: self._resolve(item, found))
        return sorted(found)

    def spans_for(self, content, limit=None):
        """Evidence spans for the facts on one page (paper §3.1, §3.3).

        Every field is linked to the passage that produced it, so a page can
        point back into the source document rather than only naming a 30,000-char
        chunk. A fact whose exact signature survived extraction has its recorded
        offsets; one reshaped by the merge resolves to a chunk and takes that
        chunk's bounds, which is weaker but still a real location. Overlapping
        spans are coalesced so a page's evidence reads as passages, not fragments.

        Returns [{start, end, chunk, how}] in document order.
        """
        raw = []

        def visit(item):
            recs = self.spans.get(_sig(item))
            if recs:
                raw.extend(recs)
                return
            # No exact span: fall back to whatever chunk the fact resolves to.
            found = set()
            self._resolve(item, found, count=False)
            for n in found:
                b = self.bounds.get(n)
                if b:
                    raw.append({"chunk": n, "start": b["start"], "end": b["end"],
                                "how": "chunk"})

        self._walk(content, visit)
        return _coalesce_spans(raw, limit=limit)

    def report(self):
        total = sum(self.stats.values())
        return {
            "resolved_by": dict(self.stats),
            "total_facts_linked": total,
            "exact_pct": round(100 * self.stats["exact"] / total, 1) if total else 0.0,
            "fallback_pct": round(
                100 * (self.stats["embedding"] + self.stats["lexical"]) / total, 1
            ) if total else 0.0,
            "unresolved_sample": self.unresolved,
        }


def _chunks_for(content, prov):
    """Which chunks contributed the facts on this page.

    `prov` is a ProvenanceResolver, or a plain {signature: [chunks]} dict for
    exact-only lookup (no fallback), or None.
    """
    if not prov:
        return []
    if isinstance(prov, ProvenanceResolver):
        return prov.chunks_for(content)
    return ProvenanceResolver(prov).chunks_for(content)


# A span narrower than this is a fragment, not a passage: widen it symmetrically
# so the text handed back reads as prose. A verbatim hit on a four-word field
# value is precise but unreadable on its own.
MIN_SPAN_CHARS = 400


def _coalesce_spans(spans, limit=None, min_chars=MIN_SPAN_CHARS):
    """Merge overlapping evidence spans into passages, in document order.

    Extraction records one span per fact, and a subject page carries many facts
    from the same neighbourhood, so the raw list is dozens of overlapping
    fragments. Widening each to a readable minimum and then merging what touches
    gives the passages the fallback actually reads (§3.5).
    """
    if not spans:
        return []
    widened = []
    for s in spans:
        start, end = int(s.get("start", 0)), int(s.get("end", 0))
        if end <= start:
            continue
        if end - start < min_chars:                  # widen symmetrically
            pad = (min_chars - (end - start)) // 2
            start, end = max(0, start - pad), end + pad
        widened.append((start, end, s.get("chunk"), s.get("how", "chunk")))
    if not widened:
        return []

    widened.sort(key=lambda t: (t[0], t[1]))
    merged = [list(widened[0])]
    for start, end, chunk, how in widened[1:]:
        cur = merged[-1]
        if start <= cur[1]:                          # touching or overlapping
            cur[1] = max(cur[1], end)
            # "verbatim" and "window" are passage-level; keep the strongest.
            if cur[3] == "chunk" and how != "chunk":
                cur[3] = how
        else:
            merged.append([start, end, chunk, how])

    out = [{"start": s, "end": e, "chunk": c, "how": h} for s, e, c, h in merged]
    if limit is not None:
        # Keep the narrowest spans: they are the ones extraction could actually
        # pin, and a whole-chunk span crowds out several real passages.
        out = sorted(sorted(out, key=lambda d: d["end"] - d["start"])[:limit],
                     key=lambda d: d["start"])
    return out


def _spans_for(content, prov, limit=None):
    """Evidence spans for one page, or [] when provenance is unavailable."""
    if not isinstance(prov, ProvenanceResolver):
        return []
    return prov.spans_for(content, limit=limit)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: DISCOVER — per-document schema (1 LLM call, domain-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

DISCOVER_PROMPT = """\
You are analyzing one passage of a document to help design a structured \
knowledge base (wiki) for the whole document.

Read the passage below and propose the categories of information it contains, \
the fields worth recording under each, and a short partial overview of what \
this passage establishes about the document.

Requirements:
- Propose as many categories as this passage naturally requires. Each should \
  capture a distinct type of information — do not merge unrelated topics, but \
  do not split artificially either.
- Category names must be generic and descriptive, so that the same name would \
  fit the same kind of content elsewhere in the document.
- Fields are the attributes worth recording for each subject in that category \
  (for a character: role, actions, relationships).
- Propose only what this passage actually shows. Another passage will be read \
  separately; do not guess at the rest of the document.
- The overview is 2-4 sentences on what this passage establishes.

Return ONLY valid JSON:
{{
  "document_type": "your best guess at what kind of document this is",
  "overview": "2-4 sentences on what this passage establishes",
  "sections": [
    {{
      "name": "category name",
      "description": "what this category captures",
      "subsections": [
        {{"name": "field or sub-category name", "description": "what goes here"}}
      ]
    }}
  ]
}}

PASSAGE ({chunk_num}/{total_chunks}):
{excerpt}

JSON:"""


def _norm_entry(name):
    """Identity for schema merging: an entry is a repeat if its content matches."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _merge_schema_proposals(proposals):
    """Merge per-chunk schema proposals into one schema (paper §3.2).

    Fixed rules, no LLM call and no ranking: repeated entries are deduplicated by
    content, and the first non-empty description is kept. Categories and their
    subsections both follow the rule, and first-seen order is preserved so the
    merge is deterministic in the chunk order discovery read.

    Near-duplicates that are not exact repeats (`Characters` vs `Principal
    Characters`) survive as separate categories. Reconciling those is the open
    question §3.2 records; compilation tolerates it because _find_section matches
    a section name to the compiled data fuzzily.
    """
    sections, order = {}, []
    doc_type, overviews = None, []

    for p in proposals:
        if not isinstance(p, dict):
            continue
        if not doc_type and p.get("document_type"):
            doc_type = p["document_type"]              # first non-empty wins
        if p.get("overview"):
            overviews.append(str(p["overview"]).strip())

        for s in p.get("sections") or []:
            if not isinstance(s, dict) or not s.get("name"):
                continue
            key = _norm_entry(s["name"])
            if not key:
                continue
            if key not in sections:
                sections[key] = {"name": str(s["name"]).strip(),
                                 "description": "", "subsections": {}, "_sub_order": []}
                order.append(key)
            tgt = sections[key]
            if not tgt["description"] and s.get("description"):
                tgt["description"] = str(s["description"]).strip()

            for sub in s.get("subsections") or []:
                if isinstance(sub, str):
                    sub = {"name": sub}
                if not isinstance(sub, dict) or not sub.get("name"):
                    continue
                sk = _norm_entry(sub["name"])
                if not sk:
                    continue
                if sk not in tgt["subsections"]:
                    tgt["subsections"][sk] = {"name": str(sub["name"]).strip(),
                                              "description": ""}
                    tgt["_sub_order"].append(sk)
                if not tgt["subsections"][sk]["description"] and sub.get("description"):
                    tgt["subsections"][sk]["description"] = str(sub["description"]).strip()

    if not order:
        return None
    return {
        "corpus_type": doc_type or "unknown",
        "sections": [
            {"name": sections[k]["name"], "description": sections[k]["description"],
             "subsections": [sections[k]["subsections"][sk]
                             for sk in sections[k]["_sub_order"]]}
            for k in order
        ],
        # Seeds the root. Provisional by construction: every part of it was
        # written against one chunk in isolation, so compilation rewrites it
        # against the full merged record once extraction completes (§3.3).
        "overview": "\n\n".join(overviews),
    }


def discover_schema(doc_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Propose the document's schema before any content is extracted (paper §3.2).

    One LLM call per chunk, each proposing the categories it sees, the fields
    worth recording under each, and a partial overview. The proposals are merged
    into a single schema by _merge_schema_proposals.

    Reading only the opening of the document instead would fix the schema from
    whatever the first pages happen to contain: on a screenplay the first 12,000
    characters are roughly the first 4% of the text, so any category that only
    appears later has no page to be filed under and is lost for every later
    query. Discovery reuses compilation's chunking (§3.3) so both phases see the
    same passages.
    """
    raw = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_document(raw)
    if len(text) < 200:
        return None

    bounds = chunk_bounds(text, size=chunk_size, overlap=chunk_overlap)
    n = len(bounds)
    proposals = []
    for i, b in enumerate(bounds):
        prompt = DISCOVER_PROMPT.format(
            excerpt=text[b["start"]:b["end"]], chunk_num=i + 1, total_chunks=n,
        )
        result = parse_json_with_retry(prompt, max_tokens=16384, phase="discover")
        if result and result.get("sections"):
            proposals.append(result)
            log.info(f"    Discover chunk {i+1}/{n}: "
                     f"{len(result['sections'])} categories proposed")

    template = _merge_schema_proposals(proposals)
    if not template:
        log.warning(f"  Schema discovery failed for {Path(doc_path).stem}")
        return None

    log.info(f"  Discovered {len(template['sections'])} categories from {n} chunks "
             f"({len(proposals)} proposals merged, type: {template['corpus_type']})")
    return template


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: EXTRACT — 2-pass chunked extraction (reuses cold_start prompts)
# ═════════════════════════════════════════════════════════════════════════════

def _template_description(template):
    lines = []
    for section in template["sections"]:
        lines.append(f"\n### {section['name']}")
        lines.append(f"  Purpose: {section.get('description', '')}")
        for sub in section.get("subsections", []):
            lines.append(f"  - {sub['name']}: {sub.get('description', '')}")
    return "\n".join(lines)


def extract(doc_path, template, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            return_provenance=False):
    """2-pass extraction → merged dict.

    With return_provenance=True returns (merged, provenance, bounds) instead:
      provenance — {fact signature: [chunk numbers it was extracted from]}
      bounds     — the chunk boundary table, saved so insert can reuse it

    The default return shape is unchanged, so callers that just want the facts
    (packed.py) are unaffected.
    """
    raw = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_document(raw)
    if len(text) < 200:
        return (None, {}, []) if return_provenance else None

    bounds = chunk_bounds(text, size=chunk_size, overlap=chunk_overlap)
    chunks = [text[b["start"]:b["end"]] for b in bounds]
    n = len(chunks)
    tmpl = _template_description(template)
    log.info(f"  {Path(doc_path).stem}: {len(text):,} chars, {n} chunks")

    merged, prov, spans = {}, {}, {}
    for i, chunk in enumerate(chunks):
        prompt = COMPILE_PASS1_PROMPT.format(
            template_description=tmpl, chunk=chunk,
            chunk_num=i + 1, total_chunks=n,
        )
        result = parse_json_with_retry(prompt, max_tokens=16384, phase="compile_p1")
        if result:
            _record_provenance(result, i + 1, prov, chunk, bounds[i]["start"], spans)
            _merge_paper(merged, result, template)
            log.info(f"    Pass1 chunk {i+1}/{n}: OK")

    if not merged:
        return (None, {}, bounds) if return_provenance else None

    existing_json = json.dumps(merged, indent=1, default=str)
    if len(existing_json) > 15000:
        existing_json = existing_json[:15000] + "\n... [truncated]"

    for i, chunk in enumerate(chunks):
        prompt = COMPILE_PASS2_PROMPT.format(
            existing=existing_json, chunk=chunk,
            chunk_num=i + 1, total_chunks=n,
        )
        result = parse_json_with_retry(prompt, max_tokens=8192, phase="compile_p2")
        if result and result != {}:
            _record_provenance(result, i + 1, prov, chunk, bounds[i]["start"], spans)
            _merge_paper(merged, result, template)
            log.info(f"    Pass2 chunk {i+1}/{n}: found new info")

    if return_provenance:
        prov_out = {k: sorted(v) for k, v in prov.items()}
        _LAST_SPANS.clear()
        _LAST_SPANS.update(spans)
        return merged, prov_out, bounds
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: ORGANIZE — recursive tree build
# ═════════════════════════════════════════════════════════════════════════════

def _one_line(text, max_len=SUMMARY_LINE_SIZE):
    line = " ".join(str(text).split())
    if len(line) > max_len:
        line = line[:max_len - 3] + "..."
    return line


_TITLE_KEYS = ("name", "title", "heading", "subject", "term", "label", "entity",
               "character", "party", "clause", "event")


def _item_title(item, fallback=""):
    """A human-readable title for an extracted item.

    Discovery names the fields, so their case and wording are the model's, not
    ours: a character subject arrives as {'Name': ..., 'Age': ...}. A
    case-sensitive lookup for 'name' misses that and falls through to str(item),
    which puts a Python dict repr in the page's H1 and in its filename. Match the
    title key case-insensitively, then settle for the first usable string value
    before giving up on the item's own words.
    """
    if isinstance(item, str):
        return _one_line(item, 60)
    if not isinstance(item, dict):
        return _one_line(str(item), 60) or fallback

    lowered = {str(k).strip().lower(): v for k, v in item.items()}
    for key in _TITLE_KEYS:
        v = lowered.get(key)
        if isinstance(v, str) and v.strip() and v != _PLACEHOLDER:
            return _one_line(v, 60)
    for k, v in item.items():
        if isinstance(v, str) and v.strip() and v != _PLACEHOLDER:
            return _one_line(v, 60)
        if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
            return _one_line(v[0], 60)
    return fallback


def _evidence_footer(spans):
    """Where this page's facts came from, written into the page itself.

    Figure 1b has every subject pointing back to its evidence spans in the source
    document, and §3.1 requires the whole index to be human-readable plain text.
    Offsets that live only in a sidecar JSON satisfy neither, so each page states
    its own provenance as character ranges into the cleaned source.
    """
    if not spans:
        return ""
    lines = ["", "---", "", "**Evidence spans** (character offsets into the source document):"]
    for s in spans:
        chunk = f", chunk {s['chunk']}" if s.get("chunk") is not None else ""
        lines.append(f"- {s['start']}–{s['end']}{chunk} ({s.get('how', 'chunk')})")
    return "\n".join(lines) + "\n"


def _field_span_note(value, prov):
    """The evidence span for one field, rendered inline (paper §3.1, §3.3).

    "Each subject records a small set of fields extracted directly from the
    source, and every field is linked to the evidence span that produced it."
    Extraction records a span per field value, so the link belongs on the field
    line, not only in a page-level footer: a subject page names where each
    individual fact came from.
    """
    if not isinstance(prov, ProvenanceResolver):
        return ""
    values = value if isinstance(value, list) else [value]
    recs = []
    for v in values:
        recs.extend(prov.spans.get(_sig(v)) or [])
    spans = _coalesce_spans(recs, limit=2, min_chars=0)
    if not spans:
        return ""
    return "  `[" + ", ".join(f"{s['start']}–{s['end']}" for s in spans) + "]`"


def _render_item(item, prov=None):
    """Render a single extracted item as clean markdown (no raw JSON)."""
    if isinstance(item, str):
        return f"- {item}{_field_span_note(item, prov)}"
    if isinstance(item, dict):
        name = _item_title(item)
        lines = []
        if name:
            lines.append(f"### {name}")
        for k, v in item.items():
            if v is None or v == "" or v == [] or v == {} or v == _PLACEHOLDER:
                continue
            note = _field_span_note(v, prov)
            if isinstance(v, list):
                v = "; ".join(_render_value(x) for x in v)
            if isinstance(v, dict):
                v = _render_value(v)
            lines.append(f"**{k}:** {v}{note}")
        return "\n".join(lines) if lines else str(item)
    return str(item)


def _render_value(value):
    """Readable, recursive representation without list limits or repr escapes."""
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_render_value(v)}" for key, v in value.items())
    if isinstance(value, list):
        return "; ".join(_render_value(v) for v in value)
    return str(value)


def _summarize_item(item):
    """One-line summary of an item for branch pages.
    Extracts all string values from any dict shape — never falls back to json.dumps.
    """
    if isinstance(item, str):
        return _one_line(item)
    if isinstance(item, dict):
        # Collect all meaningful string values from the dict
        parts = []
        for _, v in item.items():
            if not v or v == "[not present in this document]":
                continue
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for x in v[:3]:
                    if isinstance(x, str):
                        parts.append(x)
        return _one_line(" — ".join(parts)) if parts else _one_line(str(item))
    return _one_line(str(item))


MAX_SPANS_PER_PAGE = 8

# A rendered record can span several pages. These certificates let validation
# reconstruct its exact body after splitting and after balancing.
_RENDER_RECORDS = {}
_RENDER_ATOMS = {}
LINK_RESERVE = 440


def _atoms(content):
    if isinstance(content, dict):
        for value in content.values():
            yield from _atoms(value)
    elif isinstance(content, list):
        for value in content:
            yield from _atoms(value)
    elif content is not None and str(content).strip() and content != _PLACEHOLDER:
        yield str(content)


def _unique_page_path(wiki_dir, fpath, body, manifest):
    if Path(fpath).is_absolute() or ".." in Path(fpath).parts:
        raise ValueError(f"Index page path must remain relative: {fpath}")
    paths = {entry["path"] for entry in manifest}
    if fpath not in paths:
        return fpath
    path = Path(fpath)
    suffix = hashlib.sha256(body.encode()).hexdigest()[:12]
    candidate = str(path.with_name(f"{path.stem}__{suffix}{path.suffix}"))
    serial = 2
    while candidate in paths:
        candidate = str(path.with_name(f"{path.stem}__{suffix}_{serial}{path.suffix}"))
        serial += 1
    return candidate


def _directory_name(name, used):
    """Allocate sibling directories before recursion so parent paths stay exact."""
    digest = hashlib.sha256(str(name).encode()).hexdigest()[:12]
    base = safe_filename(str(name)) or f"section_{digest}"
    candidate = base if base not in used else f"{base}__{digest}"
    serial = 2
    while candidate in used:
        candidate = f"{base}__{digest}_{serial}"
        serial += 1
    used.add(candidate)
    return candidate


def _compact_branch(body, max_chars):
    """Shorten descriptions evenly, keeping every routing bullet represented."""
    lines = body.splitlines()
    bullets = [line for line in lines if line.startswith("- ")]
    if not bullets:
        return _one_line(body, max_chars)
    header = next((line for line in lines if line.startswith("# ")), "# Contents")
    head = _one_line(header, 150) + "\n\n"
    allowance = max(1, (max_chars - len(head)) // len(bullets) - 1)
    return head + "\n".join(_one_line(line, allowance) for line in bullets) + "\n"


def _bounded_spans(content, prov):
    spans = _spans_for(content, prov)
    limit = max((bound["end"] for bound in prov.bounds.values()), default=0) if isinstance(prov, ProvenanceResolver) else 0
    if limit:
        for span in spans:
            span["start"] = max(0, min(span["start"], limit))
            span["end"] = min(span["end"], limit)
    return [span for span in spans if span["end"] > span["start"]]


def _write_page(wiki_dir, fpath, body, *, level, parent, summary, content, prov,
                manifest):
    """Write one index page and record its manifest entry.

    Every page is capped at PAGE_SIZE (paper §3.2, §3.3), and a leaf carries the
    evidence spans for the facts on it, both in its manifest entry and in its own
    text. The footer is written inside the cap rather than on top of it, so the
    capacity invariant holds for the page a query actually reads.
    """
    wiki_dir = Path(wiki_dir)
    fpath = _unique_page_path(wiki_dir, fpath, body, manifest)
    spans = _bounded_spans(content, prov) if level == "leaf" else []
    footer = _evidence_footer(spans[:MAX_SPANS_PER_PAGE])
    if len(spans) > MAX_SPANS_PER_PAGE:
        footer += "Additional evidence spans are retained in the manifest.\n"
    room = PAGE_SIZE - len(footer) - (LINK_RESERVE if level == "leaf" else 0)
    entry = {
        "path": fpath, "level": level, "parent": parent,
        # The routing descriptor (§3.3). Held to S so that a routing step reading
        # f of them stays inside the routing budget C, which is what makes
        # f = C/S the real fanout bound rather than an arithmetic coincidence.
        "summary": _one_line(summary, DESCRIPTOR_CHARS),
        "chunks": _chunks_for(content, prov),
    }
    if level == "leaf":
        entry["spans"] = spans
        entry["links"] = []              # filled by _link_subjects once all pages exist
        record_id = hashlib.sha256((fpath + "\0" + body).encode()).hexdigest()
        key = str(wiki_dir.resolve())
        _RENDER_ATOMS.setdefault(key, Counter()).update(_atoms(content))
        certificate = {"id": record_id, "sha256": hashlib.sha256(body.encode()).hexdigest(),
                       "chars": len(body), "segments": []}
        _RENDER_RECORDS.setdefault(key, []).append(certificate)
        if len(body) > room:
            # Preserve every original character. Prefer line boundaries, and
            # split even a single long field if it cannot fit in one page.
            parts, offset = [], 0
            fragment_room = room - 160
            if fragment_room < 200:
                raise ValueError(f"Evidence footer leaves no fact capacity: {fpath}")
            while offset < len(body):
                end = min(len(body), offset + fragment_room)
                if end < len(body):
                    boundary = body.rfind("\n", offset, end)
                    if boundary > offset + fragment_room // 2:
                        end = boundary + 1
                parts.append((offset, end, body[offset:end]))
                offset = end
            folder = str(Path(fpath).with_suffix("")) + "__parts"
            paths = [f"{folder}/part_{n:04d}.md" for n in range(1, len(parts) + 1)]
            entry.update(level="branch", record_id=record_id, preserve_boundary=True,
                         continuations=paths, summary=_one_line(summary, DESCRIPTOR_CHARS))
            manifest.append(entry)
            heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), summary)
            branch_body = f"# {heading}\n\n" + "\n".join(
                f"- Part {n}: {_one_line(fragment, 90)}" for n, (_, _, fragment) in enumerate(parts, 1)) + "\n"
            (wiki_dir / fpath).parent.mkdir(parents=True, exist_ok=True)
            (wiki_dir / fpath).write_text(_compact_branch(branch_body, PAGE_SIZE))
            for n, (start, end, fragment) in enumerate(parts):
                prefix = f"# {_one_line(heading, 70)}\n\nPart {n + 1}/{len(parts)}\n\n"
                path = paths[n]
                text = prefix + fragment + footer
                assert len(text) <= PAGE_SIZE - LINK_RESERVE
                (wiki_dir / path).parent.mkdir(parents=True, exist_ok=True)
                (wiki_dir / path).write_text(text)
                manifest.append({"path": path, "level": "leaf", "parent": fpath,
                                 "summary": _one_line(f"{heading}: {fragment}", DESCRIPTOR_CHARS),
                                 "chunks": entry["chunks"], "spans": spans, "links": [],
                                 "record_id": record_id, "continuation_of": fpath,
                                 "continuation_index": n + 1,
                                 "continuations": paths[max(0, n - 1):n] + paths[n + 1:n + 2]})
                certificate["segments"].append({"page": path, "start": len(prefix), "end": len(prefix) + len(fragment)})
            return entry
        entry["record_id"] = record_id
        certificate["segments"].append({"page": fpath, "start": 0, "end": len(body)})
    elif len(body) > room:
        body = _compact_branch(body, room)
    (wiki_dir / fpath).parent.mkdir(parents=True, exist_ok=True)
    (wiki_dir / fpath).write_text(body + footer)
    manifest.append(entry)
    return entry


def build_subtree(wiki_dir, rel_dir, name, content, manifest, parent_path, prov=None):
    """Recursively build wiki pages. Splits when content > PAGE_SIZE.

    content: list of items, dict of sub-sections, or string.
    Depth is unlimited — the tree grows as deep as content requires.

    prov: fact-provenance table from extract(). When given, every page records
    the chunk numbers and evidence spans its facts came from.
    """
    wiki_dir = Path(wiki_dir)
    dir_path = wiki_dir / rel_dir
    dir_path.mkdir(parents=True, exist_ok=True)

    if isinstance(content, list):
        _build_list_subtree(wiki_dir, rel_dir, name, content, manifest, parent_path, prov)
    elif isinstance(content, dict):
        _build_dict_subtree(wiki_dir, rel_dir, name, content, manifest, parent_path, prov)
    elif content is not None:
        # Simple string content → single page
        _write_page(wiki_dir, rel_dir + "/" + safe_filename(name) + ".md",
                    f"# {name}\n\n{content}\n",
                    level="leaf", parent=parent_path, summary=_one_line(content),
                    content=content, prov=prov, manifest=manifest)


def _build_list_subtree(wiki_dir, rel_dir, name, items, manifest, parent_path, prov=None):
    """Build pages from a list of items. Splits recursively if too many."""
    # Filter out empty/placeholder items
    items = [x for x in items if x is not None and x != "" and x != [] and x != {} and str(x) != _PLACEHOLDER]
    if not items:
        return

    # Try rendering everything in one page
    rendered = f"# {name}\n\n" + "\n\n".join(_render_item(x, prov) for x in items) + "\n"

    if len(rendered) <= PAGE_SIZE:
        # Fits in one page → leaf
        _write_page(wiki_dir, rel_dir + "/" + safe_filename(name) + ".md", rendered,
                    level="leaf", parent=parent_path,
                    summary=f"{name}: {len(items)} items",
                    content=items, prov=prov, manifest=manifest)
        return

    # Too big → branch with children
    # If individual items are small enough to be leaves, write them individually
    if len(items) <= MAX_CHILDREN:
        # Each item gets its own leaf page, write a summary page
        summary_path = rel_dir + "/_summary.md"
        child_summaries = []
        for i, item in enumerate(items):
            item_name = _item_title(item, fallback=f"{name} item {i+1}")
            child_path = rel_dir + "/" + safe_filename(str(item_name)) + ".md"
            summ = _summarize_item(item)
            child_summaries.append(summ)
            _write_page(wiki_dir, child_path,
                        f"# {item_name}\n\n{_render_item(item, prov)}\n",
                        level="leaf", parent=summary_path, summary=summ,
                        content=item, prov=prov, manifest=manifest)

        # Write summary (branch page)
        summary_text = f"# {name}\n\nTotal: {len(items)}\n\n"
        summary_text += "\n".join(f"- {s}" for s in child_summaries) + "\n"
        _write_page(wiki_dir, summary_path, summary_text,
                    level="branch", parent=parent_path,
                    summary=f"{name}: {len(items)} items",
                    content=items, prov=prov, manifest=manifest)
    else:
        # More items than MAX_CHILDREN → group into sub-branches, recurse.
        # Groups are CONTIGUOUS: writing "keeps children in the order extraction
        # produced them" (§3.3), so a group holds a run of neighbouring subjects.
        # Dealing them round-robin (items[i::n]) instead would scatter every run
        # across every group, and since the balancing pass reorders by embedding
        # rather than by arrival, nothing downstream would put them back.
        summary_path = rel_dir + "/_summary.md"
        n_groups = (len(items) + MAX_CHILDREN - 1) // MAX_CHILDREN
        size = (len(items) + n_groups - 1) // n_groups
        groups = [items[i:i + size] for i in range(0, len(items), size)]

        group_summaries = []
        for gi, group in enumerate(groups):
            # Name the group by its first/last items
            first = _item_title(group[0], fallback=f"item {gi * size + 1}")
            last = _item_title(group[-1], fallback="")
            group_name = f"{first} — {last}" if last and last != first else f"{name} (part {gi+1})"
            sub_dir = rel_dir + "/" + safe_filename(f"group_{gi+1}")

            # Recurse — this group might itself need splitting
            build_subtree(wiki_dir, sub_dir, group_name, group, manifest, summary_path, prov)
            group_summaries.append(f"{group_name} ({len(group)} items)")

        # Write summary
        summary_text = f"# {name}\n\nTotal: {len(items)} items in {len(groups)} groups\n\n"
        summary_text += "\n".join(f"- {s}" for s in group_summaries) + "\n"
        _write_page(wiki_dir, summary_path, summary_text,
                    level="branch", parent=parent_path,
                    summary=f"{name}: {len(items)} items across {len(groups)} groups",
                    content=items, prov=prov, manifest=manifest)


def _build_dict_subtree(wiki_dir, rel_dir, name, data, manifest, parent_path, prov=None):
    """Build pages from a dict (subsections). Each key becomes a child."""
    # Filter out empty/placeholder values
    filtered = {k: v for k, v in data.items()
                if v is not None and v != "" and v != [] and v != {} and str(v) != _PLACEHOLDER}
    if not filtered:
        return

    # Try rendering as a single page
    rendered = f"# {name}\n\n"
    for k, v in filtered.items():
        if isinstance(v, list):
            rendered += f"## {k}\n" + "\n".join(_render_item(x, prov) for x in v) + "\n\n"
        elif isinstance(v, dict):
            rendered += f"## {k}\n" + _render_item(v, prov) + "\n\n"
        else:
            rendered += f"## {k}\n{v}\n\n"

    if len(rendered) <= PAGE_SIZE:
        # Fits in one page → leaf
        _write_page(wiki_dir, rel_dir + "/" + safe_filename(name) + ".md", rendered,
                    level="leaf", parent=parent_path,
                    summary=f"{name}: {', '.join(filtered.keys())}",
                    content=filtered, prov=prov, manifest=manifest)
        return

    # Too big → branch: each key becomes a child subtree
    summary_path = rel_dir + "/_summary.md"
    child_summaries = []
    sibling_dirs = set()
    for key, value in filtered.items():
        sub_dir = rel_dir + "/" + _directory_name(key, sibling_dirs)
        build_subtree(wiki_dir, sub_dir, key, value, manifest, summary_path, prov)
        child_summaries.append(_one_line(key))

    summary_text = f"# {name}\n\n{len(filtered)} sub-sections:\n\n"
    summary_text += "\n".join(f"- {s}" for s in child_summaries) + "\n"
    _write_page(wiki_dir, summary_path, summary_text,
                level="branch", parent=parent_path,
                summary=f"{name}: {len(filtered)} sub-sections",
                content=filtered, prov=prov, manifest=manifest)


# ═════════════════════════════════════════════════════════════════════════════
# LATERAL LINKS — subject ↔ subject, recorded at write time (paper §3.3, §3.5)
# ═════════════════════════════════════════════════════════════════════════════

_EVIDENCE_MARK = "\n---\n\n**Evidence spans**"

# A title matching more than this many pages is a generic word ("Purpose",
# "Notes"), not an identifying one; linking on it would wire the whole category
# together and make the link set useless for routing.
LINK_MAX_DEGREE = 40
LINK_MAX_WORDS = 4                  # longest title n-gram worth looking for


def _page_title(text):
    """The H1 a page was written with."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _ngrams(text, n_max=LINK_MAX_WORDS):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    out = set()
    for i in range(len(words)):
        for n in range(1, n_max + 1):
            if i + n <= len(words):
                out.add(" ".join(words[i:i + n]))
    return out


def _top_category(path):
    return path.split("/", 1)[0]


def _link_subjects(wiki_dir, manifest, max_links=MAX_LINKS):
    """Record the links between related subjects that queries follow laterally.

    A B+ tree links its leaves so a search can continue sideways instead of
    returning to the root; the index's analogue is a subject pointing at the
    subjects it co-occurs with, which may sit under other categories -- a
    character's page linking to the events that character appears in (§3.5).

    The signal is the document's own words, so this costs no LLM call and no
    embedding: subject A links to subject B when B's title appears in A's page
    text. That is exactly the character-in-an-event relation, and it is
    symmetric, so the pair is recorded on both pages. Cross-category links are
    preferred because a same-category neighbour is already one descent away,
    whereas the lateral hop is the only cheap route to another category.

    Links are written into the manifest and into the page text, so the plain-text
    index carries them too.
    """
    wiki_dir = Path(wiki_dir)
    leaves = [e for e in manifest if e["level"] == "leaf"]
    texts, titles = {}, {}
    for e in leaves:
        f = wiki_dir / e["path"]
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        texts[e["path"]] = t
        title = _norm_entry(_page_title(t))
        # One-character and empty titles cannot identify a subject.
        if len(title) >= 3 and len(title.split()) <= LINK_MAX_WORDS:
            titles.setdefault(title, []).append(e["path"])

    # Drop generic titles before matching, so they never become links.
    index = {t: paths for t, paths in titles.items() if len(paths) <= LINK_MAX_DEGREE}

    pairs = set()
    for path, text in texts.items():
        body = text.split(_EVIDENCE_MARK, 1)[0]
        own = _norm_entry(_page_title(text))
        for gram in _ngrams(body) & index.keys():
            if gram == own:
                continue
            for other in index[gram]:
                if other != path:
                    pairs.add((min(path, other), max(path, other)))

    related = {}
    for a, b in pairs:
        related.setdefault(a, []).append(b)
        related.setdefault(b, []).append(a)

    chunks_of = {e["path"]: set(e.get("chunks") or []) for e in leaves}
    n_links = 0
    for e in leaves:
        cands = related.get(e["path"], [])
        if not cands:
            continue
        mine = _top_category(e["path"])
        # Cross-category first, then subjects sharing the most evidence chunks:
        # co-occurrence in the same passage is the strongest signal that the two
        # subjects answer a question together.
        cands.sort(key=lambda p: (_top_category(p) == mine,
                                  -len(chunks_of.get(p, set()) & chunks_of.get(e["path"], set())),
                                  p))
        e["links"] = cands[:max_links]
        n_links += len(e["links"])

        block = "\n**Related subjects:** " + ", ".join(
            _one_line(_page_title(texts[p]) or p, 60) for p in e["links"]) + "\n"
        text = texts[e["path"]]
        body, _, footer = text.partition(_EVIDENCE_MARK)
        footer = (_EVIDENCE_MARK + footer) if footer else ""
        room = PAGE_SIZE - len(block) - len(footer)
        if len(body) > room:
            raise ValueError(f"Link footer would overwrite facts in {e['path']}; rebuild with reserved link space")
        (wiki_dir / e["path"]).write_text(body + block + footer)

    log.info(f"    Lateral links: {n_links} on {sum(1 for e in leaves if e.get('links'))} "
             f"of {len(leaves)} subjects")
    return n_links


# ═════════════════════════════════════════════════════════════════════════════
# BUILD_WIKI: full pipeline (discover → extract → organize)
# ═════════════════════════════════════════════════════════════════════════════

def _build_category(wiki_dir, sdir, sname, sdata, manifest, prov, root_path="index.md"):
    """Build one category and guarantee it exists as an internal node.

    "The tree is therefore at least three levels deep" (§3.1): root, categories,
    subjects. build_subtree only creates a `_summary.md` for a section big enough
    to split, so a compact category used to be written as a single leaf hanging
    directly off the root, giving that category a two-level tree and no node for
    a routing step to read. Here the category node always exists: either
    build_subtree made it, in which case its self-parent is repointed at the
    root, or it is inserted above whatever build_subtree produced.
    """
    cat_path = sdir + "/_summary.md"
    before = len(manifest)
    build_subtree(wiki_dir, sdir, sname, sdata, manifest, cat_path, prov)
    added = manifest[before:]
    if not added:
        return None

    existing = next((e for e in added if e["path"] == cat_path), None)
    if existing is not None:
        existing["parent"] = root_path          # build_subtree set it to itself
        return existing

    children = [e for e in added if e.get("parent") == cat_path]
    body = f"# {sname}\n\n{len(children)} pages:\n\n" + "\n".join(
        f"- {_one_line(e['summary'], DESCRIPTOR_CHARS)}" for e in children) + "\n"
    return _write_page(wiki_dir, cat_path, body, level="branch", parent=root_path,
                       summary=f"{sname}: {_count_items(sdata)} items",
                       content=sdata, prov=prov, manifest=manifest)


def build_wiki(doc_path, output_dir, doc_id, title):
    """Full build: discover schema → extract → recursive tree."""
    # Phase 1: Discover
    template = discover_schema(doc_path)
    if not template:
        log.warning(f"  Discovery failed, skipping {doc_id}")
        return None

    # Phase 2: Extract (with per-fact chunk provenance + the chunk boundaries)
    compiled, provenance, bounds = extract(doc_path, template, return_provenance=True)
    if not compiled:
        log.warning(f"  Extraction failed for {doc_id}")
        return None

    return render_extracted(doc_path, output_dir, doc_id, title, template,
                            compiled, provenance, bounds, dict(_LAST_SPANS))


def render_extracted(doc_path, output_dir, doc_id, title, template, compiled,
                     provenance, bounds, spans, *, overview=None, consolidate=True):
    """Write a complete index from extraction artifacts; no extraction calls.

    With consolidate=False this is entirely offline. Any pass-2 categories that
    were absent from discovery are explicitly retained and recorded, rather
    than guessed into a similarly named category or silently discarded.
    """
    wiki = Path(output_dir) / safe_filename(doc_id)
    wiki.mkdir(parents=True, exist_ok=True)
    if (wiki / "_manifest.json").exists():
        raise ValueError(f"Refusing to overwrite an existing index: {wiki}")
    manifest = []
    key = str(wiki.resolve())
    _RENDER_RECORDS[key] = []
    _RENDER_ATOMS[key] = Counter()
    text = clean_document(Path(doc_path).read_text(encoding="utf-8", errors="ignore"))
    if bounds and bounds[-1]["end"] != len(text):
        raise ValueError("Saved chunk boundaries do not match cleaned source length")
    resolver = ProvenanceResolver(
        provenance, chunk_texts=[text[b["start"]:b["end"]] for b in bounds],
        embed_fn=_EMBED_FN if consolidate else None, spans=spans, bounds=bounds)

    # Resolve identities once, and visit every compiled key exactly once.
    # Exact dictionary keys have precedence over normalized spelling aliases.
    sections, used, mappings = [], set(), []
    template = json.loads(json.dumps(template))
    for section in template["sections"]:
        name = section["name"]
        matched = name if name in compiled else None
        if matched is None:
            candidates = [k for k in compiled if _norm_entry(k) == _norm_entry(name)]
            if len(candidates) == 1:
                matched = candidates[0]
        if matched is not None and matched not in used:
            sections.append((matched, name))
            mappings.append({"compiled_key": matched, "category": name, "kind": "exact_or_normalized"})
            used.add(matched)
    added = []
    for name in compiled:
        if name not in used:
            sections.append((name, name)); used.add(name); added.append(name)
            mappings.append({"compiled_key": name, "category": name, "kind": "retained_extraction_category"})
            template["sections"].append({"name": name, "description": "Retained from saved extraction", "subsections": []})
    directories = set()
    for compiled_key, name in sections:
        content = compiled[compiled_key]
        if not list(_atoms(content)):
            continue
        dirname = _directory_name(name, directories)
        _build_category(wiki, dirname, name, content, manifest, resolver)

    draft = overview if overview is not None else template.get("overview", "")
    manifest.insert(0, {"path": "index.md", "level": "root", "parent": None,
                       "summary": _one_line(f"{title} — {template.get('corpus_type', template.get('document_type', 'document'))}", DESCRIPTOR_CHARS),
                       "chunks": []})
    # Root descriptions are refreshed again after balancing, using only the
    # actual immediate children. The stored complete overview remains available.
    (wiki / "consolidated.md").write_text(draft or "")
    _write_root_overview(wiki, title, draft, manifest)
    _link_subjects(wiki, manifest)
    if consolidate:
        revised = _consolidate(wiki, title, manifest, draft=draft)
        if revised:
            draft = revised
            (wiki / "consolidated.md").write_text(revised)
            _write_root_overview(wiki, title, revised, manifest)

    expected = Counter(_atoms(compiled))
    missing = expected - _RENDER_ATOMS[key]
    integrity = {
        "version": 1,
        "source": {"path": str(Path(doc_path).resolve()), "normalization": "clean_document",
                   "sha256": hashlib.sha256(text.encode()).hexdigest(), "length_chars": len(text)},
        "compiled_sha256": hashlib.sha256(json.dumps(compiled, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "unaccounted_keys": sorted(set(compiled) - used),
        "unaccounted_fields": [{"value_sha256": hashlib.sha256(value.encode()).hexdigest(), "count": count}
                               for value, count in missing.items()],
        "compiled_atomic_values": sum(expected.values()),
        "rendered_atomic_values": sum(_RENDER_ATOMS[key].values()),
        "category_mapping": mappings, "retained_extra_categories": added,
        "fields": _RENDER_RECORDS[key],
    }
    if integrity["unaccounted_keys"] or missing:
        raise ValueError(f"Index failed field accounting: {len(missing)} unrendered values")
    (wiki / "_integrity.json").write_text(json.dumps(integrity, indent=2))
    (wiki / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    (wiki / "_template.json").write_text(json.dumps(template, indent=2))
    (wiki / "_compiled.json").write_text(json.dumps(compiled, indent=2, default=str))
    (wiki / "_chunking.json").write_text(json.dumps({
        "doc_path": str(Path(doc_path).resolve()), "title": title, "doc_chars": len(text),
        "source_sha256": integrity["source"]["sha256"], "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP, "bounds": bounds}, indent=2))
    how = Counter(record["how"] for records in spans.values() for record in records)
    total = sum(how.values())
    (wiki / "_provenance.json").write_text(json.dumps({
        "signatures": provenance, "spans": spans, "linking": resolver.report(),
        "offsets": {"facts_with_spans": len(spans), "located_by": dict(how),
                    "passage_level_pct": round(100 * (how["verbatim"] + how["window"]) / total, 1) if total else 0}}, indent=2))
    log.info(f"Rendered {doc_id}: {len(manifest)} nodes, {len(integrity['fields'])} complete records, {len(added)} recovered categories")
    return {"doc_id": doc_id, "pages": len(manifest), "sections": len(sections),
            "chunks": len(bounds), "provenance": resolver.report(), "integrity": {"records": len(integrity["fields"]), "missing_fields": len(missing)}}


def _write_root_overview(wiki, title, overview, manifest):
    children = [e for e in manifest if e.get("parent") == "index.md"]
    header = f"# {title}\n\n"
    routes = "## Categories\n" + "\n".join(f"- {e['summary']}" for e in children) + "\n"
    # Many raw categories are compacted equally; balancing will replace this
    # list with descriptions of the actual small set of immediate root groups.
    routes = _compact_branch(routes, 950)
    room = 3000 - len(header) - len(routes) - 4
    text = header + _fit_root_summary(overview or "", cap=room) + "\n\n" + routes
    # Keep historical marker used by tools while the routing table remains the
    # authoritative complete list of children.
    text = text.replace("# Contents", "## Categories", 1)
    (Path(wiki) / "index.md").write_text(text)


CONSOLIDATE_PROMPT = """\
Below is all the information extracted from "{title}" organized across \
multiple wiki pages. Your job is to consolidate this into a single, dense \
reference document that would let someone answer any question about this work.
{draft_block}
Include everything important: who the key people are and what they do, \
what happens and why, specific facts (names, numbers, places, objects), \
how people relate to each other, and any other details that someone \
might ask about.

Organize it however makes sense for this document. Be thorough but concise — \
every sentence should contain information, not filler. Stay under \
{target_chars} characters: anything past that is cut.

WIKI CONTENT:
{content}

CONSOLIDATED REFERENCE:"""


REDUCE_PROMPT = """\
Below are {n} partial reference summaries of "{title}", each covering a \
different part of the document. Merge them into a single, dense reference \
document that would let someone answer any question about this work.

Keep every distinct fact. Where partials overlap, state the fact once. Where \
they conflict, keep the more specific version. Do not drop a section just \
because it appeared in only one partial.

Organize it however makes sense for this document. Be thorough but concise — \
every sentence should contain information, not filler. Stay under \
{target_chars} characters: anything past that is cut.

PARTIAL SUMMARIES:
{content}

CONSOLIDATED REFERENCE:"""


def _fit_root_summary(text, cap=ROOT_SUMMARY_CAP):
    """Hold the root's summary inside the node capacity limit (§3.1 invariants).

    _consolidate asks the model for `cap` characters and the model overshoots:
    measured across the 40-document corpus it returned 11.6K to 23.5K against a
    12,000-char answering budget, so the root alone saturated the budget and the
    descent never read a page of the tree. A target is not an invariant, so this
    enforces it. The cut lands on the last paragraph, then sentence, boundary
    that fits, so the page stays readable prose rather than a severed line.
    """
    if not text or len(text) <= cap:
        return text
    head = text[:cap]
    for sep in ("\n\n", "\n", ". "):
        i = head.rfind(sep)
        if i > cap // 2:                       # don't throw away most of the page
            return head[:i + (len(sep) if sep == ". " else 0)].rstrip() + "\n"
    return head.rstrip() + "\n"


def _consolidate(wiki_dir, title, manifest, draft=None):
    """Rewrite the root's overview against the full merged record (§3.3).

    The whole document contributes. When the pages don't fit one prompt they are
    summarized in batches and the partials reduced into a single summary, rather
    than the content past the prompt limit being dropped.

    `draft` is the provisional overview discovery produced from per-chunk
    glimpses; this call is what revises it.
    """
    wiki_dir = Path(wiki_dir)

    # Gather all page content
    all_content = []
    for entry in manifest:
        fpath = wiki_dir / entry["path"]
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            all_content.append(text)
    if not all_content:
        return None

    # The root is a node, so its summary obeys the node capacity limit. Half the
    # answering budget leaves the other half for the detail pages a descent adds.
    target = ROOT_SUMMARY_CAP
    draft_block = ""
    if draft and str(draft).strip():
        draft_block = (
            "\nA provisional overview was drafted while reading the document one "
            "passage at a time, before any content was extracted. It is a first "
            "guess: correct it wherever the extracted pages below contradict or "
            "sharpen it, and do not carry over anything they do not support.\n\n"
            "PROVISIONAL OVERVIEW:\n" + str(draft).strip() + "\n"
        )

    # Partition pages into batches that each fit a single prompt. A page larger
    # than a whole batch still gets its own batch rather than being cut.
    batches, current, size = [], [], 0
    for text in all_content:
        if current and size + len(text) > CONSOLIDATE_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)

    if len(batches) == 1:
        prompt = CONSOLIDATE_PROMPT.format(
            title=title, content="\n\n---\n\n".join(batches[0]), target_chars=target,
            draft_block=draft_block,
        )
        result = _fit_root_summary(llm(prompt, max_tokens=16384, phase="consolidate"))
        if result:
            log.info(f"  Consolidated: {len(result):,} chars from {len(all_content)} pages")
        return result

    # MAP: one partial summary per batch, so no page is left out
    partials = []
    for bi, batch in enumerate(batches):
        prompt = CONSOLIDATE_PROMPT.format(
            title=title, content="\n\n---\n\n".join(batch), target_chars=target,
            draft_block=draft_block if bi == 0 else "",
        )
        part = llm(prompt, max_tokens=16384, phase="consolidate")
        if part:
            partials.append(part)
            log.info(f"    Consolidate batch {bi+1}/{len(batches)}: {len(part):,} chars")

    if not partials:
        return None
    if len(partials) == 1:
        return _fit_root_summary(partials[0])

    # REDUCE: merge the partials into one reference
    joined = "\n\n---\n\n".join(f"[part {i+1}]\n{p}" for i, p in enumerate(partials))
    if len(joined) > CONSOLIDATE_BATCH_CHARS:
        # Pathological fan-out: reduce the partials pairwise rather than truncate
        while len(joined) > CONSOLIDATE_BATCH_CHARS and len(partials) > 1:
            merged = []
            for i in range(0, len(partials), 2):
                pair = partials[i:i + 2]
                if len(pair) == 1:
                    merged.append(pair[0])
                    continue
                p = llm(REDUCE_PROMPT.format(
                    n=len(pair), title=title, target_chars=target,
                    content="\n\n---\n\n".join(pair)), max_tokens=16384, phase="consolidate")
                merged.append(p or "\n\n".join(pair))
            partials = merged
            joined = "\n\n---\n\n".join(partials)

    result = _fit_root_summary(llm(REDUCE_PROMPT.format(
        n=len(partials), title=title, content=joined, target_chars=target,
    ), max_tokens=16384, phase="consolidate"))
    if result:
        log.info(f"  Consolidated: {len(result):,} chars from {len(all_content)} pages "
                 f"via {len(batches)} batches")
    return result


def _find_section(data, section_name):
    """Exact key first; accept a normalized alias only when unambiguous.

    Partial word overlap can select a different category even when the exact
    key occurs later. Unknown categories are retained by render_extracted.
    """
    if section_name in data:
        return data[section_name]
    matches = [key for key in data if _norm_entry(key) == _norm_entry(section_name)]
    return data[matches[0]] if len(matches) == 1 else None


def _count_items(data):
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return sum(_count_items(v) for v in data.values())
    return 1


def _max_depth(manifest):
    """Compute max depth of the tree from manifest parent pointers."""
    depth_of = {}
    for entry in manifest:
        path = entry["path"]
        parent = entry.get("parent")
        if parent is None:
            depth_of[path] = 0
        elif parent in depth_of:
            depth_of[path] = depth_of[parent] + 1
        else:
            # Walk up
            d = 0
            cur = parent
            while cur:
                d += 1
                p_entry = next((e for e in manifest if e["path"] == cur), None)
                if not p_entry or p_entry.get("parent") is None:
                    break
                cur = p_entry["parent"]
            depth_of[path] = d
    return max(depth_of.values()) if depth_of else 0


def compile_all(corpus_path, output_dir, doc_ids=None):
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_stem = {}
    for f in sorted(corpus_path.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".json"):
            if f.stem not in by_stem or f.suffix == ".txt":
                by_stem[f.stem] = f
    if doc_ids:
        by_stem = {k: v for k, v in by_stem.items() if k in doc_ids}

    results = {}
    for i, (doc_id, doc_path) in enumerate(sorted(by_stem.items())):
        log.info(f"[{i+1}/{len(by_stem)}] Building: {doc_id}")
        title = doc_id.replace("_", " ")
        stats = build_wiki(doc_path, output_dir, doc_id, title)
        results[doc_id] = stats
        if not stats:
            log.warning(f"  Failed: {doc_id}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# INSERT: re-extract only the chunk the new text landed in
# ═════════════════════════════════════════════════════════════════════════════

def _prune_chunk_contributions(data, drop):
    """Remove facts contributed ONLY by a chunk being re-extracted.

    Re-extraction returns the whole chunk's facts, not just the new ones, and the
    model paraphrases, so merging fresh output on top of the old record keeps both
    wordings of everything -- one inserted sentence inflated a real document's
    record by 28%. Re-extracting a chunk should replace what that chunk
    contributed, so its old facts are pruned first.

    `drop` holds signatures whose provenance was that chunk alone. A fact also
    seen in another chunk is still supported by that chunk and stays.
    """
    if isinstance(data, list):
        out = []
        for x in data:
            if _sig(x) in drop:
                continue
            out.append(_prune_chunk_contributions(x, drop))
        return [x for x in out if x not in ({}, [], None)]
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if _sig(v) in drop:
                continue
            pruned = _prune_chunk_contributions(v, drop)
            if pruned in ({}, [], None):
                continue
            out[k] = pruned
        return out
    return data


def insert_text(wiki_dir, new_text, position=None, doc_path=None, rebuild=True):
    """Insert new text into the document and update only what it touched.

    Reuses the chunk boundaries saved at build time: the insertion lands inside
    exactly one chunk, so only that chunk is re-extracted. Chunks before it are
    untouched; chunks after it shift by len(new_text) but their text is
    identical, so they are not re-extracted either.

    Pages whose manifest `chunks` don't include the affected chunk are left
    exactly as they were — that is the point of tagging facts with chunk numbers.

    Returns a summary dict: which chunk changed, which pages were rebuilt, and
    the cost (1 extraction call, vs len(bounds) for a full rebuild).
    """
    wiki_dir = Path(wiki_dir)
    chunking_path = wiki_dir / "_chunking.json"
    if not chunking_path.exists():
        raise FileNotFoundError(
            f"{chunking_path} not found — this digest predates chunk-boundary "
            f"tracking. Rebuild it with build_wiki() before inserting.")

    chunking = json.loads(chunking_path.read_text())
    bounds = chunking["bounds"]
    doc_path = Path(doc_path or chunking["doc_path"])
    template = json.loads((wiki_dir / "_template.json").read_text())
    compiled = json.loads((wiki_dir / "_compiled.json").read_text())
    manifest = json.loads((wiki_dir / "_manifest.json").read_text())

    text = clean_document(doc_path.read_text(encoding="utf-8", errors="ignore"))
    if position is None:
        position = len(text)
    position = max(0, min(int(position), len(text)))

    # Which chunks does the insertion land in? Chunks overlap by CHUNK_OVERLAP,
    # so a position inside an overlap belongs to two chunks and BOTH of their
    # extractions are now stale. Re-extract every chunk that contains it.
    affected = [b["n"] for b in bounds if b["start"] <= position < b["end"]]
    if not affected:                                # appending past the last chunk
        affected = [bounds[-1]["n"]]

    # Rewrite the source. An offset at or before the insertion is unmoved;
    # anything after it shifts by delta. A chunk spanning the insertion keeps its
    # start and shifts its end, so it simply grows — which is what we want.
    # The `n in affected` term covers appending at exactly the end of the last
    # chunk, where end == position and the chunk must still grow to hold the text.
    delta = len(new_text)
    updated = text[:position] + new_text + text[position:]
    _aff = set(affected)
    new_bounds = [{
        "n": b["n"],
        "start": b["start"] + delta if b["start"] > position else b["start"],
        "end": b["end"] + delta if (b["end"] > position or b["n"] in _aff) else b["end"],
    } for b in bounds]

    if not rebuild:
        return {"chunks": affected, "chars_added": delta,
                "pages_rebuilt": [], "dry_run": True}

    doc_path.write_text(updated, encoding="utf-8")

    prov_file = wiki_dir / "_provenance.json"
    stored = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    provenance = {k: set(v) for k, v in stored.get("signatures", {}).items()}
    spans = stored.get("spans", {})

    # Drop what the affected chunks previously contributed, so re-extraction
    # replaces their facts instead of stacking paraphrases on top of them.
    drop = {sig for sig, chunks in provenance.items() if set(chunks) <= set(affected)}
    if drop:
        compiled = _prune_chunk_contributions(compiled, drop)
        for sig in drop:
            provenance.pop(sig, None)
            spans.pop(sig, None)
        log.info(f"  insert: pruned {len(drop)} facts contributed only by chunk(s) {affected}")

    # Re-extract ONLY the affected chunk(s)
    tmpl = _template_description(template)
    extracted_any = False
    for n in affected:
        b = new_bounds[n - 1]
        ctext = updated[b["start"]:b["end"]]
        prompt = COMPILE_PASS1_PROMPT.format(
            template_description=tmpl, chunk=ctext,
            chunk_num=n, total_chunks=len(new_bounds),
        )
        result = parse_json_with_retry(prompt, max_tokens=16384, phase="insert_extract")
        if not result:
            log.warning(f"  insert: re-extraction of chunk {n} returned nothing")
            continue
        extracted_any = True
        _record_provenance(result, n, provenance, ctext, b["start"], spans)
        _merge_paper(compiled, result, template)

    if not extracted_any:
        return {"chunks": affected, "chars_added": delta,
                "pages_rebuilt": [], "extracted": False}

    # Rebuild only the pages built from an affected chunk
    stale = [e for e in manifest if set(e.get("chunks") or ()) & set(affected)]
    kept = [e for e in manifest if e not in stale]
    resolver = ProvenanceResolver(
        {k: sorted(v) for k, v in provenance.items()},
        chunk_texts=[updated[b["start"]:b["end"]] for b in new_bounds],
        embed_fn=_EMBED_FN,
        spans=spans,                    # so re-written pages keep their evidence
        bounds=new_bounds,
    )

    rebuilt = []
    for section in template["sections"]:
        sname = section["name"]
        sdata = _find_section(compiled, sname)
        if not sdata:
            continue
        sdir = safe_filename(sname)
        if not any(e["path"].startswith(sdir + "/") for e in stale):
            continue                                    # this section is untouched
        sub = []
        _build_category(wiki_dir, sdir, sname, sdata, sub, resolver)
        kept = [e for e in kept if not e["path"].startswith(sdir + "/")]
        kept.extend(sub)
        rebuilt.extend(e["path"] for e in sub)

    # Links point at subject pages, and the ones just rebuilt have new paths, so
    # the whole link set is recomputed rather than left dangling.
    _link_subjects(wiki_dir, kept)

    # The consolidated summary covers every chunk, so it is always stale
    consolidated = _consolidate(wiki_dir, chunking.get("title", doc_path.stem), kept)
    if consolidated:
        (wiki_dir / "consolidated.md").write_text(consolidated)
        kept = [e for e in kept if e["path"] != "consolidated.md"]
        kept.append({
            "path": "consolidated.md", "level": "root", "parent": None,
            "summary": "consolidated overview of all key information",
            "chunks": sorted({c for e in kept for c in e.get("chunks", [])}),
        })
        rebuilt.append("consolidated.md")

    chunking["bounds"] = new_bounds
    chunking["doc_chars"] = len(updated)
    chunking_path.write_text(json.dumps(chunking, indent=2))
    (wiki_dir / "_manifest.json").write_text(json.dumps(kept, indent=2))
    (wiki_dir / "_compiled.json").write_text(json.dumps(compiled, indent=2, default=str))
    (wiki_dir / "_provenance.json").write_text(json.dumps({
        "signatures": {k: sorted(v) for k, v in provenance.items()},
        "spans": spans,
        "linking": resolver.report(),
    }, indent=2))

    log.info(f"  insert: chunk(s) {affected} of {len(new_bounds)} re-extracted, "
             f"{len(rebuilt)} pages rebuilt ({len(affected)} extraction call(s) vs "
             f"{len(new_bounds)} for a full rebuild)")
    return {
        "chunks": affected, "chars_added": delta, "extracted": True,
        "pages_rebuilt": rebuilt, "pages_untouched": len(kept) - len(rebuilt),
        "extraction_calls": len(affected), "full_rebuild_calls": len(new_bounds),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TRAVERSE: manifest-driven, descend / ascend / lateral hop
# ═════════════════════════════════════════════════════════════════════════════

# The question itself must dominate navigation. Terms picked up from pages
# already read are a weak secondary signal -- see _score().
QUESTION_WEIGHT = 8

# Kept out of traverse's return value for compatibility with existing ablations.
# Query returns a snapshot of these records as its third result.
_LAST_TRAVERSAL_TRACE = {}
_LAST_EVIDENCE_TRACE = {}
_LAST_ROUTE_TRACE = {}


def _score(query_words, text, question_words=None):
    """Relevance of a page summary to the question.

    The paper (§3.4) scores summaries against *the question*. Expanding the query
    with terms from pages already read is an implementation addition, and left
    unweighted it destroys the thing it is meant to help: the mega summary alone
    contributes ~460 terms, so a 4-term question is under 1% of the expanded set
    and every question ends up scoring every page identically -- navigation stops
    depending on what was asked at all.

    So the question's own terms carry QUESTION_WEIGHT each and expansion terms
    count 1, keeping lateral hops useful without letting them swamp the query.
    """
    text_words = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
    if question_words is None:
        return len(query_words & text_words)
    return (QUESTION_WEIGHT * len(question_words & text_words)
            + len((query_words - question_words) & text_words))


ROUTE_MAX_TOKENS = 8192

ROUTE_PROMPT = """\
You are navigating a plain-text index of "{title}" to answer a question.

You are at "{here}". Below are its children, each with a one-line summary. \
Choose the ones most likely to lead to the answer, best first. Choose at most \
{k}. If none of them could lead to the answer, reply NONE.

QUESTION: {question}

CHILDREN:
{catalog}

Reply with ONLY the numbers, comma separated, best first, or NONE. Nothing else."""


LINK_PROMPT = """\
You are reading the page "{here}" in a plain-text index of "{title}", and it \
does not answer the question. The page links to these related subjects, which \
may sit under other categories.

Choose the ones likely to hold the answer, best first. Choose at most {k}. If \
none of them would help, reply NONE.

QUESTION: {question}

RELATED SUBJECTS:
{catalog}

Reply with ONLY the numbers, comma separated, best first, or NONE. Nothing else."""


def _route_catalog(entries, budget=ROUTING_BUDGET):
    """Render children as numbered routing descriptors within the routing budget.

    §3.4 defines the routing budget C as the most one routing step may read while
    choosing a direction, and S as one descriptor, giving f = C/S. Honouring C
    here is what makes that derivation real: a routing call never reads more than
    C characters, so a balanced node's f children always fit in exactly one call.
    """
    if not entries or budget <= 0:
        return "", 0
    # Numbering and newline separators count too. Shorten descriptors uniformly
    # when prompt overhead leaves less than C available, instead of silently
    # hiding the last children of an otherwise balanced node.
    n = min(len(entries), max(1, budget // 24))
    overhead = sum(len(f"{i + 1}. ") for i in range(n)) + n - 1
    width = min(DESCRIPTOR_CHARS, (budget - overhead) // n)
    if width <= 0:
        return "", 0
    lines = [f"{i + 1}. {_one_line(e.get('summary', e['path']), width)}"
             for i, e in enumerate(entries[:n])]
    return "\n".join(lines), len(lines)


def _parse_picks(raw, n):
    """Numbers the router chose, in order, deduplicated and in range."""
    if not (raw or "").strip():
        return None                       # no content: caller falls back
    if re.search(r"\bnone\b", raw, re.I) and not re.search(r"\d", raw):
        return []                         # an explicit "nothing here" is an answer
    picks = []
    for tok in re.findall(r"\d+", raw):
        i = int(tok) - 1
        if 0 <= i < n and i not in picks:
            picks.append(i)
    return picks or None                  # malformed response: lexical fallback


def llm_route_children(question, children, title="the document", here="the root",
                       k=3, prompt=ROUTE_PROMPT):
    """One routing step: the model reads the children's descriptors and judges.

    This is the paper's navigation (§3.1, §3.5, Figure 1b): "the model judges
    which summaries are relevant rather than comparing a search key", and "at each
    level, a short LLM call reads only the children's routing descriptors and
    judges which are relevant, so a descent costs a handful of brief calls rather
    than one long read of the document".

    Per LEVEL, not per question. A single call over a flat catalogue of every page
    in the index would be a re-ranker over the whole document: it reads far more
    than the routing budget, it does not use the hierarchy, and its cost grows
    with the index rather than with its depth.

    Returns the children reordered, most relevant first; [] when the model says
    none of them lead anywhere; or None when the call gave nothing usable, so the
    caller can fall back to keyword scoring rather than stall.
    """
    if not children:
        return []
    _LAST_ROUTE_TRACE.clear()
    # C bounds the complete routing input, including question/instructions.
    overhead = len(prompt.format(title=title, question=question, k=k,
                                 here=here, catalog=""))
    catalog, n = _route_catalog(children, budget=ROUTING_BUDGET - overhead)
    _LAST_ROUTE_TRACE.update({"shown_paths": [c["path"] for c in children[:n]],
                              "unshown_paths": [c["path"] for c in children[n:]],
                              "catalog": catalog, "prompt_chars": overhead + len(catalog)})
    if not n:
        _LAST_ROUTE_TRACE["reason"] = "routing_prompt_exceeds_budget"
        return None
    try:
        raw = llm(prompt.format(title=title, question=question, k=k, here=here,
                                catalog=catalog),
                  max_tokens=ROUTE_MAX_TOKENS, phase="route")
    except Exception as e:                       # a failed step must not kill a query
        log.warning(f"    route: {repr(e)[:100]}")
        _LAST_ROUTE_TRACE["error"] = repr(e)[:300]
        return None
    _LAST_ROUTE_TRACE["response"] = raw
    picks = _parse_picks(raw, n)
    if picks is None:
        log.warning("    route: model returned no content — falling back to keyword scoring")
        return None
    return [children[i] for i in picks[:k]]


def traverse(question, wiki_dir, max_chars=CONTEXT_BUDGET, exhaustive=False,
             seed_fraction=1.0, router=None):
    """Walk local child/subject links and record exactly what the answerer reads.

    NONE permits bounded lexical recovery at internal nodes. Short internal-page
    excerpts leave room for subjects; their complete descriptors are routed
    separately. Rejected/unavailable pages are never reported as read.
    """
    wiki_dir = Path(wiki_dir)
    _LAST_VISITED.clear()
    _LAST_TRAVERSAL_TRACE.clear()
    trace = _LAST_TRAVERSAL_TRACE
    trace.update({"budget_chars": max_chars, "pages_read": [], "skipped_pages": [],
                  "routes": [], "none_recoveries": 0, "context_chars": 0})
    manifest_path = wiki_dir / "_manifest.json"
    if not manifest_path.exists():
        trace["stop_reason"] = "missing_manifest"
        return ""
    manifest = json.loads(manifest_path.read_text())
    title = wiki_dir.name.replace("_", " ")
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    expanded = set(q_words)
    visited, unavailable, rejected_edges = set(), set(), set()
    collected = []
    total_chars = 0
    n_route_calls = 0
    by_path = {e["path"]: e for e in manifest}
    children = {}
    for e in manifest:
        children.setdefault(e.get("parent"), []).append(e)

    def _read(path, limit=None, reason="descent"):
        nonlocal total_chars
        if path in visited or path in unavailable:
            return False
        full = wiki_dir / path
        if not full.is_file():
            unavailable.add(path)
            trace["skipped_pages"].append({"path": path, "reason": "missing_file"})
            return False
        content = full.read_text(encoding="utf-8", errors="ignore")
        sep = len(_DIGEST_SEP) if collected else 0
        remaining = max_chars - total_chars - sep
        if children.get(path) and by_path.get(path, {}).get("parent") is not None:
            remaining = min(remaining, PAGE_SIZE // 3)
        if limit is not None:
            remaining = min(remaining, limit)
        if remaining <= 0:
            trace["skipped_pages"].append({"path": path, "reason": "context_budget"})
            return False
        if not content:
            unavailable.add(path)
            trace["skipped_pages"].append({"path": path, "reason": "empty_file"})
            return False
        excerpt = content[:remaining]
        visited.add(path)
        _LAST_VISITED.append(path)
        context_start = total_chars + sep
        collected.append(excerpt)
        total_chars += len(excerpt) + sep
        trace["pages_read"].append({"path": path, "level": by_path.get(path, {}).get("level"),
            "reason": reason, "start": 0, "end": len(excerpt), "file_chars": len(content),
            "partial": len(excerpt) < len(content), "context_start": context_start,
            "context_end": total_chars, "text": excerpt})
        expanded.update(re.findall(r"[a-zA-Z]{3,}", excerpt.lower()))
        return True

    def _budget_ok():
        return total_chars + (len(_DIGEST_SEP) if collected else 0) < max_chars

    if router is None:
        router = _ROUTER

    def _candidates(path):
        return [c for c in children.get(path, []) if c["path"] not in visited
                and c["path"] not in unavailable and (path, c["path"]) not in rejected_edges]

    def _rank(cands, current, prompt=ROUTE_PROMPT):
        nonlocal n_route_calls
        here = _one_line(by_path.get(current, {}).get("summary", current), 80)
        event = {"at": current, "kind": "children" if prompt == ROUTE_PROMPT else "links",
                 "candidate_paths": [c["path"] for c in cands]}
        lexical = sorted(cands, key=lambda c: -_score(expanded, c.get("summary", ""), q_words))
        picked = None
        if router:
            n_route_calls += 1
            _LAST_ROUTE_TRACE.clear()
            picked = router(question, cands, title=title, here=here, prompt=prompt)
            if _LAST_ROUTE_TRACE:
                event["router"] = dict(_LAST_ROUTE_TRACE)
            event["mode"] = "llm"
        if picked is None:
            picked = lexical
            event["mode"] = "lexical_fallback" if router else "lexical"
        elif not picked and prompt == ROUTE_PROMPT and trace["none_recoveries"] < LEVEL_UP_BUDGET:
            # Recovery inspects only this node's children, never a flat index.
            picked = lexical[:1]
            trace["none_recoveries"] += 1
            event["mode"] = "none_lexical_recovery"
        allowed = {c["path"]: c for c in cands}
        picked = [allowed[c["path"]] for c in picked if c.get("path") in allowed]
        selected = {c["path"] for c in picked}
        event["selected_paths"] = [c["path"] for c in picked]
        event["rejected_paths"] = [c["path"] for c in cands if c["path"] not in selected]
        trace["routes"].append(event)
        if not picked:
            rejected_edges.update((current, c["path"]) for c in cands)
        return picked

    roots = [e for e in manifest if e.get("parent") is None] or [{"path": "index.md"}]
    orient_cap = int(max_chars * seed_fraction) if seed_fraction < 1 else min(ROOT_SUMMARY_CAP, max_chars // 4)
    spent = 0
    for root in roots:
        if spent >= orient_cap:
            break
        before = total_chars
        _read(root["path"], limit=orient_cap - spent, reason="orientation")
        spent += total_chars - before
    current = next((r["path"] for r in roots if children.get(r["path"])), roots[0]["path"])
    walked = set()
    while current is not None and _budget_ok():
        state = (current, len(visited), len(unavailable), len(rejected_edges))
        if state in walked:
            trace["stop_reason"] = "navigation_cycle"
            break
        walked.add(state)
        kids = _candidates(current)
        if kids:
            target = next((c["path"] for c in _rank(kids, current) if _read(c["path"])), None)
            if target is not None:
                current = target
                continue
            if _budget_ok() and _candidates(current):
                # A selected file may be missing while siblings remain usable.
                continue
        entry = by_path.get(current, {})
        # Continuations belong to the same record and precede lateral subjects.
        continuation = [p for p in entry.get("continuations", [])
                        if p in by_path and p not in visited and p not in unavailable]
        target = next((p for p in continuation if _read(p, reason="continuation")), None)
        if target is not None:
            current = target
            continue
        links = [by_path[p] for p in entry.get("links", [])
                 if p in by_path and p not in visited and p not in unavailable
                 and (current, p) not in rejected_edges]
        if links and _budget_ok():
            target = next((c["path"] for c in _rank(links, current, prompt=LINK_PROMPT)
                           if _read(c["path"], reason="lateral")), None)
            if target is not None:
                current = target
                continue
        node, ancestors = entry.get("parent"), set()
        while node is not None and node not in ancestors and not _candidates(node):
            ancestors.add(node)
            node = by_path.get(node, {}).get("parent")
        current = None if node in ancestors else node
    if exhaustive:
        for e in sorted(manifest, key=lambda e: -_score(expanded, e.get("summary", ""), q_words)):
            if not _budget_ok():
                break
            _read(e["path"], reason="exhaustive_ablation")
    trace["context_chars"] = total_chars
    trace["routing_calls"] = n_route_calls
    trace.setdefault("stop_reason", "context_budget" if not _budget_ok() else "no_local_candidates")
    log.info(f"    Traversal: {len(visited)}/{len(manifest)} pages read, {total_chars:,} chars, "
             f"{n_route_calls} routing calls")
    return _DIGEST_SEP.join(collected)


def _call_traverse(question, wiki_dir, **kw):
    """Call traverse, passing only the arguments it actually accepts.

    Several ablations rebind the module-global `traverse` to their own retrieval
    (semantic, collapsed, big-budget). Those take max_chars but not exhaustive,
    so filter to the target's real signature rather than assuming v7's.
    """
    fn = traverse
    try:
        accepted = inspect.signature(fn).parameters
        kw = {k: v for k, v in kw.items() if k in accepted}
    except (TypeError, ValueError):
        kw = {}
    return fn(question, wiki_dir, **kw)


def _source_bounds(wiki_dir, text):
    """The chunk boundary table this index was built on.

    Saved at build time so the fallback slices exactly the passages extraction
    read. Recomputed for a tree built before that file existed, which is safe
    because chunk_bounds is deterministic in (size, overlap).
    """
    f = Path(wiki_dir) / "_chunking.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            if d.get("bounds"):
                return {b["n"]: b for b in d["bounds"]}
        except (ValueError, KeyError):
            pass
    return {b["n"]: b for b in chunk_bounds(text)}


def _evidence_text(wiki_dir, text, visited, max_chars, question=""):
    """Rank source passages only inside the provenance of reached subjects.

    Long originating chunks are windowed locally, so a relevant passage late in
    a reached chunk can survive the answering budget. Every returned character
    retains its exact cleaned-source offset. Branch provenance is never used as
    a shortcut to retrieve the whole document.
    """
    _LAST_EVIDENCE_TRACE.clear()
    trace = _LAST_EVIDENCE_TRACE
    trace.update({"budget_chars": max_chars, "offset_basis": "clean_document",
                  "passages": [], "candidate_windows": 0, "context_chars": 0})
    manifest_path = Path(wiki_dir) / "_manifest.json"
    if not manifest_path.exists():
        trace["empty_reason"] = "missing_manifest"
        return ""
    manifest = json.loads(manifest_path.read_text())
    by_path = {e["path"]: e for e in manifest}
    bounds = _source_bounds(wiki_dir, text)
    reached = [by_path[p] for p in visited if p in by_path and by_path[p].get("level") == "leaf"]
    trace["reached_subject_paths"] = [e["path"] for e in reached]
    stopwords = {"what", "when", "where", "who", "why", "how", "the", "does", "did",
                 "was", "were", "are", "his", "her", "their", "this", "that", "with", "from"}
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower())) - stopwords
    candidates = {}
    for entry in reached:
        pinned = [s for s in entry.get("spans", []) if s.get("how") != "chunk"]
        spans = pinned or [dict(s) for s in entry.get("spans", [])]
        if not pinned:
            for n in entry.get("chunks", []):
                if n in bounds:
                    spans.append({**bounds[n], "chunk": n, "how": "chunk"})
        for span in spans:
            start, end = max(0, int(span.get("start", 0))), min(len(text), int(span.get("end", 0)))
            if end <= start:
                continue
            tier = 0 if span.get("how") != "chunk" else 1
            width = min(1600, max(1, max_chars))
            cursor = start
            while cursor < end:
                finish = min(end, cursor + width)
                key = (cursor, finish)
                piece = text[cursor:finish]
                score = _score(q_words, piece)
                candidate = candidates.setdefault(key, {
                    "start": cursor, "end": finish, "tier": tier,
                    "how": span.get("how", "chunk"), "chunk": span.get("chunk"),
                    "subject_paths": [], "score": score})
                candidate["tier"] = min(candidate["tier"], tier)
                if entry["path"] not in candidate["subject_paths"]:
                    candidate["subject_paths"].append(entry["path"])
                if finish >= end:
                    break
                cursor = finish - min(200, width // 4)
    trace["candidate_windows"] = len(candidates)
    if not candidates:
        trace["empty_reason"] = "no_reached_subject_provenance" if reached else "no_subject_reached"
        return ""
    # Keep precise fact spans ahead of coarse chunk windows; within each tier
    # relevance beats source order or span length.
    ranked = sorted(candidates.values(), key=lambda s: (s["tier"], -s["score"], s["end"] - s["start"], s["start"]))
    selected, used = [], 0
    for span in ranked:
        room = max_chars - used - (len(_PASSAGE_SEP) if selected else 0)
        if room <= 0:
            break
        # Subtract already selected intervals, avoiding duplicate text even when
        # neighboring windows/provenance ranges overlap.
        uncovered = [(span["start"], span["end"])]
        for old in selected:
            fragments = []
            for start, end in uncovered:
                if old["end"] <= start or old["start"] >= end:
                    fragments.append((start, end))
                else:
                    if start < old["start"]:
                        fragments.append((start, old["start"]))
                    if end > old["end"]:
                        fragments.append((old["end"], end))
            uncovered = fragments
        for start, end in uncovered:
            room = max_chars - used - (len(_PASSAGE_SEP) if selected else 0)
            if room <= 0:
                break
            end = min(end, start + room)
            piece = text[start:end]
            if piece.strip():
                selected.append({**span, "start": start, "end": end, "text": piece})
                used += len(piece) + (len(_PASSAGE_SEP) if len(selected) > 1 else 0)
    selected.sort(key=lambda s: s["start"])
    cursor = 0
    for i, span in enumerate(selected):
        cursor += len(_PASSAGE_SEP) if i else 0
        span["context_start"] = cursor
        cursor += len(span["text"])
        span["context_end"] = cursor
    trace["passages"] = selected
    trace["context_chars"] = cursor
    return _PASSAGE_SEP.join(span["text"] for span in selected)


def query_with_traversal(question, wiki_dir, raw_doc_path, deep_retry=False):
    """Answer from bounded index/source contexts and return their audit trace.

    The third result records decisions, pages actually read, exact evidence and
    answer attempts. The original answer/source result pair remains unchanged.
    """
    import copy
    import hashlib

    wiki_dir = Path(wiki_dir)
    _LAST_EVIDENCE_TRACE.clear()
    trace = {"version": 1, "question": question, "wiki_dir": str(wiki_dir),
             "traversals": [], "answer_attempts": [], "evidence": {}}
    manifest_path = wiki_dir / "_manifest.json"
    if manifest_path.exists():
        trace["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def _answer_line(raw):
        lines = (raw or "").strip().splitlines()
        return lines[0].strip(" .\"'") if lines else "not found"

    def _is_answer(answer):
        return isinstance(answer, str) and answer.strip().lower() not in ("not found", "", "none", "n/a")

    def _walk(**kwargs):
        _LAST_VISITED.clear()
        _LAST_TRAVERSAL_TRACE.clear()
        digest = _call_traverse(question, wiki_dir, **kwargs)
        walk_trace = copy.deepcopy(_LAST_TRAVERSAL_TRACE)
        walk_trace["read_order"] = list(_LAST_VISITED)
        walk_trace["context"] = digest
        trace["traversals"].append(walk_trace)
        return digest

    def _answer_index(digest, phase):
        prompt = (
            "You are answering a question about a document using ONLY the compiled "
            "wiki digest below.\nAnswer in 10 words or fewer. If the answer is not in the "
            "digest, return exactly: not found\n\n"
            f"Question: {question}\n\nWiki Digest:\n{digest}\n\nAnswer:"
        )
        answer = _answer_line(llm(prompt, max_tokens=2048, phase=phase))
        trace["answer_attempts"].append({"phase": phase, "answer": answer,
                                         "context_chars": len(digest)})
        return answer

    digest = _walk()
    visited = list(_LAST_VISITED)
    if digest:
        predicted = _answer_index(digest, "query_traverse")
        if _is_answer(predicted):
            return predicted, "wiki", trace

    if deep_retry:
        deep = _walk(max_chars=CONTEXT_BUDGET * DEEP_RETRY_FACTOR, exhaustive=True)
        visited = list(dict.fromkeys(visited + list(_LAST_VISITED)))
        if deep and deep != digest:
            predicted = _answer_index(deep, "query_traverse_deep")
            if _is_answer(predicted):
                return predicted, "wiki_deep", trace

    if not (raw_doc_path and Path(raw_doc_path).exists()):
        trace["miss_reason"] = "missing_source_document"
        return "not found", "miss", trace
    text = clean_document(Path(raw_doc_path).read_text(encoding="utf-8", errors="ignore"))
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    trace["source_sha256"] = source_sha
    chunking_path = wiki_dir / "_chunking.json"
    if chunking_path.exists():
        expected = json.loads(chunking_path.read_text()).get("source_sha256")
        if expected and expected != source_sha:
            trace["miss_reason"] = "source_digest_mismatch"
            log.warning("    fallback: cleaned source differs from the indexed source; refusing stale offsets")
            return "not found", "miss", trace
    excerpt = _evidence_text(wiki_dir, text, visited, CONTEXT_BUDGET, question=question)
    trace["evidence"] = copy.deepcopy(_LAST_EVIDENCE_TRACE)
    trace["evidence"]["context"] = excerpt
    if not excerpt:
        trace["miss_reason"] = trace["evidence"].get("empty_reason", "no_reached_subject_provenance")
        log.warning("    fallback: no source passage available for the subjects actually read (%s)",
                    trace["miss_reason"])
        return "not found", "miss", trace
    raw_resp = llm(FALLBACK_QA_PROMPT.format(question=question, excerpt=excerpt),
                   max_tokens=4096, phase="query_fallback")
    result = parse_json(raw_resp)
    predicted = result.get("answer", "not found") if isinstance(result, dict) else _answer_line(raw_resp)
    trace["answer_attempts"].append({"phase": "query_fallback", "answer": predicted,
                                     "context_chars": len(excerpt)})
    if _is_answer(predicted):
        return predicted, "fallback", trace
    trace["miss_reason"] = "source_context_did_not_answer"
    return "not found", "miss", trace
