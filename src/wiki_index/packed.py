"""
packed.py - Wiki v8: bounded-fanout "packed" index (derived depth).

The v7 index fixes the number of levels (root, category, subject) and lets one
level grow as wide as the source document requires, so that wide level gets
scanned linearly. v8 inverts that: it fixes how wide a node can be and lets the
tree grow deeper, so per-query routing work grows like log of the leaf count
instead of linearly.

Vocabulary (kept consistent everywhere): root, category, subject, field,
source document.

BUILD (per source document):
  1. DISCOVER - reuse pipeline.discover_schema (categories). Not changed here.
  2. EXTRACT  - reuse pipeline.extract (fills subjects under categories). Not
     changed here. The extracted subjects are the leaves.
  3. PACK     - new step (this module): build a bounded-fanout tree on the
     leaves, per source document. Categories stay as the top-level grouping so
     semantic navigation is preserved.

Fanout math (all config-derived, nothing hardcoded):
  C = node budget      - routing content one node can hold (config, default 4000 tokens)
  s = child summary    - length of one child's routing summary (config, default 120 tokens)
  f = floor(C / s)     - fanout cap. With the defaults, about 33.
  depth              = ceil(log_f(N)) for N leaves. We never pick the level count.

Non-negotiables enforced here:
  - No internal node has more than f children.
  - The LLM only ever summarizes one bundle of at most f items at a time. It
    never sees the whole set at once.
  - Grouping into bundles uses cheap embeddings, not the LLM.

This module is additive. It reuses v7's discover/extract/render helpers by
import and never edits them. The `llm` name is patched by the benchmark cli the
same way it patches cold_start.llm and pipeline.llm; the cluster embedder is
injected via set_cluster_embedder so this module stays decoupled from the shared
Azure client.

Phase 2 (traversal / answering) and Phase 4 (insert / lazy re-summarization)
are added on top of the structures defined here.
"""

import re
import json
import math
import heapq
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse v7 code without editing it.
from cold_start import (
    log, safe_filename, clean_document, chunk_text, parse_json,
    FALLBACK_QA_PROMPT,
)
from pipeline import (
    discover_schema, extract, _find_section,
    _render_item, _summarize_item, _one_line,
    CONSOLIDATE_PROMPT, CONTEXT_BUDGET,
)

# Patched by the benchmark cli (bench/methods/wiki_v8/cli.py), exactly like
# cold_start.llm / pipeline.llm. Falls back to cold_start.llm when run directly.
from cold_start import llm  # noqa: F401


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG - every knob lives here, nothing is hardcoded in the algorithms below
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PackedConfig:
    # fanout math
    node_budget_tokens: int = 4000       # C: routing content one node can hold
    child_summary_tokens: int = 120      # s: one child's routing summary size

    # clustering (coherent_partition)
    cluster_embed_backend: str = "azure"  # "azure" (injected) is the default cluster embedder
    cluster_seed: int = 42                # deterministic clustering
    kmeans_iters: int = 12
    pack_workers: int = 8                 # concurrency for the routing-summary calls

    # routing at query time (used in Phase 2; declared here so config is one place)
    # cosine   : cosine only, all-mpnet-base-v2 (cached local model)
    # llm      : one LLM call per node to pick the child
    # hybrid   : cosine plus an epsilon tie-break LLM call on close candidates
    # keyword  : same lexical scorer as v7 (matched-mode shape-isolation baseline)
    route_mode: str = "cosine"
    # sentence-transformers model for cosine/hybrid routing. all-mpnet-base-v2 is
    # already cached. Documented upgrade path if cosine misroutes on
    # question-versus-summary register mismatch: "intfloat/e5-large-v2" with
    # "query: " / "passage: " prefixes (asymmetric). Swapping is a one-line change.
    route_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    route_epsilon: float = 0.05          # hybrid tie-break threshold (cosine gap)

    # traversal budget (Phase 2); kept here so limits are config-driven
    traverse_budget: int = 200           # cap on total child summaries scored per question
    beam_width: int = 2                  # branches kept per level (covers cross-cutting facts)
    max_answer_leaves: int = 6           # leaves gathered into the answer digest
    answer_char_budget: int = 12000      # QA prompt size budget (matches v7 CONTEXT_BUDGET)
    use_overview: bool = True            # prepend a per-document consolidated overview (as v7 does)

    @property
    def fanout(self) -> int:
        """f = floor(C / s). Config-derived, never hardcoded."""
        return max(2, self.node_budget_tokens // self.child_summary_tokens)

    @classmethod
    def from_file(cls, path):
        """Load overrides from a JSON config file if it exists."""
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Active config. The cli may replace this via configure().
CFG = PackedConfig()


def configure(cfg=None, **overrides):
    """Set the active PackedConfig (from an instance and/or field overrides)."""
    global CFG
    base = cfg or CFG
    if overrides:
        merged = {**asdict(base), **overrides}
        merged = {k: v for k, v in merged.items() if k in PackedConfig.__dataclass_fields__}
        base = PackedConfig(**merged)
    CFG = base
    return CFG


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING INJECTION - cluster embedder is provided by the harness
# ═════════════════════════════════════════════════════════════════════════════

# Set by the cli to azure_client.embed (returns (list_of_vectors, total_tokens)).
_CLUSTER_EMBED = None


def set_cluster_embedder(fn):
    """Inject the cluster embedding function: fn(list[str]) -> (vectors, tokens)."""
    global _CLUSTER_EMBED
    _CLUSTER_EMBED = fn


def _embed_matrix(texts):
    """Embed texts with the injected cluster embedder, L2-normalized rows.

    Returns a numpy array of shape (len(texts), dim). Imported numpy lazily so
    importing this module never requires it until a build actually runs.
    """
    import numpy as np
    if _CLUSTER_EMBED is None:
        raise RuntimeError(
            "cluster embedder not set; call set_cluster_embedder(azure_client.embed) "
            "before building a packed index")
    vecs, _tok = _CLUSTER_EMBED(list(texts))
    arr = np.asarray(vecs, dtype="float64")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN HELPERS - measure and cap routing summaries at s tokens
# ═════════════════════════════════════════════════════════════════════════════

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False  # tiktoken unavailable; use a word-based approximation
    return _ENC


def _ntok(text):
    enc = _encoder()
    if enc:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.3) + 1


def _truncate_tokens(text, max_tokens):
    text = " ".join(str(text).split())
    enc = _encoder()
    if enc:
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens]).strip()
    # word-based fallback
    words = text.split()
    approx = max(1, int(max_tokens / 1.3))
    if len(words) <= approx:
        return text
    return " ".join(words[:approx])


# ═════════════════════════════════════════════════════════════════════════════
# NODE - the packed tree data structure
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    id: str
    level: int                       # 0 = leaf
    kind: str                        # "root" | "internal" | "leaf"
    category: str
    summary: str                     # routing text, <= s tokens (internal/root)
    content: str = None              # leaf only: the actual subject/field content
    children: list = field(default_factory=list)   # child ids, len <= f for internal/root
    parent: str = None
    dirty: bool = False              # Phase 4: marks a stale path node
    # cached routing-summary vector is added at Phase 2; not persisted as a field

    def to_dict(self):
        return asdict(self)


class _IdGen:
    """Deterministic node-id generator scoped to one source document."""

    def __init__(self, doc_id):
        self.doc_id = safe_filename(doc_id)
        self._n = 0
        self.by_id = {}

    def new(self, level, kind, category, summary, content=None, children=None):
        nid = f"{self.doc_id}:n{self._n}"
        self._n += 1
        node = Node(id=nid, level=level, kind=kind, category=category,
                    summary=summary, content=content,
                    children=list(children or []))
        self.by_id[nid] = node
        return node


# ═════════════════════════════════════════════════════════════════════════════
# LEAVES - derive subject leaves from the extract output (extraction unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _leaf_payloads(category_data):
    """Flatten one category's extracted data into (content, summary) leaves.

    The extract phase leaves subjects as a list of items or a dict of fields.
    We turn each atomic subject into one leaf, reusing v7's renderers so the
    content and routing summary match what v7 would show. Extraction itself is
    not changed.
    """
    payloads = []

    def _emit(item):
        if not item or str(item) == "[not present in this document]":
            return
        content = _render_item(item)
        summary = _summarize_item(item)
        if content and summary:
            payloads.append((content, summary))

    if isinstance(category_data, list):
        for item in category_data:
            _emit(item)
    elif isinstance(category_data, dict):
        for key, value in category_data.items():
            if not value or str(value) == "[not present in this document]":
                continue
            if isinstance(value, list):
                for item in value:
                    _emit(item)
            else:
                # a single field: render as a small named subject
                content = f"### {key}\n{_render_item(value)}"
                summary = _one_line(f"{key}: {_summarize_item(value)}")
                payloads.append((content, summary))
    elif isinstance(category_data, str):
        payloads.append((category_data, _one_line(category_data)))
    return payloads


def _leaves_for_category(name, category_data, idgen):
    """Build leaf Nodes for one category."""
    leaves = []
    for content, summary in _leaf_payloads(category_data):
        leaves.append(idgen.new(level=0, kind="leaf", category=name,
                                summary=summary, content=content))
    return leaves


# ═════════════════════════════════════════════════════════════════════════════
# COHERENT PARTITION - embeddings + seeded k-means, hard cap at f
# ═════════════════════════════════════════════════════════════════════════════

def _kmeans(vecs, k, seed, iters):
    """Tiny deterministic k-means over L2-normalized rows (cosine via dot).

    Returns a list of clusters, each a list of row indices. Streaming-friendly
    in spirit (assign one pass, update, repeat); no external dependency beyond
    numpy so it runs anywhere the build runs.
    """
    import numpy as np
    n = len(vecs)
    rng = np.random.default_rng(seed)
    # k-means++ style seeding for stable, spread-out centroids
    first = int(rng.integers(0, n))
    centroids = [vecs[first]]
    for _ in range(1, k):
        d2 = np.min([1.0 - vecs @ c for c in centroids], axis=0)
        d2 = np.clip(d2, 0, None)
        total = d2.sum()
        if total <= 0:
            centroids.append(vecs[int(rng.integers(0, n))])
            continue
        probs = d2 / total
        centroids.append(vecs[int(rng.choice(n, p=probs))])
    C = np.vstack(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        sims = vecs @ C.T                 # (n, k) cosine similarity
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for j in range(len(C)):
            members = vecs[labels == j]
            if len(members):
                v = members.mean(axis=0)
                nrm = np.linalg.norm(v)
                C[j] = v / nrm if nrm else C[j]
            else:
                # reseed an empty centroid to the worst-served point
                sims_all = vecs @ C.T
                worst = int(np.argmin(np.max(sims_all, axis=1)))
                C[j] = vecs[worst]

    clusters = [[] for _ in range(len(C))]
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)
    return [c for c in clusters if c]


def coherent_partition(nodes, max_size=None, seed=None):
    """Partition nodes into coherent bundles of at most max_size (= f).

    Embeds each node's routing summary with the cluster embedder, clusters with
    seeded k-means targeting ceil(n / f) bundles, then enforces the hard cap by
    recursively splitting any bundle that still exceeds f. The cap is a ceiling,
    not a target: we prefer the natural topic cut and only force a split when a
    bundle would exceed f.
    """
    f = max_size or CFG.fanout
    seed = CFG.cluster_seed if seed is None else seed
    if len(nodes) <= f:
        return [list(nodes)]

    vecs = _embed_matrix([n.summary for n in nodes])
    k = math.ceil(len(nodes) / f)
    idx_clusters = _kmeans(vecs, k, seed=seed, iters=CFG.kmeans_iters)

    bundles = []
    for idxs in idx_clusters:
        group = [nodes[i] for i in idxs]
        if len(group) <= f:
            bundles.append(group)
        else:
            # k-means left a cluster over the cap (for example many near-duplicate
            # summaries collapsing onto one centroid). Enforce the ceiling with a
            # deterministic contiguous slice split; this always makes progress, so
            # partitioning terminates even on degenerate embeddings.
            for j in range(0, len(group), f):
                bundles.append(group[j:j + f])
    return bundles


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING SUMMARIES - one bundle at a time, cached by child-id set
# ═════════════════════════════════════════════════════════════════════════════

ROUTING_SUMMARY_PROMPT = """\
You are writing a short routing summary for one branch of a navigation tree over
a single document. Below are the summaries of this branch's children. Write what
DISTINGUISHES these children from each other, so that a reader with a question
can tell whether the answer is likely under this branch and which child to open.
Do not recap the contents. Name the specific subjects, entities, or topics that
set these children apart. Be terse.

Write at most {max_tokens} tokens, one line.

CHILDREN:
{children}

DISTINGUISHING ROUTING SUMMARY:"""


def _bundle_key(child_nodes):
    """Stable cache key from the set of child ids."""
    ids = sorted(n.id for n in child_nodes)
    return hashlib.sha1("|".join(ids).encode()).hexdigest()


_cache_lock = threading.Lock()


def _summarize_bundle(child_nodes, cache):
    """Produce an s-token routing summary for a bundle, cached by child-id set.

    Thread-safe: the cache read/write are locked but the LLM call happens outside
    the lock so bundle summaries at one level run concurrently.
    """
    key = _bundle_key(child_nodes)
    with _cache_lock:
        if key in cache:
            return cache[key]

    listed = "\n".join(f"- {_truncate_tokens(c.summary, CFG.child_summary_tokens)}"
                       for c in child_nodes)
    prompt = ROUTING_SUMMARY_PROMPT.format(
        max_tokens=CFG.child_summary_tokens, children=listed)
    raw = llm(prompt, max_tokens=512, phase="pack_summary")
    summary = _truncate_tokens((raw or "").strip() or listed[:200],
                               CFG.child_summary_tokens)
    with _cache_lock:
        cache[key] = summary
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PACK - build the bounded-fanout tree for one category, then the root
# ═════════════════════════════════════════════════════════════════════════════

def _make_internal(children, category, idgen, cache, kind="internal"):
    level = 1 + max(c.level for c in children)
    summary = _summarize_bundle(children, cache)
    node = idgen.new(level=level, kind=kind, category=category,
                     summary=summary, children=[c.id for c in children])
    return node


def _make_internals(bundles, category, idgen, cache):
    """Summarize all bundles at one level concurrently, then create the internal
    nodes sequentially (id assignment stays single-threaded).
    """
    summaries = [None] * len(bundles)
    workers = min(CFG.pack_workers, max(1, len(bundles)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_summarize_bundle, b, cache): i for i, b in enumerate(bundles)}
        for fut in as_completed(futs):
            summaries[futs[fut]] = fut.result()

    nodes = []
    for bundle, summary in zip(bundles, summaries):
        level = 1 + max(c.level for c in bundle)
        nodes.append(idgen.new(level=level, kind="internal", category=category,
                               summary=summary, children=[c.id for c in bundle]))
    return nodes


def pack_level(nodes, category, idgen, cache):
    """Recursively pack a list of nodes into a single subtree root.

    If the list has <= f items, wrap them under one internal node and return it.
    Otherwise partition into coherent bundles of at most f, summarize each bundle
    into an internal node, and recurse on those internal nodes until the top has
    <= f children. The bundle summaries at each level are produced concurrently.
    """
    f = CFG.fanout
    if len(nodes) <= f:
        return _make_internal(nodes, category, idgen, cache)

    bundles = coherent_partition(nodes, max_size=f)
    internal_nodes = _make_internals(bundles, category, idgen, cache)
    return pack_level(internal_nodes, category, idgen, cache)


def pack(leaves_by_category, idgen, cache):
    """Build the whole packed tree for one source document.

    leaves_by_category: dict[category_name -> list[Node] leaves].
    idgen: the same _IdGen that created the leaves, so internal-node ids continue
    the leaf id sequence and never collide with a leaf id. Returns the root node;
    idgen.by_id then holds every node.
    """
    category_tops = []
    for cat, leaves in leaves_by_category.items():
        leaves = [l for l in leaves if l is not None]
        if not leaves:
            continue
        top = pack_level(leaves, cat, idgen, cache)
        top.category = cat
        category_tops.append(top)

    if not category_tops:
        return None

    f = CFG.fanout
    if len(category_tops) <= f:
        root = _make_internal(category_tops, "__root__", idgen, cache, kind="root")
    else:
        # more categories than f: pack the category tops too, then mark root
        root = pack_level(category_tops, "__root__", idgen, cache)
        root.kind = "root"

    # wire parent pointers from the root down
    _assign_parents(root, idgen.by_id)
    return root


def _assign_parents(root, by_id):
    root.parent = None
    stack = [root]
    while stack:
        node = stack.pop()
        for cid in node.children:
            child = by_id.get(cid)
            if child is not None:
                child.parent = node.id
                stack.append(child)


# ═════════════════════════════════════════════════════════════════════════════
# BUILD - discover (reuse) -> extract (reuse) -> pack (new) -> persist
# ═════════════════════════════════════════════════════════════════════════════

def _store_dir(output_dir, doc_id):
    return Path(output_dir) / safe_filename(doc_id)


def build_packed_index(doc_path, output_dir, doc_id, title):
    """Full v8 build for one source document. Returns tree stats or None.

    Caches the discover/extract output and the routing-summary set on disk so
    rebuilds do not re-run extraction or re-summarize unchanged bundles.
    """
    store = _store_dir(output_dir, doc_id)
    store.mkdir(parents=True, exist_ok=True)

    # --- discover + extract (reused from v7, cached) --------------------------
    compiled_path = store / "compiled.json"
    template_path = store / "template.json"
    if compiled_path.exists() and template_path.exists():
        template = json.loads(template_path.read_text())
        compiled = json.loads(compiled_path.read_text())
        log.info(f"  {doc_id}: reusing cached extract output")
    else:
        template = discover_schema(doc_path)
        if not template:
            log.warning(f"  Discovery failed, skipping {doc_id}")
            return None
        compiled = extract(doc_path, template)
        if not compiled:
            log.warning(f"  Extraction failed for {doc_id}")
            return None
        template_path.write_text(json.dumps(template, indent=2))
        compiled_path.write_text(json.dumps(compiled, indent=2, default=str))

    # --- leaves = extracted subjects, grouped by category ---------------------
    idgen = _IdGen(doc_id)
    leaves_by_category = {}
    for section in template["sections"]:
        cat = section["name"]
        sdata = _find_section(compiled, cat)
        if not sdata:
            continue
        leaves = _leaves_for_category(cat, sdata, idgen)
        if leaves:
            leaves_by_category[cat] = leaves

    total_leaves = sum(len(v) for v in leaves_by_category.values())
    if total_leaves == 0:
        log.warning(f"  No leaves extracted for {doc_id}")
        return None

    # --- pack (new step) with a persisted routing-summary cache ---------------
    cache_path = store / "summary_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    root = pack(leaves_by_category, idgen, cache)
    cache_path.write_text(json.dumps(cache, indent=2))
    if root is None:
        log.warning(f"  Packing produced no root for {doc_id}")
        return None

    # --- persist the node store -----------------------------------------------
    nodes = list(idgen.by_id.values())
    store_payload = {
        "doc_id": doc_id, "title": title, "root": root.id,
        "config": {"C": CFG.node_budget_tokens, "s": CFG.child_summary_tokens,
                   "f": CFG.fanout, "cluster_seed": CFG.cluster_seed},
        "nodes": [n.to_dict() for n in nodes],
    }
    (store / "packed.json").write_text(json.dumps(store_payload, indent=2))

    stats = tree_stats(nodes, root, CFG.fanout, total_leaves)
    _print_stats(doc_id, stats)
    _assert_invariants(stats, CFG.fanout, leaves_by_category)
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# STATS + ACCEPTANCE ASSERTIONS
# ═════════════════════════════════════════════════════════════════════════════

def tree_stats(nodes, root, f, total_leaves):
    by_id = {n.id: n for n in nodes}
    depth = max(n.level for n in nodes)
    per_level = Counter(n.level for n in nodes)
    fanouts = [len(n.children) for n in nodes if n.kind in ("internal", "root")]
    fanout_dist = Counter(fanouts)
    max_fanout = max(fanouts) if fanouts else 0
    expected_depth = math.ceil(math.log(max(total_leaves, 2), f))

    # per-category depth (categories are the top-level grouping)
    per_cat = {}
    for cid in root.children:
        child = by_id.get(cid)
        if child is None:
            continue
        cat_leaves = _count_leaves(child, by_id)
        cat_depth = _subtree_depth(child, by_id)
        per_cat[child.category] = {
            "leaves": cat_leaves, "depth": cat_depth,
            "expected": math.ceil(math.log(max(cat_leaves, 2), f)),
        }

    return {
        "doc_leaves": total_leaves,
        "n_nodes": len(nodes),
        "depth": depth,
        "expected_depth": expected_depth,
        "nodes_per_level": dict(sorted(per_level.items())),
        "fanout_max": max_fanout,
        "fanout_dist": dict(sorted(fanout_dist.items())),
        "n_categories": len(root.children),
        "per_category": per_cat,
    }


def _count_leaves(node, by_id):
    if node.kind == "leaf":
        return 1
    return sum(_count_leaves(by_id[c], by_id) for c in node.children if c in by_id)


def _subtree_depth(node, by_id):
    if node.kind == "leaf" or not node.children:
        return 0
    return 1 + max(_subtree_depth(by_id[c], by_id) for c in node.children if c in by_id)


def _print_stats(doc_id, s):
    log.info(f"  packed[{doc_id}]: {s['doc_leaves']} leaves, {s['n_nodes']} nodes, "
             f"depth {s['depth']} (expected ~{s['expected_depth']}), "
             f"f_max {s['fanout_max']}")
    log.info(f"    nodes per level: {s['nodes_per_level']}")
    log.info(f"    fanout distribution (children -> count): {s['fanout_dist']}")
    for cat, cs in s["per_category"].items():
        log.info(f"    category '{cat}': {cs['leaves']} leaves, "
                 f"depth {cs['depth']} (expected ~{cs['expected']})")


def _assert_invariants(stats, f, leaves_by_category):
    # Hard invariant: no internal node exceeds f children.
    assert stats["fanout_max"] <= f, (
        f"fanout invariant violated: a node has {stats['fanout_max']} children > f={f}")
    # Depth should be about ceil(log_f(N)); with category grouping we check it
    # per category (each category is an independent packed subtree) and overall.
    ok = abs(stats["depth"] - (stats["expected_depth"] + 1)) <= 1 or \
        abs(stats["depth"] - stats["expected_depth"]) <= 1
    if not ok:
        log.warning(f"    depth {stats['depth']} is not within 1 of "
                    f"expected {stats['expected_depth']} (+root). Review.")
    return ok


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: bounded-fanout traversal + answering
# ═════════════════════════════════════════════════════════════════════════════
#
# Traversal is a best-first search that, at each internal node, scores ONLY that
# node's children (at most f) using their routing summaries and descends into the
# best. Backtracking, level-ups and lateral hops all draw from one shared budget
# (traverse_budget), so total work stays O(log N times budget). A no-revisit set
# guarantees termination. nodes_scored (the count of child summaries scored) is
# the headline metric: it grows like log of the leaf count, not linearly.
#
# Four routing modes (config flag route_mode), run as an ablation:
#   keyword : the same lexical scorer v7 uses (matched-mode shape-isolation floor)
#   cosine  : cosine of question vs child routing-summary vectors (all-mpnet-base-v2)
#   hybrid  : cosine, plus one LLM tie-break call when the top two are within epsilon
#   llm     : one LLM call per node to pick the child

def load_store(store_dir):
    """Load a persisted packed index: returns (payload, by_id) with Node objects."""
    data = json.loads((Path(store_dir) / "packed.json").read_text())
    by_id = {}
    for nd in data["nodes"]:
        by_id[nd["id"]] = Node(**nd)
    return data, by_id


# --- routing embedder (all-mpnet-base-v2, cached local model) -----------------

_ROUTE_MODEL = None
_ROUTE_MODEL_NAME = None


def _route_model():
    """Lazy-load the sentence-transformers routing model named in config."""
    global _ROUTE_MODEL, _ROUTE_MODEL_NAME
    if _ROUTE_MODEL is None or _ROUTE_MODEL_NAME != CFG.route_embed_model:
        from sentence_transformers import SentenceTransformer
        _ROUTE_MODEL = SentenceTransformer(CFG.route_embed_model)
        _ROUTE_MODEL_NAME = CFG.route_embed_model
    return _ROUTE_MODEL


def _route_passage(text):
    """Passage-side text for routing. all-mpnet is symmetric so no prefix. Upgrade
    path: for an asymmetric model such as intfloat/e5-large-v2, return
    'passage: ' + text here and 'query: ' + question in _route_query (one-line swap)."""
    return text


def _route_query(text):
    return text


def _route_vectors(store_dir, by_id):
    """Embed every node's routing summary once with the route model; cache to disk
    so answering does no build-time embedding work. Keyed by node-id list + model."""
    import numpy as np
    cache = Path(store_dir) / "route_vectors.npz"
    ids = list(by_id.keys())
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["ids"]) == ids and str(d["model"]) == CFG.route_embed_model:
            return {i: v for i, v in zip(d["ids"], d["vecs"])}
    texts = [_route_passage(by_id[i].summary or "") for i in ids]
    vecs = _route_model().encode(texts, normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype="float32")
    np.savez(cache, ids=np.array(ids), vecs=vecs, model=CFG.route_embed_model)
    return {i: v for i, v in zip(ids, vecs)}


def _q_vector(question):
    import numpy as np
    v = _route_model().encode([_route_query(question)], normalize_embeddings=True)[0]
    return np.asarray(v, dtype="float32")


# --- child scoring per mode ---------------------------------------------------

ROUTE_PICK_PROMPT = """\
You are navigating a document index to answer a question. Below are numbered
branches, each with a short routing summary describing what distinguishes it.
Pick the single branch most likely to contain the answer. Reply with ONLY the
branch number.

Question: {question}

Branches:
{branches}

Number:"""


def _kw_overlap(qwords, text):
    return len(qwords & set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower())))


def _llm_pick(question, children, metrics):
    branches = "\n".join(f"{i}. {c.summary}" for i, c in enumerate(children))
    raw = llm(ROUTE_PICK_PROMPT.format(question=question, branches=branches),
              max_tokens=16, phase="query_route")
    metrics["llm_route_calls"] += 1
    m = re.search(r"\d+", raw or "")
    idx = int(m.group()) if m else 0
    return idx if 0 <= idx < len(children) else 0


def _score_children(question, qwords, children, mode, rvecs, qvec, metrics):
    """Return one score per child (higher = better). Only these children are
    ever scored at this node; that count is added to nodes_scored by the caller."""
    if mode == "keyword":
        return [_kw_overlap(qwords, c.summary) for c in children]

    if mode in ("cosine", "hybrid"):
        import numpy as np
        cos = [float(np.dot(qvec, rvecs[c.id])) for c in children]
        if mode == "cosine":
            return cos
        # hybrid: if the top two are within epsilon, spend one LLM call to break
        # the tie among just the close candidates. Rare, so LLM routing stays cheap.
        order = sorted(range(len(children)), key=lambda i: -cos[i])
        if len(order) >= 2 and (cos[order[0]] - cos[order[1]]) < CFG.route_epsilon:
            close = [i for i in order if (cos[order[0]] - cos[i]) < CFG.route_epsilon]
            pick_local = _llm_pick(question, [children[i] for i in close], metrics)
            boosted = list(cos)
            boosted[close[pick_local]] = max(cos) + 1.0
            return boosted
        return cos

    if mode == "llm":
        pick = _llm_pick(question, children, metrics)
        scores = [0.0] * len(children)
        scores[pick] = 1.0
        return scores

    raise ValueError(f"unknown route_mode: {mode}")


def traverse_packed(question, by_id, root_id, mode, rvecs=None, qvec=None):
    """Bounded-width beam descent. Returns (leaf_contents, metrics).

    At each level we score ONLY the children of the current beam nodes (each at
    most f) and keep the top beam_width internal branches for the next level; any
    leaf encountered goes into a scored pool. Keeping the top few branches (not
    one) lets a question pull facts from more than one category, which is what a
    single greedy path missed. Work per question is about beam_width times f times
    depth, still far below the leaf count and still growing like log of it. The
    shared budget caps pathological cases; the document-scoped raw fallback
    handles misses.
    """
    metrics = {"nodes_scored": 0, "depth_reached": 0, "level_ups": 0,
               "hops": 0, "llm_route_calls": 0}
    qwords = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))

    beam = [by_id[root_id]]
    visited = {root_id}
    leaf_pool = []          # (score, content) gathered across branches
    depth = 0

    while beam and metrics["nodes_scored"] < CFG.traverse_budget:
        next_internal = []  # (score, node) candidates for the next level
        for node in beam:
            children = [by_id[c] for c in node.children
                        if c in by_id and c not in visited]
            if not children:
                continue
            scores = _score_children(question, qwords, children, mode,
                                     rvecs, qvec, metrics)
            metrics["nodes_scored"] += len(children)
            for child, s in zip(children, scores):
                visited.add(child.id)
                if child.kind == "leaf":
                    if child.content:
                        leaf_pool.append((float(s), child.content))
                else:
                    next_internal.append((float(s), child))
        if not next_internal:
            break
        next_internal.sort(key=lambda t: -t[0])
        beam = [n for _, n in next_internal[:CFG.beam_width]]
        depth += 1
        metrics["depth_reached"] = depth

    # gather the top-scoring leaves across the branches we opened
    leaf_pool.sort(key=lambda t: -t[0])
    seen, collected = set(), []
    for _, content in leaf_pool:
        if content in seen:
            continue
        seen.add(content)
        collected.append(content)
        if len(collected) > 1:
            metrics["hops"] += 1
        if len(collected) >= CFG.max_answer_leaves:
            break

    return collected, metrics


# --- consolidated overview (as v7 has), generated once per document -----------

def _ensure_overview(store_dir, by_id, title):
    """Return a dense per-document overview, generating and caching it once.

    Mirrors v7, whose traversal always reads a consolidated overview page. Built
    from the leaf contents in one LLM call, tagged to the build bucket (phase
    'consolidate') so it is one-time build cost, not per-question answer cost."""
    if not CFG.use_overview:
        return ""
    path = Path(store_dir) / "overview.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")

    leaves = [n.content for n in by_id.values() if n.kind == "leaf" and n.content]
    combined = "\n\n".join(leaves)
    if len(combined) > 80_000:
        combined = combined[:80_000] + "\n... [truncated]"
    prompt = CONSOLIDATE_PROMPT.format(
        title=title, content=combined, target_chars=CONTEXT_BUDGET // 2)
    overview = llm(prompt, max_tokens=16384, phase="consolidate") or ""
    path.write_text(overview)
    return overview


# --- answering (reuse v7 answer + document-scoped fallback) -------------------

def _answer_from_leaves(question, collected, raw_doc_path, overview=""):
    """Answer from the gathered leaves; fall back to the raw source document on a
    miss, scoped to that single document, exactly like v7."""
    # answer context = the dense overview (breadth, as v7 has) + the routed leaves
    # (depth). The overview is one-time build cost; routing supplies the specifics.
    if overview or collected:
        parts = []
        if overview:
            parts.append("Overview:\n" + overview)
        if collected:
            parts.append("Relevant detail pages:\n" + "\n\n---\n\n".join(collected))
        digest = "\n\n====\n\n".join(parts)
        if len(digest) > CFG.answer_char_budget:
            digest = digest[:CFG.answer_char_budget]
        qa_prompt = (
            "You are answering a question about a document using ONLY the compiled "
            "wiki digest below.\nAnswer in 10 words or fewer. If the answer is not in "
            "the digest, return exactly: not found\n\n"
            f"Question: {question}\n\nWiki Digest:\n{digest}\n\nAnswer:"
        )
        predicted = llm(qa_prompt, max_tokens=2048, phase="query_answer")
        predicted = (predicted or "").strip().splitlines()[0].strip(" .\"'") \
            if predicted else "not found"
        if predicted.lower() not in ("not found", "", "none", "n/a"):
            return predicted, "wiki"

    if raw_doc_path and Path(raw_doc_path).exists():
        text = clean_document(Path(raw_doc_path).read_text(encoding="utf-8", errors="ignore"))
        qwords = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
        chunks = chunk_text(text, size=20000, overlap=2000)
        best, best_score = chunks[0], 0
        for chunk in chunks:
            words = set(re.findall(r"[a-zA-Z]{3,}", chunk.lower()))
            score = len(qwords & words)
            if score > best_score:
                best, best_score = chunk, score
        raw = llm(FALLBACK_QA_PROMPT.format(question=question, excerpt=best),
                  max_tokens=4096, phase="query_fallback")
        result = parse_json(raw)
        if result:
            predicted = result.get("answer", "not found")
            if predicted.lower() not in ("not found", "", "none", "n/a"):
                return predicted, "fallback"
            return "not found", "miss"
        answer = (raw or "").strip().splitlines()[0].strip(" .\"'") if raw else "not found"
        if answer.lower() not in ("not found", "", "none", "n/a"):
            return answer, "fallback"

    return "not found", "miss"


def query_packed(question, store_dir, raw_doc_path, route_mode=None):
    """Answer one question over the packed index. Returns (predicted, source, metrics).

    metrics carries the Phase 3 instrumentation: nodes_scored, leaf_count,
    depth_reached, level_ups, hops, llm_route_calls.
    """
    mode = route_mode or CFG.route_mode
    data, by_id = load_store(store_dir)

    rvecs = qvec = None
    if mode in ("cosine", "hybrid"):
        rvecs = _route_vectors(store_dir, by_id)
        qvec = _q_vector(question)

    collected, metrics = traverse_packed(question, by_id, data["root"], mode, rvecs, qvec)
    metrics["leaf_count"] = sum(1 for n in by_id.values() if n.kind == "leaf")
    metrics["route_mode"] = mode
    overview = _ensure_overview(store_dir, by_id, data.get("title", ""))
    predicted, source = _answer_from_leaves(question, collected, raw_doc_path, overview)
    return predicted, source, metrics
