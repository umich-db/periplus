"""Phases 2 and 3: propose categories per chunk, then merge to one canonical list.

Phase 2 (propose): one LLM call per chunk lists the top-level categories present in
that chunk. Labels only, no content. Never more than one chunk per call. Calls run
concurrently (propose_workers) but results are collected by chunk id so the union
order is deterministic.

Phase 3 (merge): NO LLM call. Merge is a DEDUP step, not a balance step: collapse only
true near-duplicate labels and let distinct topics survive, so categories stay meaningful
(Phase 4 files subjects under them by the label-to-canonical map). This is similarity-
thresholded grouping, NOT pack's ceil(n/f) clustering (that is a compression target and
would force distinct topics into grab-bags). Merge = case/whitespace exact-dedup, then
embed each distinct label with the local all-mpnet model, connect two labels whose cosine
similarity is at or above merge_similarity, and take connected components as categories,
each named by its highest-frequency label. One defensible dedup threshold, no LLM tokens,
deterministic. pack's ceil(n/f) routine (cluster.py) is used by pack only.

Vocabulary: root, category, subject, field, source document.
"""
import sys, re, json, hashlib, pathlib, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import chunk as _chunk  # noqa: E402

PROPOSE_PROMPT = (
    "Below is one passage from a source document. List the top-level categories "
    "(topics) present in this passage. Return ONLY a JSON array of short category "
    "labels (one to three words each), no content and no explanation.\n\n"
    "Passage:\n{chunk}\n\nJSON array of category labels:"
)


def _parse_labels(text):
    """Parse a JSON array of strings; tolerate code fences and stray prose."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:  # noqa: BLE001
            pass
    return [ln.strip(" -*\t\"'") for ln in t.splitlines() if ln.strip(" -*\t\"'")]


def propose_one(chunk_text):
    # gpt-5 is a reasoning model: give headroom so hidden reasoning does not starve the
    # visible label list to an empty response.
    text, pt, ct = client.chat(PROPOSE_PROMPT.format(chunk=chunk_text),
                               max_tokens=4096, bucket="index_build")
    return _parse_labels(text), pt, ct


def propose(chunks, workers=8, cache_path=None):
    """One call per chunk, concurrent, collected by chunk id (deterministic order). An
    optional persisted cache keyed by a hash of the chunk text makes rebuilds reproducible
    and cheap: a chunk whose text is unchanged reuses its cached labels with no LLM call
    (same pattern pack uses for routing summaries). Returns (union, per_chunk, tokens)."""
    cache = {}
    if cache_path and pathlib.Path(cache_path).exists():
        cache = json.loads(pathlib.Path(cache_path).read_text())
    keys = [hashlib.sha1(c["text"].encode("utf-8")).hexdigest() for c in chunks]
    per_chunk = [None]*len(chunks)
    tok = {"in": 0, "out": 0}
    def _work(i):
        if keys[i] in cache:
            return i, cache[keys[i]], 0, 0, False
        labels, pt, ct = propose_one(chunks[i]["text"])
        return i, labels, pt, ct, True
    updated = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, labels, pt, ct, is_new in ex.map(_work, range(len(chunks))):
            per_chunk[i] = labels
            tok["in"] += pt; tok["out"] += ct
            if is_new:
                cache[keys[i]] = labels; updated = True
    if cache_path and updated:
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False))
    union = [lab for labels in per_chunk for lab in labels]   # duplicates kept, chunk order
    return union, per_chunk, tok


def _norm_key(label):
    """Case- and whitespace-insensitive key for exact deduplication."""
    return re.sub(r"\s+", " ", label.strip().lower())


def _components(n, edges):
    """Union-find connected components over a node set of size n and an edge list.
    Deterministic (attach to lower index; caller supplies edges in fixed order)."""
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [comps[k] for k in sorted(comps)]


def _singular(word):
    """Crude, deterministic singularization for lexical matching."""
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-3] + "y" if w.endswith("ies") else w[:-1]
    return w


def _tokens(label):
    return [_singular(t) for t in re.findall(r"[a-z0-9]+", label.lower())]


def _is_prefix(a, b):
    """True if token list `a` is a proper LEADING prefix of `b`. Leading (not any sublist)
    so a shared modifier folds (Siblings -> Sibling rivalry) but a shared trailing head word
    does NOT bridge distinct topics (Conflict does not join Family conflict + Sibling conflict)."""
    return 0 < len(a) < len(b) and b[:len(a)] == a


def _lexical_edges(labels):
    """Fold morphological families with no threshold: connect two labels when one's token
    phrase is a leading prefix of the other's (Siblings in 'Sibling rivalry'; High school in
    'High school life'). Same leading phrase, same category. Trivially defensible, and prefix
    (not subphrase) prevents a generic trailing word from chaining distinct topics."""
    toks = [_tokens(l) for l in labels]
    edges = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and _is_prefix(toks[i], toks[j]):
                edges.append((i, j))
    return edges


def merge(union, embed_model, similarity):
    """NO LLM. DEDUP only (not balance), so categories stay meaningful topics. Layers:
      1. case/whitespace exact-dedup, keeping frequency per distinct label,
      2. LEXICAL subphrase fold (no threshold): a shorter label phrase contained in a longer
         one is the same category (Siblings/Sibling rivalry; High school/High school life).
         This catches the morphological families and cannot cross topics.
      3. STRICT embedding fold: connect two labels whose cosine similarity >= `similarity`,
         for the true semantic synonyms lexical misses (Alcohol/Drinking). Kept strict so it
         under-merges rather than chains distinct topics.
    Categories are the connected components over layers 2+3, each named by its highest-
    frequency label. Biased strict on purpose: a missed fold costs one extra coherent
    routing box (pack absorbs it); a wrong fold misfiles subjects. Returns (categories,
    mapping, tokens) with tokens identically zero; deterministic."""
    counts = Counter(union)
    distinct = sorted(counts)

    # layer 1: exact case/whitespace dedup -> one canonical spelling per key
    by_key = {}
    for lab in distinct:
        by_key.setdefault(_norm_key(lab), []).append(lab)
    map1, dedup_count = {}, Counter()
    for variants in by_key.values():
        canon = sorted(variants, key=lambda v: (-counts[v], v))[0]
        for v in variants:
            map1[v] = canon
            dedup_count[canon] += counts[v]
    deduped = sorted(dedup_count)

    # layers 2+3: lexical subphrase edges + strict embedding edges -> connected components
    if len(deduped) <= 1:
        cats, map2 = list(deduped), {d: d for d in deduped}
    else:
        edges = list(_lexical_edges(deduped))
        vecs = client.embed(deduped, model_name=embed_model)     # L2-normalized rows
        sim = vecs @ vecs.T
        for i in range(len(deduped)):
            for j in range(i+1, len(deduped)):
                if sim[i, j] >= similarity:
                    edges.append((i, j))
        cats, map2 = [], {}
        for comp in _components(len(deduped), edges):
            members = [deduped[i] for i in comp]
            canon = sorted(members, key=lambda m: (-dedup_count[m], m))[0]
            cats.append(canon)
            for m in members:
                map2[m] = canon
        cats = sorted(set(cats))

    mapping = {orig: map2[map1[orig]] for orig in distinct}   # every raw label -> a category
    return cats, mapping, {"in": 0, "out": 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--phase", choices=["propose", "merge"], default="merge")
    a = ap.parse_args()
    cfg = _cfg.load()
    chunks = _chunk.chunk_document(_chunk.load_source(a.movie), cfg.chunk_target, cfg.B)
    cache_path = HERE/"output"/a.movie/"propose_cache.json"
    union, per_chunk, ptok = propose(chunks, workers=cfg.propose_workers, cache_path=cache_path)
    counts = Counter(union)

    if a.phase == "propose":
        print(json.dumps({
            "movie": a.movie, "n_chunks": len(chunks), "union_size": len(union),
            "distinct_labels": len(counts), "build_tokens": ptok["in"]+ptok["out"],
        }, indent=2, ensure_ascii=False)); sys.exit(0)

    # Phase 3: embedding dedup merge (no LLM). Determinism is structural: run twice, compare.
    cats1, map1, mtok = merge(union, cfg.embedding_model, cfg.merge_similarity)
    cats2, map2, _ = merge(union, cfg.embedding_model, cfg.merge_similarity)
    merge_tokens = mtok["in"] + mtok["out"]
    # acceptance asserts (fail loudly)
    assert merge_tokens == 0, f"merge made LLM calls: {merge_tokens} tokens (LLM path still wired in)"
    assert cats1 == cats2 and map1 == map2, "merge is not deterministic"
    assert all(lab in map1 and map1[lab] in cats1 for lab in counts), "some raw label maps to no category"
    folds = sorted([(lab, map1[lab]) for lab in counts if lab != map1[lab]])[:12]
    print(json.dumps({
        "movie": a.movie, "n_chunks": len(chunks),
        "union_size": len(union), "distinct_labels": len(counts),
        "merge_similarity": cfg.merge_similarity,
        "n_canonical_categories": len(cats1), "f": cfg.f,
        "top_layer_exceeds_f (will be packed later)": len(cats1) > cfg.f,
        "categories": cats1,
        "every_raw_label_mapped_to_one_category": all(lab in map1 and map1[lab] in cats1 for lab in counts),
        "example_folds(redundant -> canonical)": {k: v for k, v in folds},
        "deterministic_categories": cats1 == cats2,
        "deterministic_mapping": map1 == map2,
        "propose_build_tokens": ptok["in"]+ptok["out"],
        "merge_build_tokens": merge_tokens,
    }, indent=2, ensure_ascii=False))
