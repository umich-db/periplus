"""Phase 7: the answer path. Walk the packed index to answer one question, bounded.

Load the written index (store.load_index, no rebuild). Route from the root by scoring
each node's children (at most f routing summaries) and descending the best, with backtrack
and lateral moves, all drawing from one shared budget = budget_a * depth + budget_b. Three
bounds hold together: the shared move budget, a no-revisit set (never score or enter a node
twice), and stop-on-good-enough (a reached leaf at or above confidence_bar answers now). The
answer is produced by one LLM call over the C-token window of routing summaries and leaf
contents collected along the walk, using v7's abstain answer prompt for comparability. On a
miss the fallback re-reads only the source spans linked from the leaves reached (plus their
nearest siblings); the wide fallback (opt-in) re-reads the whole source document once.

The headline instrumentation is nodes_scored: the total routing summaries scored for the
question, which grows like log of the leaf count, not linearly. Recorded per question along
with answer_tokens, leaf_count, depth_reached, backtracks, hops, budget_spent, and fallback.

Routing embeddings are transient: cached in memory per source document across its questions,
never written as a query index.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, re, json, time, pathlib, argparse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import chunk as _chunk, store as _store  # noqa: E402

BENCH = _BENCH
SLUGS = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
         "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
         "a_dry_white_season", "a_few_good_men"]

# v7's abstain answer prompt, reproduced verbatim so the two methods answer under identical
# instructions (same model, same reasoning, same 10-word cap, same "not found"). Kept as-is
# for comparability; do not reword.
ANSWER_PROMPT = (
    "You are answering a question about a document using ONLY the compiled "
    "wiki digest below.\nAnswer in 10 words or fewer. If the answer is not in the "
    "digest, return exactly: not found\n\n"
    "Question: {question}\n\nWiki Digest:\n{digest}\n\nAnswer:"
)

ROUTE_LLM_PROMPT = (
    "A question is being routed through an index. Below are numbered groups. Pick the one "
    "group most likely to contain the answer. Return ONLY its number.\n\n"
    "Question: {question}\n\n{blocks}\n\nBest group number:"
)

_MISS = ("not found", "", "none", "n/a")


def _routing_text(node):
    if node["kind"] == "leaf":
        return node["content"] or ""
    return node["summary"] or node["category"] or ""


# ---------- routing modes (how a node's children are scored) ----------
class Router:
    """Score child nodes against one question. cosine (default) embeds with all-mpnet and
    caches each node's routing-summary embedding in `emb_cache` (a per-source-document dict,
    transient, never persisted); keyword reuses v7's scorer read-only; llm is an ablation."""

    def __init__(self, question, cfg, emb_cache):
        self.q = question
        self.mode = cfg.routing_mode
        self.embed_model = cfg.embedding_model
        self.emb_cache = emb_cache
        self.route_tokens = 0
        if self.mode == "cosine":
            self.qv = client.embed([question], model_name=self.embed_model)[0]
        elif self.mode == "keyword":
            # v7's keyword scorer, imported read-only (never modified). Lazy import so the
            # default cosine path does not pull in v7's module.
            self._v7_score = _load_v7_score()
            self.qwords = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
        elif self.mode != "llm":
            raise ValueError(f"unknown routing_mode {self.mode!r}")

    def score_nodes(self, nodes):
        if self.mode == "cosine":
            miss = [n for n in nodes if n["id"] not in self.emb_cache]
            if miss:
                vecs = client.embed([_routing_text(n) for n in miss], model_name=self.embed_model)
                for n, v in zip(miss, vecs):
                    self.emb_cache[n["id"]] = v
            import numpy as np
            M = np.stack([self.emb_cache[n["id"]] for n in nodes])
            return [float(x) for x in (M @ self.qv)]
        if self.mode == "keyword":
            return [float(self._v7_score(self.qwords, _routing_text(n))) for n in nodes]
        # llm: one pick per expansion (ablation; billed to a routing bucket, not answer)
        blocks = "\n".join(f"{i}: {' '.join(_routing_text(n).split())[:160]}" for i, n in enumerate(nodes))
        text, pt, ct = client.chat(ROUTE_LLM_PROMPT.format(question=self.q, blocks=blocks),
                                   max_tokens=2048, bucket="index_route")
        self.route_tokens += pt + ct
        m = re.search(r"\d+", text or "")
        pick = int(m.group(0)) if m else 0
        pick = pick if 0 <= pick < len(nodes) else 0
        return [1.0 if i == pick else 0.0 for i in range(len(nodes))]


_V7_SCORE = None


def _load_v7_score():
    """Return v7's pipeline._score, imported read-only. v7 is not modified."""
    global _V7_SCORE
    if _V7_SCORE is None:
        wiki_src = _BENCH/"src"/"wiki_index"
        sys.path.insert(0, str(wiki_src))
        import pipeline as _v7pipe  # noqa: E402
        _V7_SCORE = _v7pipe._score
    return _V7_SCORE


# ---------- traversal (bounded, no-revisit, stop-on-good-enough) ----------
def traverse(idx, router, cfg):
    """Route from the root, scoring at most f children per descent and following the best,
    with backtrack and lateral moves from one shared budget. Returns the leaves reached in
    reach order (the answer reads their source spans) plus the routing diagnostics. Scoring
    the children is the only work counted in nodes_scored; the answer context is built later."""
    nodes, root_id = idx["nodes"], idx["root"]
    tree_depth = nodes[root_id]["level"]
    budget_total = cfg.budget_a * tree_depth + cfg.budget_b
    budget = budget_total

    visited = {root_id}
    nodes_scored = 0
    backtracks = hops = depth_reached = 0
    reached_leaves = []
    good_enough = False

    def depth_of(node):
        return tree_depth - node["level"]

    frontier = []                                             # {id, score, parent, depth}

    def expand(node):
        nonlocal nodes_scored
        kids = [nodes[cid] for cid in node["children"] if cid not in visited]
        if not kids:
            return
        scores = router.score_nodes(kids)                     # at most f children scored
        nodes_scored += len(kids)
        for k, sc in zip(kids, scores):
            visited.add(k["id"])
            frontier.append({"id": k["id"], "score": sc, "parent": node["id"], "depth": depth_of(k)})

    last = nodes[root_id]
    expand(nodes[root_id])

    while budget > 0 and frontier and not good_enough:
        frontier.sort(key=lambda e: (-e["score"], str(e["id"])))   # deterministic best-first
        cur = frontier.pop(0)
        budget -= 1
        node = nodes[cur["id"]]
        # classify the move against the node we last expanded: continuing down the same line,
        # backtracking up to a sibling or shallower candidate, or hopping laterally to a
        # deeper node in a different branch. All three spend the one shared budget.
        if cur["parent"] != last["id"]:
            if node["level"] >= last["level"]:
                backtracks += 1
            else:
                hops += 1
        depth_reached = max(depth_reached, cur["depth"])
        if node["kind"] == "leaf":
            reached_leaves.append(node)
            if cur["score"] >= cfg.confidence_bar:
                good_enough = True
        else:
            expand(node)
        last = node

    return {"reached_leaves": reached_leaves,
            "nodes_scored": nodes_scored, "depth_reached": depth_reached,
            "backtracks": backtracks, "hops": hops,
            "budget_spent": budget_total - budget, "budget_total": budget_total,
            "route_tokens": router.route_tokens}


def _leaf_chunks(idx, leaves):
    """Deduped source-span chunk texts for a set of leaves, in order, each with its chunk id."""
    chunks = idx["chunks"]
    seen, out = set(), []
    for lf in leaves:
        cid = lf.get("source", {}).get("chunk_id")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        ck = chunks.get(cid) or chunks.get(str(cid))
        if ck:
            out.append((cid, ck["text"]))
    return out


def _digest(overview, chunk_texts, char_budget, overview_cap=1500):
    """Assemble the C-window answer context: a trimmed whole-document overview (broad-question
    coverage) followed by the actual source spans of the reached region, up to the budget."""
    pieces, used = [], 0
    ov = (overview or "").strip()
    if ov:
        ov = ov[:overview_cap]; pieces.append(ov); used += len(ov)
    for _cid, txt in chunk_texts:
        if used >= char_budget:
            break
        t = txt[:char_budget - used]; pieces.append(t); used += len(t)
    return "\n\n---\n\n".join(pieces)


# ---------- answering + fallback ----------
def _answer_call(question, digest):
    if not digest.strip():
        return "not found", 0, 0
    text, pt, ct = client.chat(ANSWER_PROMPT.format(question=question, digest=digest),
                               max_tokens=2048, bucket="index_answer")
    pred = (text or "").strip().splitlines()[0].strip(" .\"'") if text else "not found"
    return pred, pt, ct


def _sibling_leaves(idx, reached_leaves):
    """The reached leaves plus their nearest siblings (other leaves under the same parent box)."""
    nodes = idx["nodes"]
    out, seen = [], set()
    for lf in reached_leaves:
        for lid in [lf["id"]] + list(nodes.get(lf.get("parent") or "", {}).get("children", [])):
            n = nodes.get(lid)
            if n and n["kind"] == "leaf" and lid not in seen:
                seen.add(lid); out.append(n)
    return out


def _doc_leaf_topk(idx, question, cfg, emb_cache, k=None):
    """Document-wide leaf selector. Score every subject leaf in the source document by query-time
    cosine of its content against the question, and return the top k leaves (default fallback_k). The
    embeddings are computed over the leaves already in memory and cached transiently in emb_cache (the
    same per-source-document dict routing uses); nothing is stored as a persisted query index."""
    import numpy as np
    k = cfg.fallback_k if k is None else k
    leaves = [n for n in idx["nodes"].values() if n["kind"] == "leaf"]
    miss = [lf for lf in leaves if lf["id"] not in emb_cache]
    if miss:
        vecs = client.embed([_routing_text(lf) for lf in miss], model_name=cfg.embedding_model)
        for lf, v in zip(miss, vecs):
            emb_cache[lf["id"]] = v
    qv = client.embed([question], model_name=cfg.embedding_model)[0]
    scores = np.stack([emb_cache[lf["id"]] for lf in leaves]) @ qv
    order = np.argsort(-scores)[:k]
    return [leaves[int(i)] for i in order]


def _subject_digest(idx, question, cfg, emb_cache, reached_leaves, overview):
    """Compile a digest by SELECTING index's own extracted subject content over the answer-relevant
    region (the reached leaves plus the top doc_leaves leaves), the analog of v7 selecting its
    precompiled digest pages. The subjects are the compiled facts index already extracted (Name ::
    field: value); this feeds those to the answer model instead of the reached leaf's raw source chunk.
    No query-time summarize call: selection only, like v7. Deterministic, capped at digest_chars."""
    region = _doc_leaf_topk(idx, question, cfg, emb_cache, k=cfg.digest_k)
    ordered, seen = [], set()
    for lf in list(reached_leaves) + region:               # reached first, then the doc_leaves region
        if lf["id"] not in seen:
            seen.add(lf["id"]); ordered.append(lf)
    pieces, used = [], 0
    ov = (overview or "").strip()[:1500]
    if ov:
        pieces.append(ov); used += len(ov)
    for lf in ordered:
        t = (lf.get("content") or "").strip()
        if not t or used + len(t) > cfg.digest_chars:
            if used >= cfg.digest_chars:
                break
            continue
        pieces.append("- " + t); used += len(t) + 2
    return "\n".join(pieces)


def _wide_digest(slug, question, char_budget):
    """Whole-document re-read: keyword-pick the best chunk (v7 style) and answer from it."""
    text = _chunk.load_source(slug)
    qwords = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    chunks = _chunk.chunk_document(text, 4000, 16000)
    best, best_s = (chunks[0]["text"] if chunks else ""), -1
    for c in chunks:
        s = len(qwords & set(re.findall(r"[a-zA-Z]{3,}", c["text"].lower())))
        if s > best_s:
            best, best_s = c["text"], s
    return best[:char_budget]


def answer_question(idx, slug, question, cfg, emb_cache):
    router = Router(question, cfg, emb_cache)
    tr = traverse(idx, router, cfg)
    char_budget = cfg.C * 4
    overview = idx.get("overview")

    # primary answer context. raw_chunk (default): the reached region's raw source spans. digest:
    # index's own extracted subject content over the reached + doc_leaves region (the v7-style
    # compiled-digest analog). Only the text the model reads changes; model, prompt, tree are the same.
    if cfg.answer_context == "digest":
        primary = _subject_digest(idx, question, cfg, emb_cache, tr["reached_leaves"], overview)
    else:
        primary = _digest(overview, _leaf_chunks(idx, tr["reached_leaves"]), char_budget)
    pred, pt, ct = _answer_call(question, primary)
    source, fallback = "index", None
    in_tok, out_tok = pt, ct

    if pred.lower() in _MISS:                                 # miss -> widen the re-read
        if cfg.fallback_mode == "doc_leaves":                 # score every leaf in the document, top-k
            fallback = "doc_leaves"
            wide_leaves = _doc_leaf_topk(idx, question, cfg, emb_cache)
        else:                                                 # sibling (default): reached leaf + siblings
            fallback = "span"
            wide_leaves = _sibling_leaves(idx, tr["reached_leaves"])
        sd = _digest(overview, _leaf_chunks(idx, wide_leaves), char_budget)
        pred, pt, ct = _answer_call(question, sd)
        in_tok += pt; out_tok += ct
        source = "fallback" if pred.lower() not in _MISS else "miss"
        if pred.lower() in _MISS and cfg.fallback_wide:      # wide re-read (opt-in)
            fallback = "wide"
            wd = _wide_digest(slug, question, char_budget)
            pred, pt, ct = _answer_call(question, wd)
            in_tok += pt; out_tok += ct
            source = "fallback_wide" if pred.lower() not in _MISS else "miss"

    if pred.lower() in _MISS:
        pred, source = "not found", "miss"
    return {"prediction": pred, "source": source, "fallback": fallback,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "nodes_scored": tr["nodes_scored"], "depth_reached": tr["depth_reached"],
            "backtracks": tr["backtracks"], "hops": tr["hops"],
            "budget_spent": tr["budget_spent"], "budget_total": tr["budget_total"],
            "route_tokens": tr["route_tokens"]}


# ---------- per-movie driver ----------
def _leaf_count(idx):
    return sum(1 for n in idx["nodes"].values() if n["kind"] == "leaf")


def run_movie(slug, cfg):
    idx = _store.load_index(slug)
    leaf_count = _leaf_count(idx)
    emb_cache = {}                                            # transient, per source document
    questions = [json.loads(l) for l in
                 (BENCH/"data"/"questions"/"questions.jsonl").read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]

    out = HERE/"results"/f"index_{slug}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["qid"])
    fh = open(out, "a", encoding="utf-8")
    for q in qs:
        if q["qid"] in done:
            continue
        t0 = time.time()
        r = answer_question(idx, slug, q["question"], cfg, emb_cache)
        lat = round(time.time() - t0, 2)
        rec = {"method": "index", "movie": slug, "qid": q["qid"], "question": q["question"],
               "gold_answers": q["gold_answers"], "prediction": r["prediction"], "source": r["source"],
               "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
               "total_tokens": r["input_tokens"] + r["output_tokens"], "latency_s": lat,
               "build_tokens": 0,
               # ---- Phase 7 instrumentation ----
               "nodes_scored": r["nodes_scored"], "answer_tokens": r["input_tokens"] + r["output_tokens"],
               "leaf_count": leaf_count, "depth_reached": r["depth_reached"],
               "backtracks": r["backtracks"], "hops": r["hops"], "budget_spent": r["budget_spent"],
               "fallback": r["fallback"], "routing_mode": cfg.routing_mode}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        print(f"  {q['qid']} src={r['source']} scored={r['nodes_scored']} "
              f"d={r['depth_reached']} bt={r['backtracks']} hop={r['hops']} "
              f"fb={r['fallback']} :: {r['prediction'][:50]}", flush=True)
    fh.close()
    print(f"DONE {slug}: {len(qs)} questions -> {out}", flush=True)


def concat_all():
    """Concatenate per-movie result files into results/index.jsonl for eval.py."""
    agg = HERE/"results"/"index.jsonl"
    rows = []
    for slug in SLUGS:
        p = HERE/"results"/f"index_{slug}.jsonl"
        if p.exists():
            rows += [l for l in p.read_text().splitlines() if l.strip()]
    agg.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"wrote {agg} ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--concat", action="store_true")
    a = ap.parse_args()
    cfg = _cfg.load()
    if a.concat:
        concat_all()
    elif a.all:
        for slug in SLUGS:
            run_movie(slug, cfg)
        concat_all()
    elif a.movie:
        run_movie(a.movie, cfg)
    else:
        ap.error("give --movie <slug>, or --all, or --concat")
