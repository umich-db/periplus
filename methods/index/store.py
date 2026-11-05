"""Phase 6: write (persist the tree plus the whole-document overview) and load it back.

Write is the end of the build. It persists everything the answer phase and the fallback
read: the packed tree (root, category nodes, routing nodes, leaves), the leaves with their
source backpointers, the chunks with their spans and text (the fallback re-reads source
spans), and the routing-summary cache. It also builds the one whole-document overview by
summarizing the organized subjects (category digests), never by re-reading the source
document. No embeddings are stored as a query index.

Vocabulary: root, category, subject, field, source document.
"""
import sys, json, pathlib, argparse
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import chunk as _chunk, categories as _cat, extract as _extract, pack as _pack  # noqa: E402

OVERVIEW_PROMPT = (
    "Below are the categories of one source document, each with a short digest of its "
    "content. Using ONLY these, write a concise overview of the whole source document in "
    "4 to 8 sentences: what kind of document it is, and the main people, places, and events "
    "across it. Base it on the full set of categories together, not any single one; this "
    "overview gives the index global coverage for broad questions.\n\n{blocks}\n\nOverview:"
)


def _index_dir(slug):
    return HERE/"output"/slug/"index"


# ---------- overview (one LLM call over category digests, not the source document) ----------
def _category_digests(nodes, cat_top_id):
    digests = {}
    for cat, tid in cat_top_id.items():
        parts = []
        for cid in nodes[tid]["children"]:
            c = nodes[cid]
            if c["kind"] != "leaf" and c["summary"]:
                parts.append(c["summary"])
            elif c["kind"] == "leaf":
                parts.append(c["content"].split("::")[0].strip())
        digests[cat] = " | ".join(parts[:8])
    return digests


def write_overview(nodes, cat_top_id, cfg):
    digests = _category_digests(nodes, cat_top_id)
    blocks = "\n".join(f"- {cat}: {dig}" for cat, dig in sorted(digests.items()))
    text, pt, ct = client.chat(OVERVIEW_PROMPT.format(blocks=blocks),
                               max_tokens=16384, bucket="index_build")
    return (text or "").strip(), pt + ct


# ---------- write / load ----------
def write_index(slug, nodes, root_id, chunks, cfg):
    d = _index_dir(slug); d.mkdir(parents=True, exist_ok=True)
    with open(d/"nodes.jsonl", "w", encoding="utf-8") as f:
        for n in nodes.values():
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    with open(d/"chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["id"], "start_char": c["start_char"],
                                "end_char": c["end_char"], "text": c["text"]}, ensure_ascii=False) + "\n")
    (d/"meta.json").write_text(json.dumps({
        "slug": slug, "root_id": root_id, "n_nodes": len(nodes), "n_chunks": len(chunks),
        "overview": nodes[root_id].get("overview", ""), "config": cfg.as_dict(),
    }, indent=2, ensure_ascii=False))
    return d


def load_index(slug):
    """Read the written files back into memory: tree navigable from root by child ids, the
    leaves with backpointers, the chunks with spans, and the overview. No rebuild."""
    d = _index_dir(slug)
    nodes = {}
    for l in (d/"nodes.jsonl").read_text().splitlines():
        if l.strip():
            n = json.loads(l); nodes[n["id"]] = n
    chunks = {}
    for l in (d/"chunks.jsonl").read_text().splitlines():
        if l.strip():
            c = json.loads(l); chunks[c["chunk_id"]] = c
    meta = json.loads((d/"meta.json").read_text())
    return {"nodes": nodes, "root": meta["root_id"], "chunks": chunks,
            "overview": meta["overview"], "meta": meta}


# ---------- build_and_write: the whole build ends here ----------
def _get_chunks(slug, cfg):
    return _chunk.chunk_document(_chunk.load_source(slug), cfg.chunk_target, cfg.B)


def _get_subjects(slug, cfg, chunks):
    """Reuse persisted Phase 4 subjects if present (avoids re-running the extract LLM);
    otherwise run propose + merge + extract."""
    sp = HERE/"output"/slug/"subjects.jsonl"
    if sp.exists():
        return [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
    union, _, _ = _cat.propose(chunks, workers=cfg.propose_workers,
                               cache_path=HERE/"output"/slug/"propose_cache.json")
    categories, _, _ = _cat.merge(union, cfg.embedding_model, cfg.merge_similarity)
    subjects, _, _ = _extract.extract(chunks, categories, cfg)
    return subjects


def build_and_write(slug, cfg):
    chunks = _get_chunks(slug, cfg)
    subjects = _get_subjects(slug, cfg, chunks)
    nodes, root_id, cat_top_id, stats = _pack.pack(
        subjects, cfg, cache_path=HERE/"output"/slug/"routing_cache.json")
    overview, ov_tok = write_overview(nodes, cat_top_id, cfg)
    nodes[root_id]["overview"] = overview
    nodes[root_id]["summary"] = overview
    d = write_index(slug, nodes, root_id, chunks, cfg)
    return d, len(nodes), len(chunks), stats, ov_tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "load"])
    ap.add_argument("--movie", required=True)
    a = ap.parse_args()
    cfg = _cfg.load()

    if a.cmd == "build":
        d, n_nodes, n_chunks, stats, ov_tok = build_and_write(a.movie, cfg)
        print(json.dumps({"wrote": str(d), "n_nodes": n_nodes, "n_chunks": n_chunks,
                          "overview_tokens": ov_tok, "pack_summary_calls": stats["calls"]},
                         indent=2, ensure_ascii=False))
        sys.exit(0)

    # load in a fresh process, traverse, assert, report
    idx = load_index(a.movie)
    nodes, chunks = idx["nodes"], idx["chunks"]
    # traverse root -> leaf by child ids only (no rebuild)
    path, cur = [], idx["root"]
    while True:
        n = nodes[cur]
        path.append(n)
        if n["kind"] == "leaf":
            break
        cur = n["children"][0]
    # asserts
    leaves = [n for n in nodes.values() if n["kind"] == "leaf"]
    for lf in leaves:
        s = lf.get("source", {})
        assert s.get("chunk_id") is not None and s.get("start_char") is not None \
            and s.get("end_char") is not None, f"leaf {lf['id']} missing source backpointer"
    for cid, c in chunks.items():
        assert c.get("start_char") is not None and c.get("end_char") is not None and c.get("text"), \
            f"chunk {cid} missing span or text"
    files = sorted(_index_dir(a.movie).glob("*"))
    total_bytes = sum(f.stat().st_size for f in files)

    print("=== reload + traverse (fresh process, no rebuild) ===")
    print(f"nodes loaded: {len(nodes)} | chunks loaded: {len(chunks)} | leaves: {len(leaves)}")
    print("path root -> leaf:")
    for n in path:
        tag = n["kind"] + (f" [{n['category']}]" if n["category"] else "")
        rt = (n.get("summary") or n.get("content") or "").strip().replace("\n", " ")
        print(f"  L{n['level']} {tag}: {rt[:80]}")
    print("\noverview (from organized subjects, whole document):")
    print("  " + (idx["overview"][:600] or "(EMPTY)"))
    print(f"\nasserts passed: every leaf has a source backpointer; every chunk has span + text")
    print(f"\nfiles under output/{a.movie}/index/  (total {total_bytes:,} bytes):")
    for f in files:
        print(f"  {f.stat().st_size:>9,}  {f.name}")
