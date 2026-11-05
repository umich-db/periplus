"""Phase 4: extract subjects (the leaves).

Two parts. Part A is one LLM call per chunk that pulls out CONTENT only (subject name +
a few fields + the source backpointer); the model does not choose a category. Part B
files each subject under exactly one canonical category by EMBEDDING, not by asking the
model, the same way merge and pack group. Grouping is the embedding's job.

Vocabulary: root, category, subject, field, source document.
"""
import sys, re, json, pathlib, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client, config as _cfg  # noqa: E402
import chunk as _chunk  # noqa: E402
import categories as _cat  # noqa: E402

EXTRACT_PROMPT = (
    "Below is one passage from a source document. Pull out the SUBJECTS it describes. A "
    "subject is a specific person, place, event, object, or scene in the passage. For each "
    "subject return a short name and a small set of fields (key/value details drawn from "
    "the text). Do NOT assign topics or categories; return content only.\n\n"
    "Return ONLY a JSON array:\n"
    '  [{{"name": "...", "fields": {{"key": "value", ...}}}}, ...]\n\n'
    "Passage:\n{chunk}\n\nJSON array of subjects:"
)


def _parse_subjects(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for s in arr:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        fields = s.get("fields", {})
        fields = {str(k): str(v) for k, v in fields.items()} if isinstance(fields, dict) else {}
        if name:
            out.append({"name": name, "fields": fields})
    return out


# ---------- Part A: content-only extraction (LLM, one call per chunk) ----------
def extract_one(chunk):
    # 16384 build budget from the start: a structured reasoning call, a small budget gets
    # starved by hidden reasoning to an empty response.
    text, pt, ct = client.chat(EXTRACT_PROMPT.format(chunk=chunk["text"]),
                               max_tokens=16384, bucket="index_build")
    subs = []
    for s in _parse_subjects(text):
        subs.append({
            "name": s["name"], "fields": s["fields"],
            "source": {"chunk_id": chunk["id"],
                       "start_char": chunk["start_char"], "end_char": chunk["end_char"]},
        })
    return subs, pt, ct


def extract_content(chunks, workers=8):
    """One call per chunk, concurrent, collected by chunk id (deterministic order)."""
    per_chunk = [None]*len(chunks)
    tok = {"in": 0, "out": 0}
    def _work(i):
        subs, pt, ct = extract_one(chunks[i])
        return i, subs, pt, ct
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, subs, pt, ct in ex.map(_work, range(len(chunks))):
            per_chunk[i] = subs
            tok["in"] += pt; tok["out"] += ct
    subjects = [s for subs in per_chunk for s in subs]
    return subjects, tok


# ---------- Part B: category assignment (embedding, no LLM) ----------
def _subject_text(s):
    # name alone is too thin; include the field values
    return (s["name"] + " " + " ".join(str(v) for v in s["fields"].values())).strip()


def assign_categories(subjects, categories, embed_model, weak_floor):
    """Assign each subject to its nearest canonical category by cosine. Transient
    embeddings, not stored. weak_floor only sets a diagnostic flag; it never reassigns."""
    if not subjects:
        return subjects, 0
    cat_vecs = client.embed(categories, model_name=embed_model)          # cached once here
    sub_vecs = client.embed([_subject_text(s) for s in subjects], model_name=embed_model)
    sims = sub_vecs @ cat_vecs.T
    weak = 0
    for s, row in zip(subjects, sims):
        j = int(row.argmax()); score = float(row[j])
        s["category"] = categories[j]
        s["assign_score"] = round(score, 4)
        s["weak_match"] = score < weak_floor
        weak += int(s["weak_match"])
    return subjects, weak


def extract(chunks, categories, cfg):
    """Full Phase 4: content-only extraction then embedding assignment."""
    subjects, tokA = extract_content(chunks, workers=cfg.propose_workers)
    subjects, weak = assign_categories(subjects, categories, cfg.embedding_model, cfg.weak_match_floor)
    return subjects, weak, tokA


def write_dump(slug, categories, mapping, subjects, f):
    """Persist categories, the label-to-category map, and all subjects for inspection:
    structured JSON/JSONL plus a readable markdown grouped by category."""
    out = HERE/"output"/slug
    out.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    per_cat = Counter(s["category"] for s in subjects)

    (out/"categories.json").write_text(json.dumps({
        "n_categories": len(categories), "f": f, "categories": categories,
        "label_to_category": mapping,
    }, indent=2, ensure_ascii=False))
    with open(out/"subjects.jsonl", "w", encoding="utf-8") as fh:
        for s in subjects:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # readable view: categories high to low, subjects nested under each
    order = sorted(per_cat, key=lambda c: (-per_cat[c], c))
    lines = [f"# Index dump: {slug}",
             f"{len(subjects)} subjects, {len(categories)} categories, f={f}", ""]
    over = [c for c in order if per_cat[c] > f]
    lines.append("## Categories over f (pack builds a subtree here)")
    lines.append(", ".join(f"{c} ({per_cat[c]})" for c in over) or "(none)")
    lines.append("")
    by_cat = {}
    for s in subjects:
        by_cat.setdefault(s["category"], []).append(s)
    for c in order:
        lines.append(f"## {c}  ({per_cat[c]} subjects)")
        for s in sorted(by_cat[c], key=lambda x: (x["source"]["chunk_id"], x["name"])):
            fields = "; ".join(f"{k}: {v}" for k, v in s["fields"].items())
            flag = "  [weak]" if s.get("weak_match") else ""
            src = s["source"]
            lines.append(f"- **{s['name']}**"
                         + (f" - {fields}" if fields else "")
                         + f"  (chunk {src['chunk_id']}, {src['start_char']}-{src['end_char']}, "
                         + f"score {s.get('assign_score')}){flag}")
        lines.append("")
    (out/"index_dump.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    a = ap.parse_args()
    cfg = _cfg.load()
    chunks = _chunk.chunk_document(_chunk.load_source(a.movie), cfg.chunk_target, cfg.B)
    cache_path = HERE/"output"/a.movie/"propose_cache.json"
    union, _, _ = _cat.propose(chunks, workers=cfg.propose_workers, cache_path=cache_path)
    categories, mapping, _ = _cat.merge(union, cfg.embedding_model, cfg.merge_similarity)

    subjects, weak, tokA = extract(chunks, categories, cfg)

    # acceptance asserts (fail loudly)
    for s in subjects:
        src = s.get("source", {})
        assert src.get("chunk_id") is not None and src.get("start_char") is not None \
            and src.get("end_char") is not None, f"subject {s['name']!r} missing source backpointer"
    catset = set(categories)
    assert all(s["category"] in catset for s in subjects), "a subject was assigned a non-canonical category"

    per_cat = Counter(s["category"] for s in subjects)
    over_f = {c: n for c, n in per_cat.items() if n > cfg.f}
    n = len(subjects)

    dump_dir = write_dump(a.movie, categories, mapping, subjects, cfg.f)
    print(json.dumps({
        "movie": a.movie, "n_chunks": len(chunks), "n_categories": len(categories), "f": cfg.f,
        "total_subjects": n,
        "subjects_per_category (desc)": dict(sorted(per_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
        "categories_over_f (pack builds a subtree here)": over_f,
        "any_category_clears_f": bool(over_f),
        "weak_match_count": weak, "weak_match_rate": round(weak/max(1, n), 3),
        "extract_build_tokens": tokA["in"]+tokA["out"],
        "dump_written_to": str(dump_dir),
    }, indent=2, ensure_ascii=False))
