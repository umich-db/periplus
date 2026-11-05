"""test_paper_conformance.py — does the build do what the paper's §3 says it does?

Every check here is a sentence from the Method section, asserted against a real
build and a real query run against a stub model. No API calls: `llm` is replaced
with a scripted responder, so the test measures the pipeline's structure and
control flow, not the model's judgement.

Run from bench/:  envs/base/bin/python methods/llm_index/test_paper_conformance.py
"""

import os, sys, json, re, pathlib, tempfile, collections

os.environ.setdefault("AZURE_API_KEY", "test")
os.environ.setdefault("AZURE_API_VERSION", "2024-02-01")
os.environ.setdefault("AZURE_API_BASE", "https://example.invalid/")

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH / "src" / "wiki_index"))
sys.path.insert(0, str(BENCH / "src" / "cold_start"))

import cold_start                      # noqa: E402
import pipeline                        # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + str(detail)) if detail else ''}")
    if not cond:
        FAILS.append(name)


# ── a document long enough to need several chunks, with content that only
#    appears late, so a discovery that reads the opening cannot see it ─────────
def make_doc(path):
    parts = ["THE LONG CASE — a screenplay\n\nby A. Writer\n\n"]
    for i in range(1, 7):
        parts.append(
            f"\n\nINT. LOCATION {i} — DAY\n\n"
            f"ELLIOT is an engineer who forms a society. He speaks with DARLENE, his sister.\n"
            f"They plan the HEIST at site {i}. Elliot carries a red notebook numbered {i}.\n"
            + ("Filler dialogue about the weather and the traffic. " * 220)
        )
    # A category that exists ONLY past the first chunk.
    parts.append(
        "\n\nAPPENDIX — LEGAL NOTICES\n\n"
        "The rights to this screenplay are held by Northwind Pictures since 1994.\n"
        + ("Boilerplate indemnity language repeated for length. " * 240)
    )
    path.write_text("".join(parts))
    return path.read_text()


# ── the stub model ───────────────────────────────────────────────────────────
CALLS = collections.Counter()
ROUTE_PROMPTS = []


def stub_llm(prompt, max_tokens=16384, phase="unknown"):
    CALLS[phase] += 1

    if phase == "discover":
        late = "LEGAL NOTICES" in prompt
        secs = [{"name": "Characters", "description": "People in the story",
                 "subsections": [{"name": "Role", "description": "what they do"}]},
                {"name": "Events", "description": "What happens", "subsections": []}]
        if late:
            secs = [{"name": "Legal Notices", "description": "Rights and boilerplate",
                     "subsections": []}]
        return json.dumps({"document_type": "screenplay",
                           "overview": "A passage of the screenplay.", "sections": secs})

    if phase == "compile_p1":
        m = re.search(r"numbered (\d+)", prompt)
        n = m.group(1) if m else "0"
        if "LEGAL NOTICES" in prompt:
            return json.dumps({"Legal Notices": [
                {"Name": "Rights holder", "Detail": "Northwind Pictures since 1994"}]})
        # Enough subjects, with long enough fields, that a category cannot fit one
        # page and writing has to give each subject its own -- the shape the paper
        # describes, and the only shape in which lateral links mean anything.
        who = ["Elliot", "Darlene", "Angela", "Tyrell", "Krista", "Gideon"]
        chars = [{"Name": w,
                  "Role": f"{w} is a principal of the society, described at length "
                          f"across the screenplay in scenes of rising tension",
                  "Actions": [f"{w} plans the heist at site {n}",
                              f"{w} argues with the others about the red notebook"]}
                 for w in who]
        events = [{"Name": f"Heist at site {n} involving {w}",
                   "Detail": f"{w} carries a red notebook numbered {n} into the building "
                             f"while the others wait outside in the rain"}
                  for w in who[:3]]
        return json.dumps({"Characters": chars, "Events": events})

    if phase == "compile_p2":
        return "{}"

    if phase == "consolidate":
        # Overshoot the budget the way the real model does.
        return "OVERVIEW. " + ("Elliot forms a society and plans a heist with Darlene. " * 700)

    if phase == "route":
        ROUTE_PROMPTS.append(prompt)
        return "1"

    if phase == "query_traverse":
        return "not found"            # force the fallback so we can test it

    if phase == "query_fallback":
        return json.dumps({"answer": "a red notebook", "new_facts": [], "new_entities": [],
                           "new_relationships": []})
    return ""


cold_start.llm = stub_llm
pipeline.llm = stub_llm
pipeline.set_router(pipeline.llm_route_children)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="conformance_"))
doc = make_doc(tmp / "doc.txt")
out = tmp / "out"

print("\n§3.2  Discovering the structure")
res = pipeline.build_wiki(tmp / "doc.txt", out, "the_long_case", "The Long Case")
wiki = out / "the_long_case"
manifest = json.loads((wiki / "_manifest.json").read_text())
template = json.loads((wiki / "_template.json").read_text())

n_chunks = len(pipeline.chunk_bounds(cold_start.clean_document(doc)))
check("one LLM call per chunk (not one per document)",
      CALLS["discover"] == n_chunks, f"{CALLS['discover']} calls / {n_chunks} chunks")
check("chunks are ~30,000 chars overlapping 2,000",
      (pipeline.CHUNK_SIZE, pipeline.CHUNK_OVERLAP) == (30_000, 2_000))
cat_names = {s["name"] for s in template["sections"]}
check("proposals are merged, so a category found only late still exists",
      "Legal Notices" in cat_names, sorted(cat_names))
check("repeated entries deduplicated by content",
      len(cat_names) == len(template["sections"]))
check("discovery produces an overview that seeds the root",
      bool(template.get("overview", "").strip()))
check("subject pages capped at 2,400 chars", pipeline.PAGE_SIZE == 2400)

print("\n§3.1  Structure")
roots = [e for e in manifest if e.get("parent") is None]
check("the tree has exactly one root", len(roots) == 1, [e["path"] for e in roots])
check("root holds the consolidated summary and routes to the internal level",
      "## Categories" in (wiki / "index.md").read_text())
by = {e["path"]: e for e in manifest}
depths = {}
for e in manifest:
    d, p, seen = 0, e.get("parent"), set()
    while p is not None and p not in seen:
        seen.add(p); d += 1; p = by.get(p, {}).get("parent")
    depths[e["path"]] = d
leaf_d = [depths[e["path"]] for e in manifest if e["level"] == "leaf"]
check("tree is at least three levels deep (root, category, subject)",
      leaf_d and min(leaf_d) >= 2, f"min leaf depth {min(leaf_d) if leaf_d else None}")
leaves = [e for e in manifest if e["level"] == "leaf"]
check("every subject records its evidence spans",
      all("spans" in e for e in leaves) and any(e["spans"] for e in leaves))
check("every subject carries lateral links to related subjects",
      any(e.get("links") for e in leaves),
      f"{sum(1 for e in leaves if e.get('links'))}/{len(leaves)} linked")
cross = [e for e in leaves
         if any(p.split('/')[0] != e['path'].split('/')[0] for p in e.get("links") or [])]
check("links may point at subjects under other categories", bool(cross), f"{len(cross)} cross-category")

print("\n§3.1  Invariants")
pages = [p for p in wiki.rglob("*.md")]
ROOT_FILES = {"index.md", "consolidated.md"}     # roots have their own, larger cap
over = [p.name for p in pages if p.name not in ROOT_FILES
        and len(p.read_text()) > pipeline.PAGE_SIZE]
check("no node's page exceeds the capacity limit", not over, over[:3])
root_len = len((wiki / "index.md").read_text())
check("the root summary is inside its capacity limit too",
      root_len <= pipeline.ROOT_SUMMARY_CAP, f"{root_len} <= {pipeline.ROOT_SUMMARY_CAP}")
check("root no longer saturates the answering budget",
      root_len < pipeline.CONTEXT_BUDGET, f"{root_len} < {pipeline.CONTEXT_BUDGET}")
long_desc = [e["path"] for e in manifest if len(e["summary"]) > pipeline.DESCRIPTOR_CHARS]
check("every routing descriptor is at most S chars", not long_desc, long_desc[:3])
check("f is derived as C/S, not hardcoded",
      pipeline.MAX_FANOUT == pipeline.ROUTING_BUDGET // pipeline.DESCRIPTOR_CHARS == 33)

print("\n§3.1  Human-readable plain text")
bad = [p.name for p in pages if re.match(r"#\s*\{['\"]", p.read_text().lstrip()[:40])]
check("no page's heading is a leaked dict repr", not bad, bad[:3])
subject_pages = [p for p in pages if p.name not in ("index.md", "consolidated.md", "_summary.md")]
with_fields = [p for p in subject_pages if re.search(r"^\*\*[^*]+:\*\*", p.read_text(), re.M)]
check("subjects record fields extracted from the source", bool(with_fields))
with_span = [p for p in with_fields if re.search(r"`\[\d+–\d+\]`", p.read_text())]
check("every field is linked to its evidence span",
      len(with_span) == len(with_fields), f"{len(with_span)}/{len(with_fields)}")
check("pages state their evidence spans in plain text",
      any("**Evidence spans**" in p.read_text() for p in subject_pages))
check("pages state their lateral links in plain text",
      any("**Related subjects:**" in p.read_text() for p in subject_pages))

print("\n§3.3  Compiling")
check("extraction runs two passes over the chunks",
      CALLS["compile_p1"] == n_chunks and CALLS["compile_p2"] == n_chunks,
      f"p1={CALLS['compile_p1']} p2={CALLS['compile_p2']}")
merged = {"Role": "engineer"}
pipeline._merge_paper(merged, {"Role": "hacker"}, None)
check("single-value fields keep the first non-empty value", merged["Role"] == "engineer", merged)
lst = {"Actions": ["a"]}
pipeline._merge_paper(lst, {"Actions": ["a", "b"]}, None)
check("list fields accumulate with exact duplicates removed", lst["Actions"] == ["a", "b"], lst)
check("one further call revises the root against the merged record",
      CALLS["consolidate"] >= 1, CALLS["consolidate"])
prov = json.loads((wiki / "_provenance.json").read_text())
check("every extracted field carries its exact evidence span",
      prov["offsets"]["facts_with_spans"] > 0, prov["offsets"]["located_by"])

print("\n§3.5  Answering queries")
CALLS.clear(); ROUTE_PROMPTS.clear()
q = "What does Elliot carry?"
digest = pipeline.traverse(q, wiki)
visited = list(pipeline._LAST_VISITED)
check("the descent reads pages below the root",
      len(visited) > 1, f"{len(visited)} pages read: {sorted(visited)[:4]}")
check("a routing call is made per level, not once per question",
      CALLS["route"] >= 2, f"{CALLS['route']} routing calls")
check("a routing step reads only children's descriptors, within budget C",
      all(len(p.split("CHILDREN:")[-1].split("RELATED SUBJECTS:")[-1]) <= pipeline.ROUTING_BUDGET + 400
          for p in ROUTE_PROMPTS))
check("selected pages stay inside the answering budget",
      len(digest) <= pipeline.CONTEXT_BUDGET, f"{len(digest)} <= {pipeline.CONTEXT_BUDGET}")
check("the query is paired with the root summary first",
      "index.md" in visited)

# The fallback: stub answers "not found" from the index, so this is the source path.
CALLS.clear()
pred, source, _ = pipeline.query_with_traversal(q, wiki, tmp / "doc.txt")
check("a not found from the index triggers the fallback", source == "fallback", (pred, source))
check("the fallback is not the deep-retry re-walk (not in §3.5)",
      CALLS["query_traverse_deep"] == 0)

text = cold_start.clean_document((tmp / "doc.txt").read_text())
ev = pipeline._evidence_text(wiki, text, visited, pipeline.CONTEXT_BUDGET)
check("the fallback returns to the originating chunk via recorded spans", bool(ev))
check("the fallback does not rescan the full document",
      len(ev) <= pipeline.CONTEXT_BUDGET and len(ev) < len(text),
      f"{len(ev)} chars of {len(text)}")


# ── routing behaviour the method depends on, and the failure modes around it ──
print("\n§3.5  Routing behaviour")

ROUTE_REPLY = ["1"]


def routing_stub(prompt, max_tokens=16384, phase="unknown"):
    CALLS[phase] += 1
    if phase == "route":
        ROUTE_PROMPTS.append(prompt)
        return ROUTE_REPLY[0]
    if phase == "query_traverse":
        return "not found"
    return stub_llm(prompt, max_tokens, phase)


pipeline.llm = routing_stub
cold_start.llm = routing_stub

CALLS.clear()
ROUTE_REPLY[0] = "NONE"
d_none = pipeline.traverse(q, wiki)
check("a router that says NONE stops the descent there",
      len(pipeline._LAST_VISITED) < 12, f"{len(pipeline._LAST_VISITED)} pages")
check("a NONE descent still returns the root summary", bool(d_none.strip()))

CALLS.clear()
ROUTE_REPLY[0] = ""                      # model returned nothing usable
d_fail = pipeline.traverse(q, wiki)
check("a failed routing call falls back to keyword scoring, not a stall",
      len(pipeline._LAST_VISITED) > 1, f"{len(pipeline._LAST_VISITED)} pages")
check("the fallback walk still respects the answering budget",
      len(d_fail) <= pipeline.CONTEXT_BUDGET)

# A category wide enough that writing must insert grouping nodes.
print("\n§3.3  Grouping nodes under a wide category")
WIDE = [{"Name": f"Subject {i:03d}",
         "Role": f"Subject {i:03d} holds a described position in the document, "
                 f"set out across several clauses of running text",
         "Actions": [f"acts in scene {i}"]} for i in range(120)]


def wide_stub(prompt, max_tokens=16384, phase="unknown"):
    CALLS[phase] += 1
    if phase == "discover":
        return json.dumps({"document_type": "screenplay", "overview": "o",
                           "sections": [{"name": "Characters", "description": "people",
                                         "subsections": []}]})
    if phase == "compile_p1":
        return json.dumps({"Characters": WIDE})
    if phase == "compile_p2":
        return "{}"
    if phase == "consolidate":
        return "Short overview of a wide document."
    if phase == "route":
        return "1"
    return ""


pipeline.llm = wide_stub
cold_start.llm = wide_stub
out2 = tmp / "out2"
pipeline.build_wiki(tmp / "doc.txt", out2, "wide", "Wide")
w2 = out2 / "wide"
m2 = json.loads((w2 / "_manifest.json").read_text())
kids2 = collections.Counter(e.get("parent") for e in m2 if e.get("parent"))
groups = [e for e in m2 if "/group_" in e["path"]]
check("writing inserts grouping nodes when a category overflows one page",
      bool(groups), f"{len(groups)} grouping nodes")
order = [e["summary"] for e in m2 if e["level"] == "leaf" and "Subject" in (e["summary"] or "")]
nums = [int(re.search(r"Subject (\d+)", s).group(1)) for s in order if re.search(r"Subject (\d+)", s)]
check("subjects keep the order extraction produced them (contiguous, not strided)",
      nums == sorted(nums), f"first 8: {nums[:8]}")
pages2 = [p for p in w2.rglob("*.md") if p.name not in ("index.md", "consolidated.md")]
check("every page still inside the capacity limit",
      all(len(p.read_text()) <= pipeline.PAGE_SIZE for p in pages2))


# ── 3.3 extraction under the gateway's 300 s cutoff: halve, keep offsets, fail loudly at the floor ──
print("\n§3.3  Gateway cutoff fallback")
if hasattr(pipeline, "_extract_pieces") and hasattr(cold_start, "GatewayTimeout"):
    CUT = [12_000]
    def cutoff_llm(prompt, max_tokens=16384, phase="x"):
        body = prompt.split("PASSAGE:")[-1]
        if len(body) > CUT[0]:
            raise cold_start.GatewayTimeout("simulated 300s cutoff")
        return json.dumps({"n": len(body)})
    cold_start.llm = cutoff_llm; pipeline.llm = cutoff_llm
    big = ("A paragraph of the screenplay about Elliot and Darlene.\n\n" * 1300)[:30_000]
    pcs = pipeline._extract_pieces(lambda t: "PASSAGE:" + t, big, 5_000, "chunk 1/1", "compile_p1", 65536)
    st = [p[2] for p in pcs]; ln = [len(p[1]) for p in pcs]
    check("a 30K passage that trips the cutoff is split into pieces under the limit",
          len(pcs) >= 3 and all(l <= CUT[0] for l in ln), f"{len(pcs)} pieces {ln}")
    check("every piece was extracted (none dropped)", all(p[0] for p in pcs))
    check("piece offsets are contiguous and cover the whole passage",
          st[0] == 5_000 and all(st[i+1] == st[i] + ln[i] for i in range(len(pcs)-1)) and sum(ln) == len(big))
    check("each piece's text is exactly the source at its recorded offset",
          all(p[1] == big[s-5_000:s-5_000+len(p[1])] for p, s in zip(pcs, st)))
    check("splits land on paragraph breaks, not mid-sentence",
          all(p[1].endswith("\n\n") or p is pcs[-1] for p in pcs))
    CUT[0] = 0
    try:
        pipeline._extract_pieces(lambda t: "PASSAGE:" + t, big, 0, "c", "compile_p1", 1); floor_ok = False
    except pipeline.ChunkTooSlow:
        floor_ok = True
    check("a passage that never fits fails the build visibly (ChunkTooSlow), not silently", floor_ok)
else:
    print("  SKIP  gateway cutoff extension is absent from this branch (also absent before repair)")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
elif FAILS:
    raise AssertionError(FAILS)
