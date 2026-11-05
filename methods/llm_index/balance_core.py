"""Offline bounded-fanout rebalancing of an already-built document tree.

Fanout is C/S = 4000/120 = 33. Overfull children are ordered with transient local
all-mpnet embeddings and grouped in balanced contiguous slices. No LLM or API
client is needed. Set HF_HUB_OFFLINE=1 to require an already cached model.

Every internal routing descriptor is regenerated from its real child topics.
Routing pages compact each child fairly within 2400 characters rather than
cutting off later children. The root is refreshed from consolidated.md and its
immediate balanced children, targeting 3000 characters (maximum 6000). Subject
pages and the saved overview remain byte-identical.

The occupancy-window implementation in balance_window.py also borrows and
merges groups while preserving category and continuation-record boundaries.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, math, shutil, pathlib, re

BENCH = _BENCH
WIKI_SRC = _BENCH/"src"/"wiki_index"
SRC_DIGESTS = BENCH / "methods" / "llm_index" / "output" / "digests"
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIGESTS = HERE / "output" / "digests_balanced"

# Balancing is an offline operation. Do not import the extraction pipeline (and
# therefore construct an API client) merely to obtain its capacity constants.
PAGE_SIZE = 2400
ROOT_SUMMARY_CAP = 6000
ROOT_PAGE_TARGET = 3000

# --- config-derived B+ parameters (chars; mirror pipeline.ROUTING_BUDGET/DESCRIPTOR_CHARS) ---
C = 4000
S = 120
F = C // S                 # 33
OCC_LO = math.ceil(F / 2)  # 17
CLUSTER_SEED = 42

# --- local, transient all-mpnet embedder (same model the repo uses) ----------
_model = None


def embed(texts):
    """L2-normalized all-mpnet row vectors. Transient: never stored as an index."""
    global _model
    import numpy as np
    if _model is None:
        from sentence_transformers import SentenceTransformer
        offline = _os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                     local_files_only=offline)
    v = _model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def coherent_order(summaries):
    """Return child indices ordered so embedding-adjacent children are together.

    Orders by projection onto the first principal component of the all-mpnet
    embeddings (deterministic; ties broken by original index). Adjacent items in
    this order are the most similar, so contiguous cuts give coherent groups.
    """
    import numpy as np
    if len(summaries) <= 2:
        return list(range(len(summaries)))
    X = embed(summaries)
    Xc = X - X.mean(axis=0, keepdims=True)
    # first principal component via SVD (deterministic)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[0]
    order = sorted(range(len(summaries)), key=lambda i: (float(proj[i]), i))
    return order


def balanced_group_sizes(n, k):
    """k contiguous group sizes for n items, as equal as possible."""
    base, rem = divmod(n, k)
    return [base + 1 if i < rem else base for i in range(k)]


def _compact(text, limit):
    """Compact a routing descriptor, never an extracted fact page."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    prefix = text[:limit - 1]
    boundary = prefix.rfind(" ")
    if boundary >= max(8, limit // 2):
        prefix = prefix[:boundary]
    return prefix.rstrip(" ,;:") + "…"


def _fair_lines(summaries, budget):
    """Give every child a line; distribute spare space from short descriptors."""
    summaries = [" ".join(str(s).split()) for s in summaries]
    if not summaries:
        return ""
    remaining = budget - 3 * len(summaries)  # '- ' and newline per child
    if remaining < len(summaries):
        raise ValueError("Routing page cannot represent every child in its budget")
    limits = [0] * len(summaries)
    pending = list(range(len(summaries)))
    while pending:
        allowance = remaining // len(pending)
        small = [i for i in pending if len(summaries[i]) <= allowance]
        if not small:
            for offset, i in enumerate(pending):
                limits[i] = allowance + (offset < remaining % len(pending))
            break
        for i in small:
            limits[i] = len(summaries[i])
            remaining -= limits[i]
        pending = [i for i in pending if i not in small]
    return "".join(f"- {_compact(s, n)}\n" for s, n in zip(summaries, limits))


def _group_summary_page(group_name, child_summaries):
    """Bounded routing page that includes every immediate child fairly."""
    text = f"# {_compact(group_name, S)}\n\nTotal: {len(child_summaries)}\n\n"
    text += _fair_lines(child_summaries, PAGE_SIZE - len(text))
    assert len(text) <= PAGE_SIZE
    return text


def _content_name(path, summary=""):
    """Recover a real category/subject label, never an artificial group name."""
    head = re.split(r"\s+[—–]\s+|:", str(summary or ""), maxsplit=1)[0].strip()
    if head and not re.fullmatch(r"_?summary|(?:_?(?:win|bal)_)?group[_ ]?\d*", head, re.I):
        return re.sub(r"\s*\(\d+ items\)$", "", head)
    parts = pathlib.PurePosixPath(path).parts
    for part in parts[:-1]:
        if not re.fullmatch(r"_?(?:win_|bal_)?group_?\d+", part):
            return part.replace("_", " ")
    return ""


def branch_descriptor(name, child_summaries):
    """Deterministic, content-derived branch descriptor within S characters.

    A stable category label is followed by representative child labels sampled
    across the embedding order. No counts or synthetic filenames replace the
    subject matter. The full child set remains available on the branch page.
    """
    name = _compact(name, 48)
    phrases = []
    for summary in child_summaries:
        text = " ".join(str(summary or "").split())
        text = re.sub(r"\s*\(\d+ items\)$|:\s*\d+ items(?: across \d+ groups)?$", "", text)
        if not text or re.match(r"^_summary\b|^_?(?:win_|bal_)?group_\d+", text):
            continue
        # Remove the repeated category prefix, retaining the actual child topic.
        if name and text.casefold().startswith(name.casefold() + ":"):
            text = text[len(name) + 1:].strip()
        if text and text.casefold() not in {p.casefold() for p in phrases}:
            phrases.append(text)
    if not phrases:
        if not name:
            raise ValueError("Branch lacks content-derived routing information")
        return _compact(name, S)
    count = min(4, len(phrases))
    indices = [round(i * (len(phrases) - 1) / max(1, count - 1)) for i in range(count)]
    selected = [phrases[i] for i in indices]
    prefix = name + ": " if name else ""
    room = S - len(prefix) - 2 * (len(selected) - 1)
    # Use the same fair allocation as page bullets without its formatting cost.
    lines = _fair_lines(selected, room + 3 * len(selected)).splitlines()
    return prefix + "; ".join(line[2:] for line in lines)


def refresh_root_page(src_dir, manifest, doc_title=None):
    """Restore the saved overview plus only the root's immediate routing choices.

    Root pages are allowed to change during balancing; consolidated.md and all
    subject pages remain byte-identical. Keep orientation near 3K so navigation
    still has room to reach subjects under the 12K answering budget.
    """
    src_dir = pathlib.Path(src_dir)
    root = next((e for e in manifest if e.get("parent") is None), None)
    if root is None:
        raise ValueError("Cannot refresh a tree without its root")
    old_path = src_dir / root["path"]
    old = old_path.read_text() if old_path.exists() else ""
    title_line = next((line for line in old.splitlines() if line.startswith("# ")), "")
    title = doc_title or title_line.removeprefix("# ") or src_dir.name.replace("_", " ")
    saved = src_dir / "consolidated.md"
    overview = saved.read_text().strip() if saved.exists() else ""
    if not overview:
        overview = old.split("## Categories", 1)[0].split("\n", 1)[-1].strip()
    children = [e for e in manifest if e.get("parent") == root["path"]]
    header = f"# {_compact(title, S)}\n\n"
    route_header = "## Categories\n"
    # Reserve most of the page for plot orientation; the manifest retains the
    # full 120-character descriptors used by the router.
    route_budget = min(sum(min(S, len(e.get("summary", ""))) + 3 for e in children),
                       max(4 * len(children), ROOT_PAGE_TARGET // 3))
    routes = route_header + _fair_lines([e.get("summary", "") for e in children], route_budget)
    room = ROOT_PAGE_TARGET - len(header) - len(routes) - 2
    body = _compact(overview, max(0, room))
    text = header + body + "\n\n" + routes
    assert len(text) <= ROOT_PAGE_TARGET <= ROOT_SUMMARY_CAP
    return text


def _short(summary, n=28):
    w = " ".join(str(summary).split())
    return (w[: n - 1] + "…") if len(w) > n else w


class Balancer:
    """Balances one doc's manifest in place, tracking a structural report."""

    def __init__(self, doc, manifest):
        self.doc = doc
        self.manifest = [dict(e) for e in manifest]
        self.by_path = {e["path"]: e for e in self.manifest}
        if len(self.by_path) != len(self.manifest):
            raise ValueError("Duplicate page paths: rebuild lossless pages before balancing")
        self.names = {e["path"]: _content_name(e["path"], e.get("summary", ""))
                      for e in self.manifest}
        self.children = self._children_map()
        self.new_pages = {}                                 # rel_path -> text
        self.splits = []                                    # report rows
        self._gid = 0

    def _children_map(self):
        ch = {}
        for e in self.manifest:
            ch.setdefault(e.get("parent"), []).append(e["path"])
        return ch

    def _node_name(self, path):
        """Human name for a node (its summary head, or the file stem)."""
        return self.names.get(path, "")

    def _refresh(self, path):
        entry = self.by_path[path]
        if entry.get("level") == "root":
            return
        summaries = [self.by_path[c].get("summary", "") for c in self.children.get(path, [])]
        descriptor = branch_descriptor(self._node_name(path), summaries)
        entry["summary"] = descriptor
        self.new_pages[path] = _group_summary_page(descriptor, summaries)

    def _dir_of(self, path):
        d = str(pathlib.PurePosixPath(path).parent)
        return "" if d == "." else d

    def split_node(self, parent_path):
        """Insert intermediate group nodes under an over-full parent, recursively."""
        kids = self.children.get(parent_path, [])
        n = len(kids)
        if n <= F:
            return
        pname = self._node_name(parent_path) if parent_path else "root"
        base_dir = self._dir_of(parent_path) if parent_path else ""
        fanout_before = n

        # 1-3: coherent order + balanced contiguous groups
        summaries = [self.by_path[c].get("summary", pathlib.Path(c).stem) for c in kids]
        order = coherent_order(summaries)
        ordered = [kids[i] for i in order]
        k = math.ceil(n / F)
        sizes = balanced_group_sizes(n, k)

        groups, pos = [], 0
        for sz in sizes:
            groups.append(ordered[pos:pos + sz])
            pos += sz

        # 4-5: create a branch node per group, rewire children to it
        group_paths = []
        for gi, grp in enumerate(groups):
            self._gid += 1
            gdir = f"{base_dir}/_bal_group_{self._gid}" if base_dir else f"_bal_group_{self._gid}"
            gpath = f"{gdir}/_summary.md"
            child_summaries = [self.by_path[c].get("summary", "") for c in grp]
            label = "" if self.by_path.get(parent_path, {}).get("level") == "root" else pname
            gname = branch_descriptor(label, child_summaries)
            self.new_pages[gpath] = _group_summary_page(gname, child_summaries)
            gentry = {
                "path": gpath, "level": "branch", "parent": parent_path,
                "summary": gname,
            }
            self.by_path[gpath] = gentry
            self.names[gpath] = label
            # insert group node into manifest right after its parent
            self._insert_after(parent_path, gentry)
            # rewire this group's children to the new group node
            for c in grp:
                self.by_path[c]["parent"] = gpath
            self.children[gpath] = list(grp)
            group_paths.append(gpath)
            assert OCC_LO <= len(grp) <= F, f"occupancy violated: {len(grp)}"

        # parent now points only at the group nodes
        self.children[parent_path] = group_paths
        if parent_path is not None:
            self._refresh(parent_path)

        self.splits.append({
            "parent": parent_path if parent_path else "(root)",
            "parent_name": pname,
            "fanout_before": fanout_before,
            "groups": k,
            "group_sizes": sizes,
            "fanout_after": k,
        })

        # 6: recurse if the parent is still over-full, then descend into groups
        if len(self.children[parent_path]) > F:
            self.split_node(parent_path)
        for gp in group_paths:
            if len(self.children.get(gp, [])) > F:
                self.split_node(gp)

    def _insert_after(self, anchor_path, entry):
        if anchor_path is None:
            self.manifest.append(entry)
            return
        for i, e in enumerate(self.manifest):
            if e["path"] == anchor_path:
                self.manifest.insert(i + 1, entry)
                return
        self.manifest.append(entry)

    def balance(self):
        # snapshot the set of parents to consider (root + every internal node)
        parents = [p for p in self.children if p is not None]
        # root grouping: index.md's children are the categories; None-parent = roots
        for parent_path in list(parents):
            if len(self.children.get(parent_path, [])) > F:
                self.split_node(parent_path)
        # also bound the category layer under the root page(s)
        for root_path in ("index.md",):
            if len(self.children.get(root_path, [])) > F:
                self.split_node(root_path)
        def refresh(path):
            for child in self.children.get(path, []):
                if self.children.get(child):
                    refresh(child)
            self._refresh(path)
        for root in self.children.get(None, []):
            refresh(root)
        return self.manifest


def _depth_stats(manifest):
    depth = {}

    def d(path):
        if path is None:
            return -1
        if path in depth:
            return depth[path]
        e = next((x for x in manifest if x["path"] == path), None)
        depth[path] = 0 if not e else d(e.get("parent")) + 1
        return depth[path]

    ds = [d(e["path"]) for e in manifest]
    return max(ds) if ds else 0


def _max_fanout(manifest):
    from collections import Counter
    c = Counter(e.get("parent") for e in manifest if e.get("parent") is not None)
    if not c:
        return 0, None
    path, n = c.most_common(1)[0]
    return n, path


def balance_doc(doc):
    src = SRC_DIGESTS / doc
    dst = OUT_DIGESTS / doc
    mpath = src / "_manifest.json"
    if not mpath.exists():
        print(f"  {doc}: no v7 manifest, skip")
        return None
    report_path = dst / "_balance_report.json"
    if report_path.exists() and (dst / "_manifest.json").exists():
        print(f"  {doc}: already balanced, skip")
        return json.loads(report_path.read_text())

    manifest = json.loads(mpath.read_text())
    fo_before, fo_before_at = _max_fanout(manifest)
    depth_before = _depth_stats(manifest)

    bal = Balancer(doc, manifest)
    new_manifest = bal.balance()
    root = next(e for e in new_manifest if e.get("parent") is None)
    bal.new_pages[root["path"]] = refresh_root_page(src, new_manifest)

    fo_after, fo_after_at = _max_fanout(new_manifest)
    depth_after = _depth_stats(new_manifest)

    # write balanced digest: copy every original page unchanged, add new nodes
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for rel, text in bal.new_pages.items():
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (dst / "_manifest.json").write_text(json.dumps(new_manifest, indent=2))

    report = {
        "doc": doc,
        "f": F, "occ_window": [OCC_LO, F],
        "nodes_before": len(manifest), "nodes_after": len(new_manifest),
        "nodes_added": len(bal.new_pages),
        "max_fanout_before": fo_before, "max_fanout_before_at": fo_before_at,
        "max_fanout_after": fo_after, "max_fanout_after_at": fo_after_at,
        "depth_before": depth_before, "depth_after": depth_after,
        "n_splits": len(bal.splits), "splits": bal.splits,
        "occupancy_ok": all(OCC_LO <= s <= F for sp in bal.splits for s in sp["group_sizes"]),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  {doc}: fanout {fo_before}->{fo_after}  depth {depth_before}->{depth_after}  "
          f"splits={len(bal.splits)}  +{len(bal.new_pages)} nodes")
    return report


def main():
    OUT_DIGESTS.mkdir(parents=True, exist_ok=True)
    docs = sys.argv[1:] or sorted(p.name for p in SRC_DIGESTS.iterdir() if p.is_dir())
    print(f"f = C/s = {C}/{S} = {F}   occupancy window [{OCC_LO}, {F}]")
    reports = []
    for doc in docs:
        r = balance_doc(doc)
        if r:
            reports.append(r)
    (OUT_DIGESTS / "_balance_summary.json").write_text(json.dumps(reports, indent=2))
    total_splits = sum(r["n_splits"] for r in reports)
    worst_before = max((r["max_fanout_before"] for r in reports), default=0)
    worst_after = max((r["max_fanout_after"] for r in reports), default=0)
    print(f"\nBALANCED {len(reports)} docs | total splits={total_splits} | "
          f"max fanout {worst_before} -> {worst_after} (bound f={F})")


if __name__ == "__main__":
    main()
