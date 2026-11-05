"""window.py — v7_window: the WHOLE occupancy window [OCC_LO, f] = [17, 33] enforced in ONE
build-time pass, folding the split (cap) half and the floor (borrow/merge) half into a single
traversal. This is the clean one-step version Lin asked for: split-now-merge-later becomes one pass
that runs as the final phase of a v7 build, so a finished v7_window tree already satisfies the
window. No separately invoked merge step exists.

It extends llm_index/balance.py (the cap-only reshape, kept intact as the comparison
baseline) by REUSING balance's split primitives verbatim — coherent_order (PC1), balanced_group_sizes,
_group_summary_page, embed, F, OCC_LO — so the split algorithm here is byte-for-byte v7_balanced's.
The merge tier calls the same local split(), so merge reuses the split code and does not duplicate it.
The pass is index-keyed (every manifest entry is a node). Duplicate page paths
are rejected before balancing: records must first be rebuilt with unique paths,
since an overwritten subject file cannot be recovered by restructuring it.

The pass, per internal node, in one traversal (repeated to a fixpoint; deterministic node order):

  1. Overfull (> f children): split — coherent PC1 order, ceil(n/f) balanced groups each in [17, 33],
     insert grouping nodes, recurse. (balance's split, re-expressed on the index-keyed model.)
  2. Underfull (< OCC_LO children), tier 1 BORROW: redistribute from the embedding-nearest peer
     sibling under the same parent, only if the donor stays >= 17; move the donor children whose
     summaries are closest to the receiver's centroid.
  3. Underfull, tier 2 MERGE: if no sibling can donate, merge with the embedding-nearest peer
     sibling under the same parent; if the merged node exceeds 33, re-split it through the same
     split() as step 1.
  4. Underfull, tier 3 ACCEPT: a node whose entire category holds fewer than 17 subjects is left as
     is. Category nodes are the roots of their own subtrees and inherit the textbook B+ root
     exemption from the minimum; the same exemption covers a node with no peer sibling to borrow or
     merge with (it is the sole occupant of its category slice).

Scope: the floor binds on EVERY non-root, non-category internal node, which is what §3.4 states
("non-root internal nodes are kept within the window [[f/2], f] = [17, 33]"). An earlier revision
enforced it on leaf-group nodes only, leaving intermediate grouping nodes -- the ones whose children
are themselves grouping nodes -- bounded from above but not below, although a descent reads them as
routing pages exactly like any other. Borrow and merge pair like with like (leaf-group with
leaf-group, group-of-groups with group-of-groups) so a repair never mixes subjects and routing nodes
under one parent. The two exemptions are the paper's own: the root, and a category, since "balance is
local to each category" and no subject may cross one (category = the top-level path component;
subject pages never move on disk, only parent pointers and group summaries change).

Invariants asserted at the end of every build (crash rather than ship):
  - every non-root, non-category internal node is inside [17, 33] or is an exempt small category /
    sibling-less isolate;
  - no node exceeds 33 children;
  - every subject's category ancestor is unchanged by the pass;
  - subject pages, field values, and consolidated.md are byte-identical before
    and after; routing pages and index.md are refreshed from preserved content;
  - the tree is connected and acyclic; parent pointers and child lists are consistent.

Usage:
  python window.py                 # window-pass all built v7 docs
  python window.py <DocDir>..      # specific docs (safe_filename dir names)

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, math, shutil, pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
BENCH = _BENCH
BAL_DIR = BENCH / "methods" / "llm_index"
SRC_DIGESTS = BENCH / "methods" / "llm_index" / "output" / "digests"
# BENCH, not HERE. HERE is Path(__file__).resolve().parent, and .resolve()
# follows the symlink a run tree uses for code, so this wrote the balanced
# tree back into the source tree instead of the tree being built.
OUT_DIGESTS = BENCH / "methods" / "llm_index" / "output" / "digests_balanced"

sys.path.insert(0, str(BAL_DIR))
import balance_core as balance  # noqa: E402  (split primitives; zero LLM calls)
F, OCC_LO = balance.F, balance.OCC_LO


def _top(path):
    return str(pathlib.PurePosixPath(path).parts[0])


def _is_category_path(path):
    """<Category>/_summary.md exactly: one directory deep, and not a root group."""
    parts = pathlib.PurePosixPath(path).parts
    return len(parts) == 2 and parts[1] == "_summary.md" and not parts[0].startswith("_win_group_")


class WindowBalancer:
    """Index-keyed window pass over unique page paths (node id = list index)."""

    def __init__(self, doc, manifest):
        self.doc = doc
        self.entries = [dict(e) for e in manifest]          # node id = index into this list
        paths = [e["path"] for e in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate page paths: rebuild lossless pages before balancing")
        self.names = {i: balance._content_name(e["path"], e.get("summary", ""))
                      for i, e in enumerate(self.entries)}
        self.pid = {e["path"]: i for i, e in enumerate(self.entries) if e["level"] != "leaf"}
        self.kids = defaultdict(list)
        self.roots = []
        for i, e in enumerate(self.entries):
            p = e.get("parent")
            if p is None:
                self.roots.append(i)
            else:
                self.kids[self.pid[p]].append(i)
        self.new_pages = {}                                 # path -> text (branch pages to (re)write)
        self.removed_ids = set()
        self.removed_paths = set()
        self._gid = 0
        self.borrows = self.merges = self.resplits = self.overfull_splits = self.accepted_isolate = 0

    # ---------- node accessors ----------
    def _path(self, i):
        return self.entries[i]["path"]

    def _level(self, i):
        return self.entries[i]["level"]

    def _summary(self, i):
        return self.entries[i].get("summary") or pathlib.Path(self._path(i)).stem

    def _name(self, i):
        return self.names.get(i, "")

    def _protected_record(self, i):
        """Continuation containers retain record ownership and incoming links."""
        e = self.entries[i]
        return bool(e.get("preserve_boundary") or e.get("record_id"))

    def _category(self, i):
        return None if self._level(i) == "root" else _top(self._path(i))

    def _is_leaf(self, i):
        return self._level(i) == "leaf"

    def _is_leaf_group(self, i):
        k = self.kids.get(i, [])
        return bool(k) and all(self._is_leaf(c) for c in k)

    def _is_category(self, i):
        """A category node: an internal node whose parent is the root.

        Categories are the broad divisions discovery proposed. They are exempt
        from the floor for the reason §3.4 gives -- "balance is local to each
        category", so no operation may move a subject across one -- and they are
        the roots of their own subtrees, which is the textbook B+ root exemption.
        """
        # Structural, not positional: <Category>/_summary.md, one path component
        # deep, not a _win_group. "Parent is the root" broke as soon as the root
        # itself was split, because categories then sit under root groups and
        # stopped being recognised -- at which point the floor merged categories
        # into each other, the one move Section 3.4 forbids.
        return _is_category_path(self._path(i))

    def _floor_applies(self, i):
        """Does the minimum-occupancy floor bind on this node?

        §3.4 keeps "non-root internal nodes ... within the window [f/2, f]", so
        the floor is not a leaf-group rule: an intermediate grouping node whose
        children are themselves grouping nodes is a routing page a descent reads,
        and an underfull one costs the same wasted step. Only the root and the
        categories are exempt, both for reasons the paper states.
        """
        return (bool(self.kids.get(i)) and self._level(i) != "root"
                and not self._is_category(i) and not self._protected_record(i))

    def _depth(self, i):
        d, cur, seen = 0, i, set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            p = self.entries[cur].get("parent")
            cur = self.pid.get(p) if p is not None else None
            d += 1
        return d

    def _subject_count_cat(self, cat):
        return sum(1 for i, e in enumerate(self.entries)
                   if i not in self.removed_ids and e["level"] == "leaf" and _top(e["path"]) == cat)

    def _centroid(self, ids):
        import numpy as np
        vecs = balance.embed([self._summary(i) for i in ids])
        c = vecs.mean(axis=0)
        return c / (float(np.linalg.norm(c)) + 1e-9)

    def _peer_siblings(self, nid):
        """Siblings this node may borrow from or merge with.

        Like with like: a leaf-group pairs with a leaf-group and a group-of-groups
        with a group-of-groups, so an operation never mixes subjects and routing
        nodes under one parent and the level structure survives the repair. Both
        sit under the same parent, so both are inside the same category and no
        subject crosses a category boundary (§3.4).
        """
        p = self.entries[nid].get("parent")
        if p is None:
            return []
        par = self.pid.get(p)
        if par is None:
            return []
        want = self._is_leaf_group(nid)
        return [c for c in self.kids.get(par, [])
                if c != nid and self.kids.get(c) and self._is_leaf_group(c) == want
                and not self._is_category(c) and not self._protected_record(c)]

    def _buckets(self, nid):
        """Children of a leaf-group grouped by path (duplicate-path leaves move as one unit so a
        subject file never ends up under two parents)."""
        b = defaultdict(list)
        for c in self.kids[nid]:
            b[self._path(c)].append(c)
        return b

    # ---------- one shared summary refresh (name header + child-summary bullets, no LLM) ----------
    def _refresh(self, nid):
        # The root is refreshed separately from its saved overview, never as an
        # ordinary group page. Its children are now the balanced routing choices.
        if self._level(nid) == "root":
            return
        k = self.kids.get(nid, [])
        name = self._name(nid)
        summaries = [self._summary(c) for c in k]
        descriptor = balance.branch_descriptor(name, summaries)
        self.new_pages[self._path(nid)] = balance._group_summary_page(descriptor, summaries)
        self.entries[nid]["summary"] = descriptor

    def _reparent(self, child_ids, old_parent, new_parent):
        npath = self._path(new_parent)
        moved = set(child_ids)
        for c in child_ids:
            self.entries[c]["parent"] = npath
        self.kids[old_parent] = [c for c in self.kids[old_parent] if c not in moved]
        self.kids[new_parent].extend(child_ids)

    def _remove_node(self, nid):
        e = self.entries[nid]
        self.removed_ids.add(nid)
        self.removed_paths.add(e["path"])
        self.pid.pop(e["path"], None)
        self.kids.pop(nid, None)
        p = e.get("parent")
        if p is not None:
            par = self.pid.get(p)
            if par is not None:
                self.kids[par] = [c for c in self.kids[par] if c != nid]

    # ---------- split (reused primitives; called by the overfull tier AND by merge) ----------
    def split(self, nid):
        kids = list(self.kids[nid])
        n = len(kids)
        if n <= F:
            return
        order = balance.coherent_order([self._summary(c) for c in kids])
        ordered = [kids[i] for i in order]
        k = math.ceil(n / F)
        sizes = balance.balanced_group_sizes(n, k)
        # Groups live in the SPLIT NODE's own directory, not its first child's.
        # Splitting the root (36 categories > f on 12_and_Holding) used to file
        # the category groups under whichever category came first, e.g.
        # Document_Metadata/_win_group_3/, so a grouping OF categories sat inside
        # one category on disk and every subject's category attribution broke.
        # For the root that directory is "", so root groups are top-level.
        base_dir = str(pathlib.PurePosixPath(self._path(nid)).parent)
        if base_dir == ".":
            base_dir = ""
        groups, pos = [], 0
        for sz in sizes:
            groups.append(ordered[pos:pos + sz]); pos += sz
        group_ids = []
        for grp in groups:
            self._gid += 1
            gpath = (f"{base_dir}/" if base_dir else "") + f"_win_group_{self._gid}/_summary.md"
            gid = len(self.entries)
            self.entries.append({"path": gpath, "level": "branch",
                                 "parent": self._path(nid), "summary": ""})
            self.names[gid] = "" if self._level(nid) == "root" else self._name(nid)
            self.pid[gpath] = gid
            self.kids[gid] = list(grp)
            for c in grp:
                self.entries[c]["parent"] = gpath
            group_ids.append(gid)
            assert OCC_LO <= len(grp) <= F, f"split occupancy violated: {len(grp)}"
        self.kids[nid] = group_ids
        for gid in group_ids:
            self._refresh(gid)
        self._refresh(nid)
        if len(self.kids[nid]) > F:                          # too many groups -> group the groups
            self.split(nid)

    # ---------- tier 1: borrow ----------
    def _try_borrow(self, recv):
        need = OCC_LO - len(self.kids[recv])
        sibs = self._peer_siblings(recv)
        if not sibs:
            return False
        rc = self._centroid(self.kids[recv])
        sibs.sort(key=lambda s: -float(self._centroid(self.kids[s]) @ rc))
        for donor in sibs:
            cap = len(self.kids[donor]) - OCC_LO             # donor must stay >= 17
            if cap < need:
                continue
            buckets = self._buckets(donor)
            reps = list(buckets.keys())
            rv = balance.embed([self._summary(buckets[p][0]) for p in reps])
            order = sorted(range(len(reps)), key=lambda j: -float(rv[j] @ rc))
            move, moved = [], 0
            for j in order:
                grp = buckets[reps[j]]
                if moved + len(grp) > cap:
                    continue
                move += grp; moved += len(grp)
                if moved >= need:
                    break
            if moved < need:
                continue
            self._reparent(move, donor, recv)
            self._refresh(recv); self._refresh(donor)
            return True
        return False

    # ---------- tier 2: merge (reuses split when the merge overflows) ----------
    def _try_merge(self, nid):
        sibs = self._peer_siblings(nid)
        if not sibs:
            return False
        nc = self._centroid(self.kids[nid])
        donor = max(sibs, key=lambda s: float(self._centroid(self.kids[s]) @ nc))
        self._reparent(list(self.kids[donor]), donor, nid)
        self._remove_node(donor)
        self._refresh(nid)
        if len(self.kids[nid]) > F:                          # merged over f -> reuse split()
            self.split(nid)
            self.resplits += 1
        return True

    # ---------- the single window pass (to a fixpoint) ----------
    def window(self):
        guard = 0
        while True:
            guard += 1
            if guard > 200000:
                raise RuntimeError(f"{self.doc}: window pass did not converge")
            internal = [i for i in self.kids if i not in self.removed_ids and self.kids.get(i)]
            internal.sort(key=lambda i: (-self._depth(i), self._path(i)))    # deepest first, deterministic
            acted = False
            for i in internal:
                if i in self.removed_ids:
                    continue
                n = len(self.kids.get(i, []))
                if n > F:                                    # overfull -> split
                    self.split(i)
                    self.overfull_splits += 1
                    acted = True; break
                if n < OCC_LO and self._floor_applies(i):
                    cat = self._category(i)
                    if self._subject_count_cat(cat) < OCC_LO:
                        continue                             # tier 3: small category, root exemption
                    if self._try_borrow(i):
                        self.borrows += 1; acted = True; break
                    if self._try_merge(i):
                        self.merges += 1; acted = True; break
                    self.accepted_isolate += 1               # no peer sibling: isolate, accept
            if not acted:
                break
        # Refresh even unchanged branches, deepest first. This repairs legacy
        # '_summary: N items' descriptors and propagates changed child topics up
        # every ancestor after split, borrow, or merge.
        internal = [i for i in self.kids if i not in self.removed_ids and self.kids.get(i)]
        for i in sorted(internal, key=lambda i: (-self._depth(i), self._path(i))):
            self._refresh(i)
        return [e for i, e in enumerate(self.entries) if i not in self.removed_ids]


# ═══════════════════════ invariants (crash rather than ship) ═══════════════════════
def _children_map(manifest):
    by = {e["path"]: e for e in manifest if e["level"] != "leaf"}
    ch = defaultdict(list)
    for e in manifest:
        ch[e.get("parent")].append(e["path"])   # by path; leaf dups appear multiple times (occupancy)
    return by, ch


def _leaf_category_ancestor(manifest):
    """Map each subject to the category node its parent chain passes through.

    Section 3.4: balance is local to each category and no subject ever crosses
    one. A subject's category is <top>/_summary.md for its own top path
    component (subject pages never move on disk), so the check is that this node
    is still an ancestor. None means the chain no longer passes through it,
    which is a violation whatever else the tree looks like. Walking the chain
    rather than the path is what catches a cross-category reparent; identifying
    the category structurally is what survives the root being split into groups.
    """
    by = {e["path"]: e for e in manifest if e["level"] != "leaf"}
    out = {}
    for e in manifest:
        if e["level"] != "leaf":
            continue
        want = f"{_top(e['path'])}/_summary.md"
        cur, seen, hit = e.get("parent"), set(), None
        while cur is not None and cur in by and cur not in seen:
            seen.add(cur)
            if cur == want:
                hit = _top(cur); break
            cur = by[cur].get("parent")
        out[e["path"]] = hit
    return out


def _assert_structure(manifest):
    by, ch = _children_map(manifest)
    # fanout <= f on every internal node
    for p, kids in ch.items():
        if p is not None:
            assert len(kids) <= F, f"node {p} has {len(kids)} > f={F} children"
    # minimum-occupancy floor on every non-root, non-category internal node (§3.4)
    leaves = [e for e in manifest if e["level"] == "leaf"]
    cat_subj = defaultdict(int)
    for e in leaves:
        cat_subj[_top(e["path"])] += 1

    def _is_leaf_group(path):
        kids = ch.get(path) or []
        return bool(kids) and all(by.get(k, {}).get("level", "leaf") == "leaf" for k in kids)

    for p, kids in ch.items():
        if p is None or not kids:
            continue
        pe = by.get(p)
        if pe is None or pe.get("level") == "root":
            continue
        if _is_category_path(p) or pe.get("preserve_boundary") or pe.get("record_id"):
            continue                                   # category: exempt (§3.4)
        if cat_subj[_top(p)] < OCC_LO or len(kids) >= OCC_LO:
            continue
        want = _is_leaf_group(p)
        sibs = [c for c in ch.get(pe["parent"], []) if c != p and ch.get(c)
                and _is_leaf_group(c) == want and not _is_category_path(c)
                and not by.get(c, {}).get("preserve_boundary") and not by.get(c, {}).get("record_id")]
        assert not sibs, (f"internal node {p} has {len(kids)} < {OCC_LO} with a mergeable "
                          f"peer sibling still present")
    # every subject still descends through its own category node (§3.4)
    anc = _leaf_category_ancestor(manifest)
    orphans = [p for p, c in anc.items() if c is None]
    assert not orphans, f"{len(orphans)} subject(s) no longer descend through their category: {orphans[:3]}"
    # connected + acyclic on the BRANCH/ROOT skeleton (leaves are terminal, dup paths allowed)
    branch_ids = {e["path"] for e in manifest if e["level"] != "leaf"}
    roots = [e["path"] for e in manifest if e.get("parent") is None]
    assert len(roots) == 1, f"expected one root, found {len(roots)}"
    seen, stack = set(), list(roots)
    while stack:
        cur = stack.pop()
        assert cur not in seen, f"cycle at {cur}"
        seen.add(cur)
        stack.extend([c for c in ch.get(cur, []) if c in branch_ids])
    assert seen == branch_ids, f"disconnected branches: {len(branch_ids - seen)}"
    for e in manifest:
        if e.get("parent") is not None:
            assert e["parent"] in by, f"{e['path']} points at missing parent {e['parent']}"


def _assert_byte_identity(src, dst, orig_manifest, new_manifest):
    """Every subject page and saved overview remain byte-identical."""
    orig_paths = {e["path"]: e for e in orig_manifest}
    checked = set()
    for e in new_manifest:
        p = e["path"]
        if p in checked or e["level"] != "leaf":
            continue
        checked.add(p)
        if p in orig_paths:
            a, b = (src / p), (dst / p)
            assert a.exists() and b.exists(), f"missing page {p}"
            assert a.read_bytes() == b.read_bytes(), f"page changed (must be byte-identical): {p}"
    if (src / "consolidated.md").exists():
        assert (src / "consolidated.md").read_bytes() == (dst / "consolidated.md").read_bytes(), \
            "saved consolidated overview changed"


# ═══════════════════════ report helpers ═══════════════════════
def _cat_of(path, by):
    return None if by.get(path, {}).get("level") == "root" else _top(path)


def leaf_group_stats(manifest):
    by, ch = _children_map(manifest)
    leaves = [e for e in manifest if e["level"] == "leaf"]
    cat_subj = defaultdict(int)
    for e in leaves:
        cat_subj[_top(e["path"])] += 1
    lg = []
    below = 0
    single = 0
    for p, kids in ch.items():
        if p is None or by.get(p, {}).get("level") == "root":
            continue
        if kids and all(by.get(c, {}).get("level", "leaf") == "leaf" for c in kids):
            lg.append(len(kids))
            if len({_top(c) for c in kids}) == 1:
                single += 1
            entry = by.get(p, {})
            if (len(kids) < OCC_LO and cat_subj[_top(p)] >= OCC_LO
                    and not _is_category_path(p) and not entry.get("preserve_boundary")
                    and not entry.get("record_id")):
                below += 1
    exempt = sum(1 for c, n in cat_subj.items() if n < OCC_LO)
    return {"leaf_group_nodes": len(lg),
            "leaf_group_size_min": min(lg) if lg else None,
            "leaf_group_size_max": max(lg) if lg else None,
            "leaf_groups_below_floor_nonexempt": below,
            "exempt_small_categories": exempt,
            "single_category_rate_leaf_group": round(single / len(lg), 3) if lg else None}


def window_doc(doc, force=False):
    src = SRC_DIGESTS / doc
    dst = OUT_DIGESTS / doc
    mpath = src / "_manifest.json"
    if not mpath.exists():
        print(f"  {doc}: no v7 manifest, skip")
        return None
    report_path = dst / "_window_report.json"
    if report_path.exists() and (dst / "_manifest.json").exists() and not force:
        print(f"  {doc}: already windowed, skip")
        return json.loads(report_path.read_text())

    orig = json.loads(mpath.read_text())
    cat_before = _leaf_category_ancestor(orig)
    fo_before, _ = balance._max_fanout(orig)
    depth_before = balance._depth_stats(orig)

    bal = WindowBalancer(doc, orig)
    new_manifest = bal.window()
    root = next(e for e in new_manifest if e.get("parent") is None)
    bal.new_pages[root["path"]] = balance.refresh_root_page(src, new_manifest)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for rel, text in bal.new_pages.items():
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for rel in bal.removed_paths:
        if rel not in bal.new_pages and (dst / rel).exists():
            (dst / rel).unlink()
    (dst / "_manifest.json").write_text(json.dumps(new_manifest, indent=2))

    _assert_structure(new_manifest)
    assert cat_before == _leaf_category_ancestor(new_manifest), "a subject's category ancestor changed"
    _assert_byte_identity(src, dst, orig, new_manifest)

    fo_after, fo_after_at = balance._max_fanout(new_manifest)
    lg = leaf_group_stats(new_manifest)
    report = {
        "doc": doc, "f": F, "occ_window": [OCC_LO, F],
        "nodes_before": len(orig), "nodes_after": len(new_manifest),
        "nodes_added": sum(1 for e in new_manifest if e["path"] not in {x["path"] for x in orig}),
        "nodes_removed": len(bal.removed_paths),
        "overfull_splits": bal.overfull_splits, "borrows": bal.borrows, "merges": bal.merges,
        "resplits": bal.resplits, "accepted_isolate": bal.accepted_isolate,
        "max_fanout_before": fo_before, "max_fanout_after": fo_after, "max_fanout_after_at": fo_after_at,
        "depth_before": depth_before, "depth_after": balance._depth_stats(new_manifest),
        **lg, "invariants_ok": True,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  {doc}: fanout {fo_before}->{fo_after}  depth {depth_before}->{report['depth_after']}  "
          f"borrow={bal.borrows} merge={bal.merges} resplit={bal.resplits} split={bal.overfull_splits} "
          f"+{report['nodes_added']}/-{report['nodes_removed']}  lg[{lg['leaf_group_size_min']},"
          f"{lg['leaf_group_size_max']}] below={lg['leaf_groups_below_floor_nonexempt']}")
    return report


def main():
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    OUT_DIGESTS.mkdir(parents=True, exist_ok=True)
    docs = args or sorted(p.name for p in SRC_DIGESTS.iterdir() if p.is_dir())
    print(f"f = {F}   occupancy window [{OCC_LO}, {F}]   (v7_window: one build-time pass)")
    reports = [r for r in (window_doc(d, force=force) for d in docs) if r]
    (OUT_DIGESTS / "_window_summary.json").write_text(json.dumps(reports, indent=2))
    tb = sum(r["borrows"] for r in reports); tm = sum(r["merges"] for r in reports)
    trs = sum(r["resplits"] for r in reports); ts = sum(r["overfull_splits"] for r in reports)
    print(f"\nWINDOWED {len(reports)} docs | borrow={tb} merge={tm} resplit={trs} split={ts} | "
          f"max fanout after <= {F}: {all(r['max_fanout_after'] <= F for r in reports)} | "
          f"all invariants ok: {all(r['invariants_ok'] for r in reports)}")


if __name__ == "__main__":
    main()
