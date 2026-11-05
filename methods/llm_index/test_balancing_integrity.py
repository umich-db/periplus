"""Offline regression tests for lossless routing repair.

Embedding order is replaced with a deterministic test fixture. Production
balancing still uses cached all-mpnet; these tests assert structure, not clustering
quality. Run: python -m unittest discover -s methods/llm_index -p test_balancing_integrity.py
"""
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import balance_core as core
import balance_window as window


def fixture_embeddings(texts):
    import numpy as np
    # Equal vectors make ties deterministic without downloading a model or
    # confounding routing integrity checks with an embedding quality measure.
    return np.ones((len(texts), 4), dtype="float32") / 2


def tree(categories=1, counts=(40,)):
    manifest = [{"path": "index.md", "level": "root", "parent": None,
                 "summary": "The Story — screenplay"}]
    for category in range(categories):
        cat = f"Character_occupations_{category}"
        catpath = f"{cat}/_summary.md"
        manifest.append({"path": catpath, "level": "branch", "parent": "index.md",
                         "summary": f"Character occupations {category}: people and work"})
        for group, count in enumerate(counts):
            parent = catpath
            if len(counts) > 1:
                parent = f"{cat}/group_{group}/_summary.md"
                manifest.append({"path": parent, "level": "branch", "parent": catpath,
                                 "summary": f"_summary: {count} items"})
            for subject in range(count):
                path = f"{cat}/subject_{group}_{subject}.md"
                manifest.append({"path": path, "level": "leaf", "parent": parent,
                                 "summary": f"Person {category}/{group}/{subject}: history teacher and parent",
                                 "spans": [[subject, subject + 1]], "links": []})
    return manifest


class BalancingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.order = patch.object(core, "coherent_order", side_effect=lambda s: list(range(len(s))))
        self.embed = patch.object(core, "embed", side_effect=fixture_embeddings)
        self.order.start()
        self.embed.start()
        self.addCleanup(self.order.stop)
        self.addCleanup(self.embed.stop)

    def assert_routing(self, bal, manifest):
        window._assert_structure(manifest)
        for entry in manifest:
            self.assertLessEqual(len(entry["summary"]), core.S)
            self.assertNotIn("_summary:", entry["summary"])
            if entry["level"] == "branch":
                self.assertIn(entry["path"], bal.new_pages)
                self.assertLessEqual(len(bal.new_pages[entry["path"]]), core.PAGE_SIZE)
                self.assertNotIn("*(truncated)*", bal.new_pages[entry["path"]])

    def test_branch_page_preserves_every_child_under_capacity(self):
        children = [f"Subject{i:02d} " + "detailed subject information " * 10 for i in range(core.F)]
        page = core._group_summary_page("Characters and occupations", children)
        self.assertLessEqual(len(page), core.PAGE_SIZE)
        self.assertEqual(sum(line.startswith("- ") for line in page.splitlines()), core.F)
        for i in range(core.F):
            self.assertIn(f"Subject{i:02d}", page)

    def test_split_refreshes_categories_and_new_group_descriptors(self):
        original = tree()
        bal = window.WindowBalancer("story", original)
        result = bal.window()
        self.assert_routing(bal, result)
        self.assertEqual(window._leaf_category_ancestor(original), window._leaf_category_ancestor(result))
        branches = [e for e in result if e["level"] == "branch"]
        self.assertTrue(all("Character occupations" in e["summary"] for e in branches))
        self.assertEqual([e for e in original if e["level"] == "leaf"],
                         [{**e, "parent": o["parent"]} for e, o in zip(
                             [e for e in result if e["level"] == "leaf"],
                             [e for e in original if e["level"] == "leaf"])])

    def test_borrow_and_merge_refresh_unchanged_ancestors(self):
        for counts, operation in [((5, 29), "borrows"), ((10, 10), "merges")]:
            with self.subTest(counts=counts):
                bal = window.WindowBalancer("story", tree(counts=counts))
                result = bal.window()
                self.assertGreater(getattr(bal, operation), 0)
                self.assert_routing(bal, result)

    def test_continuation_containers_keep_record_boundaries(self):
        original = tree(counts=(2, 24))
        for e in original:
            if e["path"].endswith("group_0/_summary.md"):
                e.update(record_id="record-123", preserve_boundary=True)
        first = [dict(e) for e in original if e["parent"] == "Character_occupations_0/group_0/_summary.md"]
        bal = window.WindowBalancer("story", original)
        result = bal.window()
        self.assert_routing(bal, result)
        self.assertEqual(first, [e for e in result if e["parent"] == "Character_occupations_0/group_0/_summary.md"])

    def test_root_refresh_retains_overview_and_balanced_children(self):
        original = tree(categories=40, counts=(2,))
        bal = window.WindowBalancer("story", original)
        result = bal.window()
        self.assert_routing(bal, result)
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory)
            (src / "index.md").write_text("# Story\n\n## Categories\nOld stale categories")
            overview = "Ben is a history teacher investigating a wrongful killing. " * 100
            (src / "consolidated.md").write_text(overview)
            page = core.refresh_root_page(src, result)
            self.assertIn("Ben is a history teacher", page)
            self.assertNotIn("Old stale categories", page)
            self.assertLessEqual(len(page), core.ROOT_PAGE_TARGET)
            root_children = [e for e in result if e.get("parent") == "index.md"]
            self.assertEqual(sum(line.startswith("- ") for line in page.splitlines()), len(root_children))
            self.assertEqual((src / "consolidated.md").read_text(), overview)

    def test_window_doc_preserves_subject_bytes_and_saved_overview(self):
        original = tree(counts=(5, 29))
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            src = base / "src" / "story"
            src.mkdir(parents=True)
            (src / "_manifest.json").write_text(json.dumps(original))
            (src / "consolidated.md").write_text("Ben is a history teacher.")
            for e in original:
                page = src / e["path"]
                page.parent.mkdir(parents=True, exist_ok=True)
                page.write_text("# The Story\n\nSource evidence retained: αβγ\n")
            with patch.object(window, "SRC_DIGESTS", base / "src"), \
                    patch.object(window, "OUT_DIGESTS", base / "dst"):
                report = window.window_doc("story")
            self.assertTrue(report["invariants_ok"])
            dst = base / "dst" / "story"
            for e in original:
                if e["level"] == "leaf":
                    self.assertEqual((src / e["path"]).read_bytes(), (dst / e["path"]).read_bytes())
            self.assertIn("Ben is a history teacher", (dst / "index.md").read_text())

    def test_duplicate_paths_fail_instead_of_collapsing_records(self):
        original = tree(counts=(2,))
        original.append(dict(original[-1]))
        for cls in (window.WindowBalancer, core.Balancer):
            with self.assertRaisesRegex(ValueError, "Duplicate page paths"):
                cls("story", original)


if __name__ == "__main__":
    unittest.main()
