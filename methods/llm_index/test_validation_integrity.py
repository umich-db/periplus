"""Offline regression tests for corrupt builds that must not reach evaluation."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import verify_tree


class ValidationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.doc = Path(self.tmp.name)
        self.source = "Ben teaches history. Ben lives with his family."
        (self.doc / "source.txt").write_text(self.source, encoding="utf-8")
        self.pages = {
            "index.md": "# Example\n\nA teacher and his family.\n\n## Categories\n- Characters and professions\n",
            "Characters/_summary.md": "# Characters\n\n- Ben: history teacher and family relationships\n",
            "Characters/ben.md": "# Ben\n\nOccupation: history teacher.\n",
            "Characters/family.md": "# Ben's family\n\nHe lives with his family.\n",
        }
        self.manifest = [
            {"path": "index.md", "parent": None, "level": "root", "summary": "Example screenplay about Ben"},
            {"path": "Characters/_summary.md", "parent": "index.md", "level": "branch",
             "summary": "Characters: Ben's teaching profession and family"},
            {"path": "Characters/ben.md", "parent": "Characters/_summary.md", "level": "leaf",
             "summary": "Ben: history teacher", "spans": [{"start": 0, "end": 20}],
             "links": ["Characters/family.md"]},
            {"path": "Characters/family.md", "parent": "Characters/_summary.md", "level": "leaf",
             "summary": "Ben's family and home", "spans": [{"start": 21, "end": len(self.source)}],
             "links": ["Characters/ben.md"]},
        ]
        self.integrity = {
            "version": 1, "unaccounted_keys": [], "unaccounted_fields": [],
            "source": {"path": "source.txt", "normalization": "raw",
                       "length_chars": len(self.source), "sha256": self.hash(self.source)},
            "fields": [{"id": path, "chars": len(text), "sha256": self.hash(text),
                        "segments": [{"page": path, "start": 0, "end": len(text)}]}
                       for path, text in self.pages.items() if path in ("Characters/ben.md", "Characters/family.md")],
        }
        for path, text in self.pages.items():
            out = self.doc / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        self.write("_provenance.json", {"spans": {"Ben teaches history.": [{"start": 0, "end": 20}]}})
        self.write("_chunking.json", {"doc_chars": len(self.source), "bounds": [{"start": 0, "end": len(self.source)}]})
        self.save()

    @staticmethod
    def hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def write(self, name, value):
        (self.doc / name).write_text(json.dumps(value), encoding="utf-8")

    def save(self):
        self.write("_manifest.json", self.manifest)
        self.write("_integrity.json", self.integrity)

    def errors(self):
        return verify_tree.validate_doc(self.doc)["errors"]

    def assert_error(self, fragment):
        errors = self.errors()
        self.assertTrue(any(fragment in e for e in errors), errors)

    def test_intact_build_passes(self):
        self.assertEqual(self.errors(), [])

    def test_duplicate_manifest_path_rejected(self):
        self.manifest.append(copy.deepcopy(self.manifest[-1]))
        self.save()
        self.assert_error("duplicate manifest path")

    def test_parent_cycle_is_reported_without_hanging(self):
        self.manifest[1]["parent"] = "Characters/ben.md"
        self.save()
        self.assert_error("parent cycle")
        self.assert_error("unreachable manifest nodes")

    def test_missing_parent_rejected(self):
        self.manifest[-1]["parent"] = "deleted.md"
        self.save()
        self.assert_error("missing parent")

    def test_missing_page_and_unmanaged_page_rejected(self):
        (self.doc / "Characters/ben.md").unlink()
        (self.doc / "forgotten.md").write_text("# Orphan", encoding="utf-8")
        self.assert_error("missing or unreadable page")
        self.assert_error("unreachable page files")

    def test_blank_and_generic_descriptions_rejected(self):
        for description in ("", "   ", "_summary: 29 items", "Group 3", "14 items"):
            with self.subTest(description=description):
                self.manifest[1]["summary"] = description
                self.save()
                self.assert_error("routing description")

    def test_category_names_are_meaningful_even_with_counts(self):
        for description in ("Characters: 14 items", "Groups and Social Categories: 3 items", "Objects:2 items"):
            with self.subTest(description=description):
                self.manifest[-1]["summary"] = description
                self.save()
                self.assertEqual(self.errors(), [])

    def test_clipped_leaf_and_changed_field_hash_rejected(self):
        (self.doc / "Characters/ben.md").write_text("# Ben\n*(truncated)*\n", encoding="utf-8")
        self.assert_error("clipped leaf content")
        self.assert_error("content checksum mismatch")

    def test_broken_manifest_and_markdown_links_rejected(self):
        self.manifest[-1]["links"] = ["Characters/missing.md"]
        self.save()
        (self.doc / "Characters/_summary.md").write_text("[missing](missing.md)", encoding="utf-8")
        self.assert_error("broken links link")
        self.assert_error("broken page link")

    def test_broken_continuation_link_rejected(self):
        self.manifest[-1]["continuation_of"] = "missing.md"
        self.manifest[-1]["continuations"] = ["also_missing.md"]
        self.save()
        self.assert_error("broken continuation link")
        self.assert_error("broken continuations link")

    def test_continuation_cycle_rejected(self):
        self.manifest[-1]["next_page"] = "Characters/ben.md"
        self.manifest[-2]["next_page"] = "Characters/family.md"
        self.save()
        self.assert_error("continuation cycle")

    def test_continuation_owner_and_neighbor_completeness(self):
        owner, left, right = (entry["path"] for entry in self.manifest[1:])
        self.manifest[1]["continuations"] = [left, right]
        for index, entry in enumerate(self.manifest[2:], 1):
            entry.update(continuation_of=owner, continuation_index=index,
                         continuations=[right if index == 1 else left])
        self.save()
        self.assertEqual(self.errors(), [])
        self.manifest[-1]["continuations"] = []
        self.save()
        self.assert_error("broken continuation neighbor links")
        self.manifest[1]["continuations"] = [left]
        self.save()
        self.assert_error("continuation owner does not list all parts")

    def test_integrity_fields_can_span_multiple_pages(self):
        first, second = self.integrity["fields"]
        self.integrity["fields"] = [{
            "id": "one-original-record", "chars": first["chars"] + second["chars"],
            "sha256": self.hash(self.pages["Characters/ben.md"] + self.pages["Characters/family.md"]),
            "segments": first["segments"] + second["segments"],
        }]
        self.save()
        self.assertEqual(self.errors(), [])
        self.integrity["fields"][0]["segments"].reverse()
        self.save()
        self.assert_error("content checksum mismatch")

    def test_bad_segment_bounds_and_uncovered_page_rejected(self):
        self.integrity["fields"][0]["segments"][0]["start"] = -1
        self.save()
        self.assert_error("segment bounds invalid")
        self.assert_error("missing integrity coverage")

    def test_unaccounted_keys_and_fields_rejected(self):
        self.integrity["unaccounted_keys"] = ["Motivations"]
        self.integrity["unaccounted_fields"] = ["Characters/0/role"]
        self.save()
        self.assert_error("unaccounted_keys: 1")
        self.assert_error("unaccounted_fields: 1")

    def test_spans_checked_against_source_bounds(self):
        self.manifest[-1]["spans"] = [{"start": 1, "end": len(self.source) + 1}]
        self.save()
        self.assert_error("evidence span exceeds source length")
        self.write("_provenance.json", {"spans": {"bad": [{"start": -1, "end": 2}]}})
        self.assert_error("invalid evidence span bounds: provenance")

    def test_source_checksum_mismatch_rejected(self):
        (self.doc / "source.txt").write_text(self.source.replace("Ben", "Dan"), encoding="utf-8")
        self.assert_error("source checksum mismatch")

    def test_cleaned_source_checksum_without_api_imports(self):
        (self.doc / "source.txt").write_text("download headers\n<html><body>" + self.source + "</body></html>", encoding="utf-8")
        self.integrity["source"]["normalization"] = "clean_document"
        self.save()
        self.assertEqual(self.errors(), [])

    def test_integrity_optional_only_when_absent_on_legacy(self):
        (self.doc / "_integrity.json").unlink()
        self.assertEqual(self.errors(), [])
        report = verify_tree.validate_doc(self.doc, strict_integrity=True)
        self.assertIn("missing file: _integrity.json", report["errors"])
        (self.doc / "_integrity.json").write_text("{broken", encoding="utf-8")
        self.assert_error("invalid JSON in _integrity.json")

    def test_malformed_manifest_returns_errors(self):
        self.write("_manifest.json", [None, {"path": ["invalid"]}, {"path": "../outside.md"}])
        self.assert_error("manifest entry 0 must be an object")
        self.assert_error("invalid manifest path")

    def test_malformed_level_returns_errors(self):
        self.manifest[-1]["level"] = []
        self.save()
        self.assert_error("invalid node level")

    def test_json_null_is_not_a_valid_report_or_manifest(self):
        self.write("_integrity.json", None)
        self.assert_error("integrity report must be an object")
        self.write("_manifest.json", None)
        self.assert_error("manifest must be a list")

    def test_cli_json_and_legacy_positional(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = verify_tree.main([str(self.doc), "--json", "--strict-integrity"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["failing_checks"], 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify_tree.check_doc(self.doc), [])


if __name__ == "__main__":
    unittest.main()
