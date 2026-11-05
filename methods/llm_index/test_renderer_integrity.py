"""Regression tests for preserving extracted records during page generation."""
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("AZURE_API_KEY", "offline-unused")
os.environ.setdefault("AZURE_API_BASE", "https://example.invalid")
os.environ.setdefault("AZURE_API_VERSION", "2024-02-01")
BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH / "src/wiki_index"))
import pipeline


class RendererIntegrity(unittest.TestCase):
    def test_safe_directory_identity(self):
        used = set()
        names = [pipeline._directory_name(name, used) for name in ["A/B", "A?B", "???"]]
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(names))
        with self.assertRaises(ValueError):
            pipeline._unique_page_path(Path("/tmp"), "/outside.md", "body", [])
        with self.assertRaises(ValueError):
            pipeline._unique_page_path(Path("/tmp"), "../outside.md", "body", [])

    def test_exact_category_beats_earlier_partial_match(self):
        data = {"Character Relationships": ["wrong"], "Relationships": ["right"]}
        self.assertEqual(pipeline._find_section(data, "Relationships"), ["right"])
        self.assertIsNone(pipeline._find_section(data, "School Relationships"))

    def test_repeated_titles_long_lists_and_new_categories_survive(self):
        compiled = {
            "Characters": [
                {"name": "Jacob", "state": "FIRST_RECORD " + "early detail " * 40},
                {"name": "Jacob", "state": "SECOND_RECORD " + "later detail " * 40},
                {"name": "Ben", "facts": [f"UNIQUE_FACT_{i:03d} " + "detail " * 40 for i in range(28)]},
            ],
            "Causal links": [{"cause": "A distinct late cause", "effect": "A distinct late effect"}],
            "Values": {"count": 0, "truth": False, "items": [f"ITEM_{i}" for i in range(18)]},
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.txt"
            source.write_text("\n".join(pipeline._atoms(compiled)))
            text = pipeline.clean_document(source.read_text())
            bounds = pipeline.chunk_bounds(text)
            prov, spans = {}, {}
            pipeline._record_provenance(compiled, 1, prov, text, 0, spans)
            prov = {k: sorted(v) for k, v in prov.items()}
            pipeline.render_extracted(source, root, "movie", "Movie",
                                      {"sections": [{"name": "Characters"}], "overview": "The complete story."},
                                      compiled, prov, bounds, spans, consolidate=False)
            directory = root / "movie"
            manifest = json.loads((directory / "_manifest.json").read_text())
            self.assertEqual(len(manifest), len({e["path"] for e in manifest}))
            integrity = json.loads((directory / "_integrity.json").read_text())
            self.assertEqual(integrity["unaccounted_fields"], [])
            self.assertEqual(integrity["unaccounted_keys"], [])
            bodies = []
            for record in integrity["fields"]:
                body = "".join((directory / s["page"]).read_text()[s["start"]:s["end"]] for s in record["segments"])
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), record["sha256"])
                bodies.append(body)
            text = "\n".join(bodies)
            for marker in ["FIRST_RECORD", "SECOND_RECORD", "UNIQUE_FACT_027", "ITEM_17", "A distinct late effect", "## count\n0", "## truth\nFalse"]:
                self.assertIn(marker, text)
            for entry in manifest:
                page = (directory / entry["path"]).read_text()
                self.assertNotIn("*(truncated)*", page)
                self.assertLessEqual(len(page), pipeline.ROOT_SUMMARY_CAP if entry["level"] == "root" else pipeline.PAGE_SIZE)
            self.assertTrue(any(e.get("continuation_of") for e in manifest))


if __name__ == "__main__":
    unittest.main()
