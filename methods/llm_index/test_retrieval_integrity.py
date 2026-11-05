"""Offline regression tests for navigation, provenance fallback and query traces.

Run with the repository Python environment; all model calls are stubbed.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("AZURE_API_KEY", "test")
os.environ.setdefault("AZURE_API_VERSION", "2024-02-01")
os.environ.setdefault("AZURE_API_BASE", "https://example.invalid/")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "wiki_index"))
import pipeline  # noqa: E402


def entry(path, parent, summary, level="leaf", **extra):
    return {"path": path, "parent": parent, "summary": summary,
            "level": level, "links": [], "spans": [], "chunks": [], **extra}


class RetrievalIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.wiki = Path(self.tmp.name) / "movie"
        self.wiki.mkdir()
        self.old_router = pipeline._ROUTER
        pipeline.set_router(None)
        self.addCleanup(pipeline.set_router, self.old_router)

    def tree(self, entries, pages=None):
        (self.wiki / "_manifest.json").write_text(json.dumps(entries))
        for row in entries:
            content = (pages or {}).get(row["path"], row["summary"])
            if content is not None:
                file = self.wiki / row["path"]
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content)

    def small_tree(self, **leaf_extra):
        return [entry("index.md", None, "A teacher seeks justice", "root"),
                entry("characters.md", "index.md", "Characters: Ben Du Toit", "branch"),
                entry("ben.md", "characters.md", "Ben Du Toit: profession history teacher", **leaf_extra)]

    def test_none_can_reach_existing_fact_without_exceeding_budget(self):
        self.tree(self.small_tree())
        digest = pipeline.traverse("What is Ben Du Toit's profession?", self.wiki,
                                   max_chars=600, router=lambda *a, **k: [])
        self.assertIn("history teacher", digest)
        self.assertLessEqual(len(digest), 600)
        self.assertEqual(pipeline._LAST_VISITED, ["index.md", "characters.md", "ben.md"])
        self.assertTrue(any(r["mode"] == "none_lexical_recovery"
                            for r in pipeline._LAST_TRAVERSAL_TRACE["routes"]))

    def test_rejected_and_missing_pages_are_not_reported_as_read(self):
        rows = [entry("index.md", None, "Root", "root")]
        rows += [entry(f"leaf{i}.md", "index.md", f"Subject {i}") for i in range(9)]
        self.tree(rows, {"leaf0.md": None})
        pipeline.traverse("Subject", self.wiki, router=lambda *a, **k: [])
        trace = pipeline._LAST_TRAVERSAL_TRACE
        self.assertNotIn("leaf0.md", pipeline._LAST_VISITED)
        self.assertIn("leaf1.md", pipeline._LAST_VISITED)
        self.assertLessEqual(trace["none_recoveries"], pipeline.LEVEL_UP_BUDGET)
        self.assertEqual(pipeline._LAST_VISITED, [r["path"] for r in trace["pages_read"]])
        self.assertTrue(any(s["reason"] == "missing_file" for s in trace["skipped_pages"]))

    def test_navigation_pages_leave_room_for_subject_and_trace_exact_context(self):
        self.tree(self.small_tree(), {"index.md": "Overview. " * 700,
                                     "characters.md": "Ben Du Toit and other characters. " * 150})
        digest = pipeline.traverse("Ben Du Toit profession", self.wiki)
        self.assertIn("history teacher", digest)
        self.assertLessEqual(len(digest), pipeline.CONTEXT_BUDGET)
        for page in pipeline._LAST_TRAVERSAL_TRACE["pages_read"]:
            self.assertEqual(digest[page["context_start"]:page["context_end"]], page["text"])
            self.assertEqual((self.wiki / page["path"]).read_text()[page["start"]:page["end"]], page["text"])
        self.assertTrue(pipeline._LAST_TRAVERSAL_TRACE["pages_read"][1]["partial"])

    def test_routing_budget_includes_prompt_and_shows_every_balanced_child(self):
        children = [entry(f"leaf{i}.md", "index.md", "Ben history and professions " * 8)
                    for i in range(pipeline.MAX_FANOUT)]
        prompts = []
        def stub(prompt, **kwargs):
            prompts.append(prompt)
            return str(len(children))
        with patch.object(pipeline, "llm", stub):
            picked = pipeline.llm_route_children("What is Ben's profession?", children)
        self.assertEqual(picked, [children[-1]])
        self.assertLessEqual(len(prompts[0]), pipeline.ROUTING_BUDGET)
        self.assertEqual(len(pipeline._LAST_ROUTE_TRACE["shown_paths"]), len(children))

    def test_oversize_question_falls_back_without_overbudget_route_call(self):
        with patch.object(pipeline, "llm", side_effect=AssertionError("unexpected model call")):
            self.assertIsNone(pipeline.llm_route_children("q" * 5000, self.small_tree()[1:]))

    def test_late_source_answer_is_ranked_within_reached_chunk(self):
        text = "A quiet street and ordinary conversation. " * 650
        answer = "Ben Du Toit's profession is history teacher."
        text += answer + " The school closes. " * 50
        self.tree(self.small_tree(chunks=[1]))
        (self.wiki / "_chunking.json").write_text(json.dumps({
            "bounds": [{"n": 1, "start": 0, "end": len(text)}]}))
        excerpt = pipeline._evidence_text(self.wiki, text, ["index.md", "ben.md"], 1600,
                                           question="What is Ben Du Toit's profession?")
        self.assertIn(answer, excerpt)
        self.assertLessEqual(len(excerpt), 1600)
        for span in pipeline._LAST_EVIDENCE_TRACE["passages"]:
            self.assertEqual(text[span["start"]:span["end"]], span["text"])
            self.assertEqual(excerpt[span["context_start"]:span["context_end"]], span["text"])

    def test_unreached_subject_and_root_provenance_cannot_leak_source(self):
        text = "Root-wide secret. Ben is a teacher."
        rows = self.small_tree(spans=[{"start": 0, "end": len(text), "how": "verbatim"}])
        rows[0]["spans"] = rows[-1]["spans"]
        self.tree(rows)
        excerpt = pipeline._evidence_text(self.wiki, text, ["index.md", "characters.md"], 12000)
        self.assertEqual(excerpt, "")
        self.assertEqual(pipeline._LAST_EVIDENCE_TRACE["empty_reason"], "no_subject_reached")

    def test_continuation_is_read_before_unrelated_lateral_subject(self):
        rows = self.small_tree(continuations=["part2.md"], links=["other.md"])
        rows += [entry("part2.md", "characters.md", "His profession is history teacher"),
                 entry("other.md", "index.md", "Other unrelated character")]
        self.tree(rows)
        pipeline.traverse("Ben Du Toit", self.wiki)
        order = pipeline._LAST_VISITED
        self.assertLess(order.index("part2.md"), order.index("other.md"))
        part2 = next(p for p in pipeline._LAST_TRAVERSAL_TRACE["pages_read"] if p["path"] == "part2.md")
        self.assertEqual(part2["reason"], "continuation")

    def test_query_returns_serializable_exact_context_and_provenance(self):
        text = "Ben Du Toit teaches history at a school."
        raw = Path(self.tmp.name) / "source.txt"
        raw.write_text(text)
        cleaned = pipeline.clean_document(text)
        self.tree(self.small_tree(spans=[{"start": 0, "end": len(cleaned), "how": "verbatim"}]))
        def stub(prompt, phase, **kwargs):
            return json.dumps({"answer": "History teacher"}) if phase == "query_fallback" else "not found"
        with patch.object(pipeline, "llm", stub):
            answer, source, trace = pipeline.query_with_traversal("Ben's profession?", self.wiki, raw)
        self.assertEqual((answer, source), ("History teacher", "fallback"))
        self.assertEqual(trace["evidence"]["context"], cleaned)
        self.assertEqual(trace["source_sha256"], hashlib.sha256(cleaned.encode()).hexdigest())
        self.assertEqual(trace["traversals"][0]["read_order"], ["index.md", "characters.md", "ben.md"])
        json.dumps(trace)

    def test_changed_source_refuses_stale_offsets(self):
        raw = Path(self.tmp.name) / "source.txt"
        raw.write_text("Changed source")
        self.tree(self.small_tree(chunks=[1]))
        (self.wiki / "_chunking.json").write_text(json.dumps({"source_sha256": "different"}))
        with patch.object(pipeline, "llm", return_value="not found") as stub:
            answer, source, trace = pipeline.query_with_traversal("Ben?", self.wiki, raw)
        self.assertEqual((answer, source), ("not found", "miss"))
        self.assertEqual(trace["miss_reason"], "source_digest_mismatch")
        self.assertEqual(stub.call_count, 1)


class StrictAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import installs driver hooks, so restore pipeline hooks immediately.
        # Client creation is lazy; all chat calls in these tests are replaced.
        import cold_start
        previous = (pipeline.llm, pipeline._ROUTER, pipeline._EMBED_FN, cold_start.llm)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cli
        cls.cli = cli
        pipeline.llm, pipeline._ROUTER, pipeline._EMBED_FN, cold_start.llm = previous

    def test_answer_api_error_is_not_converted_to_empty_miss(self):
        with patch.object(self.cli, "_STRICT_API", True), \
                patch.object(self.cli.ac, "chat", side_effect=RuntimeError("offline failure")):
            with self.assertRaisesRegex(RuntimeError, "offline failure"):
                self.cli._llm("Question", phase="query_traverse")

    def test_empty_model_output_is_rejected_but_explicit_abstention_is_valid(self):
        with patch.object(self.cli, "_STRICT_API", True), \
                patch.object(self.cli.ac, "chat", return_value=("   ", 20, 10)):
            with self.assertRaisesRegex(RuntimeError, "empty content"):
                self.cli._llm("Question", phase="query_traverse")
        with patch.object(self.cli, "_STRICT_API", True), \
                patch.object(self.cli.ac, "chat", return_value=("not found", 20, 10)):
            self.assertEqual(self.cli._llm("Question", phase="query_traverse"), "not found")


if __name__ == "__main__":
    unittest.main()
