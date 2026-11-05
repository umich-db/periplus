"""Offline checks for observed usage, thinking metadata, retries, and resumes."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("usage_client_under_test", ROOT / "shared/azure_client.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


def response(finish_reason="stop", reasoning_tokens=70):
    raw = {"prompt_tokens": 40, "completion_tokens": 100, "total_tokens": 140,
           "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
    usage = SimpleNamespace(**raw, model_dump=lambda: dict(raw))
    message = SimpleNamespace(content="answer", reasoning_content="private reasoning")
    return SimpleNamespace(usage=usage,
                           choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        CLIENT.reset_counters()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Path(self.temp.name) / "usage.jsonl"
        self.env = patch.dict(os.environ, {"LLM_USAGE_LOG": str(self.ledger)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def call_with(self, responses, **kwargs):
        pending = iter(responses)
        self.forwarded = []

        def create(**params):
            self.forwarded.append(params)
            value = next(pending)
            if isinstance(value, Exception):
                raise value
            return value

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(CLIENT, "_get_client", return_value=fake), patch.object(CLIENT.time, "sleep"):
            return CLIENT.chat([{"role": "user", "content": "private prompt"}],
                               bucket="wiki_answer", usage_context={"qid": "movie_001", "phase": "route"},
                               extra_body={"chat_template_kwargs": {"enable_thinking": True}}, **kwargs)

    def test_usage_includes_reasoning_once_and_keeps_length_signal(self):
        self.assertEqual(self.call_with([response("length")]), ("answer", 40, 100))
        records = CLIENT.get_call_records()
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["total_tokens"], 140)
        self.assertEqual(row["reasoning_tokens"], 70)
        self.assertEqual(row["finish_reason"], "length")
        self.assertTrue(row["thinking_requested"])
        self.assertEqual(row["reasoning_content_chars"], len("private reasoning"))
        self.assertEqual(CLIENT.get_counters()["wiki_answer"]["completion"], 100)
        self.assertNotIn("usage_context", self.forwarded[0])
        text = self.ledger.read_text()
        self.assertNotIn("private reasoning", text)
        self.assertNotIn("private prompt", text)
        self.assertNotIn('"answer"', text)
        records[0]["context"]["qid"] = "changed"
        self.assertEqual(CLIENT.get_call_records()[0]["context"]["qid"], "movie_001")

    def test_retry_unknown_usage_and_durable_ledger_survive_reset(self):
        error = RuntimeError("503 do-not-record-this")
        self.call_with([error, response()])
        records = CLIENT.get_call_records()
        self.assertEqual([r["status"] for r in records], ["error", "ok"])
        self.assertIsNone(records[0]["input_tokens"])
        self.assertTrue(records[0]["usage_unknown"])
        self.assertEqual(records[1]["attempts"], 2)
        self.assertEqual(records[0]["logical_call_id"], records[1]["logical_call_id"])
        self.assertTrue(records[1]["prior_attempt_usage_unknown"])
        self.assertEqual(records[1]["failed_attempts"][0]["error_type"], "RuntimeError")
        self.assertNotIn("do-not-record-this", self.ledger.read_text())
        CLIENT.reset_counters()
        self.assertEqual(CLIENT.get_call_records(), [])
        self.assertEqual(CLIENT.get_counters(), {})
        self.call_with([response()])
        persisted = [json.loads(line) for line in self.ledger.read_text().splitlines()]
        self.assertEqual(len(persisted), 3)
        self.assertEqual(len({r["record_id"] for r in persisted}), 3)
        self.assertEqual(sum(r["total_tokens"] for r in persisted if r["status"] == "ok"), 280)

    def test_terminal_failure_keeps_prior_observed_usage(self):
        self.call_with([response(reasoning_tokens=None)])
        with self.assertRaises(RuntimeError):
            self.call_with([RuntimeError("503 unavailable")], max_retries=1)
        records = CLIENT.get_call_records()
        self.assertEqual([r["status"] for r in records], ["ok", "error"])
        self.assertIsNone(records[0]["reasoning_tokens"])
        self.assertEqual(sum(r["output_tokens"] for r in records if r["status"] == "ok"), 100)


if __name__ == "__main__":
    unittest.main()
