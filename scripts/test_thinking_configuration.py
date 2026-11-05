"""Offline checks that thinking can finish before a response is scored."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AZURE_API_KEY", "offline-test")
os.environ.setdefault("AZURE_API_BASE", "https://example.invalid")
os.environ.setdefault("AZURE_API_VERSION", "2024-02-01")
spec = importlib.util.spec_from_file_location("thinking_cli", ROOT / "methods/llm_index/cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ThinkingConfigurationTests(unittest.TestCase):
    def exercise(self, finishes):
        records, requests = [], []

        def chat(messages, **kwargs):
            requests.append(kwargs)
            finish = finishes[len(requests) - 1]
            records.append({"status": "ok", "finish_reason": finish})
            return "1", 100, 200

        with patch.multiple(cli, _FORCE_THINK=True, _NO_THINK=False,
                            _STRICT_API=True, _THINKING_MAX_TOKENS=8192,
                            _MODEL_EXTRA={"chat_template_kwargs": {"enable_thinking": True}}), \
             patch.object(cli.ac, "chat", side_effect=chat), \
             patch.object(cli.ac, "get_call_records", side_effect=lambda: list(records)):
            result = cli._llm("Choose a child", max_tokens=128, phase="route")
        return result, requests

    def test_small_route_request_has_thinking_headroom(self):
        result, requests = self.exercise(["stop"])
        self.assertEqual(result, "1")
        self.assertEqual(requests[0]["max_tokens"], 8192)
        self.assertTrue(requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertIsNone(requests[0]["reasoning_effort"])

    def test_truncated_response_is_retried_with_more_room(self):
        result, requests = self.exercise(["length", "stop"])
        self.assertEqual(result, "1")
        self.assertEqual([r["max_tokens"] for r in requests], [8192, 16384])

    def test_second_truncation_is_not_accepted_as_an_answer(self):
        with self.assertRaisesRegex(RuntimeError, "output allowance"):
            self.exercise(["length", "length"])


if __name__ == "__main__":
    unittest.main()
