#!/usr/bin/env python3
"""Rejudge saved baseline answers using the repaired run's frozen judge.

No answers are regenerated. Exactly two judge workers are used, with a separate
append-only cache and complete-set validation before publishing an accuracy.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from run_repaired_10_qwen import ENDPOINT, JUDGE_PROMPT, JUDGE_VERSION, MODEL


ROOT = Path(__file__).resolve().parents[1]
METHOD = "llm_index_gpt54_qwen_36"
EVAL = ROOT / "results/eval/llm_index_repaired10_qwen"
QUESTIONS = ROOT / "methods/llm_index/output/repaired_10/questions_10_movies.jsonl"
ANSWERS = ROOT / "methods" / METHOD / "results" / f"{METHOD}.jsonl"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.update(LLM_API_BASE=ENDPOINT, LLM_MODEL=MODEL, AZURE_DEPLOYMENT=MODEL,
                      LLM_MIN_MAX_TOKENS="4096", LLM_INDEX_STRICT_API="1")
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY missing")
    questions = jsonl(QUESTIONS)
    qids = {q["qid"] for q in questions}
    movies = {q["document"] for q in questions}
    if len(questions) != 295 or len(qids) != 295 or len(movies) != 10:
        raise RuntimeError("Expected 295 unique questions from ten movies")
    selected = [r for r in jsonl(ANSWERS) if r["qid"] in qids]
    by_qid = {r["qid"]: r for r in selected}
    if len(selected) != 295 or set(by_qid) != qids:
        raise RuntimeError("Baseline answers missing or duplicated")
    answers = []
    for question in questions:
        row = by_qid[question["qid"]]
        if (row["question"] != question["question"] or row["gold_answers"] != question["gold_answers"]
                or row["movie"] != question["document"]):
            raise RuntimeError(f"Baseline question/reference mismatch: {question['qid']}")
        if str(row.get("prediction", "")).startswith("[ERROR") or not row.get("input_tokens"):
            raise RuntimeError(f"Baseline contains an API error: {question['qid']}")
        answers.append(row)

    current = json.loads((EVAL / "config.json").read_text())
    expected = {"model": MODEL, "endpoint": ENDPOINT, "judge_model": MODEL,
                "judge_version": JUDGE_VERSION,
                "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
                "question_sha256": digest(QUESTIONS)}
    if any(current.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Baseline judge does not match the frozen repaired run")
    config = {**expected, "baseline_method": METHOD, "baseline_answers_sha256": digest(ANSWERS),
              "workers": 2, "max_tokens": 4096, "temperature": 0, "thinking": False}
    config_path = EVAL / "baseline_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Baseline rejudging configuration changed")
    atomic_json(config_path, config)

    cache_path = EVAL / "baseline_verdicts.jsonl"
    cache = {}
    # Reuse completed identical new-answer verdicts if already available, while
    # keeping all writes separate from the active repaired-answer runner.
    shared_cache = EVAL / "verdicts.jsonl"
    if shared_cache.exists():
        for row in jsonl(shared_cache):
            cache[row["key"]] = row
    baseline_keys = set()
    if cache_path.exists():
        for row in jsonl(cache_path):
            cache[row["key"]] = row
            baseline_keys.add(row["key"])
    for row in cache.values():
        if (row.get("verdict") not in {"correct", "incorrect"}
                or row.get("result") != ("right" if row["verdict"] == "correct" else "wrong")
                or row.get("miss") not in {0, 1} or not isinstance(row.get("judge_tokens"), int)):
            raise RuntimeError("Malformed cached verdict")

    spec = importlib.util.spec_from_file_location("baseline_judge_client", ROOT / "shared/azure_client.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    started = time.time()

    def judge(row):
        signature = json.dumps([JUDGE_VERSION, MODEL, ENDPOINT, JUDGE_PROMPT, row["qid"],
                                row["question"], row["gold_answers"], row["prediction"]])
        key = hashlib.sha256(signature.encode()).hexdigest()
        if key in cache:
            if cache[key]["qid"] != row["qid"]:
                raise RuntimeError("Cached verdict question mismatch")
            return cache[key]
        prediction = str(row["prediction"]).strip()
        miss = row.get("source") == "miss" or prediction.lower().strip(" .") in {"", "not found", "none", "n/a"}
        if miss:
            verdict, reason, tokens = "incorrect", "Abstained", 0
        else:
            prompt = JUDGE_PROMPT.format(question=row["question"], gold=json.dumps(row["gold_answers"]), prediction=prediction)
            verdict = None
            for _ in range(3):
                response, pt, ct = client.chat([{"role": "user", "content": prompt}], max_tokens=4096,
                                              bucket="judge", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                first = response.strip().lower().split(maxsplit=1)[0].strip(".,:;\"'*") if response.strip() else ""
                if first in {"correct", "incorrect"}:
                    verdict, reason, tokens = first, response.strip(), pt + ct
                    break
            if verdict is None:
                raise RuntimeError(f"No valid judge response for {row['qid']}")
        return {"key": key, "qid": row["qid"], "verdict": verdict,
                "result": "right" if verdict == "correct" else "wrong", "miss": int(miss),
                "judge_reason": reason, "judge_tokens": tokens}

    judgments = {}
    pool = ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(judge, row) for row in answers]
    try:
        with cache_path.open("a") as output:
            for future in as_completed(futures):
                verdict = future.result()
                if verdict["qid"] in judgments:
                    raise RuntimeError("Duplicate judged baseline qid")
                judgments[verdict["qid"]] = verdict
                if verdict["key"] not in baseline_keys:
                    output.write(json.dumps(verdict) + "\n")
                    output.flush()
                    baseline_keys.add(verdict["key"])
                if len(judgments) % 25 == 0:
                    print(f"BASELINE JUDGED {len(judgments)}/295", flush=True)
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    if set(judgments) != qids or len(judgments) != 295:
        raise RuntimeError("Baseline verdict set is incomplete; refusing accuracy")
    judged = [{**row, **judgments[row["qid"]]} for row in answers]
    historical_path = ROOT / "results/eval" / f"{METHOD}_verdicts.jsonl"
    historical = [r for r in jsonl(historical_path) if r["qid"] in qids]
    if len(historical) != 295 or {r["qid"] for r in historical} != qids:
        raise RuntimeError("Historical verdict set is incomplete")
    per_movie = defaultdict(lambda: {"questions": 0, "correct": 0, "misses": 0})
    for row in judged:
        metrics = per_movie[row["movie"]]
        metrics["questions"] += 1
        metrics["correct"] += row["result"] == "right"
        metrics["misses"] += row["miss"]
    correct = sum(r["result"] == "right" for r in judged)
    old_correct = sum(r["result"] == "right" for r in historical)
    session = client.get_counters().get("judge", {})
    summary = {
        "method": METHOD, "comparison_role": "original answers rejudged with repaired-run judge",
        "questions": 295, "movies": 10, "correct": correct,
        "accuracy_percent": round(100 * correct / 295, 3), "misses": sum(r["miss"] for r in judged),
        "sources": dict(Counter(r["source"] for r in judged)), "per_movie": dict(per_movie),
        "judge_model": MODEL, "judge_version": JUDGE_VERSION,
        "judge_tokens": sum(r["judge_tokens"] for r in judged),
        "session_judge_calls": session.get("calls", 0),
        "session_judge_input_tokens": session.get("prompt", 0),
        "session_judge_output_tokens": session.get("completion", 0),
        "wall_seconds": round(time.time() - started, 2),
        "historical_context_only": {"correct": old_correct, "questions": 295,
            "accuracy_percent": round(100 * old_correct / 295, 3),
            "misses": sum(r["miss"] for r in historical),
            "note": "Historical judge prompt, gold formatting, parsing and abstention rule differed."},
    }
    output = EVAL / "baseline_rejudged.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in judged))
    temporary.replace(output)
    atomic_json(EVAL / "baseline_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Keep credentials and remote exception bodies out of console output.
        print(f"STOPPED baseline rejudging ({type(exc).__name__}); completed verdicts retained; no new accuracy published.", file=sys.stderr)
        raise SystemExit(1)
