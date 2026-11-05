#!/usr/bin/env python3
"""Run the fixed 295-question repaired index with thinking and durable costs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import uuid

from run_repaired_10_qwen import ENDPOINT, JUDGE_PROMPT, JUDGE_VERSION, MODEL as JUDGE_MODEL

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results/eval/llm_index_repaired10_qwen"
MOVIES = {
    "10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
    "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
    "a_dry_white_season", "a_few_good_men",
}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verdict_key(row):
    signature = json.dumps([JUDGE_VERSION, JUDGE_MODEL, ENDPOINT, JUDGE_PROMPT,
                            row["qid"], row["question"], row["gold_answers"], row["prediction"]])
    return hashlib.sha256(signature.encode()).hexdigest()


def validate_answers(rows, questions, complete=True):
    expected = {q["qid"]: q for q in questions}
    found = {}
    for row in rows:
        qid = row.get("qid")
        if qid in found or qid not in expected:
            raise RuntimeError("Duplicate or unexpected answer question ID")
        question = expected[qid]
        if (row.get("question") != question["question"] or row.get("gold_answers") != question["gold_answers"]
                or row.get("movie") != question["document"]):
            raise RuntimeError(f"Answer question/reference mismatch: {qid}")
        if (str(row.get("prediction", "")).startswith("[ERROR") or row.get("source") == "error"
                or not isinstance(row.get("input_tokens"), int) or row["input_tokens"] <= 0
                or not isinstance(row.get("output_tokens"), int) or row["output_tokens"] <= 0
                or row.get("thinking_requested") is not True):
            raise RuntimeError(f"Invalid API result or thinking setting: {qid}")
        found[qid] = row
    if complete and set(found) != set(expected):
        raise RuntimeError("Answers do not exactly match the fixed question set")
    return found


def usage_summary(records):
    unique = {}
    for row in records:
        key = row["record_id"]
        if key in unique and unique[key] != row:
            raise RuntimeError("Conflicting usage ledger record IDs")
        unique[key] = row
    rows = list(unique.values())
    good = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "error"]
    known_reasoning = [r["reasoning_tokens"] for r in good if r.get("reasoning_tokens") is not None]
    logical_latencies = defaultdict(float)
    phases = defaultdict(lambda: {"successful_calls": 0, "failed_attempts": 0, "input_tokens": 0, "output_tokens": 0})
    for row in rows:
        logical_latencies[row.get("logical_call_id", row["record_id"])] = max(
            logical_latencies[row.get("logical_call_id", row["record_id"])], row.get("latency_s", 0))
        phase = phases[row.get("context", {}).get("phase") or "unknown"]
        if row["status"] == "ok":
            phase["successful_calls"] += 1
            phase["input_tokens"] += row.get("input_tokens") or 0
            phase["output_tokens"] += row.get("output_tokens") or 0
        else:
            phase["failed_attempts"] += 1
    input_tokens = sum(r.get("input_tokens") or 0 for r in good)
    output_tokens = sum(r.get("output_tokens") or 0 for r in good)
    return {
        "successful_calls": len(good), "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reported_reasoning_tokens_subset": sum(known_reasoning) if known_reasoning else None,
        "calls_reporting_reasoning_tokens": len(known_reasoning),
        "reasoning_content_chars": sum(r.get("reasoning_content_chars") or 0 for r in good),
        "reasoning_tokens_are_not_added_to_output_tokens": True,
        "failed_attempts_with_unknown_usage": len(failed),
        "successful_calls_without_usage": sum(not r.get("usage_reported", True) for r in good),
        "truncated_responses": sum(r.get("finish_reason") == "length" for r in good),
        "logical_call_latency_sum_seconds": round(sum(logical_latencies.values()), 3),
        "per_phase": dict(phases),
        "usage_scope": "Provider-reported received responses, including retries and calls from unfinished questions; failed remote-call billing is unknown.",
    }


def comparison(judged, index_manifests):
    path = PREVIOUS / "judged_answers.jsonl"
    before_rows = jsonl(path)
    before = {r["qid"]: r for r in before_rows}
    if len(before_rows) != 295 or len(before) != 295 or set(before) != {r["qid"] for r in judged}:
        return {"available": False, "reason": "Previous thinking-off judged answers are unavailable or incomplete", "source": str(path)}
    newly_correct = regressed = identical = 0
    for row in judged:
        old = before[row["qid"]]
        if old["question"] != row["question"] or old["gold_answers"] != row["gold_answers"]:
            raise RuntimeError("Thinking-off comparison has different questions or references")
        if verdict_key(old) == row["key"]:
            identical += 1
            if old["result"] != row["result"]:
                raise RuntimeError("Identical predictions received inconsistent shared judgments")
        newly_correct += old["result"] != "right" and row["result"] == "right"
        regressed += old["result"] == "right" and row["result"] != "right"
    old_correct = sum(r["result"] == "right" for r in before_rows)
    new_correct = sum(r["result"] == "right" for r in judged)
    previous_config = PREVIOUS / "config.json"
    same_index = (json.loads(previous_config.read_text()).get("index_manifests") == index_manifests
                  if previous_config.exists() else None)
    return {"available": True, "source": str(path), "questions": 295,
            "before_correct": old_correct, "after_correct": new_correct,
            "before_accuracy_percent": round(100 * old_correct / 295, 3),
            "after_accuracy_percent": round(100 * new_correct / 295, 3),
            "change_percentage_points": round(100 * (new_correct - old_correct) / 295, 3),
            "newly_correct": newly_correct, "regressed": regressed, "identical_predictions": identical,
            "same_index_manifests": same_index,
            "scope": ("Same questions, references, repaired index manifests, and Qwen judge."
                      if same_index else "Same questions, references, and Qwen judge; index content may also differ, so this is not an isolated thinking ablation.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--run-name", default="llm_index_repaired10_qwen_thinking")
    ap.add_argument("--workers", type=int, default=6)
    published_index = ROOT / "indexes/llm_index_gpt-5.4_narrativeqa"
    default_index = (published_index if (published_index / "repaired_10_questions.jsonl").exists()
                     else ROOT / "methods/llm_index/output/repaired_10_final")
    ap.add_argument("--index", type=Path, default=default_index)
    ap.add_argument("--thinking-budget", type=int, default=8192)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--build-audit", type=Path, default=ROOT / "results/eval/llm_index_repaired10_qwen_thinking/build_cost_audit.json")
    args = ap.parse_args()
    if args.workers < 1 or args.thinking_budget < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise RuntimeError("Invalid workers, thinking budget, or run name")
    args.index = args.index.resolve()
    args.build_audit = args.build_audit.resolve()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY missing")
    qfile = args.index / "repaired_10_questions.jsonl"
    if not qfile.exists():
        qfile = args.index / "questions_10_movies.jsonl"
    questions = jsonl(qfile)
    qids = {q["qid"] for q in questions}
    slugs = sorted({q["document"] for q in questions})
    if len(questions) != 295 or len(qids) != 295 or set(slugs) != MOVIES:
        raise RuntimeError("Expected exactly the original 295 questions from ten movies")
    original = jsonl(ROOT / "indexes/llm_index_gpt-5.4_narrativeqa/questions_and_ground_truth.jsonl")
    expected = {q["qid"]: q for q in original if q["document"] in MOVIES}
    if set(expected) != qids or any(q["question"] != expected[q["qid"]]["question"] or q["gold_answers"] != expected[q["qid"]]["gold_answers"] for q in questions):
        raise RuntimeError("Fixed questions or ground truth changed")
    sys.path.insert(0, str(ROOT / "methods/llm_index"))
    from verify_tree import validate_doc
    document_paths = sorted({args.index / "balanced" / Path(q["index_dir"]).name for q in questions})
    checks = [validate_doc(path, strict_integrity=True) for path in document_paths]
    if any(c["errors"] for c in checks):
        raise RuntimeError("Index validation failed")
    run_dir = ROOT / "methods" / args.run_name
    result_dir, logs = run_dir / "results", run_dir / "logs"
    eval_dir = ROOT / "results/eval" / args.run_name
    sessions = eval_dir / "sessions"
    for path in (result_dir, logs, sessions):
        path.mkdir(parents=True, exist_ok=True)
    shard_count = math.ceil(args.workers / len(slugs))
    tasks = []
    for slug in slugs:
        movie_questions = [q for q in questions if q["document"] == slug]
        for index in range(shard_count):
            subset = movie_questions[index::shard_count]
            if subset:
                tasks.append({"movie": slug, "shard_id": f"s{index:02d}", "questions": subset,
                              "name": f"{slug}_s{index:02d}"})
    code = ("scripts/run_repaired_thinking.py", "scripts/run_repaired_10_qwen.py", "shared/azure_client.py",
            "src/wiki_index/pipeline.py", "methods/llm_index/cli.py", "methods/llm_index/answer.py",
            "methods/llm_index/balance_core.py", "methods/llm_index/balance_window.py", "methods/llm_index/verify_tree.py")
    config = {"model": args.model, "endpoint": ENDPOINT, "thinking": True, "thinking_budget": args.thinking_budget,
              "thinking_output_budget_initial": args.thinking_budget,
              "thinking_budget_policy": "max(phase requested allowance, initial thinking budget)",
              "thinking_length_retry_attempts": 1, "thinking_output_budget_retry_multiplier": 2,
              "reasoning_effort": None,
              "workers": args.workers, "questions": 295, "movies": slugs, "index": str(args.index),
              "judge_model": JUDGE_MODEL, "judge_version": JUDGE_VERSION, "judge_thinking": False,
              "judge_max_tokens": 4096, "judge_parse_attempts": 3,
              "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
              "question_sha256": sha(qfile), "build_audit": str(args.build_audit),
              "index_manifests": {p.name: sha(p / "_manifest.json") for p in document_paths},
              "code_sha256": {name: sha(ROOT / name) for name in code},
              "shards": {t["name"]: [q["qid"] for q in t["questions"]] for t in tasks}}
    config_path = eval_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Run configuration changed; use a fresh --run-name")
    write_json(config_path, config)
    write_json(eval_dir / "validation.json", checks)
    env = {**os.environ, "BENCH_ROOT": str(ROOT), "BENCH_QUESTION_FILE": str(qfile),
           "LLM_INDEX_ROOT": str(args.index / "balanced"), "LLM_INDEX_RUN_NAME": args.run_name,
           "LLM_API_BASE": ENDPOINT, "LLM_MODEL": args.model, "AZURE_DEPLOYMENT": args.model,
           "LLM_INDEX_STRICT_API": "1", "LLM_INDEX_ENABLE_THINKING": "1",
           "LLM_INDEX_THINKING_MAX_TOKENS": str(args.thinking_budget),
           "LLM_MIN_MAX_TOKENS": str(args.thinking_budget), "LLM_INDEX_BUILD_COST_MODE": "reused_gpt54_unknown"}
    for name in ("CUAD", "BENCH_QUESTION_ID", "BENCH_MAX_QUESTIONS", "BENCH_SHARD_ID", "BENCH_QID_ALLOWLIST"):
        env.pop(name, None)
    session_id = uuid.uuid4().hex
    session_path = sessions / f"{session_id}.json"
    task_dir = sessions / session_id
    task_dir.mkdir()
    session = {"session_id": session_id, "status": "running", "started_at": stamp(), "stage": "answering",
               "answer_started_at": stamp(), "answer_wall_seconds": None, "judge_wall_seconds": None}
    write_json(session_path, session)
    started = answer_started = time.perf_counter()
    answers, judged = [], []

    def answer_task(task):
        output = result_dir / f"{args.run_name}_{task['name']}.jsonl"
        before = validate_answers(jsonl(output), task["questions"], complete=False)
        task_path = task_dir / f"{task['name']}.json"
        state = {"task": task["name"], "started_at": stamp(), "before_rows": len(before), "status": "running"}
        write_json(task_path, state)
        beginning = time.perf_counter()
        task_env = {**env, "BENCH_QID_ALLOWLIST": json.dumps([q["qid"] for q in task["questions"]]),
                    "BENCH_SHARD_ID": task["shard_id"], "LLM_USAGE_LOG": str(logs / f"usage_{task['name']}.jsonl")}
        try:
            if len(before) != len(task["questions"]):
                with (logs / f"{task['name']}.log").open("a") as log:
                    done = subprocess.run([args.python, str(ROOT / "methods/llm_index/answer.py"), "--movie", task["movie"]],
                                          cwd=ROOT, env=task_env, stdout=log, stderr=subprocess.STDOUT)
                if done.returncode:
                    raise RuntimeError(f"Answer process failed for {task['name']}; inspect its log")
            rows = jsonl(output)
            validate_answers(rows, task["questions"])
            state["status"] = "complete"
            print(f"ANSWERED {task['name']}: {len(rows)} ({len(rows) - len(before)} new)", flush=True)
            return rows
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            state.update(ended_at=stamp(), wall_seconds=round(time.perf_counter() - beginning, 3))
            current = jsonl(output)
            state.update(after_rows=len(current), new_rows=len(current) - len(before))
            write_json(task_path, state)

    try:
        by_qid = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(answer_task, task) for task in tasks]):
                for row in future.result():
                    if row["qid"] in by_qid:
                        raise RuntimeError("Duplicate question across shards")
                    by_qid[row["qid"]] = row
        answers = [by_qid[q["qid"]] for q in questions]
        validate_answers(answers, questions)
        write_jsonl(result_dir / f"{args.run_name}.jsonl", answers)
        session.update(answer_ended_at=stamp(), answer_wall_seconds=round(time.perf_counter() - answer_started, 3),
                       stage="judging", judge_started_at=stamp())
        write_json(session_path, session)
        judge_started = time.perf_counter()
        # Parent judge has a separate client, budget and ledger from child answerers.
        os.environ.update(LLM_API_BASE=ENDPOINT, LLM_MODEL=JUDGE_MODEL, AZURE_DEPLOYMENT=JUDGE_MODEL,
                          LLM_MIN_MAX_TOKENS="4096", LLM_USAGE_LOG=str(logs / "judge_usage.jsonl"))
        spec = importlib.util.spec_from_file_location("thinking_run_judge_client", ROOT / "shared/azure_client.py")
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        cache_path = eval_dir / "verdicts.jsonl"
        cache, own_keys = {}, set()
        for path in (PREVIOUS / "verdicts.jsonl", cache_path):
            for row in jsonl(path):
                if row.get("verdict") not in {"correct", "incorrect"} or row.get("result") != ("right" if row["verdict"] == "correct" else "wrong"):
                    raise RuntimeError("Malformed cached verdict")
                if row["key"] in cache and cache[row["key"]]["result"] != row["result"]:
                    raise RuntimeError("Conflicting shared judge cache")
                cache[row["key"]] = row
                if path == cache_path:
                    own_keys.add(row["key"])

        def judge(row):
            key = verdict_key(row)
            if key in cache:
                if cache[key]["qid"] != row["qid"]:
                    raise RuntimeError("Cached verdict question mismatch")
                return {**cache[key], "cache_hit": True, "cache_source": "this_run" if key in own_keys else "previous_thinking_off"}
            prediction = str(row["prediction"]).strip()
            miss = row.get("source") == "miss" or prediction.lower().strip(" .") in {"", "not found", "none", "n/a"}
            pt_total = ct_total = 0
            if miss:
                verdict, reason = "incorrect", "Abstained"
            else:
                prompt = JUDGE_PROMPT.format(question=row["question"], gold=json.dumps(row["gold_answers"]), prediction=prediction)
                verdict = None
                for attempt in range(3):
                    response, pt, ct = client.chat([{"role": "user", "content": prompt}], max_tokens=4096,
                        bucket="judge", extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        usage_context={"phase": "judge", "qid": row["qid"], "movie": row["movie"],
                                       "run": args.run_name, "parse_attempt": attempt + 1})
                    pt_total += pt; ct_total += ct
                    first = response.strip().lower().split(maxsplit=1)[0].strip(".,:;\"'*") if response.strip() else ""
                    # Filter by qid because this shared client is called concurrently.
                    calls = [r for r in client.get_call_records() if r["status"] == "ok" and r.get("context", {}).get("qid") == row["qid"]]
                    if first in {"correct", "incorrect"} and calls and calls[-1].get("finish_reason") != "length":
                        verdict, reason = first, response.strip()
                        break
                if verdict is None:
                    raise RuntimeError(f"No valid judge response for {row['qid']}")
            return {"key": key, "qid": row["qid"], "verdict": verdict,
                    "result": "right" if verdict == "correct" else "wrong", "miss": int(miss),
                    "judge_reason": reason, "judge_tokens": pt_total + ct_total,
                    "judge_input_tokens": pt_total, "judge_output_tokens": ct_total, "cache_hit": False}

        judgments = {}
        with cache_path.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(judge, row) for row in answers]):
                row = future.result()
                judgments[row["qid"]] = row
                if row["key"] not in own_keys:
                    output.write(json.dumps(row) + "\n"); output.flush()
                    own_keys.add(row["key"])
                if len(judgments) % 25 == 0:
                    print(f"JUDGED {len(judgments)}/295", flush=True)
        if set(judgments) != qids:
            raise RuntimeError("Incomplete judged question set")
        judged = [{**row, **judgments[row["qid"]]} for row in answers]
        write_jsonl(eval_dir / "judged_answers.jsonl", judged)
        session.update(judge_ended_at=stamp(), judge_wall_seconds=round(time.perf_counter() - judge_started, 3), status="complete", stage="complete")
    except BaseException as error:
        session.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        if session["answer_wall_seconds"] is None:
            session.update(answer_ended_at=stamp(), answer_wall_seconds=round(time.perf_counter() - answer_started, 3))
        if session["stage"] == "judging" and session["judge_wall_seconds"] is None:
            session.update(judge_ended_at=stamp(), judge_wall_seconds=round(time.perf_counter() - judge_started, 3))
        session.update(ended_at=stamp(), wall_seconds=round(time.perf_counter() - started, 3))
        write_json(session_path, session)

    prior_sessions = [json.loads(p.read_text()) for p in sessions.glob("*.json")]
    answer_usage = usage_summary([r for p in logs.glob("usage_*.jsonl") for r in jsonl(p)])
    judge_usage = usage_summary(jsonl(logs / "judge_usage.jsonl"))
    latencies = sorted(r["latency_s"] for r in answers)
    answer_usage.update(completed_question_input_tokens=sum(r["input_tokens"] for r in answers),
                        completed_question_output_tokens=sum(r["output_tokens"] for r in answers),
                        question_latency_sum_seconds=round(sum(latencies), 3),
                        question_latency_mean_seconds=round(statistics.mean(latencies), 3),
                        question_latency_median_seconds=round(statistics.median(latencies), 3),
                        question_latency_p95_seconds=latencies[math.ceil(.95 * len(latencies)) - 1],
                        active_answer_wall_seconds=round(sum(s.get("answer_wall_seconds") or 0 for s in prior_sessions), 3),
                        workers=args.workers, shard_tasks=len(tasks))
    judge_usage.update(active_judge_wall_seconds=round(sum(s.get("judge_wall_seconds") or 0 for s in prior_sessions), 3),
                       cached_verdicts_this_invocation=sum(bool(r.get("cache_hit")) for r in judged),
                       model=JUDGE_MODEL, thinking=False)
    costs = {"method": args.run_name, "model": args.model, "thinking": True, "questions": 295,
             "answering": answer_usage, "judging": judge_usage,
             "building": {"new_llm_api_input_tokens": 0, "new_llm_api_output_tokens": 0,
                          "historical_gpt54_actual_input_tokens": None, "historical_gpt54_actual_output_tokens": None,
                          "historical_gpt54_actual_wall_seconds": None,
                          "scope": "Saved GPT-5.4 extraction reused; repair and balancing are local. Historical build usage is not inferred from legacy amortized costs.",
                          "audit_path": str(args.build_audit), "audit_available": args.build_audit.exists()},
             "sessions": len(prior_sessions), "unclosed_sessions": sum(s.get("ended_at") is None for s in prior_sessions),
             "wall_scope": "Sum of persisted active invocation stages, excluding offline gaps; unclosed sessions make wall totals a lower bound."}
    if args.build_audit.exists():
        audit = json.loads(args.build_audit.read_text())
        replay = audit.get("offline_replay", {})
        timing = replay.get("timing", {})
        costs["building"]["offline_replay"] = {
            "status": replay.get("status"), "scope": replay.get("scope"),
            **{key: timing.get(key) for key in ("whole_wall_seconds", "render_wall_seconds_sum",
               "balance_wall_seconds_sum", "validation_wall_seconds", "api_input_tokens", "api_output_tokens")}}
        costs["building"]["original_build_artifact_estimate"] = audit.get("original_build", {}).get("artifact_based_lower_bound_estimate")
    write_json(eval_dir / "cost_latency.json", costs)
    per_movie = defaultdict(lambda: {"questions": 0, "correct": 0, "misses": 0})
    for row in judged:
        item = per_movie[row["movie"]]
        item["questions"] += 1; item["correct"] += row["result"] == "right"; item["misses"] += row["miss"]
    correct = sum(row["result"] == "right" for row in judged)
    summary = {"method": args.run_name, "model": args.model, "thinking": True, "questions": 295, "movies": 10,
               "correct": correct, "accuracy_percent": round(100 * correct / 295, 3),
               "misses": sum(row["miss"] for row in judged), "sources": dict(Counter(row["source"] for row in judged)),
               "input_tokens": answer_usage["input_tokens"], "output_tokens": answer_usage["output_tokens"],
               "answer_wall_seconds": answer_usage["active_answer_wall_seconds"], "per_movie": dict(per_movie),
               "cost_latency_path": str(eval_dir / "cost_latency.json")}
    write_json(eval_dir / "summary.json", summary)
    write_json(eval_dir / "comparison_thinking_off.json", comparison(judged, config["index_manifests"]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
