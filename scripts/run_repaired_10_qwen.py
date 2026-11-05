#!/usr/bin/env python3
"""Answer and score exactly the repaired original ten movies with public Qwen.

Credentials come only from the ignored repository .env or environment. The
run is resumable; API/validation failures cannot become an accuracy number.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.6-35b-a3b"
ENDPOINT = "https://api.nextgarch.org/v1"
JUDGE_VERSION = "equivalence-v1-repaired10"
JUDGE_PROMPT = """/no_think
You are an answer-equivalence judge. Given a question, the ground-truth answers,
and a predicted answer, decide if the prediction is essentially correct: it
must convey the same core fact, allowing paraphrases, aliases and equivalent
relationship directions. Matching either reference is sufficient. Do not
penalize a correct answer for harmless additional specificity.

Question: {question}
Ground truth answers: {gold}
Predicted answer: {prediction}

Reply with a single line starting exactly 'correct' or 'incorrect', followed
by a short reason. Do not answer the question yourself.
"""


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run-name", default="llm_index_repaired10_qwen")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--index", type=Path, default=ROOT / "methods/llm_index/output/repaired_10")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.update(LLM_API_BASE=ENDPOINT, LLM_MODEL=MODEL, AZURE_DEPLOYMENT=MODEL,
                      LLM_MIN_MAX_TOKENS="4096", LLM_INDEX_STRICT_API="1")
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY missing")
    qfile = args.index / "questions_10_movies.jsonl"
    questions = jsonl(qfile)
    qids = {q["qid"] for q in questions}
    slugs = sorted({q["document"] for q in questions})
    if len(questions) != 295 or len(qids) != 295 or len(slugs) != 10:
        raise RuntimeError("Expected exactly 295 unique questions from ten movies")
    sys.path.insert(0, str(ROOT / "methods/llm_index"))
    from verify_tree import validate_doc
    document_paths = sorted({str(args.index / "balanced" / Path(q["index_dir"]).name) for q in questions})
    checks = [validate_doc(path, strict_integrity=True) for path in document_paths]
    errors = [(c["document"], c["errors"]) for c in checks if c["errors"]]
    if errors:
        raise RuntimeError(f"Index validation failed: {errors[:2]}")
    run_dir = ROOT / "methods" / args.run_name
    result_dir, logs = run_dir / "results", run_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    eval_dir = ROOT / "results/eval" / args.run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "BENCH_ROOT": str(ROOT), "BENCH_QUESTION_FILE": str(qfile),
           "LLM_INDEX_ROOT": str(args.index / "balanced"), "LLM_INDEX_RUN_NAME": args.run_name,
           "LLM_INDEX_ENABLE_THINKING": "1" if args.thinking else "0"}
    config = {
        "model": MODEL, "endpoint": ENDPOINT, "thinking": args.thinking,
        "questions": 295, "movies": slugs, "judge_model": MODEL, "judge_version": JUDGE_VERSION,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "question_sha256": hashlib.sha256(qfile.read_bytes()).hexdigest(),
        "index_manifests": {Path(p).name: hashlib.sha256((Path(p) / "_manifest.json").read_bytes()).hexdigest() for p in document_paths},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "src/wiki_index/pipeline.py", "methods/llm_index/cli.py", "methods/llm_index/answer.py",
            "methods/llm_index/balance_core.py", "methods/llm_index/balance_window.py",
            "methods/llm_index/verify_tree.py", "scripts/run_repaired_10_qwen.py")},
    }
    config_path = eval_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Run configuration changed; choose a fresh --run-name")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (eval_dir / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    started = time.time()

    def answer_movie(slug):
        with (logs / f"{slug}.log").open("a") as log:
            result = subprocess.run([args.python, str(ROOT / "methods/llm_index/answer.py"), "--movie", slug],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Answer process failed for {slug}; see its log")
        rows = jsonl(result_dir / f"{args.run_name}_{slug}.jsonl")
        bad = [r for r in rows if str(r.get("prediction", "")).startswith("[ERROR") or not r.get("input_tokens")]
        if bad:
            raise RuntimeError(f"API errors in {slug}; refusing to score {len(bad)} records")
        print(f"ANSWERED {slug}: {len(rows)}", flush=True)
        return rows

    by_qid = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(answer_movie, slug) for slug in slugs]):
            for row in future.result():
                if row["qid"] in by_qid:
                    raise RuntimeError("Duplicate answer qid")
                by_qid[row["qid"]] = row
    if set(by_qid) != qids:
        raise RuntimeError("Answers do not match the fixed question set")
    answers = [by_qid[q["qid"]] for q in questions]
    (result_dir / f"{args.run_name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in answers))
    answer_wall = time.time() - started
    spec = importlib.util.spec_from_file_location("run_client", ROOT / "shared/azure_client.py")
    client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
    cache_path = eval_dir / "verdicts.jsonl"
    cache = {r["key"]: r for r in jsonl(cache_path)} if cache_path.exists() else {}

    def judge(row):
        signature = json.dumps([JUDGE_VERSION, MODEL, ENDPOINT, JUDGE_PROMPT, row["qid"], row["question"], row["gold_answers"], row["prediction"]])
        key = hashlib.sha256(signature.encode()).hexdigest()
        if key in cache:
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
                raise RuntimeError(f"No valid judge response for {row['qid']}; refusing to score")
        return {"key": key, "qid": row["qid"], "verdict": verdict,
                "result": "right" if verdict == "correct" else "wrong", "miss": int(miss),
                "judge_reason": reason, "judge_tokens": tokens}

    judgments = {}
    with cache_path.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(judge, row) for row in answers]):
            verdict = future.result()
            judgments[verdict["qid"]] = verdict
            if verdict["key"] not in cache:
                output.write(json.dumps(verdict) + "\n"); output.flush()
                cache[verdict["key"]] = verdict
            if len(judgments) % 25 == 0:
                print(f"JUDGED {len(judgments)}/295", flush=True)
    judged = [{**row, **judgments[row["qid"]]} for row in answers]
    correct = sum(r["result"] == "right" for r in judged)
    per_movie = defaultdict(lambda: {"questions": 0, "correct": 0, "misses": 0})
    for row in judged:
        m = per_movie[row["movie"]]
        m["questions"] += 1; m["correct"] += row["result"] == "right"; m["misses"] += row["miss"]
    summary = {"method": args.run_name, "questions": len(judged), "movies": 10,
               "correct": correct, "accuracy_percent": round(100 * correct / len(judged), 3),
               "misses": sum(r["miss"] for r in judged), "sources": dict(Counter(r["source"] for r in judged)),
               "answer_wall_seconds": round(answer_wall, 2), "total_wall_seconds": round(time.time() - started, 2),
               "input_tokens": sum(r["input_tokens"] for r in judged),
               "output_tokens": sum(r["output_tokens"] for r in judged), "per_movie": dict(per_movie)}
    (eval_dir / "judged_answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
