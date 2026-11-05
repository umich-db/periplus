"""Compare the repaired answers with old answers scored by the same judge."""
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/eval/llm_index_repaired10_qwen"


def rows(name):
    data = [json.loads(line) for line in (DIRECTORY / name).read_text().splitlines() if line.strip()]
    assert len(data) == 295 and len({row["qid"] for row in data}) == 295
    return {row["qid"]: row for row in data}


def main():
    old = rows("baseline_rejudged.jsonl")
    new = rows("judged_answers.jsonl")
    assert old.keys() == new.keys()
    changes = []
    per_movie = defaultdict(lambda: {"questions": 0, "before_correct": 0, "after_correct": 0})
    for qid, after in new.items():
        before = old[qid]
        before_right = before["result"] == "right"
        after_right = after["result"] == "right"
        assert before["question"] == after["question"]
        assert before["gold_answers"] == after["gold_answers"]
        # Matching prediction cache keys must carry a consistent verdict.
        if before["key"] == after["key"]:
            assert before["result"] == after["result"]
        movie = after["movie"]
        entry = per_movie[movie]
        entry["questions"] += 1
        entry["before_correct"] += before_right
        entry["after_correct"] += after_right
        changes.append({"qid": qid, "movie": movie, "question": after["question"],
                        "gold_answers": after["gold_answers"], "before": before["prediction"],
                        "after": after["prediction"], "before_correct": before_right,
                        "after_correct": after_right, "after_source": after["source"]})
    before_correct = sum(r["result"] == "right" for r in old.values())
    after_correct = sum(r["result"] == "right" for r in new.values())
    report = {
        "questions": 295, "movies": 10,
        "before_correct": before_correct, "after_correct": after_correct,
        "before_accuracy_percent": round(100 * before_correct / 295, 3),
        "after_accuracy_percent": round(100 * after_correct / 295, 3),
        "improvement_percentage_points": round(100 * (after_correct - before_correct) / 295, 3),
        "before_misses": sum(r["miss"] for r in old.values()),
        "after_misses": sum(r["miss"] for r in new.values()),
        "newly_correct": sum(not r["before_correct"] and r["after_correct"] for r in changes),
        "regressed": sum(r["before_correct"] and not r["after_correct"] for r in changes),
        "per_movie": dict(per_movie),
        "comparison": "Old and repaired predictions use the same Qwen judge prompt/settings and share identical-prediction judgments.",
        "scope": "Development evaluation on the original ten movies; all four repairs applied together.",
    }
    (DIRECTORY / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (DIRECTORY / "answer_changes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in changes))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
