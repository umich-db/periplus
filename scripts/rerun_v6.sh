#!/usr/bin/env bash
# Rebuild wiki_v6 LESS SPARSE (n_samples=10 template) -> re-answer -> judge.
# Runs alongside graphrag (shares endpoint). Gentle parallelism (-P2).
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/base/bin/python"
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)

echo "### clear old v6 ###"
rm -rf methods/wiki_v6/output/wikis methods/wiki_v6/output/_master_template.json methods/wiki_v6/build_costs.json
rm -f methods/wiki_v6/results/wiki_v6*.jsonl results/eval/wiki_v6_verdicts.jsonl results/eval/wiki_v6_summary.json

echo "### Phase 1: bootstrap master template (n_samples=10) ###"
$PY methods/wiki_v6/run.py bootstrap > methods/wiki_v6/logs/bootstrap.log 2>&1
grep -E 'V6_BOOTSTRAP_DONE|FAILED' methods/wiki_v6/logs/bootstrap.log | tail -1

echo "### Phase 2: compile 10 movies (2-pass), -P2 ###"
printf '%s\n' "${SLUGS[@]}" | xargs -P 2 -I {} bash -c '
  $0 methods/wiki_v6/run.py compile --movie "$1" > methods/wiki_v6/logs/compile_$1.log 2>&1; echo "[compiled] $1"' "$PY" {}

echo "### Phase 3: answer 10 movies, -P2 ###"
printf '%s\n' "${SLUGS[@]}" | xargs -P 2 -I {} bash -c '
  $0 methods/wiki_v6/run.py answer --movie "$1" > methods/wiki_v6/logs/answer_$1.log 2>&1' "$PY" {}
ls methods/wiki_v6/results/wiki_v6_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/wiki_v6/results/wiki_v6.jsonl

echo "### Phase 4: judge ###"
$PY shared/eval.py wiki_v6 > results/eval/eval_v6_new.log 2>&1
$PY - <<'PY'
import json,collections
d=json.load(open('results/eval/wiki_v6_summary.json'))
r=[json.loads(l) for l in open('methods/wiki_v6/results/wiki_v6.jsonl')]
src=collections.Counter(x.get('source') for x in r)
print(f"wiki_v6 NEW: acc={d['judge_accuracy']:.1%} fab={d['fabrication_pct']}% | miss={src.get('miss',0)}/{len(r)} ({100*src.get('miss',0)/len(r):.0f}%) wiki={src.get('wiki',0)} fb={src.get('fallback',0)}")
print("  (was: acc=43.0% miss=37%; old paper v6: acc=53.9% miss=23.7%)")
PY
echo "V6_RERUN_DONE"
