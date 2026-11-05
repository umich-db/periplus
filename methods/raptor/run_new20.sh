#!/bin/bash
# raptor (native, ABSTAIN policy = permissive=False) on the 20 NEW movies.
# Builds RAPTOR trees IN PARALLEL across movies (each build is internally serial on
# the slow Azure endpoint, so per-movie parallelism is the only speedup lever), then
# answers questions_30 in parallel across movies. Reuses old-10 trees + answer files;
# concats all 30 at the end. build_stats.json is rebuilt from per-slug logs (no race).
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY=envs/base/bin/python
NEW20="blade_runner buffy_the_vampire_slayer_film cherry_falls city_of_joy_film clerks collateral_damage_film colombiana ghost_world_film heist_2001_film highlander_endgame miami_vice_film minority_report_film mission_impossible_film raiders_of_the_lost_ark supergirl_film sweet_smell_of_success the_breakfast_club tinker_tailor_soldier_spy_film up_in_the_air_1940_film white_christmas_film"
ALL30="10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men $NEW20"
mkdir -p methods/raptor/logs/new20

echo "### BUILD trees for new-20, 5 parallel (existing trees auto-skip) ###"
printf '%s\n' $NEW20 | xargs -P 5 -I {} bash -c '
  '"$PY"' methods/raptor/raptor_bench.py build "$1" > methods/raptor/logs/new20/build_$1.log 2>&1
  echo "[built] $1 :: $(grep -aoE "\[build\] [a-z0-9_]+: [0-9,]+ tok, [0-9.]+ min" methods/raptor/logs/new20/build_$1.log | tail -1)"' _ {}

echo "### rebuild build_stats.json from per-slug logs (race-free) ###"
$PY - <<'PYK'
import re, json, pathlib
HERE = pathlib.Path("methods/raptor")
sp = HERE/"output"/"build_stats.json"
stats = json.loads(sp.read_text()) if sp.exists() else {}
for lg in (HERE/"logs"/"new20").glob("build_*.log"):
    txt = lg.read_text(errors="ignore")
    m = re.search(r"\[build\] ([a-z0-9_]+): ([\d,]+) tok, ([\d.]+) min", txt)
    if m:
        stats[m.group(1)] = {"build_tokens": int(m.group(2).replace(",", "")), "build_min": float(m.group(3))}
sp.write_text(json.dumps(stats, indent=2))
print(f"build_stats now covers {len(stats)} movies")
PYK

echo "### ANSWER (native, abstain) new-20, 4 parallel ###"
printf '%s\n' $NEW20 | xargs -P 4 -I {} bash -c '
  '"$PY"' methods/raptor/raptor_bench.py answer --movie "$1" > methods/raptor/logs/new20/answer_$1.log 2>&1
  echo "[answered] $1 :: $(grep -aoE "DONE [a-z0-9_]+: [0-9]+" methods/raptor/logs/new20/answer_$1.log | tail -1)"' _ {}

echo "### CONCAT all 30 -> raptor.jsonl ###"
: > methods/raptor/results/raptor.jsonl
for s in $ALL30; do
  [ -f "methods/raptor/results/raptor_$s.jsonl" ] && cat "methods/raptor/results/raptor_$s.jsonl" >> methods/raptor/results/raptor.jsonl
done
echo "COMBINED -> $(wc -l < methods/raptor/results/raptor.jsonl) rows"
echo "RAPTOR_NEW20_DONE"
