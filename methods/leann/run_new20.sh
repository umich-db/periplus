#!/usr/bin/env bash
# LEANN on the 20 NEW movies: GPU-local index build + gpt-5 QA over retrieval.
# Reuses old-10 indexes/answers; concats all 30 at the end.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/leann/bin/python"
PYB="envs/base/bin/python"
NEW20="blade_runner buffy_the_vampire_slayer_film cherry_falls city_of_joy_film clerks collateral_damage_film colombiana ghost_world_film heist_2001_film highlander_endgame miami_vice_film minority_report_film mission_impossible_film raiders_of_the_lost_ark supergirl_film sweet_smell_of_success the_breakfast_club tinker_tailor_soldier_spy_film up_in_the_air_1940_film white_christmas_film"
ALL30="10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men $NEW20"

echo "### BUILD leann indexes (GPU) for new-20 ###"
for s in $NEW20; do
  $PY methods/leann/run.py build --movie "$s" 2>&1 | grep -avE "it/s|Batches|Some weights|newly init"
done

echo "### ANSWER new-20 (-P3) ###"
printf '%s\n' $NEW20 | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/leann/run.py answer --movie "$1" > methods/leann/logs/answer_$1.log 2>&1; echo "[answered] $1"' _ {}

echo "### CONCAT all 30 -> leann.jsonl ###"
: > methods/leann/results/leann.jsonl
for s in $ALL30; do
  [ -f "methods/leann/results/leann_$s.jsonl" ] && cat "methods/leann/results/leann_$s.jsonl" >> methods/leann/results/leann.jsonl
done
echo "COMBINED -> $(wc -l < methods/leann/results/leann.jsonl) rows"
echo "LEANN_NEW20_DONE"
