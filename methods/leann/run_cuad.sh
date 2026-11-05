#!/usr/bin/env bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
export CUAD=1
export PY="envs/leann/bin/python"
SLUGS="array_biopharma_inc_license_development_and_commercializatio azulsa_20170303_f_1a_ex_10_3_9943903_ex_10_3_maintenance_agr ceres_inc_01_25_2012_ex_10_20_collaboration_agreement gooseheadinsurance_inc_04_02_2018_ex_10_6_franchise_agreemen harpoontherapeuticsinc_20200312_10_k_ex_10_18_12051356_ex_10 manufacturersservicesltd_06_05_2000_ex_10_14_outsourcing_agr mrsfieldsoriginalcookiesinc_01_29_1998_ex_10_franchise_agree phasebiopharmaceuticalsinc_20200330_10_k_ex_10_21_12086810_e revolutionmedicinesinc_20200117_s_1_ex_10_1_11948417_ex_10_1 verizonabsllc_20200123_8_k_ex_10_4_11952335_ex_10_4_service_"
mkdir -p methods/leann/logs/cuad
echo "### BUILD leann indexes (GPU) for 10 contracts ###"
for s in $SLUGS; do $PY methods/leann/run.py build --movie "$s" 2>&1 | grep -avE "it/s|Batches|Some weights|newly init"; done
echo "### ANSWER -P3 ###"
printf '%s\n' $SLUGS | xargs -P3 -I{} bash -c '$PY methods/leann/run.py answer --movie "$1" > methods/leann/logs/cuad/answer_$1.log 2>&1; echo "[answered] $1"' _ {}
: > methods/leann/results/leann_cuad.jsonl
for s in $SLUGS; do [ -f methods/leann/results/leann_cuad_$s.jsonl ] && cat methods/leann/results/leann_cuad_$s.jsonl >> methods/leann/results/leann_cuad.jsonl; done
echo "LEANN_CUAD_DONE rows=$(wc -l < methods/leann/results/leann_cuad.jsonl)"
