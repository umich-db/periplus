#!/bin/bash
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
export CUAD=1
export PY=envs/base/bin/python
SLUGS="array_biopharma_inc_license_development_and_commercializatio azulsa_20170303_f_1a_ex_10_3_9943903_ex_10_3_maintenance_agr ceres_inc_01_25_2012_ex_10_20_collaboration_agreement gooseheadinsurance_inc_04_02_2018_ex_10_6_franchise_agreemen harpoontherapeuticsinc_20200312_10_k_ex_10_18_12051356_ex_10 manufacturersservicesltd_06_05_2000_ex_10_14_outsourcing_agr mrsfieldsoriginalcookiesinc_01_29_1998_ex_10_franchise_agree phasebiopharmaceuticalsinc_20200330_10_k_ex_10_21_12086810_e revolutionmedicinesinc_20200117_s_1_ex_10_1_11948417_ex_10_1 verizonabsllc_20200123_8_k_ex_10_4_11952335_ex_10_4_service_"
FILT='HTTP Request|Retrying|Batches|it/s|Some weights|newly init|You should|warnings|Using collapsed|TreeRetriever|Tokenizer:|Threshold:|Top K:|Selection|Context Embed|Embedding Model|Num Layers|Start Layer|No device'
mkdir -p methods/raptor/logs/cuad
echo "### BUILD trees for 10 contracts, -P5 ###"
printf '%s\n' $SLUGS | xargs -P5 -I{} bash -c '$PY methods/raptor/raptor_bench.py build "$1" > methods/raptor/logs/cuad/build_$1.log 2>&1; echo "[built] $1 :: $(grep -aoE "\[build\] [a-z0-9_]+: [0-9,]+ tok" methods/raptor/logs/cuad/build_$1.log|tail -1)"' _ {}
echo "### ANSWER (native abstain), -P4 ###"
printf '%s\n' $SLUGS | xargs -P4 -I{} bash -c '$PY methods/raptor/raptor_bench.py answer --movie "$1" > methods/raptor/logs/cuad/answer_$1.log 2>&1; echo "[answered] $1"' _ {}
: > methods/raptor/results/raptor_cuad.jsonl
for s in $SLUGS; do [ -f methods/raptor/results/raptor_cuad_$s.jsonl ] && cat methods/raptor/results/raptor_cuad_$s.jsonl >> methods/raptor/results/raptor_cuad.jsonl; done
echo "RAPTOR_CUAD_DONE rows=$(wc -l < methods/raptor/results/raptor_cuad.jsonl)"
