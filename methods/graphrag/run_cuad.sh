#!/usr/bin/env bash
# GraphRAG on the 10 CUAD contracts: index each, then local+global answer. CUAD=1.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
export CUAD=1
PY="envs/graphrag/bin/python"; PYB="envs/base/bin/python"
SLUGS="array_biopharma_inc_license_development_and_commercializatio azulsa_20170303_f_1a_ex_10_3_9943903_ex_10_3_maintenance_agr ceres_inc_01_25_2012_ex_10_20_collaboration_agreement gooseheadinsurance_inc_04_02_2018_ex_10_6_franchise_agreemen harpoontherapeuticsinc_20200312_10_k_ex_10_18_12051356_ex_10 manufacturersservicesltd_06_05_2000_ex_10_14_outsourcing_agr mrsfieldsoriginalcookiesinc_01_29_1998_ex_10_franchise_agree phasebiopharmaceuticalsinc_20200330_10_k_ex_10_21_12086810_e revolutionmedicinesinc_20200117_s_1_ex_10_1_11948417_ex_10_1 verizonabsllc_20200123_8_k_ex_10_4_11952335_ex_10_4_service_"
mkdir -p methods/graphrag/logs/cuad methods/graphrag/build_costs
setup_c() {
  slug="$1"; root="methods/graphrag/runs/$slug"
  mkdir -p "$root/input"
  cp "data/contracts/txt/$slug.txt" "$root/input/$slug.txt"
  cp methods/graphrag/settings_template.yaml "$root/settings.yaml"
  [ -d "$root/prompts" ] || cp -r methods/graphrag/runs/_template/prompts "$root/prompts"
  $PYB - "$root" <<'PYK'
import os,sys,pathlib,importlib.util
spec=importlib.util.spec_from_file_location("ac", pathlib.Path(os.environ["BENCH_ROOT"])/"shared"/"azure_client.py")
ac=importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
(pathlib.Path(sys.argv[1])/".env").write_text(f"GRAPHRAG_API_KEY={ac.AZURE_API_KEY}\n")
PYK
}
echo "### STAGE A: setup + INDEX 10 contracts, -P2 ###"
for s in $SLUGS; do setup_c "$s"; done
printf '%s\n' $SLUGS | xargs -P2 -I{} bash -c '
  s="$1"; root="methods/graphrag/runs/$s"; rm -rf "$root/output" "$root/cache"
  '"$PY"' methods/graphrag/run.py build --movie "$s" > methods/graphrag/logs/cuad/build_$s.log 2>&1
  echo "[built] $s"' _ {}
echo "### STAGE B: ANSWER local+global, -P3 ###"
printf '%s\n' $SLUGS | xargs -P3 -I{} bash -c '
  '"$PY"' methods/graphrag/run.py answer --movie "$1" > methods/graphrag/logs/cuad/answer_$1.log 2>&1; echo "[answered] $1"' _ {}
ls methods/graphrag/results/graphrag_local_cuad_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_local_cuad.jsonl 2>/dev/null
ls methods/graphrag/results/graphrag_global_cuad_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_global_cuad.jsonl 2>/dev/null
echo "GRAPHRAG_CUAD_DONE local=$(wc -l < methods/graphrag/results/graphrag_local_cuad.jsonl) global=$(wc -l < methods/graphrag/results/graphrag_global_cuad.jsonl)"
