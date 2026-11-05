#!/usr/bin/env bash
# Index all 10 movies (fresh, for clean token capture) then local+global query,
# then judge graphrag_local + graphrag_global. Microsoft GraphRAG (2404.16130).
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/graphrag/bin/python"
PYB="envs/base/bin/python"
SLUGS=(10_things_i_hate_about_you 12_and_holding 1492_conquest_of_paradise 17_again_film \
 2012_film 50_50_2011_film 8mm_film 9_2005_film a_dry_white_season a_few_good_men)

setup_movie() {
  slug="$1"; root="methods/graphrag/runs/$slug"
  mkdir -p "$root/input"
  cp "data/movies/txt/$slug.txt" "$root/input/$slug.txt"
  cp methods/graphrag/settings_template.yaml "$root/settings.yaml"
  [ -d "$root/prompts" ] || cp -r methods/graphrag/runs/_template/prompts "$root/prompts"
  $PYB - "$root" <<'PYK'
import os,sys,pathlib,importlib.util
spec=importlib.util.spec_from_file_location("ac", pathlib.Path(os.environ["BENCH_ROOT"])/"shared"/"azure_client.py")
ac=importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
(pathlib.Path(sys.argv[1])/".env").write_text(f"GRAPHRAG_API_KEY={ac.AZURE_API_KEY}\n")
PYK
}

echo "### STAGE A: setup (16384-token settings) then INDEX all 10 fresh, 2 parallel ###"
for s in "${SLUGS[@]}"; do setup_movie "$s"; done   # copies settings_template (now max_completion_tokens=16384)
rm -rf methods/graphrag/build_costs methods/graphrag/build_costs.json; mkdir -p methods/graphrag/build_costs
printf '%s\n' "${SLUGS[@]}" | xargs -P 2 -I {} bash -c '
  s="$1"; root="methods/graphrag/runs/$s"
  rm -rf "$root/output" "$root/cache"     # fresh -> real token counts + retry extraction at 16384
  '"$PY"' methods/graphrag/run.py build --movie "$s" > methods/graphrag/logs/build_$s.log 2>&1
  echo "[built] $s nodes=$(grep -oE "nodes=[0-9]+" methods/graphrag/logs/build_$s.log | tail -1)"
' _ {}
# aggregate per-movie build costs (no race) -> build_costs.json
'"$PYB"' -c "
import os,json,glob,pathlib
bc={pathlib.Path(f).stem:json.load(open(f)) for f in glob.glob('methods/graphrag/build_costs/*.json')}
pathlib.Path('methods/graphrag/build_costs.json').write_text(json.dumps(bc,indent=2))
print('[stageA] indexed', sum(1 for v in bc.values() if v.get('entities',0)>0),'/',len(bc),'with entities')
"

echo "### STAGE B: ANSWER (local+global) per movie, 3 parallel ###"
printf '%s\n' "${SLUGS[@]}" | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/graphrag/run.py answer --movie "$1" > methods/graphrag/logs/answer_$1.log 2>&1
' _ {}
ls methods/graphrag/results/graphrag_local_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_local.jsonl
ls methods/graphrag/results/graphrag_global_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_global.jsonl
# build_costs is shared by both local/global views
cp methods/graphrag/build_costs.json methods/graphrag_local_build_costs.json 2>/dev/null || true
echo "[stageB] answered"

echo "### STAGE C: JUDGE graphrag_local + graphrag_global ###"
# eval.py expects methods/<m>/results/<m>.jsonl; create method dirs as views
for m in graphrag_local graphrag_global; do
  mkdir -p "methods/$m/results"
  cp "methods/graphrag/results/$m.jsonl" "methods/$m/results/$m.jsonl"
  cp methods/graphrag/build_costs.json "methods/$m/build_costs.json" 2>/dev/null || true
done
$PYB shared/eval.py graphrag_local graphrag_global > results/eval/eval_graphrag.log 2>&1
for m in graphrag_local graphrag_global; do
  f="results/eval/${m}_summary.json"
  [ -f "$f" ] && $PYB -c "import json;d=json.load(open('$f'));print(f\"{d['method']:16s} acc={d['judge_accuracy']:.1%} fab={d['fabrication_pct']}%\")"
done
echo "GRAPHRAG_ALL_DONE"
