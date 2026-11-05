#!/usr/bin/env bash
# Microsoft GraphRAG (2404.16130) on the 20 NEW movies: index each, then local+global query.
# Reuses old-10 indexes/answers; concats all 30 per-slug files at the end.
set -u
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BENCH_ROOT"
export BENCH_ROOT
PY="envs/graphrag/bin/python"
PYB="envs/base/bin/python"
NEW20=(blade_runner buffy_the_vampire_slayer_film cherry_falls city_of_joy_film clerks \
 collateral_damage_film colombiana ghost_world_film heist_2001_film highlander_endgame \
 miami_vice_film minority_report_film mission_impossible_film raiders_of_the_lost_ark \
 supergirl_film sweet_smell_of_success the_breakfast_club tinker_tailor_soldier_spy_film \
 up_in_the_air_1940_film white_christmas_film)

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

echo "### STAGE A: setup + INDEX new-20 fresh, 2 parallel ###"
mkdir -p methods/graphrag/build_costs methods/graphrag/logs
for s in "${NEW20[@]}"; do setup_movie "$s"; done
printf '%s\n' "${NEW20[@]}" | xargs -P 2 -I {} bash -c '
  s="$1"; root="methods/graphrag/runs/$s"
  rm -rf "$root/output" "$root/cache"
  '"$PY"' methods/graphrag/run.py build --movie "$s" > methods/graphrag/logs/build_$s.log 2>&1
  echo "[built] $s nodes=$(grep -oE "nodes=[0-9]+" methods/graphrag/logs/build_$s.log | tail -1)"
' _ {}
# re-aggregate build costs across ALL per-movie json (old + new)
'"$PYB"' -c "
import os,json,glob,pathlib
bc={pathlib.Path(f).stem:json.load(open(f)) for f in glob.glob('methods/graphrag/build_costs/*.json')}
pathlib.Path('methods/graphrag/build_costs.json').write_text(json.dumps(bc,indent=2))
print('[stageA] indexed', sum(1 for v in bc.values() if v.get('entities',0)>0),'/',len(bc),'with entities')
"

echo "### STAGE B: ANSWER (local+global) new-20, 3 parallel ###"
printf '%s\n' "${NEW20[@]}" | xargs -P 3 -I {} bash -c '
  '"$PY"' methods/graphrag/run.py answer --movie "$1" > methods/graphrag/logs/answer_$1.log 2>&1; echo "[answered] $1"' _ {}

echo "### CONCAT all 30 per-slug -> local/global ###"
ls methods/graphrag/results/graphrag_local_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_local.jsonl
ls methods/graphrag/results/graphrag_global_*.jsonl 2>/dev/null | grep -vE "_(judged|verdicts)\.jsonl$" | xargs -r cat > methods/graphrag/results/graphrag_global.jsonl
for m in graphrag_local graphrag_global; do
  mkdir -p "methods/$m/results"
  cp "methods/graphrag/results/$m.jsonl" "methods/$m/results/$m.jsonl"
  cp methods/graphrag/build_costs.json "methods/$m/build_costs.json" 2>/dev/null || true
done
echo "local -> $(wc -l < methods/graphrag/results/graphrag_local.jsonl) rows; global -> $(wc -l < methods/graphrag/results/graphrag_global.jsonl) rows"
echo "GRAPHRAG_NEW20_DONE"
