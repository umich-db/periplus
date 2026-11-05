#!/usr/bin/env bash
# Build an isolated run tree, so one model's results can never mix with another's.
#
#   bash scripts/make_run_tree.sh ../gpt6
#
# The tree is a BENCH_ROOT: every script in this benchmark resolves its paths
# from $BENCH_ROOT (or falls back to its own location), so pointing that at the
# new tree sends every result, log, index and figure there while the code stays
# in one place.
#
# Code, data and virtualenvs are SYMLINKED, not copied: editing shared/ or
# methods/ applies to every run tree at once, and there is no second copy to
# drift. Only the directories a run writes into are real:
#
#   methods/<m>/{results,logs,output}   per-method outputs
#   results/                            eval summaries, build times, metrics
#   report/                             figures
#   logs/                               sweep logs
#
# This is safe because every method in the sweep writes via BENCH_ROOT. A script
# that writes relative to its own __file__ would resolve the symlink and land
# back in the source tree instead; the ones that do that (wiki_v7_window's
# prepare/report chain) are study scripts, not sweep methods. Check before adding
# one to the sweep:
#
#   grep -rn "HERE\s*/" --include=*.py methods/<m>/ | grep -E "write_text|mkdir"
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:?usage: make_run_tree.sh <path>}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
[ "$DEST" != "$SRC" ] || { echo "refusing to build a run tree over the source tree" >&2; exit 2; }

echo "source: $SRC"
echo "tree:   $DEST"

# --- shared, read-only: code, data, envs -------------------------------------
for item in shared scripts src data envs requirements .env; do
  [ -e "$SRC/$item" ] || continue
  [ -e "$DEST/$item" ] && continue
  ln -s "$SRC/$item" "$DEST/$item"
done

# --- per-method: code symlinked, output directories real ---------------------
mkdir -p "$DEST/methods"
for m in "$SRC"/methods/*/; do
  name="$(basename "$m")"
  # Always create all three output directories, even when the source tree has
  # none: a method whose build artefacts were archived away still needs somewhere
  # to write them, and a missing output/ makes the build fail late instead of
  # never having been a problem.
  mkdir -p "$DEST/methods/$name" "$DEST/methods/$name/results" \
           "$DEST/methods/$name/logs" "$DEST/methods/$name/output"
  for entry in "$m".[!.]* "$m"*; do
    [ -e "$entry" ] || continue
    base="$(basename "$entry")"
    case "$base" in
      results|logs|output) mkdir -p "$DEST/methods/$name/$base"; continue ;;
      # Some runners write their build-cost record into the METHOD directory
      # rather than results/. Symlinked, that write follows the link and
      # overwrites the source tree's record: a gpt-6 run silently replaced
      # bench's gpt-5 docetl build cost this way. Copy it instead, so a build
      # that is being carried over keeps its cost and a rebuild overwrites only
      # this tree's copy.
      build_costs*.json) cp -n "$entry" "$DEST/methods/$name/$base" 2>/dev/null || true; continue ;;
    esac
    [ -e "$DEST/methods/$name/$base" ] && continue
    ln -s "$entry" "$DEST/methods/$name/$base"
  done
done

# --- real output roots -------------------------------------------------------
mkdir -p "$DEST/results/eval/build_times" "$DEST/report" "$DEST/logs/rerun"

cat > "$DEST/README.md" <<EOF
# Run tree: $(basename "$DEST")

Outputs only. Code, data and virtualenvs are symlinks into $SRC, so this tree
has no copy of the benchmark to drift from it.

Run everything against it by exporting BENCH_ROOT:

    export BENCH_ROOT=$DEST
    envs/base/bin/python scripts/rerun_all.py

or per command:

    BENCH_ROOT=$DEST bash scripts/run_method.sh naive_gpt answer

Built by scripts/make_run_tree.sh on $(date -I).
EOF

echo "methods: $(find "$DEST/methods" -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "done. export BENCH_ROOT=$DEST"
