"""Re-render the original ten movie indexes from saved extraction, then balance.

No extraction or summarization API calls. Uses the existing local MPNet model
for balancing. Outputs a separate index; the archived original stays intact.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

BENCH = Path(__file__).resolve().parents[2]
os.environ.setdefault("AZURE_API_KEY", "offline-unused")
os.environ.setdefault("AZURE_API_BASE", "https://example.invalid")
os.environ.setdefault("AZURE_API_VERSION", "2024-02-01")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(BENCH / "src/wiki_index"))
import pipeline
import balance_window

MOVIES = (
    "10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
    "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
    "a_dry_white_season", "a_few_good_men",
)


def no_api(*args, **kwargs):
    raise AssertionError("Saved extraction repair must not call an LLM")


def main():
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=BENCH / "indexes/llm_index_gpt-5.4_narrativeqa")
    ap.add_argument("--output", type=Path, default=BENCH / "methods/llm_index/output/repaired_10")
    ap.add_argument("--movies", nargs="+", choices=MOVIES)
    args = ap.parse_args()
    pipeline.llm = no_api
    selected = set(args.movies or MOVIES)
    questions = [json.loads(line) for line in (args.source / "questions_and_ground_truth.jsonl").read_text().splitlines() if line.strip()]
    subset = [q for q in questions if q["document"] in selected]
    docs = {q["document"]: q for q in subset}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "questions_10_movies.jsonl").write_text("".join(json.dumps(q) + "\n" for q in subset))
    balance_window.SRC_DIGESTS = args.output / "raw"
    balance_window.OUT_DIGESTS = args.output / "balanced"
    document_timings = []
    for slug in MOVIES:
        if slug not in selected:
            continue
        document_started = time.perf_counter()
        q = docs[slug]
        saved = args.source / q["index_dir"]
        load = lambda name: json.loads((saved / name).read_text())
        chunking, provenance = load("_chunking.json"), load("_provenance.json")
        raw = args.output / "raw" / saved.name
        raw_reused = (raw / "_manifest.json").exists()
        render_seconds = 0.0
        if not raw_reused:
            # A failed prior write is reviewable and never reused as complete.
            if raw.exists() and any(raw.iterdir()):
                raise RuntimeError(f"Incomplete prior repair at {raw}; move it before retrying")
            render_started = time.perf_counter()
            pipeline.render_extracted(
                BENCH / q["source_text"], args.output / "raw", saved.name,
                chunking.get("title", saved.name), load("_template.json"),
                load("_compiled.json"), provenance["signatures"], chunking["bounds"],
                provenance.get("spans", {}),
                overview=(saved / "consolidated.md").read_text(), consolidate=False)
            render_seconds = time.perf_counter() - render_started
        balanced = args.output / "balanced" / saved.name
        balanced_reused = ((balanced / "_window_report.json").exists()
                           and (balanced / "_manifest.json").exists())
        balance_started = time.perf_counter()
        balance_window.window_doc(saved.name)
        balance_seconds = time.perf_counter() - balance_started
        document_timings.append({
            "document": slug, "index_directory": saved.name,
            "raw_reused": raw_reused, "balanced_reused": balanced_reused,
            "render_wall_seconds": render_seconds,
            "balance_wall_seconds": balance_seconds,
            "document_wall_seconds": time.perf_counter() - document_started,
        })
        print(f"REPAIRED {slug}", flush=True)
    from verify_tree import validate_doc
    validation_started = time.perf_counter()
    reports = [validate_doc(args.output / "balanced" / Path(docs[s]["index_dir"]).name,
                            strict_integrity=True) for s in MOVIES if s in selected]
    (args.output / "validation.json").write_text(json.dumps(reports, indent=2))
    validation_seconds = time.perf_counter() - validation_started
    failures = sum(len(report["errors"]) for report in reports)
    fresh = all(not row["raw_reused"] and not row["balanced_reused"] for row in document_timings)
    timing = {
        "schema_version": 1, "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "output": str(args.output.resolve()),
        "mode": "offline_saved_extraction_rebuild",
        "status": "validation_failed" if failures else "complete",
        "fresh_rebuild_of_selected_documents": fresh,
        "documents": len(document_timings), "questions": len(subset),
        "validation_errors": failures,
        "api_input_tokens": 0, "api_output_tokens": 0, "api_total_tokens": 0,
        "api_token_scope": "Incremental local rendering and balancing only; original GPT-5.4 discovery/extraction/consolidation is reused and is not free.",
        "render_wall_seconds_sum": sum(row["render_wall_seconds"] for row in document_timings),
        "balance_wall_seconds_sum": sum(row["balance_wall_seconds"] for row in document_timings),
        "document_wall_seconds_sum": sum(row["document_wall_seconds"] for row in document_timings),
        "validation_wall_seconds": validation_seconds,
        "whole_wall_seconds": time.perf_counter() - started,
        "timing_definition": "Sequential invocation from main entry through validation/report writing; includes source/artifact reads and first local MPNet model load, excludes interpreter/module import startup and original API build. Cached stages are marked reused and do not constitute a fresh build.",
        "per_document": document_timings,
    }
    # Keep the latest result convenient, while retaining previous measurements
    # if a later invocation reuses these outputs rather than rebuilding them.
    (args.output / "build_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    with (args.output / "build_timing_runs.jsonl").open("a") as handle:
        handle.write(json.dumps(timing) + "\n")
    print(f"Validated {len(reports)} movies; {len(subset)} questions; {failures} errors")
    print(f"Offline rebuild wall: {timing['whole_wall_seconds']:.3f}s; fresh={fresh}; API tokens=0 (saved extraction reused)")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
