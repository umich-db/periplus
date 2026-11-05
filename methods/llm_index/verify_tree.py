"""Validate a built index without importing the pipeline or calling any model.

The legacy positional and ``--all`` CLI remain supported. ``--json`` emits a
machine-readable report; ``--strict-integrity`` requires repair metadata even
for a legacy build. If ``_integrity.json`` exists, its checks always apply.

Integrity v1 records contain ``id``, ``sha256``, optional ``chars``, and ordered
``segments: [{page, start, end}]``. Offsets are Python character offsets into
UTF-8 page text. Concatenating those slices must reproduce the original rendered
record's hash, independently of how it was split into continuation pages.
"""

import argparse
import ast
import collections
import hashlib
from html.parser import HTMLParser
import json
import os
import pathlib
import re
from urllib.parse import unquote, urlsplit


BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[2])
DIGESTS = BENCH / "methods" / "llm_index" / "output" / "digests"
PAGE_SIZE, ROOT_CAP, FANOUT, DESCRIPTOR_CHARS = 2400, 6000, 33, 120
_COUNT_DESCRIPTION = re.compile(
    r"\d+\s+(?:items?|subjects?|children|entries|pages?)\.?", re.I)
_GENERIC_DESCRIPTION = re.compile(
    r"(?:_?(?:summary|(?:win_)?group)[_\s-]*\d*)"
    r"(?:\s*:\s*\d+\s+(?:items?|subjects?|children|entries|pages?))?", re.I)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path, errors, required=False):
    if not path.exists():
        if required:
            errors.append(f"missing file: {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"invalid JSON in {path.name}: {exc}")
        return None


def _safe_page_path(doc, name):
    if not isinstance(name, str) or not name or "\\" in name:
        return None
    rel = pathlib.PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
        return None
    path = doc / name
    try:
        path.resolve().relative_to(doc.resolve())
    except ValueError:
        return None
    return path


def _clean_source(text):
    """Use the actual pure cleaner without importing its API-client module."""
    source = pathlib.Path(__file__).resolve().parents[2] / "src" / "cold_start" / "cold_start.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {"_Strip", "clean_document"}]
    if len(nodes) != 2:
        raise ValueError("cannot locate the source cleaner")
    scope = {"re": re, "HTMLParser": HTMLParser}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["clean_document"](text)


def _source_length(doc, integrity, chunking, errors):
    length = chunking.get("doc_chars") if isinstance(chunking, dict) else None
    if length is not None and (not _integer(length) or length < 0):
        errors.append("invalid source length in _chunking.json")
        length = None
    if not isinstance(integrity, dict) or "source" not in integrity:
        return length
    source = integrity["source"]
    if not isinstance(source, dict):
        errors.append("integrity source must be an object")
        return length
    declared = source.get("length_chars")
    if declared is not None:
        if not _integer(declared) or declared < 0:
            errors.append("invalid integrity source length_chars")
        else:
            if length is not None and length != declared:
                errors.append("source length differs between integrity and chunking metadata")
            length = declared
    if not source.get("path"):
        if source.get("sha256"):
            errors.append("source checksum supplied without source.path")
        return length
    if not isinstance(source["path"], str):
        errors.append("source.path must be a string")
        return length
    path = pathlib.Path(source["path"])
    if not path.is_absolute():
        path = doc / path
    try:
        # Match the pipeline's decoding. Cleaning is explicitly declared; a raw
        # checksum and a cleaned-source checksum must never be confused.
        text = path.read_text(encoding="utf-8", errors="ignore")
        normalization = source.get("normalization", "raw")
        if normalization == "clean_document":
            text = _clean_source(text)
        elif normalization not in (None, "raw", "none"):
            raise ValueError(f"unsupported normalization: {normalization!r}")
    except (OSError, ValueError, SyntaxError) as exc:
        errors.append(f"cannot verify source: {exc}")
        return length
    if length is not None and len(text) != length:
        errors.append(f"source length mismatch: {len(text)} != {length}")
    if source.get("sha256") is not None:
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != source["sha256"]:
            errors.append("source checksum mismatch")
    return len(text)


def _check_spans(spans, label, length, errors):
    if not isinstance(spans, list):
        errors.append(f"invalid evidence spans list: {label}")
        return
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"invalid evidence span: {label}[{index}]")
            continue
        start, end = span.get("start"), span.get("end")
        if not _integer(start) or not _integer(end) or not 0 <= start < end:
            errors.append(f"invalid evidence span bounds: {label}[{index}]")
        elif length is not None and end > length:
            errors.append(f"evidence span exceeds source length: {label}[{index}]")


def _check_integrity(integrity, by, texts, errors, metrics):
    if not isinstance(integrity, dict):
        errors.append("integrity report must be an object")
        return
    if integrity.get("version") != 1:
        errors.append("unsupported integrity report version (expected 1)")
    for name in ("unaccounted_keys", "unaccounted_fields"):
        if name not in integrity:
            errors.append(f"integrity report missing {name}")
        elif not isinstance(integrity[name], list):
            errors.append(f"integrity {name} must be a list")
        elif integrity[name]:
            errors.append(f"integrity {name}: {len(integrity[name])} unaccounted")
    records = integrity.get("fields")
    if not isinstance(records, list):
        errors.append("integrity fields must be a list")
        return
    metrics["integrity_records"] = len(records)
    ids, covered = set(), set()
    for index, record in enumerate(records):
        label = f"integrity field {index}"
        if not isinstance(record, dict):
            errors.append(f"{label} must be an object")
            continue
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"{label} missing stable id")
        elif identifier in ids:
            errors.append(f"duplicate integrity field id: {identifier}")
        else:
            ids.add(identifier)
            label = f"integrity field {identifier}"
        segments = record.get("segments")
        if not isinstance(segments, list) or not segments:
            errors.append(f"{label} has no preservation segments")
            continue
        pieces = []
        for number, segment in enumerate(segments):
            if not isinstance(segment, dict):
                errors.append(f"{label} segment {number} must be an object")
                continue
            page, start, end = segment.get("page"), segment.get("start"), segment.get("end")
            if not isinstance(page, str) or page not in by or page not in texts:
                errors.append(f"{label} segment refers to missing page: {page!r}")
                continue
            if by[page].get("level") != "leaf":
                errors.append(f"{label} preservation segment is not on a leaf: {page}")
            if by[page].get("record_id") is not None and by[page]["record_id"] != identifier:
                errors.append(f"{label} preservation segment belongs to another record: {page}")
            if not _integer(start) or not _integer(end) or not 0 <= start < end <= len(texts[page]):
                errors.append(f"{label} segment bounds invalid: {page}")
                continue
            pieces.append(texts[page][start:end])
            covered.add(page)
        value = "".join(pieces)
        if hashlib.sha256(value.encode("utf-8")).hexdigest() != record.get("sha256"):
            errors.append(f"{label} content checksum mismatch")
        chars = record.get("chars")
        if chars is not None and (not _integer(chars) or chars != len(value)):
            errors.append(f"{label} character count mismatch")
    uncovered = sorted(p for p, entry in by.items() if entry.get("level") == "leaf" and p not in covered)
    if uncovered:
        errors.append(f"leaf pages missing integrity coverage ({len(uncovered)}): {uncovered[:5]}")


def _check_links(by, texts, errors):
    single_links = ("next", "previous", "next_page", "previous_page", "continuation_of")
    continuation_parts = collections.defaultdict(list)
    for path, entry in by.items():
        for key in ("links", "continuations"):
            links = entry.get(key, [])
            if not isinstance(links, list):
                errors.append(f"invalid {key} list: {path}")
                continue
            for target in links:
                if not isinstance(target, str) or target not in by:
                    errors.append(f"broken {key} link: {path} -> {target!r}")
        for key in single_links:
            target = entry.get(key)
            if target is not None and (not isinstance(target, str) or target not in by):
                errors.append(f"broken continuation link ({key}): {path} -> {target!r}")
        owner = entry.get("continuation_of")
        if isinstance(owner, str) and owner in by:
            continuation_parts[owner].append(path)
            if by[owner].get("level") != "branch":
                errors.append(f"continuation owner must be a branch: {path} -> {owner}")
            if entry.get("record_id") != by[owner].get("record_id"):
                errors.append(f"continuation record identity mismatch: {path}")
            if not _integer(entry.get("continuation_index")) or entry["continuation_index"] < 1:
                errors.append(f"invalid continuation index: {path}")
        for forward, backward in (("next", "previous"), ("next_page", "previous_page")):
            target = entry.get(forward)
            if isinstance(target, str) and target in by:
                reverse = by[target].get(backward)
                if reverse is not None and reverse != path:
                    errors.append(f"inconsistent continuation links: {path} -> {target}")
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", texts.get(path, "")):
            target = target.strip().strip("<>")
            try:
                parsed = urlsplit(target)
            except ValueError:
                errors.append(f"invalid page link: {path} -> {target}")
                continue
            if parsed.scheme or parsed.netloc or not parsed.path.endswith(".md"):
                continue
            decoded = unquote(parsed.path)
            # Standard Markdown paths are relative to the page. Legacy index
            # authors also use root-relative manifest paths without a slash.
            relative = os.path.normpath(str(pathlib.PurePosixPath(path).parent / decoded))
            if relative not in by and decoded not in by:
                errors.append(f"broken page link: {path} -> {decoded}")
    for owner, parts in continuation_parts.items():
        if not all(_integer(by[p].get("continuation_index")) for p in parts):
            continue
        ordered = sorted(parts, key=lambda p: by[p]["continuation_index"])
        if [by[p]["continuation_index"] for p in ordered] != list(range(1, len(parts) + 1)):
            errors.append(f"missing or duplicate continuation index: {owner}")
        if by[owner].get("continuations") != ordered:
            errors.append(f"continuation owner does not list all parts in order: {owner}")
        for index, path in enumerate(ordered):
            expected = ordered[max(0, index - 1):index] + ordered[index + 1:index + 2]
            if by[path].get("continuations") != expected:
                errors.append(f"broken continuation neighbor links: {path}")
    for key in ("next", "next_page"):
        done = set()
        for start in by:
            chain, current = set(), start
            while isinstance(current, str) and current in by and current not in done:
                if current in chain:
                    errors.append(f"continuation cycle ({key}): {current}")
                    break
                chain.add(current)
                current = by[current].get(key)
            done.update(chain)


def validate_doc(directory, strict_integrity=False):
    """Return errors/warnings/metrics; malformed artifacts do not abort the run."""
    doc = pathlib.Path(directory)
    report = {"document": str(doc), "errors": [], "warnings": [], "metrics": {}}
    errors, warnings, metrics = report["errors"], report["warnings"], report["metrics"]
    manifest = _read_json(doc / "_manifest.json", errors, required=True)
    if not isinstance(manifest, list):
        errors.append("manifest must be a list")
        return report
    by = {}
    for number, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            errors.append(f"manifest entry {number} must be an object")
            continue
        path = entry.get("path")
        if _safe_page_path(doc, path) is None:
            errors.append(f"invalid manifest path: {path!r}")
            continue
        if path in by:
            errors.append(f"duplicate manifest path: {path}")
            continue
        by[path] = entry
    metrics["nodes"] = len(manifest)
    metrics["unique_pages"] = len(by)
    roots = [p for p, e in by.items() if e.get("parent") is None]
    if roots != ["index.md"]:
        errors.append(f"expected one root index.md: {roots}")
    children = collections.defaultdict(list)
    texts = {}
    for path, entry in by.items():
        parent = entry.get("parent")
        if parent is not None:
            if not isinstance(parent, str) or parent not in by:
                errors.append(f"missing parent: {path} -> {parent!r}")
            else:
                children[parent].append(path)
                if by[parent].get("level") == "leaf":
                    errors.append(f"leaf used as parent: {parent}")
        level = entry.get("level")
        if level not in ("root", "branch", "leaf") or (level == "root") != (parent is None):
            errors.append(f"invalid node level or root designation: {path}")
        summary = entry.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            errors.append(f"blank routing description: {path}")
        elif _COUNT_DESCRIPTION.fullmatch(summary.strip()) or _GENERIC_DESCRIPTION.fullmatch(summary.strip()):
            errors.append(f"generic routing description: {path}: {summary}")
        elif len(summary) > DESCRIPTOR_CHARS:
            errors.append(f"routing description exceeds {DESCRIPTOR_CHARS} chars: {path}")
        try:
            texts[path] = (doc / path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"missing or unreadable page: {path}: {exc}")
            continue
        cap = ROOT_CAP if level == "root" else PAGE_SIZE
        if len(texts[path]) > cap:
            errors.append(f"page exceeds {cap} chars: {path} ({len(texts[path])})")
        if level == "leaf" and re.search(r"\(truncated\)|\[truncated\]", texts[path], re.I):
            errors.append(f"clipped leaf content: {path}")
        if re.match(r"#\s*\{['\"]", texts[path].lstrip()[:40]):
            errors.append(f"non-readable dictionary heading: {path}")
    if "index.md" in texts and "## Categories" not in texts["index.md"]:
        errors.append("root does not route to categories")

    # Iterative parent traversal detects disconnected cycles without hanging or
    # overflowing Python's recursion limit on a malformed tree.
    done = set()
    for start in by:
        chain, current = set(), start
        while isinstance(current, str) and current in by and current not in done:
            if current in chain:
                errors.append(f"parent cycle: {current}")
                break
            chain.add(current)
            current = by[current].get("parent")
        done.update(chain)
    reachable, frontier, depths = set(), ["index.md"] if "index.md" in by else [], {"index.md": 0}
    while frontier:
        path = frontier.pop()
        if path in reachable:
            continue
        reachable.add(path)
        for child in children.get(path, []):
            if child not in depths:
                depths[child] = depths[path] + 1
            frontier.append(child)
    orphaned = sorted(set(by) - reachable)
    if orphaned:
        errors.append(f"unreachable manifest nodes ({len(orphaned)}): {orphaned[:5]}")
    unmanaged = sorted(str(p.relative_to(doc)) for p in doc.rglob("*.md")
                       if str(p.relative_to(doc)) not in by and p.name != "consolidated.md")
    if unmanaged:
        errors.append(f"unreachable page files ({len(unmanaged)}): {unmanaged[:5]}")
    leaves = [p for p, e in by.items() if e.get("level") == "leaf"]
    metrics.update(leaves=len(leaves), reachable_nodes=len(reachable),
                   max_fanout=max((len(c) for c in children.values()), default=0),
                   max_depth=max(depths.values(), default=0))
    if not leaves:
        errors.append("tree contains no leaf subjects")
    if any(depths.get(p, 0) < 2 for p in leaves):
        errors.append("subject leaves must be below a category (at least three levels)")
    if (doc / "_window_report.json").exists() or (doc / "_balance_report.json").exists():
        if metrics["max_fanout"] > FANOUT:
            errors.append(f"balanced tree fanout exceeds {FANOUT}: {metrics['max_fanout']}")

    _check_links(by, texts, errors)
    integrity_path = doc / "_integrity.json"
    integrity = _read_json(integrity_path, errors, required=strict_integrity)
    if integrity_path.exists() and integrity is None:
        errors.append("integrity report must be an object")
    if not integrity_path.exists() and not strict_integrity:
        warnings.append("legacy build: no _integrity.json preservation report")
    chunking = _read_json(doc / "_chunking.json", errors)
    length = _source_length(doc, integrity, chunking, errors)
    metrics["source_chars"] = length
    if length is None:
        warnings.append("source length unavailable; span upper bounds cannot be checked")
    for path, entry in by.items():
        spans = entry.get("spans", [])
        if entry.get("level") == "leaf" and not spans:
            errors.append(f"subject missing evidence spans: {path}")
        _check_spans(spans, path, length, errors)
    provenance = _read_json(doc / "_provenance.json", errors, required=True)
    if isinstance(provenance, dict):
        spans = provenance.get("spans", {})
        if isinstance(spans, dict):
            for number, records in enumerate(spans.values()):
                _check_spans(records, f"provenance fact {number}", length, errors)
        else:
            errors.append("provenance spans must be an object")
    elif (doc / "_provenance.json").exists():
        errors.append("provenance must be an object")
    if isinstance(chunking, dict):
        _check_spans(chunking.get("bounds", []), "chunk boundaries", length, errors)
    elif (doc / "_chunking.json").exists():
        errors.append("chunking metadata must be an object")
    if integrity is not None:
        _check_integrity(integrity, by, texts, errors, metrics)
    return report


def check_doc(directory, strict_integrity=False):
    """Backward-compatible printable validator returning the failing checks."""
    report = validate_doc(directory, strict_integrity=strict_integrity)
    _print_report(report)
    return report["errors"]


def _print_report(report):
    metrics = report["metrics"]
    print(f"== {pathlib.Path(report['document']).name}: {metrics.get('nodes', 0)} nodes, "
          f"{metrics.get('leaves', 0)} subjects, max fanout {metrics.get('max_fanout', 0)}")
    for error in report["errors"]:
        print(f"  FAIL  {error}")
    for warning in report["warnings"]:
        print(f"  info  {warning}")
    if not report["errors"]:
        print("  PASS  structural, evidence, and available integrity checks")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*")
    parser.add_argument("--all", action="store_true", help="check every built digest")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--strict-integrity", action="store_true", help="require a repair integrity report")
    args = parser.parse_args(argv)
    if args.all or not args.directories:
        docs = sorted(p for p in DIGESTS.iterdir() if (p / "_manifest.json").exists()) if DIGESTS.exists() else []
    else:
        docs = [pathlib.Path(a) for a in args.directories]
    reports = [validate_doc(d, strict_integrity=args.strict_integrity) for d in docs]
    total = sum(len(r["errors"]) for r in reports)
    if args.json:
        print(json.dumps({"documents": reports, "failing_checks": total}, indent=2))
    else:
        for report in reports:
            _print_report(report)
        print(f"{len(docs)} document(s), {total} failing check(s)")
        if not docs:
            print("no built documents yet")
    return (1 if total else 0) if docs else 2


if __name__ == "__main__":
    raise SystemExit(main())
