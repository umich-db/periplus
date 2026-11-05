"""
cold_start.py — Wiki v6: Cold-start 3-phase pipeline.

Phase 1 — BOOTSTRAP (corpus-level, one-time)
  1a. SAMPLER      — picks N documents from the corpus
  1b. PROPOSER     — LLM proposes wiki outline for each sampled doc
  1c. RECONCILER   — merges N outlines into one unified master template

Phase 2 — MULTI-STEP EXPANSION (per-document, at ingest time)
  2a. Pass 1       — coarse extraction: fill master template from document chunks
  2b. Pass 2       — deepening: re-read doc with Pass 1 output, catch what was missed

Phase 3 — QUERY-TIME REFINEMENT (on-demand)
  3a. Answer from wiki (standard v3-style flat read)
  3b. If "not found" → fallback to raw document with question
  3c. Auto-backfill: write newly discovered facts/entities into the wiki

Output format matches v3/v5.1 query pipeline (index.md, facts.md,
relationships.md, timeline.md, characters/*.md, events/*.md).
"""

import json
import os
import random
import re
import time
import logging
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

# ─── Config ──────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parents[2]
load_dotenv(BASE / ".env")
client = None


def _get_client():
    global client
    if client is None:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE"), timeout=300.0,
        )
    return client


MODEL = "gpt-5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("cold_start")

# ─── LLM call tracking ──────────────────────────────────────────────────────

_llm_log = []


def llm(prompt, max_tokens=16384, phase="unknown"):
    input_chars = len(prompt)
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            elapsed = time.time() - t0
            content = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            _llm_log.append({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL, "phase": phase,
                "input_chars": input_chars, "output_chars": len(content),
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "elapsed_s": round(elapsed, 2),
            })
            time.sleep(0.3)
            return content
        except BadRequestError:
            _llm_log.append({
                "timestamp": datetime.now().isoformat(),
                "model": MODEL, "phase": phase,
                "input_chars": input_chars, "output_chars": 0,
                "error": "BadRequestError",
            })
            return ""
        except Exception as e:
            log.warning(f"  LLM error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return ""


def get_llm_log():
    return list(_llm_log)


def llm_cost_summary():
    by_phase = {}
    for entry in _llm_log:
        p = entry["phase"]
        if p not in by_phase:
            by_phase[p] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        by_phase[p]["calls"] += 1
        by_phase[p]["input_tokens"] += entry.get("input_tokens") or 0
        by_phase[p]["output_tokens"] += entry.get("output_tokens") or 0
    return by_phase


# ─── JSON parsing ────────────────────────────────────────────────────────────

def parse_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def parse_json_with_retry(prompt, max_tokens, phase):
    """Call LLM, parse JSON, retry once on failure."""
    raw = llm(prompt, max_tokens=max_tokens, phase=phase)
    result = parse_json(raw)
    if result is None and raw:
        log.warning(f"  {phase}: invalid JSON, retrying...")
        raw = llm(prompt, max_tokens=max_tokens, phase=f"{phase}_retry")
        result = parse_json(raw)
    return result


# ─── Text cleaning ───────────────────────────────────────────────────────────

class _Strip(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._t, self._skip = [], 0
    def handle_starttag(self, t, a):
        if t in ("script", "style"):
            self._skip += 1
    def handle_endtag(self, t):
        if t in ("script", "style") and self._skip > 0:
            self._skip -= 1
    def handle_data(self, d):
        if self._skip == 0:
            self._t.append(d)
    def text(self):
        t = "".join(self._t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


def clean_document(raw_text):
    """Strip wget headers + HTML tags → clean text."""
    # Strip wget/curl download headers (lines before first HTML tag)
    idx = -1
    for marker in ["<html", "<HTML", "<!DOCTYPE", "<!doctype", "<pre>"]:
        pos = raw_text.find(marker)
        if pos >= 0 and (idx < 0 or pos < idx):
            idx = pos
    if idx > 0:
        raw_text = raw_text[idx:]

    p = _Strip()
    p.feed(raw_text)
    text = p.text()

    # Remove common boilerplate patterns
    # Gutenberg headers/footers
    for pat in [
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG .+?\*\*\*",
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG .+?\*\*\*",
    ]:
        m = re.search(pat, text)
        if m:
            text = text[m.end():]
            break

    end_m = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", text)
    if end_m:
        text = text[:end_m.start()]

    return text.strip()


def chunk_text(text, size=30000, overlap=2000):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += max(1, size - overlap)
    return chunks


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name)).strip("_")[:80]


# ═════════════════════════════════════════════════════════════════════════════
# META-PROMPTS — domain-agnostic
# ═════════════════════════════════════════════════════════════════════════════

PROPOSER_PROMPT = """\
You are analyzing a document to design a structured knowledge base (wiki) for it.

Read the document excerpt below. Your task is to propose a hierarchical outline \
that captures ALL important information. The outline should work for other \
documents of the same kind.

Requirements:
- Propose 5-10 top-level sections.
- Each top-level section should have 2-4 sub-sections (fields).
- Section names must be generic and descriptive.
- Think about: who/what is mentioned, what happens, what relationships exist, \
  what facts are stated, what structure the document follows.

Return ONLY valid JSON:
{{
  "document_type": "your best guess at what kind of document this is",
  "sections": [
    {{
      "name": "section name",
      "description": "what this section captures",
      "subsections": [
        {{"name": "subsection name", "description": "what goes here"}}
      ]
    }}
  ]
}}

{excerpt}

JSON:"""

RECONCILER_PROMPT = """\
You are merging multiple proposed wiki outlines into one unified template.

Below are {n} proposed outlines from different documents in the same corpus.
Produce ONE unified outline that works across all documents.

Rules:
- KEEP sections appearing in ≥{threshold} of {n} proposals (by similar purpose, \
  not exact name).
- DROP sections from only one proposal.
- MERGE near-duplicate sections.
- Final: 5-10 top-level sections, each with 2-4 sub-sections.
- Names must remain generic (no domain jargon).

Return ONLY valid JSON:
{{
  "corpus_type": "your assessment of what this corpus contains",
  "sections": [
    {{
      "name": "section name",
      "description": "what this section captures",
      "subsections": [
        {{"name": "subsection name", "description": "what goes here"}}
      ]
    }}
  ]
}}

PROPOSED OUTLINES:
{outlines}

UNIFIED JSON:"""

COMPILE_PASS1_PROMPT = """\
Extract ALL relevant information from this document chunk and organize it \
according to the template below. Return ONLY valid JSON.

Template:
{template_description}

Rules:
- Be thorough — extract EVERY named entity, event, fact, and relationship.
- For lists, aim for completeness. Include names, numbers, specifics.
- If a section does not apply, use: "[not present in this document]"
- Do NOT fabricate. Only extract what is stated or clearly implied.
- Return JSON with section names as keys, subsection names as nested keys.

DOCUMENT CHUNK ({chunk_num}/{total_chunks}):
{chunk}

JSON:"""

COMPILE_PASS2_PROMPT = """\
You previously extracted information from this document. Below is what you found \
so far, and the same document chunk. Your task is to FIND WHAT WAS MISSED.

Focus on:
- Entities mentioned but not captured
- Relationships between entities that are implied but not listed
- Facts that are specific (names, numbers, places) but were overlooked
- Events that were skipped or only partially described
- Motivations, consequences, and causal links

Return ONLY the NEW information as JSON (same schema). Do not repeat what is \
already in the existing extraction. If nothing was missed, return {{}}.

EXISTING EXTRACTION:
{existing}

DOCUMENT CHUNK ({chunk_num}/{total_chunks}):
{chunk}

NEW INFORMATION (JSON):"""

FALLBACK_QA_PROMPT = """\
You are answering a question about a document. The wiki did not contain the answer, \
so you are reading the source document directly.

Answer in 10 words or fewer. If the answer is not in the document, return exactly: not found

Also extract any facts relevant to this question that should be added to the wiki. \
Return as JSON:
{{
  "answer": "your answer in 10 words or fewer",
  "new_facts": ["fact 1", "fact 2", ...],
  "new_entities": [
    {{"name": "entity name", "role": "who/what they are", "actions": ["what they do"]}}
  ],
  "new_relationships": [
    {{"person_a": "name", "person_b": "name", "type": "relationship", "details": "specifics"}}
  ]
}}

Question: {question}

Document excerpt (most relevant section):
{excerpt}

JSON:"""


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: BOOTSTRAP — discover master template from corpus
# ═════════════════════════════════════════════════════════════════════════════

def sample_documents(corpus_path, n=5, seed=42, min_size=5000):
    """Randomly sample N documents from the corpus, skipping failed downloads."""
    corpus_path = Path(corpus_path)
    by_stem = {}
    for f in sorted(corpus_path.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".json"):
            if f.stem not in by_stem or f.suffix == ".txt":
                by_stem[f.stem] = f
    # Filter out tiny files (wget logs with no actual content)
    valid = [(stem, path) for stem, path in sorted(by_stem.items())
             if path.stat().st_size >= min_size]
    log.info(f"Corpus: {len(by_stem)} docs, {len(valid)} with content ≥{min_size} bytes")
    rng = random.Random(seed)
    sampled = rng.sample(valid, min(n, len(valid)))
    log.info(f"Sampled {len(sampled)} documents")
    for doc_id, path in sampled:
        log.info(f"  - {doc_id} ({path.stat().st_size:,} bytes)")
    return sampled


def propose_structure(doc_path, excerpt_chars=12000):
    """Propose a wiki outline for one document."""
    raw = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_document(raw)
    excerpt = text[:excerpt_chars]

    prompt = PROPOSER_PROMPT.format(excerpt=excerpt)
    result = parse_json_with_retry(prompt, max_tokens=4096, phase="proposer")

    if result and "sections" in result:
        log.info(f"  Proposed {len(result['sections'])} sections "
                 f"(type: {result.get('document_type', '?')})")
        return result
    log.warning(f"  Failed to get valid outline for {Path(doc_path).stem}")
    return None


def reconcile_outlines(outlines, threshold_pct=40):
    """Merge N candidate outlines into one unified template."""
    n = len(outlines)
    threshold = max(1, int(n * threshold_pct / 100))

    outlines_text = ""
    for i, outline in enumerate(outlines):
        outlines_text += f"\n--- Outline {i+1} ---\n{json.dumps(outline, indent=2)}\n"

    prompt = RECONCILER_PROMPT.format(n=n, threshold=threshold, outlines=outlines_text)
    result = parse_json_with_retry(prompt, max_tokens=8192, phase="reconciler")

    if result and "sections" in result:
        log.info(f"  Unified template: {len(result['sections'])} sections")
        for s in result["sections"]:
            subs = [sub["name"] for sub in s.get("subsections", [])]
            log.info(f"    - {s['name']}: {subs}")
        return result
    log.warning("  Failed to reconcile outlines")
    return None


def bootstrap_template(corpus_path, n_samples=5, seed=42, threshold_pct=40):
    """Phase 1: sample → propose → reconcile → master template."""
    sampled = sample_documents(corpus_path, n=n_samples, seed=seed)

    outlines = []
    for doc_id, doc_path in sampled:
        log.info(f"Proposing structure for: {doc_id}")
        outline = propose_structure(doc_path)
        if outline:
            outline["_doc_id"] = doc_id
            outlines.append(outline)

    if len(outlines) < 2:
        log.error("Too few valid outlines to reconcile")
        return None, outlines, [d[0] for d in sampled]

    log.info(f"Reconciling {len(outlines)} outlines (threshold={threshold_pct}%)...")
    template = reconcile_outlines(outlines, threshold_pct=threshold_pct)
    return template, outlines, [d[0] for d in sampled]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: MULTI-STEP EXPANSION — compile each document with 2 passes
# ═════════════════════════════════════════════════════════════════════════════

def _template_description(template):
    """Convert template JSON to text for prompts."""
    lines = []
    for section in template["sections"]:
        lines.append(f"\n### {section['name']}")
        lines.append(f"  Purpose: {section.get('description', '')}")
        for sub in section.get("subsections", []):
            lines.append(f"  - {sub['name']}: {sub.get('description', '')}")
    return "\n".join(lines)


def _merge_extraction_into(base, new_data, template):
    """Merge new_data dict into base dict, deduplicating lists."""
    if not new_data or not isinstance(new_data, dict):
        return
    for key, value in new_data.items():
        if not value or value == "[not present in this document]" or value == {}:
            continue
        if key not in base:
            base[key] = value
            continue
        # Merge
        if isinstance(base[key], list) and isinstance(value, list):
            existing = {json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x)
                        for x in base[key]}
            for item in value:
                sig = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
                if sig not in existing:
                    base[key].append(item)
                    existing.add(sig)
        elif isinstance(base[key], dict) and isinstance(value, dict):
            _merge_extraction_into(base[key], value, template)
        elif isinstance(value, str) and value != "[not present in this document]":
            if not base[key] or base[key] == "[not present in this document]":
                base[key] = value


def compile_document_2pass(doc_path, template, chunk_size=30000, chunk_overlap=2000):
    """Phase 2: two-pass extraction for one document.

    Pass 1: Fill template from each chunk.
    Pass 2: Re-read each chunk with Pass 1 output, find what was missed.
    """
    raw = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_document(raw)
    if len(text) < 200:
        log.warning(f"  Document too short: {doc_path}")
        return None

    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    n_chunks = len(chunks)
    tmpl_desc = _template_description(template)
    log.info(f"  {Path(doc_path).stem}: {len(text):,} chars, {n_chunks} chunks")

    # ── Pass 1: Coarse extraction ────────────────────────────────────────────
    merged = {}
    for i, chunk in enumerate(chunks):
        prompt = COMPILE_PASS1_PROMPT.format(
            template_description=tmpl_desc,
            chunk=chunk, chunk_num=i+1, total_chunks=n_chunks,
        )
        result = parse_json_with_retry(prompt, max_tokens=16384, phase="compile_p1")
        if result:
            _merge_extraction_into(merged, result, template)
            log.info(f"    Pass1 chunk {i+1}/{n_chunks}: OK")
        else:
            log.warning(f"    Pass1 chunk {i+1}/{n_chunks}: failed")

    if not merged:
        return None

    # ── Pass 2: Deepening — show what we already have, ask for gaps ──────────
    existing_json = json.dumps(merged, indent=1, default=str)
    # Truncate existing if too large for prompt
    if len(existing_json) > 15000:
        existing_json = existing_json[:15000] + "\n... [truncated]"

    for i, chunk in enumerate(chunks):
        prompt = COMPILE_PASS2_PROMPT.format(
            existing=existing_json,
            chunk=chunk, chunk_num=i+1, total_chunks=n_chunks,
        )
        result = parse_json_with_retry(prompt, max_tokens=8192, phase="compile_p2")
        if result and result != {}:
            _merge_extraction_into(merged, result, template)
            log.info(f"    Pass2 chunk {i+1}/{n_chunks}: found new info")
        else:
            log.info(f"    Pass2 chunk {i+1}/{n_chunks}: nothing new")

    return merged


def compile_corpus(corpus_path, template, output_dir, doc_ids=None):
    """Run Phase 2 on all (or specified) documents. Write v3-compatible wikis."""
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_stem = {}
    for f in sorted(corpus_path.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".json"):
            if f.stem not in by_stem or f.suffix == ".txt":
                by_stem[f.stem] = f
    if doc_ids:
        by_stem = {k: v for k, v in by_stem.items() if k in doc_ids}

    results = {}
    for i, (doc_id, doc_path) in enumerate(sorted(by_stem.items())):
        log.info(f"[{i+1}/{len(by_stem)}] Compiling: {doc_id}")
        compiled = compile_document_2pass(doc_path, template)
        if compiled:
            title = doc_id.replace("_", " ")
            stats = write_v3_wiki(output_dir, doc_id, title, compiled)
            results[doc_id] = stats
        else:
            log.warning(f"  Failed: {doc_id}")
            results[doc_id] = None
    return results


# ═════════════════════════════════════════════════════════════════════════════
# WIKI WRITER — produce v3-compatible markdown files
# ═════════════════════════════════════════════════════════════════════════════

def _find_section(data, candidates):
    """Fuzzy-match a section name in compiled data."""
    for key in data:
        kl = key.lower().strip()
        for c in candidates:
            if c.lower() == kl or c.lower() in kl or kl in c.lower():
                return data[key]
    return {}


def _to_list_of_dicts(data, name_key="name"):
    """Normalize various formats to a list of dicts."""
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({name_key: item})
        return out
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        items.append(item)
                    elif isinstance(item, str):
                        items.append({name_key: item})
            elif isinstance(v, dict):
                v.setdefault(name_key, k)
                items.append(v)
        return items
    return []


def _to_flat_strings(data):
    """Normalize to flat list of strings."""
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and v != "[not present in this document]":
                        out.append(v)
                    elif isinstance(v, list):
                        out.extend(str(x) for x in v if x and str(x) != "[not present in this document]")
        return out
    if isinstance(data, dict):
        out = []
        for v in data.values():
            if isinstance(v, str) and v != "[not present in this document]":
                out.append(v)
            elif isinstance(v, list):
                out.extend(_to_flat_strings(v))
        return out
    return []


def write_v3_wiki(output_dir, doc_id, title, compiled):
    """Convert compiled JSON → v3-compatible wiki pages."""
    movie_dir = Path(output_dir) / safe_filename(doc_id)
    (movie_dir / "characters").mkdir(parents=True, exist_ok=True)
    (movie_dir / "events").mkdir(parents=True, exist_ok=True)

    # Extract sections by fuzzy name matching
    entities_raw = _find_section(compiled, [
        "key entities", "entities", "named entities", "characters",
        "people", "participants", "dramatis personae", "key characters",
    ])
    events_raw = _find_section(compiled, [
        "key events", "events", "narrative events", "plot events",
        "significant events", "story events", "occurrences",
    ])
    facts_raw = _find_section(compiled, [
        "factual details", "facts", "key facts", "specific details",
        "key details", "important details", "data points",
    ])
    rels_raw = _find_section(compiled, [
        "relationships", "connections", "interactions",
        "entity relationships", "dynamics", "character relationships",
    ])
    timeline_raw = _find_section(compiled, [
        "timeline", "chronology", "sequence", "temporal structure",
        "narrative structure", "structural elements", "scenes",
    ])

    entities = _to_list_of_dicts(entities_raw)
    events = _to_list_of_dicts(events_raw, name_key="name")
    facts = _to_flat_strings(facts_raw)
    rels = _to_list_of_dicts(rels_raw)
    timeline = _to_list_of_dicts(timeline_raw)

    # Deduplicate facts
    seen = set()
    unique_facts = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            unique_facts.append(f)
    facts = unique_facts

    # ── Write character pages ────────────────────────────────────────────────
    char_files = {}
    for ent in entities:
        name = ent.get("name", "")
        if not name or name == "[not present in this document]":
            continue
        fname = safe_filename(name) + ".md"
        path = movie_dir / "characters" / fname

        def _as_list(v):
            if isinstance(v, list): return v
            if isinstance(v, str) and v != "[not present in this document]": return [v]
            return []

        role = ent.get("role", ent.get("description", "(unknown)"))
        actions = _as_list(ent.get("actions", []))
        e_rels = _as_list(ent.get("relationships", []))
        motivations = _as_list(ent.get("motivations", []))
        quotes = _as_list(ent.get("key_quotes", ent.get("quotes", [])))
        emotions = ent.get("emotional_states", ent.get("emotional_state", []))
        if isinstance(emotions, list):
            emotions = ", ".join(str(e) for e in emotions) if emotions else "(not recorded)"
        outcome = ent.get("outcome", "(not recorded)")

        content = f"""# {name}
Movie: {title}
Type: Character

## Role
{role if role and role != '[not present in this document]' else '(unknown)'}

## Actions
{chr(10).join(f'- {a}' for a in actions[:30]) or '- (none recorded)'}

## Relationships
{chr(10).join(f'- {r}' for r in e_rels[:15]) or '- (none recorded)'}

## Motivations
{chr(10).join(f'- {m}' for m in motivations[:10]) or '- (none recorded)'}

## Key Quotes / Dialogue
{chr(10).join(f'- {q}' for q in quotes[:10]) or '- (none recorded)'}

## Emotional Arc
{emotions}

## Outcome
{outcome if outcome and outcome != '[not present in this document]' else '(not recorded)'}
"""
        path.write_text(content)
        char_files[name] = fname

    # ── Write event pages ────────────────────────────────────────────────────
    event_files = []
    for i, ev in enumerate(events, 1):
        name = ev.get("name", f"Event {i}")
        if name == "[not present in this document]":
            continue
        fname = safe_filename(name) + ".md"
        path = movie_dir / "events" / fname

        involved = ev.get("characters_involved", ev.get("entities_involved", []))
        if isinstance(involved, list):
            involved = ", ".join(str(c) for c in involved)

        content = f"""# {name}
Movie: {title}
Type: Event
Sequence: {ev.get('sequence_order', i)}

## What Happens
{ev.get('description', '(no description)')}

## Characters Involved
{involved or '(unknown)'}

## Location
{ev.get('location', '(unknown)')}

## Consequences
{ev.get('consequences', '(not recorded)')}
"""
        path.write_text(content)
        event_files.append((name, fname))

    # ── Write facts.md ───────────────────────────────────────────────────────
    (movie_dir / "facts.md").write_text(
        f"# Facts — {title}\n\nTotal facts: {len(facts)}\n\n## All Extracted Facts\n"
        + "\n".join(f"- {f}" for f in facts) + "\n"
    )

    # ── Write relationships.md ───────────────────────────────────────────────
    rel_lines = ""
    for r in rels:
        if isinstance(r, dict):
            a = r.get("person_a", r.get("entity_a", r.get("source", "?")))
            b = r.get("person_b", r.get("entity_b", r.get("target", "?")))
            rtype = r.get("type", r.get("relationship", "?"))
            details = r.get("details", r.get("description", ""))
            rel_lines += f"- **{a}** ↔ **{b}** ({rtype}): {details}\n"
        elif isinstance(r, str):
            rel_lines += f"- {r}\n"

    (movie_dir / "relationships.md").write_text(
        f"# Relationships — {title}\n\n## Character Relationships\n"
        + (rel_lines or "(none recorded)\n")
    )

    # ── Write timeline.md ────────────────────────────────────────────────────
    scene_lines = ""
    for i, s in enumerate(timeline, 1):
        if isinstance(s, dict):
            loc = s.get("location", s.get("setting", "?"))
            summary = s.get("summary", s.get("description", "?"))
            scene_lines += f"{i}. **{loc}** — {summary}\n"
        elif isinstance(s, str):
            scene_lines += f"{i}. {s}\n"

    (movie_dir / "timeline.md").write_text(
        f"# Timeline — {title}\n\n## Scene-by-Scene Chronology\n"
        + (scene_lines or "(no scenes recorded)\n")
    )

    # ── Write index.md ───────────────────────────────────────────────────────
    char_links = ""
    for name, fname in sorted(char_files.items()):
        ent = next((e for e in entities if e.get("name") == name), {})
        role = ent.get("role", ent.get("description", ""))
        if role == "[not present in this document]":
            role = ""
        char_links += f"- [[characters/{fname}]] — {role}\n"

    event_links = "\n".join(f"- [[events/{fname}]] — {name}" for name, fname in event_files)

    (movie_dir / "index.md").write_text(f"""# {title}
Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Overview
- Characters: {len(char_files)}
- Events: {len(event_files)}
- Facts: {len(facts)}
- Scenes: {len(timeline)}
- Relationships: {len(rels)}

## Characters
{char_links or '(none)'}

## Key Events
{event_links or '(none)'}

## Other Pages
- [[timeline.md]] — Scene-by-scene chronology ({len(timeline)} scenes)
- [[relationships.md]] — Character relationships ({len(rels)} connections)
- [[facts.md]] — All extracted facts ({len(facts)} facts)
""")

    total_pages = 1 + len(char_files) + len(event_files) + 3
    log.info(f"  Wiki for {doc_id}: {total_pages} pages "
             f"({len(char_files)} chars, {len(event_files)} events, {len(facts)} facts)")
    return {
        "doc_id": doc_id, "pages": total_pages,
        "characters": len(char_files), "events": len(event_files),
        "facts": len(facts), "relationships": len(rels),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: QUERY-TIME REFINEMENT — fallback + auto-backfill
# ═════════════════════════════════════════════════════════════════════════════

def query_with_refinement(question, wiki_dir, raw_doc_path, answer1="", answer2=""):
    """Phase 3: answer from wiki, fallback to raw doc, backfill wiki.

    Returns (predicted_answer, source, backfill_stats).
    source: "wiki" if answered from wiki, "fallback" if from raw doc, "miss" if neither.
    """
    wiki_dir = Path(wiki_dir)
    # Step 1: Try answering from wiki (v3-style flat read)
    digest = _load_v3_digest(wiki_dir)
    if digest:
        qa_prompt = (
            "You are answering a question about a document using ONLY the compiled "
            "wiki digest below.\nAnswer in 10 words or fewer. If the answer is not in the "
            "digest, return exactly: not found\n\n"
            f"Question: {question}\n\nWiki Digest:\n{digest}\n\nAnswer:"
        )
        predicted = llm(qa_prompt, max_tokens=2048, phase="query_wiki")
        predicted = (predicted or "").strip().splitlines()[0].strip(" .\"'") if predicted else "not found"

        if predicted.lower() not in ("not found", "", "none", "n/a"):
            return predicted, "wiki", {}

    # Step 2: Fallback to raw document
    raw = Path(raw_doc_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_document(raw)
    # Find most relevant section by simple keyword overlap
    excerpt = _find_relevant_excerpt(text, question, max_chars=20000)

    fallback_prompt = FALLBACK_QA_PROMPT.format(question=question, excerpt=excerpt)
    raw_resp = llm(fallback_prompt, max_tokens=4096, phase="query_fallback")
    result = parse_json(raw_resp)

    if not result:
        # Try to extract just the answer
        answer = (raw_resp or "").strip().splitlines()[0].strip(" .\"'") if raw_resp else "not found"
        return answer, "fallback", {}

    predicted = result.get("answer", "not found")
    if predicted.lower() in ("not found", "", "none", "n/a"):
        return "not found", "miss", {}

    # Step 3: Auto-backfill wiki with new discoveries
    backfill_stats = _backfill_wiki(wiki_dir, result)

    return predicted, "fallback", backfill_stats


def _load_v3_digest(wiki_dir):
    """Load wiki as flat text (same as v3 query pipeline)."""
    parts = []
    for fname in ["index.md", "facts.md", "relationships.md", "timeline.md"]:
        fpath = wiki_dir / fname
        if fpath.exists():
            parts.append(fpath.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n---\n\n".join(parts) if parts else ""


def _find_relevant_excerpt(text, question, max_chars=20000):
    """Find the chunk of text most relevant to the question via keyword overlap."""
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    chunks = chunk_text(text, size=max_chars, overlap=2000)
    best, best_score = chunks[0], 0
    for chunk in chunks:
        words = set(re.findall(r"[a-zA-Z]{3,}", chunk.lower()))
        score = len(q_words & words)
        if score > best_score:
            best, best_score = chunk, score
    return best


def _backfill_wiki(wiki_dir, fallback_result):
    """Write newly discovered facts/entities/relationships back into the wiki."""
    stats = {"facts_added": 0, "entities_added": 0, "relationships_added": 0}
    wiki_dir = Path(wiki_dir)

    # Backfill facts
    new_facts = fallback_result.get("new_facts", [])
    if new_facts and isinstance(new_facts, list):
        facts_path = wiki_dir / "facts.md"
        if facts_path.exists():
            existing = facts_path.read_text(encoding="utf-8", errors="ignore")
            existing_lower = existing.lower()
            added = []
            for fact in new_facts:
                if isinstance(fact, str) and fact.lower() not in existing_lower:
                    added.append(fact)
            if added:
                # Append to facts.md
                with open(facts_path, "a") as f:
                    for fact in added:
                        f.write(f"- {fact}\n")
                # Update count in header
                count_match = re.search(r"Total facts: (\d+)", existing)
                if count_match:
                    old_count = int(count_match.group(1))
                    new_text = existing.replace(
                        f"Total facts: {old_count}",
                        f"Total facts: {old_count + len(added)}"
                    )
                    facts_path.write_text(new_text + "\n".join(f"- {f}" for f in added) + "\n")
                stats["facts_added"] = len(added)

    # Backfill entities as character pages
    new_entities = fallback_result.get("new_entities", [])
    if new_entities and isinstance(new_entities, list):
        chars_dir = wiki_dir / "characters"
        chars_dir.mkdir(exist_ok=True)
        for ent in new_entities:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            fname = safe_filename(ent["name"]) + ".md"
            path = chars_dir / fname
            if not path.exists():
                role = ent.get("role", "(discovered during query)")
                actions = ent.get("actions", [])
                if isinstance(actions, str):
                    actions = [actions]
                content = f"""# {ent['name']}
Type: Character

## Role
{role}

## Actions
{chr(10).join(f'- {a}' for a in actions) or '- (none recorded)'}

## Relationships
- (none recorded)

## Motivations
- (none recorded)

## Key Quotes / Dialogue
- (none recorded)

## Emotional Arc
(not recorded)

## Outcome
(not recorded)
"""
                path.write_text(content)
                stats["entities_added"] += 1
                # Add to index.md
                _append_to_index(wiki_dir, "characters", fname, ent["name"], role)

    # Backfill relationships
    new_rels = fallback_result.get("new_relationships", [])
    if new_rels and isinstance(new_rels, list):
        rels_path = wiki_dir / "relationships.md"
        if rels_path.exists():
            existing = rels_path.read_text(encoding="utf-8", errors="ignore")
            existing_lower = existing.lower()
            with open(rels_path, "a") as f:
                for r in new_rels:
                    if not isinstance(r, dict):
                        continue
                    a = r.get("person_a", "?")
                    b = r.get("person_b", "?")
                    rtype = r.get("type", "?")
                    details = r.get("details", "")
                    line = f"- **{a}** ↔ **{b}** ({rtype}): {details}"
                    if line.lower() not in existing_lower:
                        f.write(line + "\n")
                        stats["relationships_added"] += 1

    return stats


def _append_to_index(wiki_dir, section_type, fname, name, desc):
    """Add a new entry to index.md."""
    index_path = wiki_dir / "index.md"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8", errors="ignore")
    link_line = f"- [[{section_type}/{fname}]] — {desc}"
    if link_line not in text:
        if section_type == "characters":
            marker = "## Key Events"
        else:
            marker = "## Other Pages"
        text = text.replace(marker, f"{link_line}\n{marker}")
        index_path.write_text(text)
