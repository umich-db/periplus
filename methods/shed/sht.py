"""SHED Structured Heading Tree (SHT) — screenplay adaptation.

SHED builds a document's heading hierarchy then lets an agent navigate it. For
PDFs it infers headings from *visual* features (font/color/caps) via Huridocs.
Screenplays already expose their structure textually: scene headings like
``INT. KITCHEN - NIGHT`` (and bare ``PADUA HIGH SCHOOL - DAY``). So we recover
the SHT structurally — "local-first, no external APIs" (build_tokens = 0).

Each scene becomes a section node: {id, heading, content}. The numbered list of
headings is the Table of Contents handed to the agent; ``read_section(id)``
returns a node's content.
"""

from __future__ import annotations

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])


import re

# time-of-day / continuation tokens that terminate a scene heading
_TIME = (r"DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|NOON|MIDNIGHT|"
         r"CONTINUOUS|LATER|MOMENTS LATER|SAME|SUNSET|SUNRISE")
_INT_EXT = re.compile(r"^(INT|EXT|INT/EXT|I/E)[\.\s/]", re.I)
_TIME_TAIL = re.compile(rf"\b({_TIME})\b\.?\s*$", re.I)
_LETTERS = re.compile(r"[A-Za-z]")


def is_scene_heading(line: str) -> bool:
    """Heuristic screenplay scene-heading detector."""
    s = line.strip()
    if len(s) < 4 or len(s) > 90:
        return False
    # 1) explicit slugline
    if _INT_EXT.match(s):
        return True
    # 2) bare location slugline: mostly UPPERCASE and ends in a time-of-day,
    #    e.g. "PADUA HIGH PARKING LOT - DAY"
    letters = _LETTERS.findall(s)
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    if upper_ratio >= 0.85 and "-" in s and _TIME_TAIL.search(s):
        return True
    return False


def build_sht(text: str) -> list[dict]:
    """Segment a screenplay into ordered section nodes by scene heading.

    Returns a list of {id, heading, content}. id "0" is the title/preamble
    block before the first scene heading (kept so nothing is lost).
    """
    lines = text.splitlines()
    sections: list[dict] = []
    cur_head = "TITLE / FRONT MATTER"
    cur_lines: list[str] = []

    def flush(idx):
        content = "\n".join(cur_lines).strip()
        sections.append({"id": str(idx), "heading": cur_head, "content": content})

    idx = 0
    for line in lines:
        if is_scene_heading(line):
            flush(idx)            # close previous section
            idx += 1
            cur_head = line.strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    flush(idx)                    # final section

    # drop an empty leading preamble (some scripts open straight on a slugline)
    if sections and not sections[0]["content"]:
        sections = sections[1:]
    return sections


def numbered_toc(sections: list[dict]) -> str:
    """Render the agent-facing Table of Contents: ``id | heading``."""
    return "\n".join(f"{s['id']} | {s['heading']}" for s in sections)


if __name__ == "__main__":
    # validation: segment every movie and report scene counts
    import pathlib
    txt_dir = _BENCH / "data" / "movies" / "txt"
    total = 0
    for p in sorted(txt_dir.glob("*.txt")):
        secs = build_sht(p.read_text(encoding="utf-8"))
        total += len(secs)
        sizes = [len(s["content"]) for s in secs]
        avg = sum(sizes) // max(len(sizes), 1)
        print(f"{p.stem:32s} sections={len(secs):4d}  avg_chars={avg:5d}  "
              f"max_chars={max(sizes) if sizes else 0}")
    print(f"\ntotal sections across 10 movies: {total}")
