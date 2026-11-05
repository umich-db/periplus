"""Scrape the 10 NarrativeQA screenplay story_urls (imsdb), clean to plaintext,
best-effort scene-split, and write data/movies/txt/<slug>.txt and
data/movies/json/<slug>.json. Consistent slugs reused everywhere."""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import urllib.request, re, json, pathlib, time

BASE = _BENCH/"data"/"movies"
TXT = BASE/"txt"; JSON = BASE/"json"
TXT.mkdir(parents=True, exist_ok=True); JSON.mkdir(parents=True, exist_ok=True)

# slug -> (canonical NarrativeQA title, document_id, imsdb url)
MOVIES = {
 "10_things_i_hate_about_you": ("10 Things I Hate About You","78e0e28b686c7157d598082bfaa8aaaab821b78b","http://www.imsdb.com/scripts/10-Things-I-Hate-About-You.html"),
 "12_and_holding": ("12 and Holding","bbea33d8cce51af36cf7aab15b83d1372f4523aa","http://www.imsdb.com/scripts/12-and-Holding.html"),
 "1492_conquest_of_paradise": ("1492: Conquest of Paradise","c41dba67fbd19688656d31ccffec145c12f15b94","http://www.imsdb.com/scripts/1492-Conquest-of-Paradise.html"),
 "17_again_film": ("17 Again (film)","16a6a06abf4ff93fe3143b66c45dbeae6550816b","http://www.imsdb.com/scripts/17-Again.html"),
 "2012_film": ("2012 (film)","6a14d1ba1aa1a6719840cb98a8feb6a711291aa4","http://www.imsdb.com/scripts/2012.html"),
 "50_50_2011_film": ("50/50 (2011 film)","9292406d5193d3402195df7e5647fd168da2d15a","http://www.imsdb.com/scripts/50-50.html"),
 "8mm_film": ("8mm (film)","6ed5d860df4edd3d32f1aca52ec369f0d21039a0","http://www.imsdb.com/scripts/8MM.html"),
 "9_2005_film": ("9 (2005 film)","2fe5752889751cbb4bbeab3deba4f6e315f90f0a","http://www.imsdb.com/scripts/9.html"),
 "a_dry_white_season": ("A Dry White Season","08ac6c0dca91fee93d27c7caf85bb57e17d4fc46","http://www.imsdb.com/scripts/Dry-White-Season%2C-A.html"),
 "a_few_good_men": ("A Few Good Men","dd40f45256efa1ec710101bcb925c02e634532e2","http://www.imsdb.com/scripts/A-Few-Good-Men.html"),
}

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def extract_script(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("td", class_="scrtext") or soup.find("pre")
    if node is None:
        return ""
    return node.get_text()

def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines to 2; strip trailing spaces
    lines = [ln.rstrip() for ln in text.split("\n")]
    out, blanks = [], 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks <= 2:
                out.append("")
        else:
            blanks = 0
            out.append(ln)
    # drop leading/trailing blank lines
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)

# scene heading: INT./EXT./INT-EXT/I/E slug lines (screenplay convention)
SCENE_RE = re.compile(r"^\s*((?:INT|EXT|INT\.?/EXT|I/E|INT/EXT)[\.\s:].{0,80})$", re.IGNORECASE)

def split_scenes(text):
    lines = text.split("\n")
    scenes = []
    cur_head, cur_buf = None, []
    def flush():
        if cur_head is not None or cur_buf:
            body = "\n".join(cur_buf).strip()
            scenes.append({"heading": (cur_head or "").strip(), "text": body})
    for ln in lines:
        if SCENE_RE.match(ln) and ln.strip().upper() == ln.strip():
            flush()
            cur_head, cur_buf = ln.strip(), []
        else:
            cur_buf.append(ln)
    flush()
    # drop empty leading scene with no heading & tiny body
    scenes = [s for s in scenes if s["heading"] or len(s["text"]) > 40]
    return scenes

results = []
for slug, (title, did, url) in MOVIES.items():
    try:
        html = fetch(url)
        raw = extract_script(html)
        cleaned = clean(raw)
        scenes = split_scenes(cleaned)
        (TXT/f"{slug}.txt").write_text(cleaned, encoding="utf-8")
        obj = {
            "title": title,
            "slug": slug,
            "document_id": did,
            "raw_text": cleaned,
            "char_count": len(cleaned),
            "scenes": scenes,
            "source": "narrativeqa",
            "story_url": url,
        }
        (JSON/f"{slug}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append((slug, len(cleaned), len(scenes)))
        print(f"OK  {slug:30s} chars={len(cleaned):>7d} scenes={len(scenes)}", flush=True)
    except Exception as e:
        results.append((slug, -1, 0))
        print(f"ERR {slug:30s} {repr(e)[:160]}", flush=True)

ok = [r for r in results if r[1] > 0]
print(f"\nDONE: {len(ok)}/{len(MOVIES)} scraped")
