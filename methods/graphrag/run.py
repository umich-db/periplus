"""graphrag runner — Microsoft GraphRAG (arXiv 2404.16130) via official graphrag
3.1.0, Azure gpt-5 + text-embedding-3-large. Captures build vs answer tokens by
wrapping litellm. Produces BOTH local and global search answers per question.

Usage:
  run.py build  --movie <slug>   # index (entities->relationships->Leiden communities->reports)
  run.py answer --movie <slug>   # local_search + global_search per question

Outputs:
  build_costs.json                       per-movie build tokens + node/edge/community counts
  results/graphrag_local_<slug>.jsonl    local-search answers (benchmark schema)
  results/graphrag_global_<slug>.jsonl   global-search answers
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import sys, json, time, pathlib, argparse, asyncio
import litellm
import pandas as pd

BENCH = _BENCH
MROOT = BENCH/"methods"/"graphrag"
RUNS = MROOT/"runs"
RES = MROOT/"results"; RES.mkdir(parents=True, exist_ok=True)
BUILD_COSTS = MROOT/"build_costs.json"
BCDIR = MROOT/"build_costs"; BCDIR.mkdir(parents=True, exist_ok=True)  # per-movie (avoids parallel-write race)
sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import add_build
import os
_CUAD = bool(os.environ.get("CUAD"))
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
N_MOVIES = len(list((BENCH/"data"/("contracts/txt" if _CUAD else "movies/txt")).glob("*.txt")))

SLUG2TITLE = {
 "10_things_i_hate_about_you":"10 Things I Hate About You","12_and_holding":"12 and Holding",
 "1492_conquest_of_paradise":"1492: Conquest of Paradise","17_again_film":"17 Again (film)",
 "2012_film":"2012 (film)","50_50_2011_film":"50/50 (2011 film)","8mm_film":"8mm (film)",
 "9_2005_film":"9 (2005 film)","a_dry_white_season":"A Dry White Season","a_few_good_men":"A Few Good Men",
}

# ---- litellm token capture ----
_tok = {"prompt": 0, "completion": 0}
def _reset(): _tok["prompt"] = 0; _tok["completion"] = 0
def _add(resp):
    u = getattr(resp, "usage", None)
    if u:
        _tok["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        _tok["completion"] += getattr(u, "completion_tokens", 0) or 0

# The UMich gpt-5 Azure endpoint returns HTTP 500 on ANY streaming request, but
# graphrag's local/global search hardcode stream=True. So we transparently force
# stream=False and replay the full response as a single streaming chunk that
# graphrag_llm can still iterate. (Non-streaming gpt-5 works fine on this endpoint.)
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta

def _as_stream(r):
    content = r.choices[0].message.content or ""
    fr = getattr(r.choices[0], "finish_reason", None) or "stop"
    chunk = ModelResponseStream(model=getattr(r, "model", None) or "gpt-5",
        choices=[StreamingChoices(index=0, delta=Delta(content=content, role="assistant"),
                                  finish_reason=fr)])
    return chunk

_oc = litellm.completion
def _wc(*a, **k):
    if k.get("stream"):
        k = dict(k); k["stream"] = False; k.pop("stream_options", None)
        r = _oc(*a, **k); _add(r)
        return iter([_as_stream(r)])
    r = _oc(*a, **k); _add(r); return r
litellm.completion = _wc
_oac = litellm.acompletion
async def _wac(*a, **k):
    if k.get("stream"):
        k = dict(k); k["stream"] = False; k.pop("stream_options", None)
        r = await _oac(*a, **k); _add(r)
        async def _gen():
            yield _as_stream(r)
        return _gen()
    r = await _oac(*a, **k); _add(r); return r
litellm.acompletion = _wac

import graphrag.api as gapi
from graphrag.config.load_config import load_config

def cfg(slug):
    return load_config(RUNS/slug)

def cmd_build(slug):
    root = RUNS/slug
    bc = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}
    done = (root/"output"/"community_reports.parquet").exists()
    if not done:
        _reset(); t0 = time.time()
        config = cfg(slug)
        asyncio.run(gapi.build_index(config=config))
        bt_p, bt_c = _tok["prompt"], _tok["completion"]
        dt = round(time.time()-t0, 1)
        add_build("graphrag", dt/60, n_total=N_MOVIES)
    else:
        bt_p = bt_c = 0; dt = 0  # already indexed elsewhere; tokens unknown
    # node/edge/community counts
    o = root/"output"
    counts = {}
    for name, f in [("entities","entities"),("relationships","relationships"),
                    ("communities","communities"),("community_reports","community_reports")]:
        p = o/f"{f}.parquet"
        counts[name] = int(len(pd.read_parquet(p))) if p.exists() else 0
    entry = {"build_tokens_total": bt_p+bt_c, "build_in": bt_p, "build_out": bt_c,
             "build_s": dt, **counts}
    (BCDIR/f"{slug}.json").write_text(json.dumps(entry, indent=2))   # per-movie, no race
    print(f"[build] {slug}: tokens={bt_p+bt_c} nodes={counts['entities']} edges={counts['relationships']} "
          f"communities={counts['communities']} reports={counts['community_reports']}", flush=True)

async def _noop():
    return

def _load_tables(slug):
    o = RUNS/slug/"output"
    def rd(n):
        p = o/f"{n}.parquet"; return pd.read_parquet(p) if p.exists() else None
    return dict(entities=rd("entities"), communities=rd("communities"),
               community_reports=rd("community_reports"), text_units=rd("text_units"),
               relationships=rd("relationships"), covariates=None)

def cmd_answer(slug):
    config = cfg(slug)
    t = _load_tables(slug)
    bcf = BCDIR/f"{slug}.json"
    boot = json.loads(bcf.read_text()).get("build_tokens_total", 0) if bcf.exists() else 0
    questions = [json.loads(l) for l in (BENCH/"data/questions"/_QFILE).read_text().splitlines() if l.strip()]
    qs = [q for q in questions if q["movie"] == slug]
    amort = round(boot/max(1, len(qs)))
    # guard: if this movie failed to index, write "not found" for all (don't crash on None tables)
    if t["entities"] is None or t["community_reports"] is None:
        for mode in ("local", "global"):
            with open(RES/f"graphrag_{mode}{_SFX}_{slug}.jsonl", "w", encoding="utf-8") as f:
                for q in qs:
                    f.write(json.dumps({"method": f"graphrag_{mode}", "movie": slug, "qid": q["qid"],
                        "question": q["question"], "gold_answers": q["gold_answers"], "prediction": "not found",
                        "source": "miss", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "latency_s": 0, "build_tokens": amort}, ensure_ascii=False)+"\n")
        print(f"SKIP answer {slug}: not indexed (wrote not-found)"); return
    INSTR = " Answer in 10 words or fewer. If not found, say: not found"

    outs = {"local": open(RES/f"graphrag_local{_SFX}_{slug}.jsonl", "w", encoding="utf-8"),
            "global": open(RES/f"graphrag_global{_SFX}_{slug}.jsonl", "w", encoding="utf-8")}
    for q in qs:
        for mode in ("local", "global"):
            _reset(); t0 = time.time()
            ans = None
            for attempt in range(4):  # retry transient Azure 500s
                try:
                    if mode == "local":
                        ans, _ = asyncio.run(gapi.local_search(config=config, entities=t["entities"], communities=t["communities"],
                            community_reports=t["community_reports"], text_units=t["text_units"],
                            relationships=t["relationships"], covariates=t["covariates"],
                            community_level=2, response_type="single phrase", query=q["question"]+INSTR))
                    else:
                        ans, _ = asyncio.run(gapi.global_search(config=config, entities=t["entities"], communities=t["communities"],
                            community_reports=t["community_reports"], community_level=2,
                            dynamic_community_selection=False, response_type="single phrase", query=q["question"]+INSTR))
                    break
                except Exception as e:  # noqa: BLE001
                    msg = repr(e)[:120]
                    if attempt < 3 and any(s in msg.lower() for s in ("internalserver", "500", "timeout", "503", "rate", "overload")):
                        time.sleep(5*(attempt+1)); continue
                    ans = f"[ERROR: {msg}]"; break
            lat = round(time.time()-t0, 2)
            rec = {"method": f"graphrag_{mode}", "movie": slug, "qid": q["qid"], "question": q["question"],
                   "gold_answers": q["gold_answers"], "prediction": str(ans).strip(),
                   "input_tokens": _tok["prompt"], "output_tokens": _tok["completion"],
                   "total_tokens": _tok["prompt"]+_tok["completion"], "latency_s": lat, "build_tokens": amort}
            outs[mode].write(json.dumps(rec, ensure_ascii=False)+"\n"); outs[mode].flush()
        print(f"  {q['qid']} done (local+global)", flush=True)
    for f in outs.values(): f.close()
    print(f"DONE answer {slug}", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build","answer"]); ap.add_argument("--movie", required=True)
    a = ap.parse_args()
    cmd_build(a.movie) if a.cmd == "build" else cmd_answer(a.movie)

if __name__ == "__main__":
    main()
