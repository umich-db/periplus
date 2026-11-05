"""Doctopus on the long-document QA bench.

Faithful adaptation of Doctopus (mutong184/Doctopus): budget-aware, per-item
strategy selection via a GROUP KNAPSACK. Doctopus fills a (document, attribute)
table cell-by-cell, picking the cheapest strategy per cell that still maximises
predicted quality under a cost budget. Here each *question* is a cell, and the
strategies are an answer-generation ladder of increasing cost/quality:

    extractive  - best sentence from the top chunk, NO LLM call (~free)
    rag@1/5/10  - retrieve k chunks -> one gpt-5 answer (cost grows with k)

Quality is estimated with the bench's own LLM judge (P(correct), not token-F1),
calibrated on a held-out validation split and bucketed by retrieval confidence
-- the QA analog of Doctopus estimating per-cell strategy accuracy on a
validation set. The group_knap_sack routine is lifted verbatim from the
Doctopus port (doctopus/Doctopus/main.py) so the optimiser is the real thing.

Pipeline: build (chunk+embed) -> calibrate (judge on val) -> optimise (knapsack
sweep over total token budget) -> execute+judge (cached) -> curve + a results
jsonl at one operating point for shared/eval.py.

Run:  envs/base/bin/python methods/doctopus/run.py
Then: envs/base/bin/python shared/eval.py doctopus
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import json, time, pathlib, importlib.util, re, os
from concurrent.futures import ThreadPoolExecutor
_CUAD = bool(os.environ.get("CUAD"))
_DOCDIR = "contracts/txt" if _CUAD else "movies/txt"
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
import numpy as np

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH / "shared" / "azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
import sys; sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import time_build
from latencyprobe import measure as measure_latency
import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------- config -----
METHOD = "doctopus"
MROOT = BENCH / "methods" / METHOD
EMB = MROOT / "output" / "emb"; EMB.mkdir(parents=True, exist_ok=True)
CACHE = MROOT / "output" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUTDIR = MROOT / "output"
RESULTS = MROOT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
BUILD_COSTS = MROOT / f"build_costs{_SFX}.json"

# Full bench. CUAD uses the 10 contract slugs; otherwise all 30 movies.
if _CUAD:
    MOVIES = sorted(p.stem for p in (BENCH/"data"/_DOCDIR).glob("*.txt"))
else:
    MOVIES = ["10_things_i_hate_about_you", "12_and_holding", "1492_conquest_of_paradise",
          "17_again_film", "2012_film", "50_50_2011_film", "8mm_film", "9_2005_film",
          "a_dry_white_season", "a_few_good_men",
          "blade_runner", "buffy_the_vampire_slayer_film", "cherry_falls",
          "city_of_joy_film", "clerks", "collateral_damage_film", "colombiana",
          "ghost_world_film", "heist_2001_film", "highlander_endgame",
          "miami_vice_film", "minority_report_film", "mission_impossible_film",
          "raiders_of_the_lost_ark", "supergirl_film", "sweet_smell_of_success",
          "the_breakfast_club", "tinker_tailor_soldier_spy_film",
          "up_in_the_air_1940_film", "white_christmas_film"]
# (name, k)  k=0 -> extractive, no LLM
STRATEGIES = [("extractive", 0), ("rag@1", 1), ("rag@3", 3), ("rag@5", 5), ("rag@10", 10)]
VAL_FRAC = 0.4          # share of questions used to calibrate quality
N_BUCKETS = 3           # retrieval-confidence buckets for P(correct|strategy,bucket)
COST_DIVISOR = 1000    # 50->1000: bounds knapsack DP size m=budget/COST_DIVISOR (OOM at 50 on 30-movie test set, m~106k)
BUDGET_POINTS = 10      # points along the budget sweep
WORKERS = 4          # lowered 8->4: caps peak memory during optimise+execute (OOM-killed at 8 under concurrent load)
CHUNK_SIZE, OVERLAP = 1000, 100
ANSWER_MAXTOK = 2048

SYS = ("You answer a question about a movie using ONLY the screenplay excerpts "
       "provided. Answer in 10 words or fewer. If the answer is not in the "
       "excerpts, return exactly: not found")

# Same judge prompt the bench's shared/eval.py uses, so calibration and the
# final score agree.
JUDGE_PROMPT = """You are an answer-equivalence judge. Given a question, the gold answer(s), and a predicted answer, decide if the prediction is essentially correct (conveys the same core fact, even if worded differently).

Question: {q}
Ground truth answer(s): {gold}
Predicted answer: {pred}

Reply with a single line: "correct" or "incorrect", then a one-line reason."""


# ------------------------------------------- group knapsack (from Doctopus) ---
def group_knap_sack(n, m, items):
    """Verbatim from doctopus/Doctopus/main.py. items: (group, cost, value).
    Returns (max_value, selected_items, selected_strategies) where each
    selected_strategy is (group, index_within_group)."""
    items.sort(key=lambda x: x[0])
    groups, current_group = [], None
    for item in items:
        g, c, w = item
        if g != current_group:
            groups.append([]); current_group = g
        groups[-1].append((c, w, g))
    t = len(groups)
    dp = [[-float('inf')] * (m + 1) for _ in range(t + 1)]
    dp[0][0] = 0
    choice = [[[] for _ in range(m + 1)] for _ in range(t + 1)]
    strategy_choice = [[[] for _ in range(m + 1)] for _ in range(t + 1)]
    for k in range(1, t + 1):
        group = groups[k - 1]
        for j in range(m + 1):
            max_val, best_prev_j, best_item, best_strategy = -float('inf'), -1, None, None
            for idx, (c, w, g) in enumerate(group):
                if j >= c and dp[k - 1][j - c] + w > max_val:
                    max_val = dp[k - 1][j - c] + w
                    best_prev_j, best_item, best_strategy = j - c, (g, c, w), (g, idx)
            if max_val != -float('inf'):
                dp[k][j] = max_val
                choice[k][j] = choice[k - 1][best_prev_j].copy(); choice[k][j].append(best_item)
                strategy_choice[k][j] = strategy_choice[k - 1][best_prev_j].copy()
                strategy_choice[k][j].append(best_strategy)
    max_value = max(dp[t])
    if max_value == -float('inf'):
        return 0, [], []
    mi = dp[t].index(max_value)
    return max_value, choice[t][mi], strategy_choice[t][mi]


# ------------------------------------------------------------ build index ----
def chunk_tokens(text, size=CHUNK_SIZE, overlap=OVERLAP):
    toks = ENC.encode(text); chunks, i = [], 0
    while i < len(toks):
        chunks.append(ENC.decode(toks[i:i + size]))
        if i + size >= len(toks):
            break
        i += size - overlap
    return chunks


def embed_batch(texts, bucket):
    vecs, tot = [], 0
    for i in range(0, len(texts), 64):
        v, t = ac.embed(texts[i:i + 64], bucket=bucket); vecs.extend(v); tot += t
    return np.array(vecs, dtype=np.float32), tot


def build_movie(slug, text, build_costs):
    cache = EMB / f"{slug}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["chunks"]), d["vecs"]
    chunks = chunk_tokens(text)
    vecs, tot = embed_batch(chunks, bucket="doctopus_build")
    np.savez(cache, chunks=np.array(chunks, dtype=object), vecs=vecs)
    build_costs[slug] = {"build_tokens_total": tot, "n_chunks": len(chunks)}
    BUILD_COSTS.write_text(json.dumps(build_costs, indent=2))
    print(f"  built {slug}: {len(chunks)} chunks, {tot:,} embed tokens", flush=True)
    return chunks, vecs


def topk(vecs, qv, k):
    sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-sims)
    return order[:k], sims[order[0]] if len(order) else 0.0


# --------------------------------------------------------- disk-cached LLM ---
_jcache, _acache = {}, {}


def _load_cache(path, d):
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); d[r["key"]] = r["val"]


_load_cache(CACHE / "answers.jsonl", _acache)
_load_cache(CACHE / "judge.jsonl", _jcache)
_afh = open(CACHE / "answers.jsonl", "a", encoding="utf-8")
_jfh = open(CACHE / "judge.jsonl", "a", encoding="utf-8")


def _persist(fh, key, val):
    fh.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n"); fh.flush()


_SENT = re.compile(r"(?<=[.!?])\s+")


def run_strategy(idx, name, k, question):
    """Execute a strategy. Returns dict: prediction, input_tokens, output_tokens, conf."""
    key = f"{idx['qid']}::{name}"
    if key in _acache:
        return _acache[key]
    chunks, vecs, qv = idx["chunks"], idx["vecs"], idx["qv"]
    if k == 0:  # extractive: no LLM, best sentence of top chunk
        order, conf = topk(vecs, qv, 1)
        top = chunks[int(order[0])] if len(order) else ""
        sents = [s.strip() for s in _SENT.split(top) if s.strip()] or [top]
        sv, _ = ac.embed(sents, bucket="doctopus_build")
        sv = np.array(sv, dtype=np.float32)
        s = sv @ qv / (np.linalg.norm(sv, axis=1) * np.linalg.norm(qv) + 1e-9)
        pred = sents[int(np.argmax(s))][:200]
        out = {"prediction": pred, "input_tokens": 0, "output_tokens": 0, "conf": float(conf)}
    else:
        order, conf = topk(vecs, qv, k)
        ctx = "\n\n---\n\n".join(chunks[int(i)] for i in order)
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"EXCERPTS:\n{ctx}\n\nQUESTION: {question}\nANSWER:"}]
        try:
            pred, pt, ct = ac.chat(msgs, max_tokens=ANSWER_MAXTOK, bucket="doctopus_answer")
        except Exception as e:
            pred, pt, ct = f"[ERROR: {repr(e)[:80]}]", 0, 0
        out = {"prediction": pred.strip(), "input_tokens": pt, "output_tokens": ct, "conf": float(conf)}
    _acache[key] = out; _persist(_afh, key, out)
    return out


def judge(qid, question, gold_answers, prediction):
    key = f"{qid}::{prediction[:160]}"
    if key in _jcache:
        return _jcache[key]
    gold = " | ".join(gold_answers)
    if not prediction or prediction.lower() in ("not found", "") or prediction.startswith("[ERROR"):
        v = 0
    else:
        try:
            jt, _, _ = ac.chat([{"role": "user",
                                 "content": JUDGE_PROMPT.format(q=question, gold=gold, pred=prediction)}],
                               max_tokens=2048, bucket="judge")
        except Exception:
            jt = ""  # never let a single judge call crash the whole calibration
        line0 = jt.strip().split("\n")[0].lower()
        v = 1 if ("incorrect" not in line0 and "correct" in line0) else 0
    _jcache[key] = v; _persist(_jfh, key, v)
    return v


def est_cost_tokens(idx, k):
    """Deterministic, pre-computable strategy cost = LLM input tokens (extractive=0)."""
    if k == 0:
        return 0
    chunks, vecs, qv = idx["chunks"], idx["vecs"], idx["qv"]
    order, _ = topk(vecs, qv, k)
    ctx = "\n\n---\n\n".join(chunks[int(i)] for i in order)
    msg = f"{SYS}\nEXCERPTS:\n{ctx}\n\nQUESTION: {idx['question']}\nANSWER:"
    return len(ENC.encode(msg))


# ------------------------------------------------------------------- main ----
def main():
    movies = {p.stem: p.read_text(encoding="utf-8")
              for p in (BENCH / "data"/_DOCDIR).glob("*.txt") if p.stem in MOVIES}
    questions = [json.loads(l) for l in (BENCH / "data/questions"/_QFILE).read_text().splitlines()
                 if l.strip()]
    questions = [q for q in questions if q["movie"] in MOVIES]
    build_costs = json.loads(BUILD_COSTS.read_text()) if BUILD_COSTS.exists() else {}

    print(f"== build ({len(movies)} movies) ==", flush=True)
    with time_build("doctopus", n_total=len(movies)) as _bt:
        index = {slug: build_movie(slug, text, build_costs) for slug, text in movies.items()}
        _bt.built(len(index))

    # per-question retrieval context (embed each question once)
    qvs, _ = embed_batch([q["question"] for q in questions], bucket="doctopus_build")
    idxs = []
    for q, qv in zip(questions, qvs):
        chunks, vecs = index[q["movie"]]
        idxs.append({"qid": q["qid"], "movie": q["movie"], "question": q["question"],
                     "gold": q["gold_answers"], "chunks": chunks, "vecs": vecs, "qv": qv})

    # deterministic per-movie val/test split
    val, test = [], []
    by_movie = {}
    for ix in idxs:
        by_movie.setdefault(ix["movie"], []).append(ix)
    for slug, lst in by_movie.items():
        lst.sort(key=lambda x: x["qid"])
        nval = max(1, int(round(len(lst) * VAL_FRAC)))
        val += lst[:nval]; test += lst[nval:]
    print(f"== calibrate (val={len(val)}, test={len(test)}) ==", flush=True)

    # --- calibration: run every strategy on val, judge, record (conf, verdict) ---
    def cal_one(args):
        ix, (name, k) = args
        r = run_strategy(ix, name, k, ix["question"])
        v = judge(ix["qid"], ix["question"], ix["gold"], r["prediction"])
        return (name, r["conf"], v)
    tasks = [(ix, s) for ix in val for s in STRATEGIES]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        cal = list(ex.map(cal_one, tasks))

    # bucket edges from val retrieval confidence
    confs = sorted(c for _, c, _ in cal)
    edges = [confs[int(len(confs) * (b + 1) / N_BUCKETS) - 1] for b in range(N_BUCKETS)] if confs else [1.0]

    def bucket(conf):
        for b, e in enumerate(edges):
            if conf <= e:
                return b
        return N_BUCKETS - 1

    # P(correct | strategy, bucket) with global-per-strategy fallback
    from collections import defaultdict
    cell = defaultdict(lambda: [0, 0]); glob = defaultdict(lambda: [0, 0])
    for name, c, v in cal:
        cell[(name, bucket(c))][0] += v; cell[(name, bucket(c))][1] += 1
        glob[name][0] += v; glob[name][1] += 1

    def pcorrect(name, conf):
        s, n = cell[(name, bucket(conf))]
        if n:
            return s / n
        s, n = glob[name]
        return s / n if n else 0.0
    print("  P(correct) per strategy (val):",
          {nm: round(glob[nm][0] / glob[nm][1], 3) if glob[nm][1] else 0 for nm, _ in STRATEGIES}, flush=True)

    # --- per-test-question strategy table: (cost_tokens, predicted p) ---
    table = []  # one entry per test question: list of (name, k, cost, p)
    for ix in test:
        row = []
        for name, k in STRATEGIES:
            cost = est_cost_tokens(ix, k)
            conf = topk(ix["vecs"], ix["qv"], max(k, 1))[1]
            row.append((name, k, cost, pcorrect(name, conf)))
        table.append(row)

    # --- budget sweep via group knapsack ---
    max_budget = sum(max(c for _, _, c, _ in row) for row in table)
    budgets = [int(max_budget * (b + 1) / BUDGET_POINTS) for b in range(BUDGET_POINTS)]
    print(f"== optimise + execute ({BUDGET_POINTS} budgets, max={max_budget:,} tok) ==", flush=True)

    curve = []
    chosen_at = {}  # budget -> {qid: (name,k)}
    for budget in budgets:
        items = []
        for gi, row in enumerate(table):
            for (name, k, cost, p) in row:
                items.append((gi + 1, cost // COST_DIVISOR, p))
        _, _, sel = group_knap_sack(len(items), budget // COST_DIVISOR, items)
        chosen = {}
        for (g, within) in sel:
            ix = test[g - 1]; name, k, cost, p = table[g - 1][within]
            chosen[ix["qid"]] = (name, k)
        chosen_at[budget] = chosen

    # execute (cached) every distinct (qid, strategy) the sweep selected, then judge
    need = {(qid, nm, k) for ch in chosen_at.values() for qid, (nm, k) in ch.items()}
    ixby = {ix["qid"]: ix for ix in test}

    def exec_one(t):
        qid, nm, k = t
        ix = ixby[qid]
        r = run_strategy(ix, nm, k, ix["question"])
        v = judge(qid, ix["question"], ix["gold"], r["prediction"])
        return (qid, nm), {**r, "verdict": v}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        execd = dict(ex.map(exec_one, need))

    # curve points
    qcount = {m: sum(1 for ix in test if ix["movie"] == m) for m in MOVIES}
    for budget in budgets:
        ch = chosen_at[budget]
        acc = used = correct = 0
        for ix in test:
            nm, k = ch.get(ix["qid"], ("extractive", 0))
            r = execd[(ix["qid"], nm)]
            used += r["input_tokens"] + r["output_tokens"]; correct += r["verdict"]
        acc = correct / len(test) if test else 0
        mix = {}
        for nm, k in ch.values():
            mix[nm] = mix.get(nm, 0) + 1
        curve.append({"budget_tokens": budget, "judge_accuracy": round(acc, 4),
                      "answer_tokens_used": used, "n_correct": correct, "n_test": len(test),
                      "strategy_mix": mix})
        print(f"  budget {budget:>9,} -> acc {acc:.3f}  tok {used:>8,}  mix {mix}", flush=True)

    (OUTDIR / "curve.json").write_text(json.dumps(curve, indent=2))
    with open(OUTDIR / "curve.csv", "w") as f:
        f.write("budget_tokens,judge_accuracy,answer_tokens_used,n_correct,n_test\n")
        for c in curve:
            f.write(f"{c['budget_tokens']},{c['judge_accuracy']},{c['answer_tokens_used']},"
                    f"{c['n_correct']},{c['n_test']}\n")

    # operating point = knee: cheapest budget within 1% of peak accuracy
    peak = max(c["judge_accuracy"] for c in curve)
    best = min((c for c in curve if c["judge_accuracy"] >= peak - 0.01),
               key=lambda c: c["answer_tokens_used"])
    ch = chosen_at[best["budget_tokens"]]
    with open(RESULTS / f"{METHOD}{_SFX}.jsonl", "w", encoding="utf-8") as f:
        for ix in test:
            nm, k = ch.get(ix["qid"], ("extractive", 0))
            r = execd[(ix["qid"], nm)]
            bt = build_costs.get(ix["movie"], {}).get("build_tokens_total", 0)
            amort = round(bt / max(1, qcount.get(ix["movie"], 1)))
            f.write(json.dumps({
                "method": METHOD, "movie": ix["movie"], "qid": ix["qid"], "question": ix["question"],
                "gold_answers": ix["gold"], "prediction": r["prediction"], "source": nm,
                "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                "total_tokens": r["input_tokens"] + r["output_tokens"],
                "build_tokens": amort}, ensure_ascii=False) + "\n")

    print(f"\nDONE {METHOD}: curve -> {OUTDIR/'curve.csv'}", flush=True)
    print(f"operating point: budget {best['budget_tokens']:,} acc {best['judge_accuracy']} "
          f"mix {best['strategy_mix']}", flush=True)
    print(f"results -> {RESULTS/(METHOD+'.jsonl')}  (now run: shared/eval.py {METHOD})", flush=True)

    # single-thread per-question latency: re-run each sampled question's chosen
    # strategy FRESH (uncached) at concurrency 1, since the exec above was threaded.
    def _answer_one(ix):
        nm, k = ch.get(ix["qid"], ("extractive", 0))
        if k == 0:
            order, _ = topk(ix["vecs"], ix["qv"], 1)
            top = ix["chunks"][int(order[0])] if len(order) else ""
            sents = [s.strip() for s in _SENT.split(top) if s.strip()] or [top]
            ac.embed(sents, bucket="doctopus_answer")
        else:
            order, _ = topk(ix["vecs"], ix["qv"], k)
            ctx = "\n\n---\n\n".join(ix["chunks"][int(i)] for i in order)
            ac.chat([{"role": "system", "content": SYS},
                     {"role": "user", "content": f"EXCERPTS:\n{ctx}\n\nQUESTION: {ix['question']}\nANSWER:"}],
                    max_tokens=ANSWER_MAXTOK, bucket="doctopus_answer")
    measure_latency(METHOD, _answer_one, test)


if __name__ == "__main__":
    main()
