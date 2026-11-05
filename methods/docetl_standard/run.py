"""docetl_standard — DocETL standard pipeline: a `map` operation that answers
each question over its full screenplay, via the model in .env (litellm,
default reasoning). Matches the paper's DocETL (full-doc, ~12.7M tokens).

Token capture via litellm monkeypatch. build_tokens=0 (stateless map).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os, json, time, pathlib, importlib.util
import litellm
import yaml as _yaml

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
import sys; sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import write as _write_build
from latencyprobe import measure as measure_latency
_write_build("docetl_standard", 0.0, coverage="none")  # no index build (single map pass)

_CUAD = bool(os.environ.get("CUAD"))
_DOCDIR = "contracts/txt" if _CUAD else "movies/txt"
_QFILE = "cuad_questions.jsonl" if _CUAD else "questions_30.jsonl"
_SFX = "_cuad" if _CUAD else ""
METHOD = "docetl_standard"
MROOT = BENCH/"methods"/METHOD
WORK = MROOT/"output"; WORK.mkdir(parents=True, exist_ok=True)
OUT = MROOT/"results"/f"{METHOD}{_SFX}.jsonl"; OUT.parent.mkdir(parents=True, exist_ok=True)

# --- creds into env for litellm, matching the endpoint's SHAPE ---------------
#
# The model was pinned to "azure/gpt-5" here. Two things break when the benchmark
# moves off that: the model no longer follows .env, and litellm's azure/ provider
# authenticates the Azure way (api-key header, deployment in the path,
# api-version) while the current umgpt endpoint is an OpenAI-compatible Portkey
# gateway that wants Authorization: Bearer. That mismatch is an auth error on
# every call, and because this runner writes "not found" when a call fails, the
# result was 885 rows, 0 tokens and a run that looked complete.
#
# So pick the provider from the URL, exactly as shared/azure_client.py does, and
# take the model from the shared config instead of hardcoding it.
_BASE = (ac.AZURE_API_BASE or "").rstrip("/")
_OPENAI_SHAPED = _BASE.endswith("/v1") or "openai.azure.com" not in _BASE

if _OPENAI_SHAPED:
    os.environ["OPENAI_API_KEY"] = ac.AZURE_API_KEY
    os.environ["OPENAI_BASE_URL"] = _BASE
    os.environ["OPENAI_API_BASE"] = _BASE
    MODEL = f"openai/{ac.LLM_MODEL}"
else:
    os.environ["AZURE_API_KEY"] = ac.AZURE_API_KEY
    os.environ["AZURE_API_BASE"] = _BASE
    os.environ["AZURE_API_VERSION"] = ac.AZURE_API_VERSION
    MODEL = f"azure/{ac.LLM_MODEL}"
print(f"[docetl] model={MODEL} via {_BASE}", flush=True)

# --- litellm token capture, and retry on the gateway's transient 404 ---------
#
# The umgpt gateway drops a large fraction of requests with a 404 "Resource not
# found" (37-42% measured on 2026-09-23). shared/azure_client.py retries those,
# but DocETL does not go through it -- it calls litellm directly, and litellm
# raises NotFoundError straight out, which kills the whole pipeline run rather
# than one question. The token-capture wrapper already sits on every call, so
# the retry goes here, with the same policy: short capped backoff, because this
# is gateway routing rather than backpressure.
_tok = {"p": 0, "c": 0}
_oc = litellm.completion
_RETRIES, _DROPPED = 12, {"n": 0}

def _wc(*a, **k):
    last = None
    for attempt in range(_RETRIES):
        try:
            r = _oc(*a, **k)
            u = getattr(r, "usage", None)
            if u:
                _tok["p"] += getattr(u, "prompt_tokens", 0) or 0
                _tok["c"] += getattr(u, "completion_tokens", 0) or 0
            return r
        except Exception as e:                        # noqa: BLE001
            last = e
            msg = str(e).lower()
            if not ("404" in msg or "resource not found" in msg):
                raise
            time.sleep(min(0.4 * (2 ** attempt), 5.0))
    _DROPPED["n"] += 1
    raise last

litellm.completion = _wc

PROMPT = """You answer a question about a movie using ONLY the screenplay below. \
Answer in 10 words or fewer. If the answer is not in the screenplay, return exactly: not found

SCREENPLAY:
{{ input.document }}

QUESTION: {{ input.question }}
ANSWER:"""

# gpt-6-astra on this gateway refuses ANY request that carries function tools,
# whatever reasoning_effort is set to or whether it is set at all -- the error
# text ("set reasoning_effort to 'none'") is misleading, and 'none' fails too.
# DocETL's default output mode ("tools") sends a function schema for its output,
# so every call was rejected. Its other mode, "structured_output", asks for the
# same schema through response_format json_schema, which the gateway accepts and
# which returns clean JSON. The schema, prompt and parse are otherwise the same,
# so the method is unchanged; only the transport for the schema differs. No
# reasoning_effort is sent, so the model's default reasoning applies, as it does
# for every other method.

def build_pipeline_yaml(dataset_path, out_path):
    cfg = {
        "default_model": MODEL,
        "datasets": {"qa": {"type": "file", "path": str(dataset_path)}},
        "operations": [{
            "name": "answer_question", "type": "map", "prompt": PROMPT,
            "output": {"schema": {"answer": "str"}, "mode": "structured_output"},
            "litellm_completion_kwargs": {"max_completion_tokens": 2048, "drop_params": True},
        }],
        "pipeline": {
            "steps": [{"name": "qa", "input": "qa", "operations": ["answer_question"]}],
            "output": {"type": "file", "path": str(out_path)},
        },
    }
    return cfg

def main():
    # dataset: one item per question, full screenplay attached.
    #
    # This used to map a hardcoded list of twenty movies and splice in rows for
    # the other ten from an earlier run, to avoid re-paying for them. In a fresh
    # run tree there is nothing to splice: the ten movies fall through to
    # `by_qid.get(qid, {})`, and every one of their 295 questions is written as
    # "not found" -- a third of the corpus silently scored as a miss.
    #
    # The intent generalises. Map whatever this output file does not already
    # hold, which is everything on a clean run and only the gaps on a resume,
    # and does not depend on which movies happened to be new in mid-2026.
    movies = {p.stem: p.read_text(encoding="utf-8") for p in (BENCH/"data"/_DOCDIR).glob("*.txt")}
    all_questions = [json.loads(l) for l in (BENCH/"data/questions"/_QFILE).read_text().splitlines() if l.strip()]
    preserved = {}
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                # Only a real answer counts as done; a "not found" from a failed
                # earlier run must be re-mapped, not preserved.
                if int(r.get("total_tokens") or 0) > 0:
                    preserved[r["qid"]] = r
    questions = [q for q in all_questions if q["qid"] not in preserved]
    print(f"[docetl] mapping {len(questions)}/{len(all_questions)} questions "
          f"({len(preserved)} reused)", flush=True)
    items = [{"qid": q["qid"], "movie": q["movie"], "question": q["question"],
              "document": movies[q["movie"]], "gold_answers": q["gold_answers"]} for q in questions]
    ds_path = WORK/"dataset.json"; ds_path.write_text(json.dumps(items, ensure_ascii=False))
    out_path = WORK/"docetl_out.json"
    cfg = build_pipeline_yaml(ds_path, out_path)
    yml = WORK/"pipeline.yaml"; yml.write_text(_yaml.safe_dump(cfg, sort_keys=False))

    from docetl.runner import DSLRunner
    t0 = time.time()
    runner = DSLRunner.from_yaml(str(yml))
    cost = runner.load_run_save()
    dt = round(time.time()-t0, 1)
    print(f"docetl run cost=${cost} time={dt}s tokens in={_tok['p']:,} out={_tok['c']:,}", flush=True)

    # read output, join back to questions, amortize tokens evenly
    out_items = json.loads(out_path.read_text())
    by_qid = {it["qid"]: it for it in out_items}
    n = len(questions)
    amort_in = round(_tok["p"]/max(1, n)); amort_out = round(_tok["c"]/max(1, n))
    # write ALL 30 movies: preserved old-10 rows verbatim + freshly-mapped new-20 rows
    with open(OUT, "w", encoding="utf-8") as f:
        for q in all_questions:
            if q["qid"] in preserved:
                f.write(json.dumps(preserved[q["qid"]], ensure_ascii=False)+"\n"); continue
            it = by_qid.get(q["qid"], {})
            pred = str(it.get("answer", "not found")).strip()
            f.write(json.dumps({"method": METHOD, "movie": q["movie"], "qid": q["qid"], "question": q["question"],
                "gold_answers": q["gold_answers"], "prediction": pred, "source": "map",
                "input_tokens": amort_in, "output_tokens": amort_out, "total_tokens": amort_in+amort_out,
                "latency_s": 0, "build_tokens": 0}, ensure_ascii=False)+"\n")
    (MROOT/"build_costs.json").write_text(json.dumps({"_total": {"build_tokens_total": 0,
        "answer_in": _tok["p"], "answer_out": _tok["c"]}}, indent=2))
    print(f"DONE {METHOD}: {n} -> {OUT}", flush=True)

    # single-thread per-question latency: the map is one LLM call per question;
    # time it directly (the run above batches all questions through docetl).
    from jinja2 import Template
    def _answer_one(it):
        prompt = Template(PROMPT).render(input=it)
        litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}],
                           max_completion_tokens=2048, drop_params=True)
    measure_latency(METHOD, _answer_one, items)

if __name__ == "__main__":
    main()
