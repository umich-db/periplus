"""moar — DocETL MOAR optimizer (ucbepic.github.io/docetl/optimization/moar/)
applied to the docetl_standard map pipeline, reduced settings:
  sample_size=20, max_iterations=20, models=['azure/gpt-5'] (only deployed model
  on umgpt -> the "5 models" search dimension collapses to 1; noted as deviation).

Optimizer token spend -> build_tokens. Optimized-pipeline answers -> answer tokens.
Token capture via litellm monkeypatch (build during optimize, answer during final run).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os, json, time, pathlib, importlib.util, re
import litellm, yaml as _yaml

BENCH = _BENCH
import sys as _sys; _sys.path.insert(0, str(BENCH / "shared"))
from corpus import DOCS as _DOCS  # noqa: E402
_CORPUS_KEY = "cuad" if __import__("os").environ.get("CUAD") else "nqa"

spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
import sys; sys.path.insert(0, str(BENCH/"shared"))
from buildtimer import write as _write_build
from latencyprobe import measure as measure_latency

METHOD = "moar"; MROOT = BENCH/"methods"/METHOD
WORK = MROOT/"output"; WORK.mkdir(parents=True, exist_ok=True)
OUT = MROOT/"results"/f"{METHOD}.jsonl"; OUT.parent.mkdir(parents=True, exist_ok=True)
(MROOT/"logs").mkdir(exist_ok=True)

# Provider chosen from the endpoint's shape, and the model taken from .env,
# for the same reason as methods/docetl_standard/run.py: litellm's azure/
# provider authenticates the Azure way, while the current umgpt endpoint is an
# OpenAI-compatible Portkey gateway wanting Authorization: Bearer.
_BASE = (ac.AZURE_API_BASE or "").rstrip("/")
if _BASE.endswith("/v1") or "openai.azure.com" not in _BASE:
    os.environ["OPENAI_API_KEY"] = ac.AZURE_API_KEY
    os.environ["OPENAI_BASE_URL"] = _BASE
    os.environ["OPENAI_API_BASE"] = _BASE
    MOAR_MODEL = f"openai/{ac.LLM_MODEL}"
else:
    os.environ["AZURE_API_KEY"] = ac.AZURE_API_KEY
    os.environ["AZURE_API_BASE"] = _BASE
    os.environ["AZURE_API_VERSION"] = ac.AZURE_API_VERSION
    MOAR_MODEL = f"azure/{ac.LLM_MODEL}"
print(f"[moar] model={MOAR_MODEL} via {_BASE}", flush=True)

# litellm token capture (bucket switches build->answer)
_tok = {"p": 0, "c": 0}; _phase = {"v": "build"}
_buckets = {"build": {"p": 0, "c": 0}, "answer": {"p": 0, "c": 0}}
_oc = litellm.completion
def _wc(*a, **k):
    r = _oc(*a, **k); u = getattr(r, "usage", None)
    if u:
        b = _buckets[_phase["v"]]
        b["p"] += getattr(u, "prompt_tokens", 0) or 0; b["c"] += getattr(u, "completion_tokens", 0) or 0
    return r
litellm.completion = _wc

PROMPT = """You answer a question about a movie using ONLY the screenplay below. \
Answer in 10 words or fewer. If the answer is not in the screenplay, return exactly: not found

SCREENPLAY:
{{ input.document }}

QUESTION: {{ input.question }}
ANSWER:"""

def _tokset(t): return set(re.findall(r"[a-z0-9']+", str(t).lower()))
def _f1(p, g):
    P, G = _tokset(p), _tokset(g)
    if not P or not G: return 0.0
    tp = len(P & G); pr, rc = tp/len(P), tp/len(G)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

def eval_fn(results_path):
    """Score optimizer candidate: avg best-token-F1 of 'answer' vs gold_answers."""
    items = json.loads(pathlib.Path(results_path).read_text())
    if not items: return {"score": 0.0}
    s = sum(max((_f1(it.get("answer", ""), g) for g in it.get("gold_answers", [])), default=0.0) for it in items)
    return {"score": s/len(items)}

def base_cfg(dataset_path, out_path):
    return {
        "default_model": MOAR_MODEL,
        "datasets": {"qa": {"type": "file", "path": str(dataset_path)}},
        "operations": [{"name": "answer_question", "type": "map", "prompt": PROMPT,
            "output": {"schema": {"answer": "str"}, "mode": "structured_output"},
            "litellm_completion_kwargs": {"max_completion_tokens": 2048, "drop_params": True}}],
        "pipeline": {"steps": [{"name": "qa", "input": "qa", "operations": ["answer_question"]}],
                     "output": {"type": "file", "path": str(out_path)}},
    }

def main():
    movies = {p.stem: p.read_text(encoding="utf-8") for p in (BENCH/"data/movies/txt").glob("*.txt")}
    questions = list(_DOCS[_CORPUS_KEY].questions)
    items = [{"qid": q["qid"], "movie": q["movie"], "question": q["question"],
              "document": movies[q["movie"]], "gold_answers": q["gold_answers"]} for q in questions]
    # sample 20 (stratified-ish: every ~15th) for the optimizer
    sample = items[::max(1, len(items)//20)][:20]
    samp_path = WORK/"sample20.json"; samp_path.write_text(json.dumps(sample, ensure_ascii=False))
    full_path = WORK/"full.json"; full_path.write_text(json.dumps(items, ensure_ascii=False))
    opt_out = WORK/"opt_out.json"; final_out = WORK/"moar_out.json"

    from docetl.api import Pipeline
    import docetl.api as dapi
    cfg = base_cfg(samp_path, opt_out)
    yml = WORK/"base.yaml"; yml.write_text(_yaml.safe_dump(cfg, sort_keys=False))

    from docetl.runner import DSLRunner
    # MOAR optimize on the 20-sample (build phase)
    _phase["v"] = "build"; t0 = time.time(); _build_t0 = t0
    pipe = Pipeline.from_yaml(str(yml)) if hasattr(Pipeline, "from_yaml") else None
    if pipe is None:
        # construct Pipeline object from config dict
        from docetl.api import Dataset, MapOp, PipelineStep, PipelineOutput
        pipe = Pipeline(name="moar", datasets={"qa": Dataset(type="file", path=str(samp_path))},
                        operations=[MapOp(name="answer_question", type="map", prompt=PROMPT,
                                          output={"schema": {"answer": "str"}},
                                          litellm_completion_kwargs={"max_completion_tokens": 2048, "drop_params": True})],
                        steps=[PipelineStep(name="qa", input="qa", operations=["answer_question"])],
                        output=PipelineOutput(type="file", path=str(opt_out)),
                        default_model=MOAR_MODEL)
    result = pipe.optimize(method="moar", eval_fn=eval_fn, metric_key="score",
                           models=[MOAR_MODEL], agent_model=MOAR_MODEL,
                           max_iterations=20, dataset_path=str(samp_path), save_dir=str(WORK/"moar_search"))
    build_p, build_c = _buckets["build"]["p"], _buckets["build"]["c"]
    _write_build("moar", (time.time()-_build_t0)/60, coverage="optimizer")
    print(f"MOAR optimize done in {round(time.time()-t0,1)}s; build tokens in={build_p:,} out={build_c:,}", flush=True)

    best = result.best()
    print(f"best plan accuracy={getattr(best,'accuracy',None)}", flush=True)
    opt_pipe = best.pipeline
    # point optimized pipeline at FULL dataset + final output, run (answer phase)
    opt_pipe.datasets = {"qa": __import__("docetl.api", fromlist=["Dataset"]).Dataset(type="file", path=str(full_path))}
    opt_pipe.output.path = str(final_out)
    _phase["v"] = "answer"
    opt_pipe.run()
    ans_p, ans_c = _buckets["answer"]["p"], _buckets["answer"]["c"]

    out_items = {it["qid"]: it for it in json.loads(final_out.read_text())}
    n = len(questions); amort_bt = round((build_p+build_c)/max(1, n))
    ai, ao = round(ans_p/max(1, n)), round(ans_c/max(1, n))
    with open(OUT, "w", encoding="utf-8") as f:
        for q in questions:
            it = out_items.get(q["qid"], {})
            pred = str(it.get("answer", "not found")).strip()
            f.write(json.dumps({"method": METHOD, "movie": q["movie"], "qid": q["qid"], "question": q["question"],
                "gold_answers": q["gold_answers"], "prediction": pred, "source": "moar",
                "input_tokens": ai, "output_tokens": ao, "total_tokens": ai+ao,
                "latency_s": 0, "build_tokens": amort_bt}, ensure_ascii=False)+"\n")
    (MROOT/"build_costs.json").write_text(json.dumps({"_optimizer": {"build_tokens_total": build_p+build_c,
        "build_in": build_p, "build_out": build_c, "answer_in": ans_p, "answer_out": ans_c}}, indent=2))
    print(f"DONE {METHOD}: {n} -> {OUT} | build={build_p+build_c:,} answer={ans_p+ans_c:,}", flush=True)

    # single-thread per-question latency: time the OPTIMIZED plan's answer call per
    # question (the opt_pipe.run() above answered all questions in one batch).
    from jinja2 import Template
    try:
        _ops = [o for o in opt_pipe.operations if getattr(o, "type", None) == "map"] or list(opt_pipe.operations)
        _aop = _ops[-1]
        _aprompt, _akw = _aop.prompt, dict(getattr(_aop, "litellm_completion_kwargs", None) or {})
        _amodel = getattr(opt_pipe, "default_model", None) or MOAR_MODEL
    except Exception:  # noqa: BLE001 -- fall back to the base map if the plan shape differs
        _aprompt, _akw, _amodel = PROMPT, {"max_completion_tokens": 2048, "drop_params": True}, MOAR_MODEL
    def _answer_one(it):
        litellm.completion(model=_amodel, messages=[{"role": "user", "content": Template(_aprompt).render(input=it)}], **_akw)
    measure_latency(METHOD, _answer_one, items)

if __name__ == "__main__":
    main()
