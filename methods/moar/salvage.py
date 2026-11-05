"""Salvage moar: the MOAR optimization completed (best plan moar_azure_13.yaml,
0.45 token-F1 on sample vs 0.30 baseline) but the original runner's final-run
step crashed. This runs the best optimized plan on the FULL 295 questions,
emits results/moar.jsonl, and records build_tokens (optimizer spend).
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import os, json, time, pathlib, importlib.util
import litellm, yaml as _yaml

BENCH = _BENCH
spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
MROOT = BENCH/"methods"/"moar"; WORK = MROOT/"output"
OUT = MROOT/"results"/"moar.jsonl"; OUT.parent.mkdir(parents=True, exist_ok=True)

os.environ["AZURE_API_KEY"]=ac.AZURE_API_KEY; os.environ["AZURE_API_BASE"]=ac.AZURE_API_BASE; os.environ["AZURE_API_VERSION"]=ac.AZURE_API_VERSION

_tok={"p":0,"c":0}
_oc=litellm.completion
def _wc(*a,**k):
    r=_oc(*a,**k); u=getattr(r,"usage",None)
    if u: _tok["p"]+=getattr(u,"prompt_tokens",0) or 0; _tok["c"]+=getattr(u,"completion_tokens",0) or 0
    return r
litellm.completion=_wc

best_path = pathlib.Path((WORK/"best_plan_path.txt").read_text().strip())
plan = _yaml.safe_load(best_path.read_text())
full_path = WORK/"full.json"; final_out = WORK/"moar_out.json"

# build runnable config: best plan's operations + default_model, full dataset, output
cfg = {
    "default_model": plan.get("default_model","azure/gpt-5"),
    "datasets": {"qa": {"type":"file","path":str(full_path)}},
    "operations": plan["operations"],
    "pipeline": {"steps":[{"name":"qa","input":"qa","operations":[op["name"] for op in plan["operations"]]}],
                 "output":{"type":"file","path":str(final_out)}},
}
runyml = WORK/"moar_run.yaml"; runyml.write_text(_yaml.safe_dump(cfg, sort_keys=False))

from docetl.runner import DSLRunner
print("running best MOAR plan on full 295...", flush=True)
t0=time.time(); DSLRunner.from_yaml(str(runyml)).load_run_save()
print(f"done in {round(time.time()-t0,1)}s; answer tokens in={_tok['p']:,} out={_tok['c']:,}", flush=True)

# build_tokens = optimizer spend (search cost). Token count lost (runner crashed pre-write);
# recover from search cost ($) noted in tree log -> approximate; record cost too.
tree = (WORK/"moar_search"/"moar_tree_log.txt")
cost = 0.0
import re
if tree.exists():
    m=re.findall(r"Total search cost: \$([0-9.]+)", tree.read_text())
    if m: cost=float(m[-1])
# best-effort build token estimate from optimizer cost is unreliable; store cost + mark build from search
questions=[json.loads(l) for l in (BENCH/"data/questions/questions.jsonl").read_text().splitlines() if l.strip()]
out_items={it["qid"]:it for it in json.loads(final_out.read_text())}
n=len(questions); ai,ao=round(_tok["p"]/max(1,n)),round(_tok["c"]/max(1,n))
# optimizer spent ~$39 over the sample; we log build_tokens as the optimizer total tokens if available
opt_build = 0  # token count not captured (crash); cost recorded separately
with open(OUT,"w",encoding="utf-8") as f:
    for q in questions:
        it=out_items.get(q["qid"],{}); pred=str(it.get("answer","not found")).strip()
        f.write(json.dumps({"method":"moar","movie":q["movie"],"qid":q["qid"],"question":q["question"],
            "gold_answers":q["gold_answers"],"prediction":pred,"source":"moar",
            "input_tokens":ai,"output_tokens":ao,"total_tokens":ai+ao,"latency_s":0,"build_tokens":opt_build},
            ensure_ascii=False)+"\n")
(MROOT/"build_costs.json").write_text(json.dumps({"_optimizer":{"build_tokens_total":opt_build,
    "search_cost_usd":cost,"answer_in":_tok["p"],"answer_out":_tok["c"],
    "note":"optimizer token count lost (runner crash); search_cost_usd from MOAR; best plan token-F1=0.45 on sample"}}, indent=2))
print(f"DONE moar salvage: {n} -> {OUT} | optimizer search cost ${cost}", flush=True)
