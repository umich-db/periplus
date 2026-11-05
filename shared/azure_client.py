"""Shared Azure OpenAI client for the long-document benchmark.

ALL llm calls in this benchmark go through this module. It reads credentials
from ~/index/.env (never hardcoded, never printed), returns token usage from
the response, accumulates a global token counter, and retries with backoff on
rate limits.

Each per-method venv imports this module BY PATH, e.g.:

    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "azure_client", "/home/phdonn/index/bench/shared/azure_client.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
"""
from __future__ import annotations

import os
import time
import pathlib
import threading
import copy
import json
import uuid
from collections import defaultdict

# --- load .env (the only source of credentials) ------------------------------
_ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / ".env"  # bench/.env -> ../.env


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH)
    except Exception:
        # minimal fallback parser if python-dotenv is absent
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)


_load_env()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or None
# deployment name == model param on Azure; user confirmed default "gpt-5"
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT")  # may be None until set

# Optional separate chat endpoint. The NGA cluster serves chat (fast, local, no
# embeddings); umgpt serves the embeddings. When LLM_API_BASE is set, chat goes
# there and embeddings stay on AZURE_*; unset, everything uses AZURE_* as before.
LLM_API_BASE = os.getenv("LLM_API_BASE")
LLM_API_KEY = os.getenv("LLM_API_KEY") or AZURE_API_KEY
LLM_MODEL = os.getenv("LLM_MODEL") or AZURE_DEPLOYMENT
# Reasoning models spend the completion allowance thinking before emitting any
# visible text, so a small max_tokens returns an EMPTY string rather than an
# error. Measured on the NGA cluster: at 2048 both models return "" for routing
# and qwen-moe returns "" for plain QA; 4096 is the floor for any visible answer.
# The methods here ask for 1024, which would silently yield empty answers that
# every caller reads as "not found" — a full run would look clean and score zero.
# So clamp up when talking to such an endpoint. 0 disables the clamp.
LLM_MIN_MAX_TOKENS = int(os.getenv("LLM_MIN_MAX_TOKENS", "4096" if LLM_API_BASE else "0"))

# --- global token accounting --------------------------------------------------
_lock = threading.Lock()
# bucket -> {"prompt": int, "completion": int, "calls": int}
_counters: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
_call_records: list[dict] = []


def get_call_records() -> list[dict]:
    """Copy observed calls: status=ok has usage; status=error has unknown usage.

    Error records represent failed client attempts, which can include SDK-level
    retries invisible to this wrapper. They must not be treated as zero billing.
    Reasoning tokens, when reported, are a subset of output_tokens, not additive.
    """
    with _lock:
        return copy.deepcopy(_call_records)


def _record_call(record: dict):
    record = {"record_id": uuid.uuid4().hex, **record}
    # Usage metadata only: neither messages nor generated/reasoning text are
    # recorded. Each movie process should receive its own ledger path.
    with _lock:
        _call_records.append(copy.deepcopy(record))
        ledger = os.environ.get("LLM_USAGE_LOG")
        if ledger:
            path = pathlib.Path(ledger)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()


def _usage_metadata(usage):
    """Keep numeric usage metadata without provider text or arbitrary objects."""
    if usage is None:
        return {}
    raw = (usage.model_dump() if hasattr(usage, "model_dump") else
           dict(usage) if isinstance(usage, dict) else vars(usage))

    def numeric(value):
        if isinstance(value, dict):
            return {str(k): numeric(v) for k, v in value.items()
                    if v is None or isinstance(v, (bool, int, float, dict, list))}
        if isinstance(value, list):
            return [numeric(v) for v in value
                    if v is None or isinstance(v, (bool, int, float, dict, list))]
        return value

    return numeric(raw)


def _bump(bucket: str, prompt: int, completion: int):
    with _lock:
        c = _counters[bucket]
        c["prompt"] += int(prompt or 0)
        c["completion"] += int(completion or 0)
        c["calls"] += 1


def get_counters() -> dict:
    """Return a snapshot copy of all token buckets."""
    with _lock:
        return {k: dict(v) for k, v in _counters.items()}


def reset_counters():
    with _lock:
        _counters.clear()
        _call_records.clear()


# --- client (lazy singleton) --------------------------------------------------
_client = None


def _get_client():
    """Client for whichever endpoint AZURE_API_BASE points at.

    UMich has moved between two shapes of endpoint, which authenticate
    differently, so pick by URL rather than by configuration:

      .../v1          OpenAI-compatible gateway (currently Portkey). Bearer auth,
                      plain /v1/chat/completions, model name passed straight
                      through. This is the current umgpt toolkit endpoint.
      *.openai.azure.com
                      Azure OpenAI. `api-key` header, deployment in the URL path,
                      api-version query parameter.

    Pointing the Azure client at the /v1 gateway fails with a Portkey 401 —
    the key is fine, it is just sent in the header Azure uses and the gateway
    wants Authorization: Bearer.
    """
    global _client
    if _client is None:
        if LLM_API_BASE:
            from openai import OpenAI
            _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE.rstrip("/"),
                             timeout=300.0)
            return _client
        if not (AZURE_API_KEY and AZURE_API_BASE):
            raise RuntimeError("AZURE_API_KEY / AZURE_API_BASE missing from .env")
        base = AZURE_API_BASE.rstrip("/")
        if base.endswith("/v1") or "openai.azure.com" not in base:
            from openai import OpenAI
            _client = OpenAI(
                api_key=AZURE_API_KEY,
                base_url=base,
                organization=OPENAI_ORG_ID,
                timeout=300.0,
            )
        else:
            from openai import AzureOpenAI
            _client = AzureOpenAI(
                api_key=AZURE_API_KEY,
                azure_endpoint=base,
                api_version=AZURE_API_VERSION,
                organization=OPENAI_ORG_ID,
            )
    return _client


# gpt-5 / reasoning deployments reject temperature!=1 and want
# max_completion_tokens instead of max_tokens. Our only deployment is gpt-5, so
# we initialize to the gpt-5-correct params (no temperature, max_completion_tokens)
# to avoid wasting probing 400s — the umgpt endpoint is slow to return errors.
# The adaptation logic below still flips these back if a model disagrees.
_TEMP_OK = False
_USE_MAX_COMPLETION = True

_RETRYABLE = ("rate limit", "429", "timeout", "temporarily", "overloaded",
              "520", "503", "502", "500", "connection",
              # The umgpt gateway returns a transient 404 "Resource not found"
              # under concurrency: measured 6 failures in 24 identical calls at
              # 8-way, for a model the same process had just called successfully
              # and which /v1/models lists. It is gateway routing, not a missing
              # deployment. Left unretried it raises, and every method in this
              # benchmark catches the exception and writes the error string as
              # its prediction -- so a quarter of the answers would be scored
              # wrong and the run would look complete while being badly corrupt.
              # A genuinely wrong model name still fails, just after the backoff.
              "404", "resource not found")


_REASONING_OK = True


class QuotaExhausted(RuntimeError):
    """The gateway refused the request because the API key is out of usage.

    Portkey returns 412 "API Key Usage Limit Exceeded". Nothing about the request
    is wrong and no retry will help, so this must not be absorbed: every runner
    in the benchmark catches Exception and writes an error string or "not found"
    as the prediction, and a dead key would otherwise fill a sweep with rows that
    score as zero-token misses and look complete. Measured 2026-09-23: 27 of 30
    documents "built" in seconds with zero tokens this way. Runners should let
    this propagate; rerun_all probes for it before every phase.
    """


def chat(messages, model: str | None = None, temperature: float = 0,
         max_tokens: int = 1024, bucket: str = "method", max_retries: int = 12,
         reasoning_effort: str | None = None, extra_body: dict | None = None,
         usage_context: dict | None = None):
    """Call chat completions on whichever endpoint is configured.

    Returns (text, prompt_tokens, completion_tokens). Tokens are also added to
    the named `bucket` in the global counter (use bucket="judge" for eval).

    Every model in play here is a reasoning model -- gpt-5 on umgpt, and both
    NGA models, which return their thinking in `reasoning_content`. Hidden
    reasoning counts against completion_tokens, so max_tokens has to leave
    headroom or the visible answer comes back empty: 64 tokens to an NGA model
    yields content=None because reasoning consumed the whole allowance.
    """
    global _TEMP_OK, _USE_MAX_COMPLETION, _REASONING_OK
    started = time.perf_counter()
    logical_call_id = uuid.uuid4().hex
    client = _get_client()
    dep = model or LLM_MODEL
    context = copy.deepcopy(usage_context or {})
    thinking_requested = (extra_body or {}).get("chat_template_kwargs", {}).get("enable_thinking")
    if LLM_MIN_MAX_TOKENS and max_tokens < LLM_MIN_MAX_TOKENS:
        max_tokens = LLM_MIN_MAX_TOKENS
    delay = 2.0
    last_err = None
    failed_attempts = []
    for attempt in range(max_retries):
        kwargs = {"model": dep, "messages": messages}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if _TEMP_OK and temperature is not None:
            kwargs["temperature"] = temperature
        if _REASONING_OK and reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if _USE_MAX_COMPLETION:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            pt = getattr(u, "prompt_tokens", 0)
            ct = getattr(u, "completion_tokens", 0)
            _bump(bucket, pt, ct)
            usage = _usage_metadata(u)
            details = usage.get("completion_tokens_details") or {}
            _record_call({
                "status": "ok", "logical_call_id": logical_call_id,
                "bucket": bucket, "model": dep,
                "input_tokens": pt, "output_tokens": ct,
                "total_tokens": usage.get("total_tokens", (pt or 0) + (ct or 0)),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "reasoning_content_chars": len(getattr(resp.choices[0].message, "reasoning_content", None) or ""),
                "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                "thinking_requested": thinking_requested,
                "latency_s": time.perf_counter() - started,
                "attempts": attempt + 1, "failed_attempts": list(failed_attempts),
                "usage_reported": u is not None,
                "prior_attempt_usage_unknown": bool(failed_attempts),
                "raw_usage": usage, "context": context,
            })
            return text, pt, ct
        except Exception as e:  # noqa: BLE001
            failure = {"attempt": attempt + 1, "error_type": type(e).__name__,
                       "status_code": getattr(e, "status_code", None)}
            failed_attempts.append(failure)
            _record_call({
                "status": "error", "logical_call_id": logical_call_id,
                "bucket": bucket, "model": dep,
                "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "reasoning_tokens": None, "reasoning_content_chars": None,
                "finish_reason": None, "thinking_requested": thinking_requested,
                "latency_s": time.perf_counter() - started,
                "attempts": attempt + 1, "usage_reported": False,
                "usage_unknown": True, "raw_usage": {}, "context": context,
                **failure,
            })
            msg = str(e).lower()
            last_err = e
            if "412" in msg and ("usage limit" in msg or "limit exceeded" in msg):
                raise QuotaExhausted(f"API key usage limit exceeded ({dep}): {str(e)[:160]}") from e
            # Azure content-filter 400 is deterministic: retrying is pointless and
            # an unguarded raise kills whole-cohort methods (e.g. doctopus calibration).
            # Degrade to an empty answer so every caller handles it uniformly.
            if ("content_filter" in msg or "content management policy" in msg
                    or "responsibleaipolicy" in msg):
                _bump(bucket, 0, 0)
                return "", 0, 0
            # adapt to gpt-5 param quirks without failing the run
            if "max_tokens" in msg and "max_completion_tokens" in msg and not _USE_MAX_COMPLETION:
                _USE_MAX_COMPLETION = True
                continue
            if "temperature" in msg and _TEMP_OK:
                _TEMP_OK = False
                continue
            if "reasoning_effort" in msg and _REASONING_OK:
                _REASONING_OK = False
                continue
            if any(tok in msg for tok in _RETRYABLE) and attempt < max_retries - 1:
                # The gateway's transient 404 is a routing miss, not backpressure:
                # an immediate retry of the same request succeeds. Starting it at
                # the rate-limit backoff (2s, doubling) turned a 4s call into 30
                # to 60s, and at the measured hit rate that is hours across a full
                # sweep. Back off from 0.4s for this one, keep 2s for everything
                # else, since a 429 or a 503 does mean slow down.
                transient_404 = "404" in msg or "resource not found" in msg
                # Capped, because the retry count is now high enough that an
                # uncapped doubling would reach minutes on the last attempt.
                time.sleep(min(0.4 * (2 ** attempt), 5.0) if transient_404 else delay)
                if not transient_404:
                    delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")


_embed_client = None


def _get_embed_client():
    """Embeddings client — always the AZURE_* endpoint.

    The chat endpoint may be a model-serving cluster with no embedding model
    (the NGA cluster is), so embeddings are kept on the endpoint that has one
    rather than following LLM_API_BASE.
    """
    global _embed_client
    if _embed_client is None:
        if not (AZURE_API_KEY and AZURE_API_BASE):
            raise RuntimeError("AZURE_API_KEY / AZURE_API_BASE missing from .env")
        base = AZURE_API_BASE.rstrip("/")
        if base.endswith("/v1") or "openai.azure.com" not in base:
            from openai import OpenAI
            _embed_client = OpenAI(api_key=AZURE_API_KEY, base_url=base,
                                   organization=OPENAI_ORG_ID, timeout=300.0)
        else:
            from openai import AzureOpenAI
            _embed_client = AzureOpenAI(api_key=AZURE_API_KEY, azure_endpoint=base,
                                        api_version=AZURE_API_VERSION,
                                        organization=OPENAI_ORG_ID)
    return _embed_client


def embed(texts, model: str | None = None, bucket: str = "method", max_retries: int = 6):
    """Embed a list of strings via the Azure embedding deployment.

    Returns (list_of_vectors, total_tokens). Tokens added to `bucket`.
    """
    client = _get_embed_client()
    dep = model or AZURE_EMBED_DEPLOYMENT
    if not dep:
        raise RuntimeError("AZURE_EMBED_DEPLOYMENT is not set in .env")
    if isinstance(texts, str):
        texts = [texts]
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=dep, input=texts)
            vecs = [d.embedding for d in resp.data]
            tot = getattr(resp.usage, "total_tokens", 0)
            _bump(bucket, tot, 0)
            return vecs, tot
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            last_err = e
            if any(tok in msg for tok in _RETRYABLE) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError(f"embed failed after {max_retries} retries: {last_err}")


# --- tiktoken helper ----------------------------------------------------------
_enc = None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Pre-call token estimate via tiktoken (encoding only, no network)."""
    global _enc
    if _enc is None:
        import tiktoken
        try:
            _enc = tiktoken.encoding_for_model(model)
        except Exception:
            _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text or ""))


if __name__ == "__main__":
    # ping: one tiny call, print ONLY "Azure OK" + usage, never the key
    txt, pt, ct = chat(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=16, bucket="ping",
    )
    print("Azure OK")
    print(f"ping usage: prompt={pt} completion={ct} total={pt + ct}")
