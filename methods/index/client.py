"""Single wiring point to the shared clients for the `index` method.

Reuses bench/shared/azure_client.py (gpt-5, existing reasoning_effort defaults, token
buckets) for the LLM, and a local sentence-transformers model for the transient
pack-time clustering embeddings. Nothing here is stored as a query-time vector index.

Vocabulary: root, category, subject, field, source document.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[2])

import pathlib, importlib.util

BENCH = _BENCH
_spec = importlib.util.spec_from_file_location("azure_client", BENCH/"shared"/"azure_client.py")
_ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ac)


def chat(prompt, max_tokens=2048, bucket="index_build", system=None):
    """One LLM call. Keeps the client's default reasoning_effort (do not override,
    to stay comparable with v7). Returns (text, prompt_tokens, completion_tokens)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    try:
        text, pt, ct = _ac.chat(msgs, max_tokens=max_tokens, bucket=bucket)
    except Exception as e:  # noqa: BLE001
        return f"[LLM ERROR: {repr(e)[:120]}]", 0, 0
    return (text or "").strip(), pt, ct


def count_tokens(text):
    return _ac.count_tokens(text)


def get_counters():
    return _ac.get_counters()


# --- local, transient embedder for pack-time clustering (Phase 5) ---
_model = None

def embed(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    """L2-normalized row vectors from a local SBERT model. Transient: never stored."""
    global _model
    if _model is None or getattr(_model, "_name", None) != model_name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name); _model._name = model_name
    import numpy as np
    v = _model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")
