"""Probe the Azure embeddings endpoint with candidate deployment names.

Reuses the working chat Azure config. Tries each name in order, sends "ping",
and on the first 200 sets AZURE_EMBED_DEPLOYMENT in ~/index/.env. Never prints
the key. On all-404, prints the exact request URL + error body.
"""

# --- repo root: BENCH_ROOT env wins, else resolved from this file's location ---
import os as _os, pathlib as _pl
_BENCH = _pl.Path(_os.environ.get("BENCH_ROOT") or _pl.Path(__file__).resolve().parents[1])

import os, pathlib, json
import importlib.util

_HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("azure_client", _HERE / "azure_client.py")
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

CANDIDATES = ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
ENV_PATH = _BENCH/".env"

from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=ac.AZURE_API_KEY,
    azure_endpoint=ac.AZURE_API_BASE,
    api_version=ac.AZURE_API_VERSION,
    organization=ac.OPENAI_ORG_ID,
)

def req_url(dep):
    base = ac.AZURE_API_BASE.rstrip("/")
    return f"{base}/openai/deployments/{dep}/embeddings?api-version={ac.AZURE_API_VERSION}"

winner = None
errors = []
for dep in CANDIDATES:
    try:
        resp = client.embeddings.create(model=dep, input="ping")
        dim = len(resp.data[0].embedding)
        winner = (dep, dim)
        break
    except Exception as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None) or str(e)
        errors.append((dep, status, body, req_url(dep)))

def set_env_var(key, value):
    lines = []
    found = False
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f"{key}="):
            lines[i] = f'{key}="{value}"'
            found = True
            break
    if not found:
        lines.append(f'{key}="{value}"')
    ENV_PATH.write_text("\n".join(lines) + "\n")

if winner:
    dep, dim = winner
    set_env_var("AZURE_EMBED_DEPLOYMENT", dep)
    print(f"EMBED OK: deployment={dep} dim={dim}")
    print(f"set AZURE_EMBED_DEPLOYMENT={dep} in {ENV_PATH}")
else:
    print("ALL CANDIDATES FAILED (404 or other). Details:")
    for dep, status, body, url in errors:
        print(f"\n--- {dep} ---")
        print(f"status: {status}")
        print(f"request URL: {url}")
        print(f"error body: {body}")
