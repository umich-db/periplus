"""Checkpointed, resumable eval harness for the index method's parallel evals.

Infrastructure only. It does not touch any build mode, the answer path, the judge, or the per-question
results schema, the caller's work function owns all of that. It only adds: append each completed
question to a per-configuration checkpoint file (JSONL, flushed + fsynced) as it finishes, skip
already-completed questions on restart, and assemble the final records in deterministic task order.

Keying: one checkpoint file per configuration, named from the settings that change the answer
(build_mode, fallback_mode, routing_mode, fallback_k, plus an optional caller tag), so two different
configurations never read each other's checkpoints. Pass fresh=True to ignore the checkpoint and rerun.

Usage:
    tag = config_tag(cfg, extra="myeval")
    records = run_checkpointed(tag, tasks, key_fn, work_fn)     # records aligned to tasks order
where key_fn(task) -> a stable unique string (e.g. f"{slug}::{qid}") and work_fn(task) -> a dict (the
per-question result in whatever schema the caller assembles later).

Vocabulary: root, category, subject, field, source document.
"""
import os, sys, json, pathlib, threading
import concurrent.futures as cf

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = HERE/"results"/"checkpoints"


def config_tag(cfg, extra=None):
    """A filesystem-safe tag from the settings that change the answer. Two configs that differ in any
    of these get different checkpoint files."""
    parts = [str(getattr(cfg, "build_mode", "na")),
             "fb-" + str(getattr(cfg, "fallback_mode", "na")),
             "route-" + str(getattr(cfg, "routing_mode", "na")),
             "k" + str(getattr(cfg, "fallback_k", "na"))]
    if extra:
        parts.append(str(extra))
    return "__".join(p.replace("/", "-").replace(" ", "_") for p in parts)


class Checkpoint:
    """Append-only per-question checkpoint. Thread-safe writes; loads completed records on open."""

    def __init__(self, tag, ckpt_dir=None, fresh=False):
        self.tag = tag
        d = pathlib.Path(ckpt_dir or DEFAULT_CKPT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d/f"ckpt_{tag}.jsonl"
        self._lock = threading.Lock()
        self.records = {}                                    # key -> record (dict, includes _key)
        if fresh and self.path.exists():
            self.path.unlink()
        elif self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["_key"]] = r              # last write wins (idempotent re-runs)
        self._fh = open(self.path, "a", encoding="utf-8")

    def done(self, key):
        return key in self.records

    def get(self, key):
        return self.records.get(key)

    def put(self, key, record):
        r = dict(record); r["_key"] = key
        line = json.dumps(r, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n"); self._fh.flush(); os.fsync(self._fh.fileno())
            self.records[key] = r

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def run_checkpointed(tag, tasks, key_fn, work_fn, fresh=False, workers=12, ckpt_dir=None,
                     progress_every=30, log=print):
    """Run work_fn over tasks in parallel, checkpointing each completed task. Resumes from an existing
    checkpoint (skips done keys). Returns the per-task records in the SAME order as `tasks` (each is the
    stored dict minus the internal _key; None only if a task neither ran nor was checkpointed). `tasks`
    must be built in a deterministic order so resume is well defined."""
    ckpt = Checkpoint(tag, ckpt_dir, fresh)
    todo = [t for t in tasks if not ckpt.done(key_fn(t))]
    log(f"[{tag}] {len(tasks) - len(todo)}/{len(tasks)} loaded from checkpoint, running {len(todo)}")
    counter = [0]; plock = threading.Lock()

    def _run(t):
        rec = work_fn(t)
        ckpt.put(key_fn(t), rec)
        with plock:
            counter[0] += 1
            if progress_every and counter[0] % progress_every == 0:
                log(f"[{tag}] {counter[0]}/{len(todo)} this run")

    if todo:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_run, todo))
    ckpt.close()

    out = []
    for t in tasks:
        r = ckpt.get(key_fn(t))
        out.append({k: v for k, v in r.items() if k != "_key"} if r else None)
    return out


# ---------- self-test (mock work, no API): resume, skip, assembly-equivalence, no collision ----------
def _selftest():
    import tempfile, shutil
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ckpt_selftest_"))
    try:
        tasks = [("m", i) for i in range(100)]              # deterministic order
        key_fn = lambda t: f"{t[0]}::{t[1]}"
        computed = []                                       # which tasks actually ran work_fn
        clock = threading.Lock()

        def work(t):
            with clock:
                computed.append(t)
            return {"slug": t[0], "qid": t[1], "value": t[1] * 2, "verdict": "correct" if t[1] % 2 else "incorrect"}

        # Phase A: partial run over the first 40 tasks
        run_checkpointed("cfgA", tasks[:40], key_fn, work, ckpt_dir=tmp, progress_every=0, log=lambda *_: None)
        assert len(computed) == 40, computed
        # Phase B: full run, resuming -> only the remaining 60 recompute
        computed.clear()
        recsB = run_checkpointed("cfgA", tasks, key_fn, work, ckpt_dir=tmp, progress_every=0, log=lambda *_: None)
        assert len(computed) == 60, f"resume recomputed {len(computed)}, expected 60"
        assert len(recsB) == 100 and all(r is not None for r in recsB)
        assert all(recsB[i]["value"] == i * 2 for i in range(100)), "assembled values wrong / out of order"

        # assembly-equivalence: a clean run produces the same assembled records
        computed.clear()
        recsClean = run_checkpointed("cfgClean", tasks, key_fn, work, ckpt_dir=tmp, progress_every=0, log=lambda *_: None)
        assert len(computed) == 100
        strip = lambda rs: [{k: v for k, v in r.items()} for r in rs]
        assert strip(recsB) == strip(recsClean), "resumed assembly != clean assembly"

        # no collision: a different config tag has its own file and shares nothing
        before = set(computed); computed.clear()
        run_checkpointed("cfgB", tasks[:10], key_fn, work, ckpt_dir=tmp, progress_every=0, log=lambda *_: None)
        assert len(computed) == 10, "cfgB should not see cfgA/cfgClean checkpoints"
        assert (tmp/"ckpt_cfgA.jsonl").exists() and (tmp/"ckpt_cfgB.jsonl").exists() and (tmp/"ckpt_cfgClean.jsonl").exists()

        # fresh=True ignores the checkpoint and reruns
        computed.clear()
        run_checkpointed("cfgA", tasks, key_fn, work, ckpt_dir=tmp, fresh=True, progress_every=0, log=lambda *_: None)
        assert len(computed) == 100, "fresh=True should recompute all"
        print("SELFTEST OK: resume skips done, assembly matches clean run, tags do not collide, fresh reruns")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- a mock driver for the real process-kill test (no API): resumable over slow tasks ----------
def _mock_driver():
    import time
    n = int(os.environ.get("MOCK_N", "80"))
    delay = float(os.environ.get("MOCK_DELAY", "0.1"))
    ckpt_dir = os.environ["MOCK_CKPT_DIR"]
    tasks = [("mock", i) for i in range(n)]
    key_fn = lambda t: f"{t[0]}::{t[1]}"

    def work(t):
        time.sleep(delay)                                   # stand-in for a slow answer+judge
        return {"slug": t[0], "qid": t[1], "value": t[1]}

    recs = run_checkpointed("mockcfg", tasks, key_fn, work, ckpt_dir=ckpt_dir, workers=8, progress_every=10)
    if all(r is not None for r in recs):
        print(f"MOCK COMPLETE: {len(recs)} records assembled")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mock-driver":
        _mock_driver()
    else:
        _selftest()
