"""Config loader for the `index` method. All values live in config.json; `f`
(fanout) is DERIVED as floor(C / s), never hardcoded.

Vocabulary: root, category, subject, field, source document.
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent


class Config:
    def __init__(self, d):
        self.C = int(d["C"])                     # routing window in tokens (reliable read window)
        self.s = int(d["s"])                     # routing summary size in tokens
        self.chunk_target = int(d["chunk_target"])  # close a chunk at first blank-line seam at/after this
        self.B = int(d["B"])                     # build batch in tokens; HARD CAP for oversized blocks
        self.embedding_model = d["embedding_model"]
        self.merge_similarity = float(d["merge_similarity"])  # dedup: fold labels above this cosine
        self.weak_match_floor = float(d["weak_match_floor"])  # DIAGNOSTIC only: flag (not reassign) below this
        self.cluster_seed = int(d["cluster_seed"])
        self.fallback_wide = bool(d["fallback_wide"])
        self.propose_workers = int(d.get("propose_workers", 8))
        # Phase 7 (answer path): traversal budget = budget_a * depth + budget_b; a reached
        # leaf at or above confidence_bar is good enough to stop on; routing_mode picks how
        # children are scored (cosine default, keyword for an apples-to-apples v7 run, llm ablation).
        self.budget_a = int(d.get("budget_a", 2))
        self.budget_b = int(d.get("budget_b", 2))
        self.confidence_bar = float(d.get("confidence_bar", 0.5))
        self.routing_mode = d.get("routing_mode", "cosine")
        # build_mode: per_category (default, the shipped pack) or global_sort (the B+ tree
        # baseline: sort every subject by embedding order, group fixed-size adjacent, bottom-up).
        self.build_mode = d.get("build_mode", "per_category")
        # fallback_mode: how the answer phase widens on a not-found. sibling (default, shipped) re-reads
        # the reached leaf plus its nearest siblings; doc_leaves scores every subject leaf in the source
        # document by query-time cosine and re-reads the top fallback_k spans (transient, not persisted).
        self.fallback_mode = d.get("fallback_mode", "sibling")
        self.fallback_k = int(d.get("fallback_k", 5))
        # answer_context: what text the answer model reads. raw_chunk (default, shipped) feeds the
        # reached leaf's raw source chunk. digest (new) selects index's own extracted subject content
        # over the answer-relevant region (reached leaves + top doc_leaves leaves) into a compiled
        # digest, the analog of v7's selected precompiled digest, to close the v7 answer-context gap.
        self.answer_context = d.get("answer_context", "raw_chunk")
        self.digest_k = int(d.get("digest_k", 40))          # leaves gathered into the digest
        self.digest_chars = int(d.get("digest_chars", 12000))  # digest char budget (~v7's 12k)

    @property
    def f(self):                                 # fanout, DERIVED
        return self.C // self.s

    def as_dict(self):
        return {"C": self.C, "s": self.s, "f": self.f, "chunk_target": self.chunk_target,
                "B": self.B, "embedding_model": self.embedding_model,
                "merge_similarity": self.merge_similarity, "weak_match_floor": self.weak_match_floor,
                "cluster_seed": self.cluster_seed, "fallback_wide": self.fallback_wide,
                "propose_workers": self.propose_workers,
                "budget_a": self.budget_a, "budget_b": self.budget_b,
                "confidence_bar": self.confidence_bar, "routing_mode": self.routing_mode,
                "build_mode": self.build_mode, "fallback_mode": self.fallback_mode,
                "fallback_k": self.fallback_k, "answer_context": self.answer_context,
                "digest_k": self.digest_k, "digest_chars": self.digest_chars}


def load(path=None):
    path = pathlib.Path(path) if path else HERE / "config.json"
    return Config(json.loads(path.read_text()))
