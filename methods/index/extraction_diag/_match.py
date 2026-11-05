"""Shared string/fuzzy coverage check used by both Part A and Part B, so "does the gold answer appear
in this text" is computed identically for the source document, the answer leaf's chunk, and the
re-extracted subjects. No LLM. Vocabulary: root, category, subject, field, source document."""
import re

STOP = set(("the a an of to in on at and or is are was were be been being he she it they them his her "
            "their its for with as by from that this which who whom what when where why how not no do "
            "does did will would can could has have had say says said get got go went into out up down "
            "over about after before during under above between within a not you your my me we our i").split())


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def content_tokens(s):
    return [w for w in norm(s).split() if len(w) >= 3 and w not in STOP]


def entities(gold):
    """Multiword capitalized sequences in the ORIGINAL gold (proper-noun answers)."""
    out = []
    for e in re.findall(r"[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*", gold or ""):
        en = norm(e)
        if en and en not in STOP and any(len(w) >= 4 for w in en.split()):
            out.append(en)
    return out


def presence(gold_variants, text):
    """True if the gold answer's content appears in `text` by substring, proper-noun entity, or a
    >=0.6 fraction of its distinctive content tokens."""
    tn = norm(text)
    tw = set(tn.split())
    best_cov, substr, ent = 0.0, False, False
    for g in gold_variants:
        gt = set(content_tokens(g))
        if gt:
            best_cov = max(best_cov, sum(1 for w in gt if w in tw) / len(gt))
        gn = norm(g)
        if gn and len(gn) <= 45 and gn in tn:
            substr = True
        for e in entities(g):
            if re.search(r"\b" + re.escape(e) + r"\b", tn):
                ent = True
    return {"present": bool(substr or best_cov >= 0.6 or ent),
            "cov": round(best_cov, 2), "substr": substr, "entity": ent}
