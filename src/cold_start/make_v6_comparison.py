"""Generate comparison charts: v6 (cold-start) vs v5.1 vs v3 vs all other systems.
Charts: judge accuracy, hallucination rate, token F1.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

BASE = Path("/Users/phdonn/narrativeqa")
V6_OUT = BASE / "llm-wiki" / "cold_start" / "output"
LLM_WIKI = BASE / "llm-wiki"
V5 = LLM_WIKI / "v5"
QA = LLM_WIKI / "qa_results"

# ── Load all system data ─────────────────────────────────────────────────────

with open(BASE / "fact_intuition_split/master_advanced_metrics.json") as f:
    master = json.load(f)

with open(QA / "wiki_qa_v3_summary.json") as f:
    wiki_v3 = json.load(f)

with open(V5 / "wiki_qa_v5_1_summary.json") as f:
    wiki_v51 = json.load(f)

with open(V6_OUT / "wiki_qa_v6_summary.json") as f:
    wiki_v6 = json.load(f)

with open(BASE / "hybrid_db_schema_v1/hybrid_results/summary_hybrid_v1_judged.json") as f:
    hybrid_v1 = json.load(f)

with open(BASE / "adaptive_schema/qa_results/summary_adaptive_schema.json") as f:
    adaptive_v1 = json.load(f)

with open(BASE / "adaptive_schema_v2/results/summary_adaptive_schema_v2.json") as f:
    adaptive_v2 = json.load(f)

with open(BASE / "adaptive_schema_v3/results/summary_adaptive_schema_v3.json") as f:
    adaptive_v3 = json.load(f)

with open(BASE / "gpt5_full_script/summary_gpt5_full.json") as f:
    gpt5_full = json.load(f)

with open(BASE / "baseline_testing_llm/summary_all.json") as f:
    baselines = json.load(f)

# ── Build system list ────────────────────────────────────────────────────────

CATEGORIES = {
    "LLM Wiki v6 (cold-start)": "wiki_new",
    "LLM Wiki QA v5.1": "wiki",
    "LLM Wiki QA v3": "wiki",
    "GPT-5 Raw Full": "parametric",
    "Gemini Raw Full": "parametric",
    "GPT-5 Full Script": "parametric",
    "DocETL Map-Only": "docetl",
    "DocETL Split-Map-Reduce": "docetl",
    "DocETL Gemini": "docetl",
    "DocETL GPT-4o": "docetl",
    "DB v1": "db",
    "DB v2 (Few-Shot)": "db",
    "DB v2 (Full Script)": "db",
    "DB v4 (Full Schema)": "db",
    "RAG (Routed)": "rag",
    "GraphRAG": "rag",
    "Hybrid (DB+RAG)": "hybrid",
    "Hybrid DB+Schema v1": "hybrid",
    "Adaptive Schema v1": "adaptive",
    "Adaptive Schema v2": "adaptive",
    "Adaptive Schema v3": "adaptive",
}

CAT_COLORS = {
    "wiki_new": "#D32F2F",
    "wiki": "#FF6F00",
    "parametric": "#1565C0",
    "docetl": "#2E7D32",
    "db": "#6A1B9A",
    "rag": "#00838F",
    "hybrid": "#C62828",
    "adaptive": "#EF6C00",
}

systems = []

# From master_advanced_metrics
for key, data in master.items():
    systems.append({
        "name": data["system"],
        "judge": data["all"]["judge_accuracy"],
        "hall": data["all"]["hallucination_rate"],
        "f1": data["all"]["token_f1"],
    })

# Wiki systems
systems.append({"name": "LLM Wiki QA v3", "judge": wiki_v3["judge_accuracy"],
                "hall": wiki_v3["hallucination_rate"], "f1": wiki_v3["avg_f1"]})
systems.append({"name": "LLM Wiki QA v5.1", "judge": wiki_v51["judge_accuracy"],
                "hall": wiki_v51["hallucination_rate"], "f1": wiki_v51["avg_f1"]})
systems.append({"name": "LLM Wiki v6 (cold-start)", "judge": wiki_v6["judge_accuracy"],
                "hall": wiki_v6["hallucination_rate"], "f1": wiki_v6["avg_f1"]})

# Other systems
systems.append({"name": "Hybrid DB+Schema v1", "judge": hybrid_v1["judge_accuracy"],
                "hall": None, "f1": hybrid_v1["avg_f1"]})
systems.append({"name": "Adaptive Schema v1", "judge": None,
                "hall": adaptive_v1["hallucination_rate"], "f1": adaptive_v1["avg_f1"]})
systems.append({"name": "Adaptive Schema v2", "judge": None,
                "hall": adaptive_v2["hallucination_rate"], "f1": adaptive_v2["avg_f1"]})
systems.append({"name": "Adaptive Schema v3", "judge": None,
                "hall": adaptive_v3["hallucination_rate"], "f1": adaptive_v3["avg_f1"]})
systems.append({"name": "GPT-5 Full Script", "judge": None,
                "hall": gpt5_full["hallucination_rate"], "f1": gpt5_full["avg_f1"]})
systems.append({"name": "Gemini Raw Full", "judge": None,
                "hall": baselines["gemini_raw_full"]["hallucination_rate"],
                "f1": baselines["gemini_raw_full"]["avg_f1"]})
systems.append({"name": "DocETL Gemini", "judge": None,
                "hall": baselines["docetl_gemini"]["hallucination_rate"],
                "f1": baselines["docetl_gemini"]["avg_f1"]})
systems.append({"name": "DocETL GPT-4o", "judge": None,
                "hall": baselines["docetl_gpt4o"]["hallucination_rate"],
                "f1": baselines["docetl_gpt4o"]["avg_f1"]})


def get_color(name):
    return CAT_COLORS.get(CATEGORIES.get(name, "other"), "#aaaaaa")

def is_highlight(name):
    return CATEGORIES.get(name, "other") in ("wiki", "wiki_new")


# ── Print head-to-head ───────────────────────────────────────────────────────
print("=" * 70)
print("v6 (COLD-START) vs v5.1 vs v3")
print("=" * 70)
for sys in systems:
    if "Wiki" in sys["name"] or "v6" in sys["name"]:
        j = f"{sys['judge']:.1%}" if sys['judge'] is not None else "—"
        h = f"{sys['hall']:.1%}" if sys['hall'] is not None else "—"
        print(f"  {sys['name']:30s}  judge={j}  f1={sys['f1']:.4f}  hall={h}")

v6_j = wiki_v6["judge_accuracy"]
v51_j = wiki_v51["judge_accuracy"]
v3_j = wiki_v3["judge_accuracy"]
print(f"\n  v6 vs v5.1: {v6_j - v51_j:+.1%} judge accuracy")
print(f"  v6 vs v3:   {v6_j - v3_j:+.1%} judge accuracy")

# v6 source breakdown
src = wiki_v6.get("source_breakdown", {})
print(f"\n  v6 answer sources: wiki={src.get('wiki',0)} fallback={src.get('fallback',0)} miss={src.get('miss',0)}")
bf = wiki_v6.get("backfill_totals", {})
print(f"  v6 backfill: facts={bf.get('facts_added',0)} entities={bf.get('entities_added',0)} rels={bf.get('relationships_added',0)}")

# ── Chart 1: Judge Accuracy ─────────────────────────────────────────────────
judge_sys = sorted([s for s in systems if s.get("judge") is not None],
                   key=lambda s: s["judge"], reverse=True)

fig, ax = plt.subplots(figsize=(13, 8))
names = [s["name"] for s in judge_sys]
vals = [s["judge"] * 100 for s in judge_sys]
colors = [get_color(n) for n in names]
y = np.arange(len(names))
bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5, height=0.7)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Judge Accuracy (%)", fontsize=12)
ax.set_title("Judge Accuracy: All Systems\nv6 cold-start (red) vs v5.1/v3 (orange)",
             fontsize=13, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)
ax.axvline(50, color="#999", linestyle="--", alpha=0.4)
for i, v in enumerate(vals):
    w = "bold" if is_highlight(names[i]) else "normal"
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight=w)
for lbl in ax.get_yticklabels():
    n = lbl.get_text()
    if "v6" in n: lbl.set_fontweight("bold"); lbl.set_color("#D32F2F")
    elif "Wiki" in n: lbl.set_fontweight("bold"); lbl.set_color("#FF6F00")
plt.tight_layout()
plt.savefig(V6_OUT / "v6_judge_accuracy.png", dpi=150, bbox_inches="tight"); plt.close()
print("\nSaved v6_judge_accuracy.png")

# ── Chart 2: Hallucination ──────────────────────────────────────────────────
hall_sys = sorted([s for s in systems if s.get("hall") is not None],
                  key=lambda s: s["hall"])
fig, ax = plt.subplots(figsize=(13, 11))
names = [s["name"] for s in hall_sys]
vals = [s["hall"] * 100 for s in hall_sys]
colors = [get_color(n) for n in names]
y = np.arange(len(names))
bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5, height=0.7)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Hallucination Rate (%) — lower is better", fontsize=12)
ax.set_title("Hallucination Rate: All Systems\nv6 cold-start (red) vs v5.1/v3 (orange)",
             fontsize=13, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)
for i, v in enumerate(vals):
    w = "bold" if is_highlight(names[i]) else "normal"
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, fontweight=w)
for lbl in ax.get_yticklabels():
    n = lbl.get_text()
    if "v6" in n: lbl.set_fontweight("bold"); lbl.set_color("#D32F2F")
    elif "Wiki" in n: lbl.set_fontweight("bold"); lbl.set_color("#FF6F00")
plt.tight_layout()
plt.savefig(V6_OUT / "v6_hallucination.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved v6_hallucination.png")

# ── Chart 3: Token F1 ───────────────────────────────────────────────────────
f1_sys = sorted(systems, key=lambda s: s["f1"], reverse=True)
fig, ax = plt.subplots(figsize=(13, 11))
names = [s["name"] for s in f1_sys]
vals = [s["f1"] for s in f1_sys]
colors = [get_color(n) for n in names]
y = np.arange(len(names))
bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5, height=0.7)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Token F1 Score", fontsize=12)
ax.set_title("Token F1: All Systems\nv6 cold-start (red) vs v5.1/v3 (orange)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(True, axis="x", alpha=0.3)
for i, v in enumerate(vals):
    w = "bold" if is_highlight(names[i]) else "normal"
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, fontweight=w)
for lbl in ax.get_yticklabels():
    n = lbl.get_text()
    if "v6" in n: lbl.set_fontweight("bold"); lbl.set_color("#D32F2F")
    elif "Wiki" in n: lbl.set_fontweight("bold"); lbl.set_color("#FF6F00")
plt.tight_layout()
plt.savefig(V6_OUT / "v6_token_f1.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved v6_token_f1.png")

# ── Chart 4: v6 per-movie breakdown ─────────────────────────────────────────
movies = list(wiki_v6["per_movie"].keys())
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, metric, label in zip(axes, ["judge_accuracy", "avg_f1", "hallucination_rate"],
                              ["Judge Accuracy", "Token F1", "Hallucination Rate"]):
    v6_vals = [wiki_v6["per_movie"][m][metric] for m in movies]
    v51_vals = [wiki_v51["per_movie"].get(m, {}).get(metric, 0) for m in movies]
    v3_vals = [wiki_v3["per_movie"].get(m, {}).get(metric, 0) for m in movies]

    x = np.arange(len(movies))
    w = 0.25
    ax.bar(x - w, v3_vals, w, label="v3", color="#FF6F00", alpha=0.8)
    ax.bar(x, v51_vals, w, label="v5.1", color="#FFB74D", alpha=0.8)
    ax.bar(x + w, v6_vals, w, label="v6", color="#D32F2F", alpha=0.8)
    ax.set_xticks(x)
    short = [m[:15] + "..." if len(m) > 15 else m for m in movies]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Per-Movie Comparison: v6 (cold-start) vs v5.1 vs v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(V6_OUT / "v6_per_movie.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved v6_per_movie.png")

print(f"\nAll charts saved to {V6_OUT}")
