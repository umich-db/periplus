"""Per-question $ cost (input vs output) + cost-vs-accuracy. GPT-5 list pricing."""
import pathlib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P_IN, P_OUT = 1.25, 10.00  # GPT-5 USD/1M
OUT = pathlib.Path("report"); OUT.mkdir(exist_ok=True)
# method, in_tok, out_tok, build_tok, n, med_s, p95_s, acc
R = [
 ("wiki_v7",  1155495, 255286, 2127028, 295, 10.81, 51.71, 55.9),
 ("wiki_v7_guess", 861490, 155148, 2127028, 295, 6.74, 23.28, 69.8),
 ("raptor_native",  325101, 254466, 1375305, 295, 10.80, 25.66, 64.8),
 ("raptor_abstain", 325691, 214099, 1375305, 295,  8.17, 18.39, 38.0),
 ("wiki_v6",  4059038, 303412, 1366286, 295,  9.97, 37.88, 54.6),
 ("docetl_std",12115945, 9735,       0, 295,  1.94,  4.60, 57.3),
 ("naive_gpt",12044706,137365,       0, 295,  7.64, 29.80, 52.9),
 ("rag",       1468835,180822,  462400, 295,  8.39, 34.47, 35.4),
 ("leann",     5748813,  5199,       0, 295,  2.31, 30.17, 51.2),
 ("graphrag",  2651541,  8972, 2357445, 295,  1.99, 61.77, 43.7),
 ("shed",      5307982,585624,       0, 295, 26.30, 62.01, 45.8),
 ("doctopus",  1749554,106983,  462400, 177,  8.70, 13.71, 52.0),
]
OURS = {"wiki_v7", "wiki_v6", "wiki_v7_guess"}
# Display-only rename: brand our method as "index_*" on the chart; data keys stay "wiki_*".
DISPLAY = {"wiki_v6": "index_v6", "wiki_v7": "index_v7",
           "wiki_v7_guess": "index_v7_guess"}
def disp(m): return DISPLAY.get(m, m)
def cin(it,n): return it/n/1e6*P_IN
def cout(ot,n): return ot/n/1e6*P_OUT
rows = sorted(R, key=lambda r: cin(r[1],r[4])+cout(r[2],r[4]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

# --- Panel 1: stacked $/question, input + output ---
labels=[r[0] for r in rows]; y=range(len(rows))
ci=[cin(r[1],r[4]) for r in rows]; co=[cout(r[2],r[4]) for r in rows]
bi=ax1.barh(y, ci, color=["#C44E52" if r[0] in OURS else "#4C72B0" for r in rows], edgecolor="white")
bo=ax1.barh(y, co, left=ci, color=["#E89A9C" if r[0] in OURS else "#9DB8DC" for r in rows], hatch="//", edgecolor="white")
ax1.set_yticks(y); ax1.set_yticklabels([f"{disp(l)}  ◄ours" if l in OURS else disp(l) for l in labels]); ax1.invert_yaxis()
for i,r in enumerate(rows):
    t=ci[i]+co[i]; ax1.text(t+0.0015, i, f"${t:.4f}", va="center", fontsize=9,
        fontweight="bold" if r[0] in OURS else "normal")
ax1.set_xlabel("Cost per question (USD)"); ax1.set_xlim(0, max(ci[i]+co[i] for i in range(len(rows)))*1.18)
ax1.set_title("Per-Question Answer Cost  (GPT-5 list: $1.25/1M in, $10/1M out)", fontweight="bold")
ax1.legend(handles=[Patch(facecolor="#4C72B0",label="input tokens"),
                    Patch(facecolor="#9DB8DC",hatch="//",label="output tokens"),
                    Patch(facecolor="#C44E52",label="our method")], loc="lower right", frameon=False)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)

# --- Panel 2: cost vs accuracy ($/q log x) ---
for r in rows:
    x=cin(r[1],r[4])+cout(r[2],r[4]); ours=r[0] in OURS
    ax2.scatter(x, r[7], s=230 if ours else 150,
        color="#C44E52" if ours else "#4C72B0", edgecolor="white", zorder=3,
        marker="*" if ours else "o")
    ax2.annotate(disp(r[0]), (x, r[7]), xytext=(7,5), textcoords="offset points",
        fontsize=9, fontweight="bold" if ours else "normal")
ax2.set_xscale("log"); ax2.set_xlabel("Cost per question (USD, log)"); ax2.set_ylabel("Judge accuracy (%)")
ax2.set_title("Cost vs Accuracy  (upper-left = cheap & accurate)", fontweight="bold")
ax2.grid(True, which="both", ls=":", alpha=0.4)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
ax2.text(0.99,0.02,"★ = our method", transform=ax2.transAxes, ha="right", fontsize=9, color="#C44E52")

fig.suptitle("Cost per Question by Method — NarrativeQA, 295 Qs (build cost excluded; one-time)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"cost_per_question.png", dpi=150); print("wrote", OUT/"cost_per_question.png")
