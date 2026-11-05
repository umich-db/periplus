"""make_tree_diagram.py — before/after balancing tree diagram for 2012, in context.

Shows the REAL 2012 tree, not just one path: root -> 11 categories -> Story
Overview -> its 16 sub-sections. Several of those sub-sections were ALREADY
grouped by v7 (Themes 79->4 groups, Key Events 96->4 groups). Only "Synopsis
(Detailed by Act/Sequence)" sat as 73 flat children (a dict node, which v7 does
not cap), so it was the single node over f=33. The bottom zoom shows that one
node before and after balancing. All node text is copied from the built manifests.

Run from bench/:  envs/base/bin/python shared/make_tree_diagram.py
"""
import textwrap, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTD = pathlib.Path("report"); OUTD.mkdir(parents=True, exist_ok=True)

ROOT_C = "#4C72B0"
CAT_C = "#8172B3"
LEAF_C = "#8C8C8C"
GROUP_C = "#55A868"
OVER_C = "#C44E52"
BOUND_C = "#2F6F4E"
ALREADY_C = "#CCB974"   # v7 already-grouped siblings
DIM = "#B7B7C9"


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.5, bold=False, ec="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.4",
                                fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=4)


def edge(ax, x1, y1, x2, y2, color="#aaa", lw=0.8):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1)


def wrap(s, n):
    return "\n".join(textwrap.wrap(s, n))


fig, ax = plt.subplots(figsize=(17, 11.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# ===================== TOP: the real 2012 tree (context) =====================
ax.text(50, 99, "The real 2012 (film) tree in v7  —  a wide, multi-level tree",
        ha="center", fontsize=14, fontweight="bold", color="#333")

box(ax, 50, 94, 18, 3.6, "2012 (film)  ·  root", ROOT_C, bold=True, fs=9.5)

# 11 categories
CATS = ["Document\nMetadata", "Story\nOverview", "Chronology", "Characters", "Settings",
        "Organizations", "Scientific /\nTechnical", "Scene\nIndex", "Audio-\nVisual",
        "Dialogue", "Continuity"]
cxs = [6 + i * 8.8 for i in range(len(CATS))]
for x, name in zip(cxs, CATS):
    hl = name.startswith("Story")
    edge(ax, 50, 92.2, x, 88.2)
    box(ax, x, 86.4, 8.0, 4.6, name, CAT_C if hl else DIM,
        tc="white" if hl else "#333", bold=hl, fs=5.8,
        ec=OVER_C if hl else "white")
ax.text(50, 82.8, "11 categories under the root  (Story Overview highlighted)",
        ha="center", fontsize=8, style="italic", color="#666")

# Story Overview's 16 sub-sections (a representative row)
sox = cxs[1] + 4
box(ax, sox, 78.5, 20, 3.4, 'Story Overview:  16 sub-sections', CAT_C, bold=True, fs=7.5)
edge(ax, cxs[1], 84.1, sox, 80.2, color=CAT_C, lw=1.1)

SUBS = [
    ("Premise /\nLogline", DIM, "#333"),
    ("Genre &\nTone\n(9 items)", DIM, "#333"),
    ("Themes & Motifs\n79 → already\n4 groups ✓", ALREADY_C, "#333"),
    ("Synopsis\n(Short)", DIM, "#333"),
    ("Synopsis (Detailed\nby Act/Sequence)\n73 FLAT ✗", CAT_C, "white"),
    ("Key Events &\nTurning Pts\n96 → already\n4 groups ✓", ALREADY_C, "#333"),
    ("… +10 more\nsub-sections", DIM, "#333"),
]
sbx = [8 + i * 13.3 for i in range(len(SUBS))]
for x, (name, fc, tc) in zip(sbx, SUBS):
    edge(ax, sox, 76.8, x, 73.2, color="#c7bfe0")
    star = name.endswith("✗")
    box(ax, x, 70.3, 12.2, 6.0, name, fc, tc=tc, bold=star, fs=6.0,
        ec=OVER_C if star else "white")
ax.text(50, 65.6,
        "Story Overview has 16 sub-sections. v7's list-grouping already split the wide ones "
        "(Themes 79→4 groups, Key Events 96→4 groups).",
        ha="center", fontsize=8, style="italic", color="#666")
ax.text(50, 63.3,
        "Only 'Synopsis (Detailed by Act/Sequence)' stayed FLAT at 73 children — a dict node, which v7 does not cap. That is the one node over f.",
        ha="center", fontsize=8, style="italic", color=OVER_C)

# ===================== divider =====================
ax.plot([3, 97], [59, 59], color="#ccc", lw=1)
ax.text(50, 57, "Zoom: the one node balancing changed  —  Synopsis (Detailed by Act/Sequence)",
        ha="center", fontsize=12.5, fontweight="bold", color="#333")
ax.axvline(50, ymin=0.03, ymax=0.53, color="#eee", lw=1, ls="--")
ax.text(25, 53.5, "BEFORE — v7", ha="center", fontsize=12, fontweight="bold", color="#333")
ax.text(75, 53.5, "AFTER — v7_balanced", ha="center", fontsize=12, fontweight="bold", color=BOUND_C)

# ---------- BEFORE: 73 flat ----------
box(ax, 25, 47, 30, 5.2, 'Synopsis (Detailed by Act/Sequence)\nrouting summary: "73 sub-sections"',
    CAT_C, bold=True, fs=7.8)
box(ax, 25, 40.5, 17, 3.2, "fanout = 73  >  f = 33", OVER_C, bold=True, fs=8.5)
SUBJ = [
    ("Prologue (Cosmic Setup)", "Mozart's Adagio, a planetary alignment, a solar event."),
    ("2009 – Discovery & Warning", "In monsoon India, Prof. West arrives at a New Delhi institute."),
    ("2010 – Global Political Response", "At the G8 in Spain, President Wilson clears the room."),
    ("2011 – Covert Preparations", "In London, MI-6 solicits a Saudi Prince for 1 billion Euros."),
    ("Sequence 2 – Yellowstone", "Jackson drives his kids Noah and Lilly through Yellowstone."),
]
sxs = [6.8, 16.4, 26.0, 35.6, 45.2]
for x, (title, summ) in zip(sxs, SUBJ):
    edge(ax, 25, 37.4, x, 30)
    box(ax, x, 24, 9.0, 11, wrap(title, 16) + "\n" + wrap(summ, 18), LEAF_C, fs=5.7)
ax.text(25, 15.5, "…  all 73 subject pages hang off one node  …", ha="center",
        fontsize=8.5, style="italic", color=OVER_C)

# ---------- AFTER: 3 bounded groups ----------
box(ax, 75, 47, 30, 5.2, 'Synopsis (Detailed by Act/Sequence)\nsummary: "73 items across 3 groups"',
    CAT_C, bold=True, fs=7.8)
GROUPS = [
    (61, "Group 1", "At their tent, Jackson stru… … While Kate, Noah, Lilly, an…", 25),
    (75, "Group 2", "A 'breach in the zoological… … At the G8 Summit in Spain, …", 24),
    (89, "Group 3", "CNN reports Manfred Picard … … Lama Yak La rings his monas…", 24),
]
for gx, gname, gsumm, gn in GROUPS:
    edge(ax, 75, 44.4, gx, 34, color=BOUND_C, lw=1.2)
    box(ax, gx, 26.5, 13, 12,
        f"{gname}  (new node)\n" + wrap('summary: "' + gsumm + '"', 22) + f"\n{gn} subjects  ≤ f",
        GROUP_C, fs=6.3)
ax.text(75, 16.5, "3 intermediate nodes added · the 73 subject pages are unchanged",
        ha="center", fontsize=8, style="italic", color="#555")
box(ax, 75, 12, 33, 4, "every node ≤ f = 33   ·   each group in occupancy window [17, 33]",
    BOUND_C, bold=True, fs=8.5)

cap = ("Only the Synopsis node changed. Everything above it (the 11 categories, the other 15 sub-sections of "
       "Story Overview, and the already-grouped Themes/Key Events) is byte-identical in v7_balanced. Balancing "
       "embeds each of the 73 subjects' routing summaries with all-mpnet, orders them by similarity, and cuts "
       "them into 3 balanced groups of [25, 24, 24], inserting one intermediate node per group whose summary is "
       "generated the v7 way (first … last child + count). The split is topical, not chronological: '2010 – "
       "Global Political Response' lands in Group 2 with the political/assassination scenes, not beside '2009' "
       "and '2011' (which went to Group 3).")
fig.text(0.012, 0.012, cap, fontsize=8, color="#333", va="bottom", wrap=True)

fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.085)
fig.savefig(OUTD / "v7_balanced_tree_2012.png", dpi=150)
plt.close(fig)
print("wrote", OUTD / "v7_balanced_tree_2012.png")
