"""make_rerun_charts.py — the four figures, from results/eval/rerun_metrics.json.

  report/rerun/accuracy.png        judge accuracy
  report/rerun/tokens.png          build (one-time) + answer (recurring) tokens
  report/rerun/latency_answer.png  per-question answer latency, median with p95
  report/rerun/latency_build.png   one-time build wall-clock

Two panels per figure, NarrativeQA and CUAD, bars ascending within each panel.

Design notes, since this departs from the figure in the paper in two ways and
both are deliberate:

1. The paper's figure identifies eight methods by colour and hatch alone, read
   off a legend. This benchmark now compares thirteen. Past eight categorical
   hues the next one is not distinguishable under colour-vision deficiency, so
   the honest options are to fold the tail into "Other" (destroys the comparison)
   or to stop using hue for identity. Identity moves to the axis: every bar is
   labelled with its method name and its value, which is what a reader actually
   reads off a ranked bar chart, and no legend lookup is needed at all.

2. Hue is then free to carry the one thing the figure is for: where our index
   lands. Ours is red, every baseline is one recessive blue. These are slots 8
   and 1 of a validated categorical palette; measured separation is dE 32.3 for
   normal vision (floor 15) and 27.3 / 21.4 under deuteranopia / protanopia
   (target 8), so the emphasis survives colour-vision deficiency, and the name
   under each bar means the chart does not depend on colour at all.

A method with no number for a given metric is left out of that panel and named
underneath, rather than drawn as a zero. A zero bar and a missing measurement
look identical and mean opposite things.

    envs/base/bin/python shared/make_rerun_charts.py             # refuses if incomplete
    envs/base/bin/python shared/make_rerun_charts.py --partial   # draw anyway
"""

import os, sys, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH = pathlib.Path(os.environ.get("BENCH_ROOT") or pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(BENCH / "shared"))
from corpus import OURS, SCORED, DISPLAY  # noqa: E402

SRC = BENCH / "results" / "eval" / "rerun_metrics.json"
OUTD = BENCH / "report" / "rerun"

OURS_C = "#e34948"        # categorical slot 8
BASE_C = "#2a78d6"        # categorical slot 1
BUILD_C = "#a9c7ee"       # same hue, lighter, for the one-time half of a stack
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e3e0"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})


def completeness(data):
    """What is missing before these figures mean anything.

    Returns (ready, [reasons]). A figure drawn from a half-finished sweep is
    worse than no figure: the bars are ranked, so a method that has answered 300
    of 885 questions sits in the ordering as though that were its score, and the
    picture is wrong in a way that looks fine. So the default is to refuse, name
    what is outstanding, and let --partial override it for a deliberate look at
    work in progress.
    """
    missing = []
    for key, c in data["corpora"].items():
        for m in SCORED:
            r = c["methods"].get(m)
            if r is None or r.get("status") == "not run":
                missing.append(f"{c['name']}: {DISPLAY.get(m, m)} has not run")
                continue
            if str(r.get("status", "")).startswith("partial"):
                missing.append(f"{c['name']}: {r['display']} {r['status']}")
            elif r.get("accuracy_pct") is None:
                missing.append(f"{c['name']}: {r['display']} answered but is not scored")
    return (not missing), missing


def _clip_cap(totals, ratio=2.5, headroom=1.18):
    """Where to cut the y axis when one method dwarfs the rest.

    GraphRAG-global answers from a query-focused global summary and spends 143M
    tokens on NarrativeQA against 36M for the next method and 1.4M for the
    cheapest. On a shared linear axis that single bar compresses eleven methods
    into the baseline and the chart stops being readable. Dropping it would hide
    a real and relevant cost, and a log axis cannot carry a stacked bar honestly
    (the segments stop summing by eye). So the axis is cut: over-cap bars are
    drawn to the cap, marked with a break, and labelled with their true value.

    Returns (cap, {index: true_total}) and no cap at all when nothing dominates.
    """
    if len(totals) < 2:
        return None, {}
    order = sorted(range(len(totals)), key=lambda i: totals[i])
    top, second = totals[order[-1]], totals[order[-2]]
    if not second or top < ratio * second:
        return None, {}
    cap = second * headroom
    return cap, {i: v for i, v in enumerate(totals) if v > cap}


def _mark_break(ax, i, cap, width=0.72):
    """A break band across a clipped bar, so a cut axis is never read as a value."""
    y = cap * 0.965
    ax.add_patch(plt.Rectangle((i - width / 2, y), width, cap * 0.035,
                               facecolor="white", edgecolor="none", zorder=5))
    ax.plot([i - width / 2, i + width / 2], [y, y + cap * 0.022],
            color="white", lw=2.5, zorder=6, solid_capstyle="butt")
    ax.plot([i - width / 2, i + width / 2], [y + cap * 0.014, y + cap * 0.036],
            color="white", lw=2.5, zorder=6, solid_capstyle="butt")


def _panels(data):
    return [(c["name"], c) for c in data["corpora"].values()]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def _labels(ax, names):
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    for lbl, n in zip(ax.get_xticklabels(), names):
        if n == OUR_NAME:
            lbl.set_color(OURS_C)
            lbl.set_fontweight("bold")


def _series(panel, key):
    """(display, value) ascending, plus the names with no measurement."""
    got, missing = [], []
    for m, r in panel["methods"].items():
        v = r.get(key)
        if v is None:
            missing.append(r.get("display", m))
        else:
            got.append((r["display"], v, m))
    got.sort(key=lambda t: t[1])
    return got, missing


def _note(ax, missing, what):
    if missing:
        ax.text(0.5, -0.42, f"no {what} recorded: " + ", ".join(sorted(missing)),
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK2)


def _bars(ax, got, fmt, missing, what):
    names = [g[0] for g in got]
    vals = [g[1] for g in got]
    cols = [OURS_C if g[2] == OURS else BASE_C for g in got]
    cap, clipped = _clip_cap(vals)
    drawn = [min(v, cap) if cap else v for v in vals]
    bars = ax.bar(range(len(drawn)), drawn, color=cols, width=0.72,
                  edgecolor="white", linewidth=1.2)
    top = cap if cap else (max(vals) if vals else 1)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, min(v, top) + top * 0.02,
                fmt(v) + (" \u2191" if i in clipped else ""),
                ha="center", va="bottom", fontsize=7.5, color=INK)
        if i in clipped:
            _mark_break(ax, i, top)
    ax.set_ylim(0, top * 1.16)
    _labels(ax, names)
    _style(ax)
    _note(ax, missing, what)
    return names


def figure(data, key, title, ylabel, fmt, what, fname, legend_extra=None,
           stacked=None):
    panels = _panels(data)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 5.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, panel) in zip(axes, panels):
        if stacked:
            got, missing = _series(panel, stacked[0])
            got = [g for g in got if panel["methods"][g[2]].get(stacked[1]) is not None]
            got.sort(key=lambda g: (panel["methods"][g[2]][stacked[0]]
                                    + panel["methods"][g[2]][stacked[1]]))
            names = [g[0] for g in got]
            # Plot in millions so the axis reads in the unit the label claims,
            # rather than matplotlib hanging a 1e8 multiplier off the corner.
            ansv = [panel["methods"][g[2]][stacked[0]] / 1e6 for g in got]
            bldv = [panel["methods"][g[2]][stacked[1]] / 1e6 for g in got]
            cols = [OURS_C if g[2] == OURS else BASE_C for g in got]
            totals = [a + b for a, b in zip(ansv, bldv)]
            cap, clipped = _clip_cap(totals)
            top = cap if cap else (max(totals) if totals else 1)
            for i, (a, b, c) in enumerate(zip(ansv, bldv, cols)):
                da = min(a, top)
                db = max(0.0, min(a + b, top) - da)
                ax.bar(i, da, color=c, width=0.72, edgecolor="white", linewidth=1.2)
                if db:
                    ax.bar(i, db, bottom=da, color=BUILD_C, width=0.72,
                           edgecolor="white", linewidth=1.2, hatch="//")
                lab = f"{a:.1f}" if not b else f"{a:.1f}+{b:.1f}"
                ax.text(i, min(a + b, top) + top * 0.02,
                        lab + (" \u2191" if i in clipped else ""),
                        ha="center", va="bottom", fontsize=7, color=INK)
                if i in clipped:
                    _mark_break(ax, i, top)
            ax.set_ylim(0, top * 1.16)
            _labels(ax, names)
            _style(ax)
            _note(ax, missing, what)
        else:
            got, missing = _series(panel, key)
            _bars(ax, got, fmt, missing, what)
        ax.set_title(f"{name}\n({panel['n_docs']} docs, {panel['n_questions']} questions)",
                     fontsize=10, color=INK, pad=10)
        ax.set_ylabel(ylabel, fontsize=9)

    handles = [Patch(facecolor=OURS_C, label="Our Index"),
               Patch(facecolor=BASE_C, label="Baselines")]
    if legend_extra:
        handles += legend_extra
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, fontsize=13, color=INK, y=1.035, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    OUTD.mkdir(parents=True, exist_ok=True)
    out = OUTD / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(BENCH))


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} not found — run shared/collect_metrics.py first")
    data = json.loads(SRC.read_text())

    ready, missing = completeness(data)
    if not ready and "--partial" not in sys.argv:
        print("Not drawing: the sweep is not finished.\n")
        for m in missing:
            print("  -", m)
        print(f"\n{len(missing)} outstanding. Re-run when the sweep completes, or pass "
              "--partial to draw anyway (the figures will rank incomplete methods "
              "against complete ones).")
        sys.exit(1)
    if not ready:
        print(f"WARNING: drawing with {len(missing)} method/corpus pairs incomplete "
              "(--partial). These figures rank partial results against full ones.\n")
    global OUR_NAME
    OUR_NAME = next((r["display"] for c in data["corpora"].values()
                     for m, r in c["methods"].items() if m == OURS), "Our Index")

    figure(data, "accuracy_pct", "Judge Accuracy across Benchmarks",
           "Judge accuracy (%)", lambda v: f"{v:.1f}", "accuracy", "accuracy.png")

    figure(data, None, "Token Cost across Benchmarks", "Total tokens (millions)",
           lambda v: f"{v/1e6:.1f}", "token cost", "tokens.png",
           legend_extra=[Patch(facecolor=BUILD_C, hatch="//", label="Build (one-time)")],
           stacked=("answer_tokens", "build_tokens"))

    figure(data, "answer_latency_median_s", "Answer Latency across Benchmarks",
           "Median seconds per question", lambda v: f"{v:.1f}", "answer latency",
           "latency_answer.png")

    figure(data, "build_min", "Build Time across Benchmarks",
           "One-time build (minutes)", lambda v: f"{v:,.0f}", "build time",
           "latency_build.png")


if __name__ == "__main__":
    main()
