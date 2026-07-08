"""Generate all figures for Lecture 1: Why Math for AI + The Course Map.

Style: cream-paper palette matching slides/mfai-theme.css.
Saves PNG to figures/lec01/ and SVG (text kept as text) to figures/lec01/svg/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

PAPER = "#F7F3E9"; INK = "#161513"; MUTED = "#5F5C54"; RULE = "#B5AE9B"
RUST = "#B85A3E"; SAGE = "#5F8573"; SLATE = "#37535F"
WINE = "#8E2A3B"; OCHRE = "#B5945A"; PAPER_ALT = "#EFEADA"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size": 13, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",          # keep SVG text as selectable text, not paths
    "savefig.dpi": 200,
})

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "figures" / "lec01"
SVG = PNG / "svg"
PNG.mkdir(parents=True, exist_ok=True)
SVG.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    bbox = "tight" if tight else None
    fig.savefig(PNG / f"{name}.png", bbox_inches=bbox)
    fig.savefig(SVG / f"{name}.svg", format="svg", bbox_inches=bbox)
    plt.close(fig)
    print(f"  wrote {name}.png + svg/{name}.svg")


# ---- 1. the loss that became NaN --------------------------------------------
def nan_loss():
    rng = np.random.default_rng(7)
    steps = np.arange(0, 900)
    loss = 4.2 * np.exp(-steps / 260) + 0.55 + rng.normal(0, 0.045, steps.size)
    loss[600:618] = np.geomspace(loss[599], 5e4, 18)   # the blow-up
    loss[618:] = np.nan                                 # and then... nothing

    fig, ax = plt.subplots(figsize=(9.5, 4.2), layout="constrained")
    ax.semilogy(steps, loss, color=SLATE, lw=2)
    ax.axvline(617, color=WINE, ls="--", lw=1.3)
    ax.axvspan(617, 900, color=WINE, alpha=0.06)
    ax.text(645, 60, "step 618 onwards:\nloss = nan, forever", color=WINE, fontsize=12)
    ax.annotate("training going nicely…", xy=(430, 0.85), xytext=(140, 12),
                color=MUTED, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("a real failure: the loss that became NaN", color=INK)
    ax.set_xlim(0, 900)
    _clean(ax)
    save(fig, "nan_loss")


# ---- 2. same learning rate, two personalities --------------------------------
def lr_two_problems():
    eta = 0.8
    iters = np.arange(0, 31)

    def gd_loss(a):
        x, out = 1.0, []
        for _ in iters:
            out.append(0.5 * a * x * x)
            x = x * (1 - eta * a)
        return np.array(out)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")

    a1.semilogy(iters, gd_loss(0.02), "o-", color=SAGE, ms=4, lw=1.8)
    a1.set_title("problem A · $\\eta = 0.8$ crawls", color=SAGE)
    a1.text(15, 6.5e-3, "after 30 steps:\nstill 38% of the\nstarting loss", color=MUTED, fontsize=11)

    a2.semilogy(iters, gd_loss(3.2), "o-", color=WINE, ms=4, lw=1.8)
    a2.set_title("problem B · $\\eta = 0.8$ explodes", color=WINE)
    a2.text(2.5, 1e8, "each step lands\nfarther from the\nminimum than the last", color=MUTED, fontsize=11)

    for ax in (a1, a2):
        ax.set_xlabel("gradient-descent iteration")
        ax.set_ylabel("loss")
        _clean(ax)

    fig.suptitle("identical code, identical learning rate — only the problem's curvature differs ($\\to$ L17)",
                 fontsize=13, color=INK)
    save(fig, "lr_two_problems")


# ---- 3. king - man + woman = queen -------------------------------------------
def embedding_arithmetic():
    pts = {
        "man":   np.array([1.00, 1.00]),
        "woman": np.array([1.45, 2.25]),
        "king":  np.array([3.20, 1.35]),
        "queen": np.array([3.65, 2.60]),
    }
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for name, p in pts.items():
        ax.plot(*p, "o", color=SLATE, ms=8, zorder=4)
        dy = -0.22 if name in ("man", "king") else 0.14
        ax.text(p[0], p[1] + dy, name, ha="center", fontsize=15, color=INK,
                style="italic", zorder=5)

    # the "gender direction", twice
    for a, b in [("man", "woman")]:
        ax.annotate("", xy=pts[b], xytext=pts[a],
                    arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=2.2, mutation_scale=16))
    ax.text(0.92, 1.68, "woman − man", color=SAGE, fontsize=12, ha="right")

    # king + (woman - man) lands on queen
    ax.annotate("", xy=pts["queen"], xytext=pts["king"],
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=2.2, ls="--", mutation_scale=16))
    ax.text(3.78, 1.95, "king − man + woman", color=RUST, fontsize=12)
    circ = plt.Circle(pts["queen"], 0.16, fill=False, color=RUST, lw=1.8, zorder=3)
    ax.add_patch(circ)

    ax.text(2.3, 0.35, "a 2-D shadow of a 300-dimensional embedding space ($\\to$ L3)",
            ha="center", color=MUTED, fontsize=11)
    ax.set_xlim(0.2, 4.9); ax.set_ylim(0.1, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "embedding_arithmetic")


# ---- 4. the AI stack, annotated with its math ---------------------------------
def ai_stack():
    stages = [
        ("DATA",       "vectors · matrices",        "Module 1 · L3–L6",   "features on\nwrong scales"),
        ("MODEL",      "linear maps · distributions", "Modules 1 & 3",            "singular\nmatrix"),
        ("LOSS",       "likelihood · cross-entropy",  "L14 · L22–L24",     "loss = nan"),
        ("OPTIMIZER",  "gradients · convexity",       "Modules 2 & 4",            "divergence"),
        ("DEPLOYMENT", "floating point · compression", "L2 · L23",             "float16\noverflow"),
    ]
    xs = [1.45, 4.05, 6.65, 9.25, 11.85]

    fig, ax = plt.subplots(figsize=(13, 4.4), layout="constrained")
    ax.set_xlim(0, 13.3); ax.set_ylim(0, 4.6); ax.axis("off")

    ax.text(6.65, 4.28, "what breaks without the math", ha="center",
            color=WINE, fontsize=12, style="italic")

    for (name, math_, module, fail), x in zip(stages, xs):
        box = FancyBboxPatch((x - 1.1, 1.95), 2.2, 0.95, boxstyle="round,pad=0.05",
                             facecolor=PAPER_ALT, edgecolor=SLATE, lw=1.6)
        ax.add_patch(box)
        ax.text(x, 2.43, name, ha="center", va="center", fontsize=15,
                color=SLATE, fontweight="bold")
        ax.text(x, 1.52, math_, ha="center", va="center", fontsize=11.5, color=INK)
        ax.text(x, 1.12, module, ha="center", va="center", fontsize=10.5,
                color=RUST, style="italic")
        ax.text(x, 3.62, f"“{fail}”".replace("\n", "\n"), ha="center",
                va="center", fontsize=10.5, color=WINE)

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2 - 1.2, 2.43), xytext=(x1 + 1.2, 2.43),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=15))

    ax.text(6.65, 0.35, "the math each stage runs on — and the lectures that deliver it",
            ha="center", color=MUTED, fontsize=12, style="italic")
    save(fig, "ai_stack")


# ---- 5. the 26-lecture narrative arc ------------------------------------------
def course_arc():
    arcs = [
        (1, 6,  SLATE, "numbers & shapes", "storing and transforming data"),
        (7, 11, RUST,  "change",           "derivatives, gradients, autodiff"),
        (12, 15, SAGE, "uncertainty",      "distributions, MLE, priors"),
        (16, 21, OCHRE, "search",          "optimization, free & constrained"),
        (22, 26, WINE, "communication",    "entropy, compression, sequences"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.8), layout="constrained")
    ax.set_xlim(0, 27.6); ax.set_ylim(-1.35, 2.45); ax.axis("off")

    for lo, hi, color, name, desc in arcs:
        ax.add_patch(plt.Rectangle((lo - 0.45, -0.18), hi - lo + 0.9, 0.36,
                                   facecolor=color, edgecolor="none", alpha=0.9))
        mid = (lo + hi) / 2
        ax.text(mid, 0.92, name, ha="center", fontsize=14, color=color, fontweight="bold")
        ax.text(mid, 0.52, desc, ha="center", fontsize=10, color=MUTED, style="italic")

    for lec in range(1, 27):
        ax.vlines(lec - 0.5, -0.18, 0.18, color=PAPER, lw=1.2) if lec > 1 else None
        ax.text(lec, -0.48, str(lec), ha="center", fontsize=8, color=MUTED)

    for i, qx in enumerate([6.5, 11.5, 15.5, 21.5], start=1):
        ax.plot(qx, -0.78, marker="^", ms=7, color=INK)
        ax.text(qx, -1.12, f"quiz {i}", ha="center", fontsize=9, color=MUTED)

    ax.plot(26, 1.62, marker="*", ms=20, color=RUST)
    ax.text(25.4, 1.98, "L26: train the n-gram language model — every arc ends here",
            ha="right", fontsize=12, color=RUST, style="italic")
    ax.annotate("", xy=(26, 0.28), xytext=(26, 1.42),
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.5, mutation_scale=13))
    save(fig, "course_arc")


# ---- 6. eigenvectors: the directions that don't turn ---------------------------
def eigenvector_demo():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    V = np.c_[np.cos(th), np.sin(th)]
    u = np.array([1, 1]) / np.sqrt(2)     # eigenvalue 3
    w = np.array([1, -1]) / np.sqrt(2)    # eigenvalue 1

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")

    def style_of(v):
        if abs(v @ u) > 0.999:
            return RUST, 2.4
        if abs(v @ w) > 0.999:
            return SAGE, 2.4
        return RULE, 1.4

    def arrow(ax, v, color, lw):
        ax.annotate("", xy=tuple(v), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=13))

    for ax in (a1, a2):
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.axhline(0, color=RULE, lw=0.7); ax.axvline(0, color=RULE, lw=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    for v in V:
        c, lw = style_of(v)
        arrow(a1, v, c, lw)
        arrow(a2, A @ v, c, lw)

    # eigen-direction guide lines on the "after" panel
    a2.plot([-3.3, 3.3], [-3.3, 3.3], ls=":", color=RUST, lw=1, alpha=0.6)
    a2.plot([-3.3, 3.3], [3.3, -3.3], ls=":", color=SAGE, lw=1, alpha=0.6)
    a2.text(2.15, 2.75, "stretched ×3,\nnot turned", color=RUST, fontsize=11)
    a2.text(1.55, -1.75, "kept ×1,\nnot turned", color=SAGE, fontsize=11)
    a2.text(-3.3, 3.1, "every other\ndirection turns", color=MUTED, fontsize=10, ha="left")

    a1.set_title("16 directions, before", color=INK)
    a2.set_title("after applying A — two directions refuse to turn", color=INK)
    fig.suptitle("A = [[2, 1], [1, 2]] acting on the unit circle", fontsize=13, color=MUTED)
    save(fig, "eigenvector_demo")


if __name__ == "__main__":
    print("Generating Lecture 1 figures...")
    nan_loss()
    lr_two_problems()
    embedding_arithmetic()
    ai_stack()
    course_arc()
    eigenvector_demo()
    print("Done: all Lecture 1 figures saved to", PNG)
