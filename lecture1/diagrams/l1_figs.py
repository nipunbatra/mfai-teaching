#!/usr/bin/env python3
"""Figures for L1 (Why Math for AI + the Course Map).
Run from repo root:  python3 lecture1/diagrams/l1_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture1/figures/.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
PAPER_ALT='#EFEEEB'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture1/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')


# ---- 1. the loss that became NaN --------------------------------------------
def nan_loss():
    rng = np.random.default_rng(7)
    steps = np.arange(0, 900)
    loss = 4.2 * np.exp(-steps / 260) + 0.55 + rng.normal(0, 0.045, steps.size)
    loss[600:618] = np.geomspace(loss[599], 5e4, 18)   # the blow-up
    loss[618:] = np.nan                                 # and then... nothing
    fig, ax = plt.subplots(figsize=(9.5, 4.0), layout="constrained")
    ax.semilogy(steps, loss, color=INK, lw=2)
    ax.axvline(617, color=RED, ls="--", lw=1.3)
    ax.axvspan(617, 900, color=RED, alpha=0.06)
    ax.text(645, 60, "step 618 onwards:\nloss = nan, forever", color=RED, fontsize=12)
    ax.annotate("training going nicely…", xy=(430, 0.85), xytext=(140, 12),
                color=MUTED, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlabel("training step"); ax.set_ylabel("loss (log scale)")
    ax.set_xlim(0, 900); ax.tick_params(length=3)
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    a1.semilogy(iters, gd_loss(0.02), "o-", color=TEAL, ms=4, lw=1.8)
    a1.set_title("problem A · $\\eta = 0.8$ crawls", color=TEAL)
    a1.text(13, 6.5e-3, "after 30 steps:\nstill 38% of the\nstarting loss", color=MUTED, fontsize=11)
    a2.semilogy(iters, gd_loss(3.2), "o-", color=RED, ms=4, lw=1.8)
    a2.set_title("problem B · $\\eta = 0.8$ explodes", color=RED)
    a2.text(2.5, 1e8, "each step lands\nfarther from the\nminimum than the last", color=MUTED, fontsize=11)
    for ax in (a1, a2):
        ax.set_xlabel("gradient-descent iteration"); ax.set_ylabel("loss"); ax.tick_params(length=3)
    save(fig, "lr_two_problems")


# ---- 3. king - man + woman = queen -------------------------------------------
def embedding_arithmetic():
    pts = {
        "man":   np.array([1.00, 1.00]),
        "woman": np.array([1.45, 2.25]),
        "king":  np.array([3.20, 1.35]),
        "queen": np.array([3.65, 2.60]),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for name, p in pts.items():
        ax.plot(*p, "o", color=INK, ms=8, zorder=4)
        dy = -0.22 if name in ("man", "king") else 0.14
        ax.text(p[0], p[1] + dy, name, ha="center", fontsize=15, color=INK,
                style="italic", zorder=5)
    ax.annotate("", xy=pts["woman"], xytext=pts["man"],
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2, mutation_scale=16))
    ax.text(0.92, 1.68, "woman − man", color=TEAL, fontsize=12, ha="right")
    ax.annotate("", xy=pts["queen"], xytext=pts["king"],
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2.2, ls="--", mutation_scale=16))
    ax.text(3.78, 1.95, "king − man + woman", color=ACC, fontsize=12)
    ax.add_patch(plt.Circle(pts["queen"], 0.16, fill=False, color=ACC, lw=1.8, zorder=3))
    ax.text(2.3, 0.35, "a 2-D shadow of a 300-dimensional embedding space ($\\to$ L3)",
            ha="center", color=MUTED, fontsize=11)
    ax.set_xlim(0.2, 4.9); ax.set_ylim(0.1, 3.1)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "embedding_arithmetic")


# ---- 4. the AI stack, annotated with its math ---------------------------------
def ai_stack():
    stages = [
        ("DATA",       "vectors · matrices",           "M1 · L3–L6",    "features on\nwrong scales"),
        ("MODEL",      "linear maps · distributions",  "M1 & M3",       "singular\nmatrix"),
        ("LOSS",       "likelihood · cross-entropy",   "L14 · L22–L24", "loss = nan"),
        ("OPTIMIZER",  "gradients · convexity",        "M2 & M4",       "divergence"),
        ("DEPLOYMENT", "floating point · compression", "L2 · L23",      "float16\noverflow"),
    ]
    xs = [1.45, 4.05, 6.65, 9.25, 11.85]
    fig, ax = plt.subplots(figsize=(13, 4.4), layout="constrained")
    ax.set_xlim(0, 13.3); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(6.65, 4.28, "what breaks without the math", ha="center",
            color=RED, fontsize=12, style="italic")
    for (name, math_, module, fail), x in zip(stages, xs):
        box = FancyBboxPatch((x - 1.1, 1.95), 2.2, 0.95, boxstyle="round,pad=0.05",
                             facecolor=PAPER_ALT, edgecolor=INK, lw=1.6)
        ax.add_patch(box)
        ax.text(x, 2.43, name, ha="center", va="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(x, 1.52, math_, ha="center", va="center", fontsize=11.5, color=INK)
        ax.text(x, 1.12, module, ha="center", va="center", fontsize=10.5,
                color=ACC, style="italic")
        ax.text(x, 3.62, f"“{fail}”", ha="center", va="center", fontsize=10.5, color=RED)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2 - 1.2, 2.43), xytext=(x1 + 1.2, 2.43),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=15))
    ax.text(6.65, 0.35, "the math each stage runs on — and the lectures that deliver it",
            ha="center", color=MUTED, fontsize=12, style="italic")
    save(fig, "ai_stack")


# ---- 5. the 26-lecture narrative arc ------------------------------------------
def course_arc():
    arcs = [
        (1, 6,   INK,   "numbers & shapes", "storing and transforming data"),
        (7, 11,  ACC,   "change",           "derivatives, gradients, autodiff"),
        (12, 15, TEAL,  "uncertainty",      "distributions, MLE, priors"),
        (16, 21, BLUE,  "search",           "optimization, free & constrained"),
        (22, 26, RED,   "communication",    "entropy, compression, sequences"),
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
        if lec > 1:
            ax.vlines(lec - 0.5, -0.18, 0.18, color="white", lw=1.2)
        ax.text(lec, -0.48, str(lec), ha="center", fontsize=8, color=MUTED)
    for i, qx in enumerate([6.5, 11.5, 15.5, 21.5], start=1):
        ax.plot(qx, -0.78, marker="^", ms=7, color=INK)
        ax.text(qx, -1.12, f"quiz {i}", ha="center", fontsize=9, color=MUTED)
    ax.plot(26, 1.62, marker="*", ms=20, color=ACC)
    ax.text(25.4, 1.98, "L26: train the n-gram language model — every arc ends here",
            ha="right", fontsize=12, color=ACC, style="italic")
    ax.annotate("", xy=(26, 0.28), xytext=(26, 1.42),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.5, mutation_scale=13))
    save(fig, "course_arc")


# ---- 6. eigenvectors: the directions that don't turn ---------------------------
def eigenvector_demo():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    V = np.c_[np.cos(th), np.sin(th)]
    u = np.array([1, 1]) / np.sqrt(2)     # eigenvalue 3
    w = np.array([1, -1]) / np.sqrt(2)    # eigenvalue 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    def style_of(v):
        if abs(v @ u) > 0.999: return ACC, 2.4
        if abs(v @ w) > 0.999: return TEAL, 2.4
        return MUTED, 1.1
    def arrow(ax, v, color, lw):
        ax.annotate("", xy=tuple(v), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=13))
    for ax in (a1, a2):
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5); ax.axvline(0, color=MUTED, lw=0.5, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    for v in V:
        c, lw = style_of(v)
        arrow(a1, v, c, lw)
        arrow(a2, A @ v, c, lw)
    a2.plot([-3.3, 3.3], [-3.3, 3.3], ls=":", color=ACC, lw=1, alpha=0.6)
    a2.plot([-3.3, 3.3], [3.3, -3.3], ls=":", color=TEAL, lw=1, alpha=0.6)
    a2.text(2.15, 2.75, "stretched ×3,\nnot turned", color=ACC, fontsize=11)
    a2.text(1.55, -1.75, "kept ×1,\nnot turned", color=TEAL, fontsize=11)
    a2.text(-3.3, 3.1, "every other\ndirection turns", color=MUTED, fontsize=10, ha="left")
    a1.set_title("16 directions, before", color=INK)
    a2.set_title("after applying A — two refuse to turn", color=INK)
    save(fig, "eigenvector_demo")


if __name__ == "__main__":
    for f in (nan_loss, lr_two_problems, embedding_arithmetic, ai_stack, course_arc, eigenvector_demo):
        f()
    print("done: L1 figures")
