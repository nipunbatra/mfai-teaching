#!/usr/bin/env python3
"""Figures for L3 (Vectors: the Geometry of Data).
Run from repo root:  uv run --no-project --with matplotlib,numpy python3 lecture3/diagrams/l3_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture3/figures/.

GloVe numbers (cosine matrix, analogy ranks) were computed once from the real
model — gensim `glove-wiki-gigaword-50` (trained on 6B tokens), 2026-08-11:
    cos(a,b) = a.dot(b) / (norm(a)*norm(b))
and are embedded below verbatim so this script stays offline-reproducible.
They match the stored outputs of psdv-teaching/notebooks/embeddings-angle.ipynb.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
PAPER_ALT='#EFEEEB'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture3/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')


def arrow(ax, tip, tail=(0, 0), color=INK, lw=2.4, ls='-', ms=17, z=4):
    ax.annotate("", xy=tip, xytext=tail, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, mutation_scale=ms, shrinkA=0, shrinkB=0))

def plane(ax, xlim, ylim, grid=True):
    """A clean 2-D plane: faint integer grid + axes through the origin."""
    if grid:
        for gx in range(int(np.floor(xlim[0])), int(np.ceil(xlim[1])) + 1):
            ax.axvline(gx, color=MUTED, lw=0.4, alpha=0.25, zorder=0)
        for gy in range(int(np.floor(ylim[0])), int(np.ceil(ylim[1])) + 1):
            ax.axhline(gy, color=MUTED, lw=0.4, alpha=0.25, zorder=0)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")


# ---- 1. one vector, two readings: arrow / data row ---------------------------
def vector_two_views():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    # (a) the arrow (3, 2)
    plane(a, (-0.9, 4.3), (-0.9, 3.3))
    arrow(a, (3, 2), color=TEAL, lw=3)
    a.plot([3, 3], [0, 2], ls='--', color=MUTED, lw=1.4)
    a.plot([0, 3], [2, 2], ls='--', color=MUTED, lw=1.4)
    a.text(3.12, 2.16, r"$\mathbf{x}=(3,\,2)$", color=TEAL, fontsize=15, fontweight='bold')
    a.text(3, -0.38, r"$x_1 = 3$", ha='center', color=INK, fontsize=13)
    a.text(-0.16, 2, r"$x_2 = 2$", ha='right', va='center', color=INK, fontsize=13)
    a.set_title("an arrow: direction + length", fontsize=14, color=INK)
    # (b) a dataset is a quiver of arrow tips
    pts = np.array([(8, 6), (7, 7.5), (3, 4), (4.5, 3), (6, 8), (9, 8.5), (2, 6), (5.5, 5)])
    plane(b, (-0.9, 10.6), (-0.9, 9.6), grid=False)
    b.plot(pts[:, 0], pts[:, 1], 'o', color=INK, ms=7, zorder=3)
    arrow(b, (8, 6), color=ACC, lw=2.6)
    arrow(b, (3, 4), color=BLUE, lw=2.6)
    b.text(8.15, 5.35, "Asha = (8, 6)", color=ACC, fontsize=13, fontweight='bold')
    b.text(2.2, 4.5, "Vikram = (3, 4)", color=BLUE, fontsize=13, fontweight='bold')
    b.text(9.9, -0.55, "maths", ha='right', color=MUTED, fontsize=12)
    b.text(-0.35, 8.9, "physics", rotation=90, va='top', color=MUTED, fontsize=12)
    b.set_title("a dataset: every row is an arrow tip", fontsize=14, color=INK)
    save(fig, "vector_two_views")


# ---- 2. addition tip-to-tail + scalar stretching -----------------------------
def vector_ops():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    plane(a, (-0.9, 4.2), (-0.9, 3.8))
    u, v = np.array([2, 1]), np.array([1, 2])
    arrow(a, u, color=TEAL, lw=2.8)
    arrow(a, v, color=BLUE, lw=2.8)
    arrow(a, u + v, tail=u, color=BLUE, lw=2.0, ls='--')
    arrow(a, u + v, color=ACC, lw=3)
    a.text(1.45, 0.32, r"$\mathbf{u}$", color=TEAL, fontsize=16, fontweight='bold')
    a.text(0.28, 1.35, r"$\mathbf{v}$", color=BLUE, fontsize=16, fontweight='bold')
    a.text(2.72, 2.05, r"$\mathbf{v}$ again", color=BLUE, fontsize=12)
    a.text(2.6, 3.2, r"$\mathbf{u}+\mathbf{v}=(3,\,3)$", color=ACC, fontsize=15, fontweight='bold')
    a.set_title("addition: walk u, then walk v", fontsize=14, color=INK)

    plane(b, (-2.9, 4.9), (-1.9, 2.9))
    b.plot([-2.9, 4.9], [-1.45, 2.45], color=MUTED, lw=0.9, ls=':', zorder=1)
    arrow(b, (4, 2), color=ACC, lw=2.4)
    arrow(b, (2, 1), color=TEAL, lw=3.2)
    arrow(b, (1, 0.5), color=GREEN, lw=2.4)
    arrow(b, (-2, -1), color=RED, lw=2.4)
    b.text(3.4, 2.35, r"$2\mathbf{u}$", color=ACC, fontsize=15, fontweight='bold')
    b.text(2.05, 0.55, r"$\mathbf{u}$", color=TEAL, fontsize=16, fontweight='bold')
    b.text(0.75, 0.86, r"$\frac{1}{2}\mathbf{u}$", color=GREEN, fontsize=14, fontweight='bold')
    b.text(-2.15, -0.75, r"$-\mathbf{u}$", color=RED, fontsize=15, fontweight='bold', ha='right')
    b.set_title("scaling: stretch, shrink, flip — same line", fontsize=14, color=INK)
    save(fig, "vector_ops")


# ---- 3. L1 walk vs L2 flight -------------------------------------------------
def taxicab():
    fig, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    plane(ax, (-0.6, 4.4), (-0.6, 3.1))
    ax.plot([0, 3], [0, 0], color=BLUE, lw=3.2, solid_capstyle='round', zorder=3)
    ax.plot([3, 3], [0, 2], color=BLUE, lw=3.2, solid_capstyle='round', zorder=3)
    arrow(ax, (3, 2), color=TEAL, lw=3.2)
    ax.plot(3, 2, 'o', color=ACC, ms=10, zorder=5)
    ax.text(3.14, 2.1, "(3, 2)", color=ACC, fontsize=15, fontweight='bold')
    ax.text(1.5, -0.42, r"walk the streets:  $|3|+|2| = 5$", color=BLUE,
            ha='center', fontsize=13.5, fontweight='bold')
    ax.text(1.05, 1.35, "fly straight:\n" + r"$\sqrt{3^2+2^2}=\sqrt{13}\approx 3.61$",
            color=TEAL, fontsize=13.5, fontweight='bold')
    save(fig, "taxicab")


# ---- 4. unit balls of L1, L2, Linf ------------------------------------------
def unit_balls():
    fig, ax = plt.subplots(figsize=(7.0, 5.4), layout="constrained")
    plane(ax, (-1.55, 1.55), (-1.55, 1.55), grid=False)
    t = np.linspace(0, 2 * np.pi, 400)
    # L-inf square
    sq = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)])
    ax.plot(sq[:, 0], sq[:, 1], color=MUTED, lw=2.2, ls='--', zorder=2,
            label=r"$\|\mathbf{x}\|_\infty=1$   square")
    # L1 diamond
    dm = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)])
    ax.fill(dm[:, 0], dm[:, 1], color=ACC, alpha=0.10, zorder=1)
    ax.plot(dm[:, 0], dm[:, 1], color=ACC, lw=2.6, zorder=3,
            label=r"$\|\mathbf{x}\|_1=1$   diamond")
    # L2 circle
    ax.fill(np.cos(t), np.sin(t), color=TEAL, alpha=0.10, zorder=1)
    ax.plot(np.cos(t), np.sin(t), color=TEAL, lw=2.6, zorder=3,
            label=r"$\|\mathbf{x}\|_2=1$   circle")
    for p in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        ax.plot(*p, 'o', color=INK, ms=5, zorder=5)
    ax.text(1.07, 0.07, "(1, 0)", color=INK, fontsize=11.5)
    ax.text(0.08, 1.08, "(0, 1)", color=INK, fontsize=11.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]  # circle, diamond, square
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, fontsize=13.5, loc='center left',
              bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    save(fig, "unit_balls")


# ---- 5. three angles, three cosines ------------------------------------------
def angle_gallery():
    cases = [(30, "0.87", "similar", TEAL), (90, "0", "unrelated", MUTED),
             (150, "−0.87", "opposite", RED)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), layout="constrained")
    for ax, (deg, cval, word, col) in zip(axes, cases):
        plane(ax, (-1.5, 1.5), (-0.45, 1.35), grid=False)
        th = np.radians(deg)
        arrow(ax, (1.25, 0), color=INK, lw=2.6)
        arrow(ax, (1.25 * np.cos(th), 1.25 * np.sin(th)), color=col, lw=2.6)
        arc = np.linspace(0, th, 60)
        ax.plot(0.38 * np.cos(arc), 0.38 * np.sin(arc), color=col, lw=1.6)
        ax.text(0.56 * np.cos(th / 2), 0.56 * np.sin(th / 2) + 0.02,
                f"{deg}°", color=col, fontsize=13, fontweight='bold',
                ha='center', va='center')
        ax.set_title(f"cos {deg}° = {cval}  —  {word}", fontsize=14.5,
                     color=col, fontweight='bold')
    save(fig, "angle_gallery")


# ---- 6. the law-of-cosines triangle ------------------------------------------
def law_cosines():
    x, y = np.array([4.0, 0.9]), np.array([1.4, 2.7])
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    plane(ax, (-0.5, 4.8), (-0.5, 3.4), grid=False)
    arrow(ax, x, color=TEAL, lw=3)
    arrow(ax, y, color=BLUE, lw=3)
    arrow(ax, x, tail=y, color=ACC, lw=2.6, ls='--')
    th_x, th_y = np.arctan2(x[1], x[0]), np.arctan2(y[1], y[0])
    arc = np.linspace(th_x, th_y, 60)
    ax.plot(0.62 * np.cos(arc), 0.62 * np.sin(arc), color=INK, lw=1.6)
    mid = (th_x + th_y) / 2
    ax.text(0.88 * np.cos(mid), 0.88 * np.sin(mid), r"$\theta$", fontsize=17, color=INK,
            ha='center', va='center')
    ax.text(2.35, 0.28, r"$\mathbf{x}$,  length $\|\mathbf{x}\|$", color=TEAL,
            fontsize=14.5, fontweight='bold')
    ax.text(0.28, 1.75, r"$\mathbf{y}$,  length $\|\mathbf{y}\|$", color=BLUE,
            fontsize=14.5, fontweight='bold', ha='left')
    ax.text(2.95, 2.15, r"$\mathbf{x}-\mathbf{y}$,  length $\|\mathbf{x}-\mathbf{y}\|$",
            color=ACC, fontsize=14.5, fontweight='bold')
    save(fig, "law_cosines")


# ---- 7 & 8. projection: the shadow, generic and with numbers -----------------
def _projection(ax, x, y, annotate_numbers):
    yy = y / np.linalg.norm(y)
    p = (x @ yy) * yy
    lo, hi = -0.6, 1.32
    ax.plot([lo * y[0], hi * y[0]], [lo * y[1], hi * y[1]], color=MUTED, lw=0.9, ls=':', zorder=1)
    arrow(ax, x, color=TEAL, lw=3)
    arrow(ax, y, color=BLUE, lw=2.6)
    arrow(ax, p, color=ACC, lw=3.4)
    ax.plot([x[0], p[0]], [x[1], p[1]], ls='--', color=RED, lw=1.8, zorder=3)
    # right-angle marker at p
    s = 0.17
    e1, e2 = yy, np.array([-yy[1], yy[0]])
    if (x - p) @ e2 < 0: e2 = -e2
    corner = [p - s * e1, p - s * e1 + s * e2, p + s * e2]
    ax.plot([c[0] for c in corner], [c[1] for c in corner], color=INK, lw=1.2, zorder=3)
    if annotate_numbers:
        ax.text(1.32, 1.26, r"$\mathbf{x}=(3,\,2)$", color=TEAL,
                fontsize=14.5, fontweight='bold', ha='center')
        ax.text(2.45, 0.68, r"$\mathbf{y}=(4,\,2)$", color=BLUE,
                fontsize=14.5, fontweight='bold')
        ax.text(p[0] + 0.16, p[1] - 0.24, r"$\mathrm{proj}=(3.2,\,1.6)$", color=ACC,
                fontsize=14.5, fontweight='bold')
        ax.text(3.12, 2.28, r"residual $(-0.2,\,0.4)$", color=RED, fontsize=13.5, ha='left')
    else:
        ax.text(x[0] - 0.1, x[1] + 0.13, r"$\mathbf{x}$", color=TEAL, fontsize=17, fontweight='bold')
        ax.text(y[0] + 0.12, y[1] - 0.14, r"$\mathbf{y}$", color=BLUE, fontsize=17, fontweight='bold')
        ax.text(p[0] + 0.05, p[1] - 0.32, r"$\mathrm{proj}_{\mathbf{y}}(\mathbf{x})$" + "\nthe shadow",
                color=ACC, fontsize=13.5, fontweight='bold')
        ax.text((x[0] + p[0]) / 2 + 0.14, (x[1] + p[1]) / 2, "residual\n(dropped\nstraight down)",
                color=RED, fontsize=11.5)

def projection_shadow():
    fig, ax = plt.subplots(figsize=(6.8, 4.5), layout="constrained")
    plane(ax, (-0.7, 4.6), (-0.7, 3.2), grid=False)
    _projection(ax, np.array([2.1, 2.5]), np.array([3.4, 1.02]), False)
    save(fig, "projection_shadow")

def projection_numbers():
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
    plane(ax, (-0.7, 5.2), (-0.7, 3.0))
    _projection(ax, np.array([3.0, 2.0]), np.array([4.0, 2.0]), True)
    save(fig, "projection_numbers")


# ---- 9. GloVe-50 cosine-similarity heatmap -----------------------------------
GLOVE_WORDS = ["king", "queen", "man", "woman", "prince", "princess", "cricket", "python"]
GLOVE_COS = np.array([  # real GloVe-50 values (see module docstring)
    [1.000, 0.784, 0.531, 0.411, 0.824, 0.602, 0.421, 0.185],
    [0.784, 1.000, 0.537, 0.600, 0.782, 0.852, 0.359, 0.084],
    [0.531, 0.537, 1.000, 0.886, 0.507, 0.441, 0.361, 0.256],
    [0.411, 0.600, 0.886, 1.000, 0.439, 0.577, 0.190, 0.139],
    [0.824, 0.782, 0.507, 0.439, 1.000, 0.743, 0.262, -0.080],
    [0.602, 0.852, 0.441, 0.577, 0.743, 1.000, 0.130, -0.011],
    [0.421, 0.359, 0.361, 0.190, 0.262, 0.130, 1.000, 0.209],
    [0.185, 0.084, 0.256, 0.139, -0.080, -0.011, 0.209, 1.000],
])

def cosine_heatmap():
    cmap = LinearSegmentedColormap.from_list("mfai", [TEAL, PAPER_ALT, ACC])
    fig, ax = plt.subplots(figsize=(7.9, 6.6), layout="constrained")
    im = ax.imshow(GLOVE_COS, cmap=cmap, vmin=-1, vmax=1)
    n = len(GLOVE_WORDS)
    ax.set_xticks(range(n), GLOVE_WORDS, rotation=45, ha='right', fontsize=12.5)
    ax.set_yticks(range(n), GLOVE_WORDS, fontsize=12.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            v = GLOVE_COS[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=11,
                    color='white' if v > 0.9 else INK,
                    fontweight='bold' if (i != j and v >= 0.78) else 'normal')
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label(r"cosine similarity  $\cos\theta$", fontsize=12)
    cb.outline.set_visible(False)
    save(fig, "cosine_heatmap")


# ---- 10. span: a line, then the whole plane ----------------------------------
def span_fig():
    u, v = np.array([2.0, 1.0]), np.array([0.5, 1.6])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    plane(a, (-3.4, 5.4), (-2.2, 3.2), grid=False)
    a.plot([-3.4, 5.4], [-1.7, 2.7], color=TEAL, lw=1.0, ls=':', zorder=1)
    for c in (-1.5, -0.5, 0.5, 1.5, 2.5):
        a.plot(*(c * u), 'o', color=TEAL, ms=5, alpha=0.55, zorder=2)
    arrow(a, u, color=TEAL, lw=3)
    a.text(2.1, 0.5, r"$\mathbf{u}$", color=TEAL, fontsize=16, fontweight='bold')
    a.plot(1.0, 2.4, 'X', color=RED, ms=11, zorder=4)
    a.text(1.2, 2.42, "unreachable", color=RED, fontsize=12.5, va='center')
    a.text(1.0, -1.85, r"span$\{\mathbf{u}\}$: stuck on a line", color=INK,
           fontsize=14, ha='center')
    plane(b, (-1.6, 6.4), (-1.3, 5.4), grid=False)
    for al in range(-1, 4):
        b.plot([al * u[0] - 2 * v[0], al * u[0] + 3 * v[0]],
               [al * u[1] - 2 * v[1], al * u[1] + 3 * v[1]], color=MUTED, lw=0.5, alpha=0.4)
    for be in range(-1, 4):
        b.plot([be * v[0] - 1.5 * u[0], be * v[0] + 3 * u[0]],
               [be * v[1] - 1.5 * u[1], be * v[1] + 3 * u[1]], color=MUTED, lw=0.5, alpha=0.4)
    arrow(b, u, color=TEAL, lw=3)
    arrow(b, v, color=BLUE, lw=3)
    x = 2 * u + 1.5 * v
    arrow(b, 2 * u, color=TEAL, lw=2.0, ls='--')
    arrow(b, x, tail=2 * u, color=BLUE, lw=2.0, ls='--')
    b.plot(*x, 'o', color=ACC, ms=10, zorder=5)
    b.text(x[0] + 0.15, x[1] + 0.1, r"$\mathbf{x} = 2\mathbf{u} + 1.5\mathbf{v}$",
           color=ACC, fontsize=14.5, fontweight='bold')
    b.text(2.15, 0.48, r"$\mathbf{u}$", color=TEAL, fontsize=16, fontweight='bold')
    b.text(0.02, 1.55, r"$\mathbf{v}$", color=BLUE, fontsize=16, fontweight='bold')
    b.text(2.5, -0.95, r"span$\{\mathbf{u},\mathbf{v}\}$: the whole plane", color=INK,
           fontsize=14, ha='center')
    save(fig, "span_fig")


# ---- 11. same arrow, two bases -----------------------------------------------
def basis_coords():
    x = np.array([3.0, 2.0])
    s2 = np.sqrt(2)
    b1, b2 = np.array([1, 1]) / s2, np.array([-1, 1]) / s2
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.7), layout="constrained")
    plane(a, (-1.2, 4.3), (-1.2, 3.3))
    arrow(a, (1, 0), color=INK, lw=2.2)
    arrow(a, (0, 1), color=INK, lw=2.2)
    a.text(1.02, -0.34, r"$\mathbf{e}_1$", color=INK, fontsize=14)
    a.text(-0.42, 1.02, r"$\mathbf{e}_2$", color=INK, fontsize=14)
    arrow(a, x, color=ACC, lw=3.2)
    a.plot([3, 3], [0, 2], ls='--', color=MUTED, lw=1.3)
    a.plot([0, 3], [2, 2], ls='--', color=MUTED, lw=1.3)
    a.text(3.1, 2.14, r"coordinates $(3,\,2)$", color=ACC, fontsize=14.5, fontweight='bold')
    a.set_title(r"basis $\mathbf{e}_1, \mathbf{e}_2$", fontsize=14, color=INK)
    plane(b, (-1.2, 4.3), (-1.2, 3.3), grid=False)
    for c in range(-4, 7):
        q = c * 0.5 * s2
        b.plot([q * b1[0] - 4 * b2[0], q * b1[0] + 4 * b2[0]],
               [q * b1[1] - 4 * b2[1], q * b1[1] + 4 * b2[1]], color=TEAL, lw=0.5, alpha=0.35)
        b.plot([q * b2[0] - 5 * b1[0], q * b2[0] + 5 * b1[0]],
               [q * b2[1] - 5 * b1[1], q * b2[1] + 5 * b1[1]], color=TEAL, lw=0.5, alpha=0.35)
    arrow(b, b1, color=TEAL, lw=2.4)
    arrow(b, b2, color=TEAL, lw=2.4)
    b.text(0.78, 0.42, r"$\mathbf{b}_1$", color=TEAL, fontsize=14)
    b.text(-0.95, 0.52, r"$\mathbf{b}_2$", color=TEAL, fontsize=14)
    arrow(b, x, color=ACC, lw=3.2)
    p1 = (x @ b1) * b1
    b.plot([p1[0], x[0]], [p1[1], x[1]], ls='--', color=MUTED, lw=1.3)
    b.text(2.45, 2.55, r"coordinates $(3.54,\,-0.71)$", color=ACC, fontsize=14.5,
           fontweight='bold', ha='center')
    b.set_title(r"basis $\mathbf{b}_1, \mathbf{b}_2$ (rotated $45°$)", fontsize=14, color=INK)
    fig.suptitle("the same arrow — the numbers depend on the basis", fontsize=15, color=INK)
    save(fig, "basis_coords")


# ---- 12. random vectors in high d are nearly orthogonal ----------------------
def high_d_angles():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.2, 4.2), layout="constrained")
    for d, col in [(3, TEAL), (30, BLUE), (300, ACC)]:
        a = rng.standard_normal((4000, d)); b_ = rng.standard_normal((4000, d))
        cos = np.sum(a * b_, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b_, axis=1))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ax.hist(ang, bins=np.arange(0, 181, 2), density=True, histtype='step',
                lw=2.6, color=col, label=f"d = {d}")
    ax.axvline(90, color=MUTED, lw=1.2, ls='--')
    ax.text(90, ax.get_ylim()[1] * 1.02, "90°", ha='center', color=MUTED, fontsize=12)
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("angle between two random vectors (degrees)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=13)
    save(fig, "high_d_angles")


if __name__ == "__main__":
    for f in (vector_two_views, vector_ops, taxicab, unit_balls, angle_gallery,
              law_cosines, projection_shadow, projection_numbers, cosine_heatmap,
              span_fig, basis_coords, high_d_angles):
        f()
    print("done: L3 figures")
