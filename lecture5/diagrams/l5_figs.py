#!/usr/bin/env python3
"""Figures for L5 (Eigendecomposition).
Run from repo root:  uv run --no-project --with matplotlib,numpy python3 lecture5/diagrams/l5_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture5/figures/.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture5/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')

A = np.array([[2.0, 1.0], [1.0, 2.0]])

def arrow(ax, tip, color, lw=2.4, tail=(0, 0), ls='-', alpha=1.0, z=4, ms=14):
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                alpha=alpha, mutation_scale=ms, shrinkA=0, shrinkB=0))

def clean(ax, lim, aspect=True):
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    if aspect: ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
    ax.axvline(0, color=MUTED, lw=0.5, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


# ---- 1. repeated application: x, Ax, A^2 x drift onto the eigen-line ----------
def repeated_application():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")

    # left: the raw iterates x, Ax, A^2x — each ~3x longer AND closer to y=x
    x = np.array([1.0, 0.0])
    pts = [x, A @ x, A @ A @ x]                      # (1,0), (2,1), (5,4)
    cols = [MUTED, TEAL, ACC]
    labs = [r"$x$", r"$Ax$", r"$A^2x$"]
    clean(a1, ((-0.6, 7.0), (-0.6, 6.2)))
    a1.plot([-0.5, 6.6], [-0.5, 6.6], ls=":", color=ACC, lw=1.2, alpha=0.6, zorder=1)
    a1.text(5.75, 4.85, "eigen-line\n$y=x$", color=ACC, fontsize=11, ha="left")
    for p, c, l in zip(pts, cols, labs):
        arrow(a1, p, c)
        a1.annotate(l, xy=p, xytext=(8, -2), textcoords="offset points", color=c, fontsize=14)
    a1.set_title("each application: ×≈3 longer, less turned", color=INK, fontsize=13)

    # right: the same iterates, normalized — the direction converges to 45°
    clean(a2, ((-0.15, 1.25), (-0.15, 1.25)))
    th = np.linspace(-0.1, np.pi / 2 + 0.1, 100)
    a2.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.8, alpha=0.45, zorder=1)
    a2.plot([0, 1.2], [0, 1.2], ls=":", color=ACC, lw=1.2, alpha=0.6, zorder=1)
    xs, xk = [], np.array([1.0, 0.0])
    for _ in range(7):
        xs.append(xk / np.linalg.norm(xk))
        xk = A @ xk
    shades = [MUTED, "#5A7B7C", TEAL, "#3E8A6E", GREEN, "#7F9A2E", ACC]
    for k, (v, c) in enumerate(zip(xs, shades)):
        arrow(a2, v, c, lw=2.0)
        if k <= 2:
            a2.annotate(f"$k={k}$", xy=v, xytext=(9, -3), textcoords="offset points",
                        color=c, fontsize=11)
    a2.annotate(r"$k\geq 3$", xy=xs[-1], xytext=(10, 4), textcoords="offset points",
                color=ACC, fontsize=11)
    a2.text(0.72, 1.08, r"$v_1=\frac{1}{\sqrt{2}}\binom{1}{1}$", color=ACC, fontsize=12)
    a2.set_title("normalized: the direction locks onto $v_1$", color=INK, fontsize=13)
    save(fig, "repeated_application")


# ---- 2. the eigenvalue dial: stretch / flip / rotate ---------------------------
def eigen_gallery():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.6, 4.4), layout="constrained")
    s = 1 / np.sqrt(2)

    # (a) our A: lambda = 3 stretches, lambda = 1 keeps
    clean(a1, ((-3.3, 3.3), (-3.3, 3.3)))
    a1.plot([-3.1, 3.1], [-3.1, 3.1], ls=":", color=ACC, lw=1, alpha=0.5, zorder=1)
    a1.plot([-3.1, 3.1], [3.1, -3.1], ls=":", color=TEAL, lw=1, alpha=0.5, zorder=1)
    v1 = np.array([s, s]); w = np.array([s, -s])
    arrow(a1, v1, ACC, lw=1.6, ls="--", alpha=0.65)          # before
    arrow(a1, 3 * v1, ACC)                                   # after: honestly x3
    arrow(a1, w, TEAL)                                       # lambda=1: after == before
    a1.annotate(r"$Av_1 = 3v_1$", xy=(-0.05, 2.35), color=ACC, fontsize=12, ha="right")
    a1.annotate(r"$Aw = w$", xy=(0.85, -1.35), color=TEAL, fontsize=12)
    a1.set_title(r"$\lambda > 0$: stretch", color=INK, fontsize=13)

    # (b) reflection F = [[0,1],[1,0]]: lambda = -1 reverses the arrow
    clean(a2, ((-2.4, 2.4), (-2.4, 2.4)))
    a2.plot([-2.3, 2.3], [2.3, -2.3], ls=":", color=RED, lw=1, alpha=0.5, zorder=1)
    u = 1.6 * np.array([s, -s])
    arrow(a2, u, RED, lw=1.6, ls="--", alpha=0.6)            # before
    arrow(a2, -u, RED)                                       # after: reversed
    arrow(a2, 1.6 * np.array([s, s]), TEAL)                  # lambda=+1 direction kept
    a2.annotate(r"$Fu = -u$", xy=(-2.25, 0.85), color=RED, fontsize=12)
    a2.annotate(r"$\lambda = +1$", xy=(1.0, 1.45), color=TEAL, fontsize=12)
    a2.annotate("same line,\narrow reversed", xy=(0.3, -1.95), color=MUTED, fontsize=10.5)
    a2.set_title(r"$\lambda < 0$: flip", color=INK, fontsize=13)

    # (c) rotation by 35 degrees: every direction turns — no real eigenvector
    clean(a3, ((-2.4, 2.4), (-2.4, 2.4)))
    th = np.linspace(0, 2 * np.pi, 200)
    a3.plot(1.8 * np.cos(th), 1.8 * np.sin(th), color=MUTED, lw=0.7, alpha=0.4, zorder=1)
    ang = np.deg2rad(35.0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        v = 1.8 * np.array([np.cos(t), np.sin(t)])
        arrow(a3, v, MUTED, lw=1.2, ls="--", alpha=0.55, ms=10)
        arrow(a3, R @ v, TEAL, lw=1.6, ms=11)
    a3.set_title("rotation: every direction turns", color=INK, fontsize=13)
    a3.annotate(r"$\lambda = \cos 35°\pm i \sin 35°$", xy=(-2.3, -2.25), color=MUTED, fontsize=11)
    save(fig, "eigen_gallery")


# ---- 3. spectral theorem picture: circle -> ellipse, axes = eigenvectors -------
def spectral_ellipse():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0), layout="constrained")
    th = np.linspace(0, 2 * np.pi, 240)
    C = np.vstack([np.cos(th), np.sin(th)])
    s = 1 / np.sqrt(2)

    # (a) symmetric A: eigenvectors orthogonal AND they are the ellipse axes
    E = A @ C
    clean(a1, ((-3.4, 3.4), (-3.05, 3.05)))
    a1.plot(C[0], C[1], color=MUTED, lw=1.1, ls="--", alpha=0.6)
    a1.plot(E[0], E[1], color=TEAL, lw=2.2)
    arrow(a1, (3 * s, 3 * s), ACC)
    arrow(a1, (s, -s), GREEN)
    a1.annotate(r"$3v_1$", xy=(1.55, 2.25), color=ACC, fontsize=13)
    a1.annotate(r"$1\cdot v_2$", xy=(0.8, -1.15), color=GREEN, fontsize=13)
    a1.annotate("unit circle", xy=(-1.75, 0.7), color=MUTED, fontsize=11)
    a1.set_title(r"$A=A^\top$: eigenvectors $\perp$ — they ARE the axes", color=INK, fontsize=13)

    # (b) shear S: eigenvectors exist but are not orthogonal, not the axes
    S = np.array([[2.0, 1.0], [0.0, 1.0]])
    E2 = S @ C
    clean(a2, ((-3.4, 3.4), (-3.05, 3.05)))
    a2.plot(C[0], C[1], color=MUTED, lw=1.1, ls="--", alpha=0.6)
    a2.plot(E2[0], E2[1], color=TEAL, lw=2.2)
    # true ellipse axes (singular directions), for contrast
    U, sv, Vt = np.linalg.svd(S)
    for i, c in enumerate((MUTED, MUTED)):
        ax_v = U[:, i] * sv[i]
        a2.plot([-ax_v[0], ax_v[0]], [-ax_v[1], ax_v[1]], ls=":", color=c, lw=1.1, alpha=0.8)
    arrow(a2, (2.0, 0.0), ACC)                                # eigvec (1,0), lambda=2
    arrow(a2, (-s, s), GREEN)                                 # eigvec (-1,1)/sqrt2, lambda=1
    a2.annotate(r"$\lambda=2$", xy=(1.35, 0.18), color=ACC, fontsize=13)
    a2.annotate(r"$\lambda=1$", xy=(-1.5, 0.95), color=GREEN, fontsize=13)
    a2.annotate("ellipse axes\n(dotted) ≠ eigenvectors", xy=(0.45, -2.5), color=MUTED, fontsize=10.5)
    a2.set_title(r"$S\neq S^\top$: eigenvectors skewed, off-axis", color=INK, fontsize=13)
    save(fig, "spectral_ellipse")


if __name__ == "__main__":
    for f in (repeated_application, eigen_gallery, spectral_ellipse):
        f()
    print("done: L5 figures")
