#!/usr/bin/env python3
"""Figures for L13 (Multivariate Gaussians).
Run from repo root:  uv run --no-project --with matplotlib,numpy python3 lecture13/diagrams/l13_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture13/figures/.
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
OUT='lecture13/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')

# the lecture's running covariance — L5's matrix, returning as Σ
SIG = np.array([[2.0, 1.0], [1.0, 2.0]])
MU = np.array([3.0, 5.0])
rng = np.random.default_rng(13)


def ellipse_pts(mu, cov, c, n=240):
    """Points on the level set (x-mu)^T cov^-1 (x-mu) = c^2 : semi-axes c*sqrt(lam)."""
    lam, Q = np.linalg.eigh(cov)               # ascending
    th = np.linspace(0, 2 * np.pi, n)
    E = Q @ (c * np.sqrt(lam)[:, None] * np.vstack([np.cos(th), np.sin(th)]))
    return mu[0] + E[0], mu[1] + E[1]


def pdf2(X, Y, mu, cov):
    Si = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    dx, dy = X - mu[0], Y - mu[1]
    q = Si[0, 0] * dx * dx + 2 * Si[0, 1] * dx * dy + Si[1, 1] * dy * dy
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


# ---- 1. sampled clouds match the theory ellipses (isotropic / diagonal / full) ----
def gallery_clouds():
    covs = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]),   r"isotropic  $\Sigma = I$"),
        (np.array([[2.25, 0.0], [0.0, 0.36]]), r"diagonal  $\Sigma=\mathrm{diag}(2.25,\ 0.36)$"),
        (SIG,                                   r"full  $\Sigma=[[2,1],[1,2]]$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), layout="constrained")
    for ax, (cov, title) in zip(axes, covs):
        pts = rng.multivariate_normal([0, 0], cov, size=900)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, alpha=0.18, linewidths=0)
        for c, lw, al in ((1.0, 1.8, 0.9), (2.0, 2.4, 1.0)):
            ex, ey = ellipse_pts(np.zeros(2), cov, c)
            ax.plot(ex, ey, color=ACC, lw=lw, alpha=al)
        ax.plot(0, 0, 'o', color=RED, ms=6, zorder=5)
        ax.set_xlim(-4.6, 4.6); ax.set_ylim(-4.6, 4.6); ax.set_aspect("equal")
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(labelsize=10)
        ax.set_title(title, color=INK, fontsize=12.5)
    axes[0].annotate(r"$1\sigma$ and $2\sigma$ ellipses", xy=(0.6, 2.35), color=ACC, fontsize=11)
    save(fig, "gallery_clouds")


# ---- 2. project = marginalize: joint contours + both marginals on the walls -------
def joint_marginal():
    fig = plt.figure(figsize=(8.6, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(4.2, 1.25), height_ratios=(1.25, 4.2),
                          wspace=0.02, hspace=0.02)
    axJ = fig.add_subplot(gs[1, 0])
    axX = fig.add_subplot(gs[0, 0], sharex=axJ)
    axY = fig.add_subplot(gs[1, 1], sharey=axJ)

    xs = np.linspace(-1.5, 7.5, 220); ys = np.linspace(0.5, 9.5, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = pdf2(X, Y, MU, SIG)
    axJ.contour(X, Y, Z, levels=7, colors=TEAL, linewidths=1.4)
    pts = rng.multivariate_normal(MU, SIG, size=350)
    axJ.scatter(pts[:, 0], pts[:, 1], s=6, color=INK, alpha=0.15, linewidths=0)
    axJ.plot(*MU, 'o', color=RED, ms=6)
    axJ.annotate(r"$\boldsymbol{\mu}=(3,5)$", xy=MU, xytext=(6, -14),
                 textcoords="offset points", color=RED, fontsize=11)
    axJ.set_xlabel("$x$"); axJ.set_ylabel("$y$")
    axJ.set_xlim(-1.5, 7.5); axJ.set_ylim(0.5, 9.5)

    # top wall: p(x) = N(3, 2)
    px = np.exp(-(xs - 3) ** 2 / 4) / np.sqrt(4 * np.pi)
    axX.fill_between(xs, px, color=ACC, alpha=0.25, lw=0)
    axX.plot(xs, px, color=ACC, lw=2.2)
    axX.set_ylim(0, 0.34); axX.axis("off")
    axX.annotate(r"$p(x) = \mathcal{N}(3,\ 2)$", xy=(4.6, 0.22), color=ACC, fontsize=12)
    axX.annotate("", xy=(0.6, 0.02), xytext=(0.6, 0.30),
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    axX.annotate("squash\n$y$ flat", xy=(-1.4, 0.10), color=MUTED, fontsize=10)

    # right wall: p(y) = N(5, 2)
    py = np.exp(-(ys - 5) ** 2 / 4) / np.sqrt(4 * np.pi)
    axY.fill_betweenx(ys, py, color=GREEN, alpha=0.22, lw=0)
    axY.plot(py, ys, color=GREEN, lw=2.2)
    axY.set_xlim(0, 0.34); axY.axis("off")
    axY.annotate(r"$p(y)$" + "\n" + r"$=\mathcal{N}(5,2)$", xy=(0.11, 8.35), color=GREEN, fontsize=12)
    save(fig, "joint_marginal")


# ---- 3. slice = condition: three slices, means on a line, same width --------------
def conditional_slices():
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.6, 4.9), layout="constrained")
    xs = np.linspace(-1.5, 7.5, 220); ys = np.linspace(0.5, 9.5, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = pdf2(X, Y, MU, SIG)
    aL.contour(X, Y, Z, levels=7, colors=MUTED, linewidths=1.0, alpha=0.75)
    cols = [TEAL, GREEN, ACC]
    y0s = [3.0, 5.0, 7.0]
    for y0, c in zip(y0s, cols):
        aL.axhline(y0, color=c, lw=2.2)
        m = 3 + 0.5 * (y0 - 5)                       # conditional mean of x
        aL.plot(m, y0, 'o', color=c, ms=7, zorder=5)
        aL.annotate(rf"$y={y0:.0f}$", xy=(6.6, y0 + 0.15), color=c, fontsize=11)
    yy = np.linspace(1.4, 8.6, 50)
    aL.plot(3 + 0.5 * (yy - 5), yy, ls="--", color=RED, lw=1.6)
    aL.annotate("slice means fall\non a straight line", xy=(-1.1, 7.7), color=RED, fontsize=11)
    aL.set_xlabel("$x$"); aL.set_ylabel("$y$")
    aL.set_xlim(-1.5, 7.5); aL.set_ylim(0.5, 9.5)
    aL.set_title("cut the hill along $y = y_0$", color=INK, fontsize=13)

    var = 1.5   # 2 - 1^2/2
    for y0, c in zip(y0s, cols):
        m = 3 + 0.5 * (y0 - 5)
        pdf = np.exp(-(xs - m) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
        aR.plot(xs, pdf, color=c, lw=2.4, label=rf"$p(x \mid y={y0:.0f}) = \mathcal{{N}}({m:.0f},\ 1.5)$")
    aR.legend(fontsize=11, frameon=False, loc="upper left")
    aR.set_xlim(-1.5, 7.5); aR.set_ylim(0, 0.47)
    aR.set_xlabel("$x$"); aR.set_ylabel(r"$p(x \mid y_0)$")
    aR.set_title("each slice, renormalized: mean moves, width doesn't", color=INK, fontsize=13)
    save(fig, "conditional_slices")


# ---- 4. Mahalanobis: two points, same Euclidean distance, different verdicts ------
def mahalanobis_scene():
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.6, 5.2), layout="constrained")
    pts = rng.multivariate_normal([0, 0], SIG, size=900)
    A = np.array([1.0, 1.0]); B = np.array([1.0, -1.0])

    for ax in (aL, aR):
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=INK, alpha=0.15, linewidths=0)
        ax.plot(*A, 'o', color=GREEN, ms=9, zorder=6)
        ax.plot(*B, 's', color=RED, ms=9, zorder=6)
        ax.plot(0, 0, 'o', color=INK, ms=5, zorder=6)
        ax.set_xlim(-4.3, 4.3); ax.set_ylim(-4.3, 4.3); ax.set_aspect("equal")
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(labelsize=10)
        ax.annotate("$A=(1,1)$", xy=A, xytext=(10, 4), textcoords="offset points",
                    color=GREEN, fontsize=12)
        ax.annotate("$B=(1,-1)$", xy=B, xytext=(10, -10), textcoords="offset points",
                    color=RED, fontsize=12)

    th = np.linspace(0, 2 * np.pi, 200)
    r = np.sqrt(2)
    aL.plot(r * np.cos(th), r * np.sin(th), ls="--", color=BLUE, lw=2.0)
    aL.annotate("Euclidean circle\nradius $\\sqrt{2}$: a tie", xy=(-4.05, 3.3), color=BLUE, fontsize=11.5)
    aL.set_title(r"Euclidean: $\Vert A\Vert=\Vert B\Vert=\sqrt{2}$ — tie", color=INK, fontsize=13)

    for c, col, lab, xy in ((np.sqrt(2 / 3), GREEN, r"$d_M=0.82$ through $A$", (-4.05, 3.6)),
                            (np.sqrt(2.0), RED, r"$d_M=1.41$ through $B$", (-4.05, 2.85))):
        ex, ey = ellipse_pts(np.zeros(2), SIG, c)
        aR.plot(ex, ey, color=col, lw=2.2)
        aR.annotate(lab, xy=xy, color=col, fontsize=11.5)
    aR.set_title(r"Mahalanobis: $B$ is $\sqrt{3}\times$ farther out", color=INK, fontsize=13)
    save(fig, "mahalanobis_scene")


# ---- 5. CLT: standardized sums of uniforms morph into the bell --------------------
def clt_uniforms():
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), layout="constrained")
    ns = [1, 2, 4, 16]
    grid = np.linspace(-4, 4, 300)
    bell = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    for ax, n in zip(axes, ns):
        S = rng.random((60000, n)).sum(axis=1)
        Zs = (S - n / 2) / np.sqrt(n / 12)         # standardize: mean 0, var 1
        ax.hist(Zs, bins=61, range=(-4, 4), density=True, color=TEAL, alpha=0.75, lw=0)
        ax.plot(grid, bell, color=ACC, lw=2.2)
        ax.set_xlim(-4, 4); ax.set_ylim(0, 0.44)
        ax.set_yticks([]); ax.set_xticks([-3, 0, 3]); ax.tick_params(labelsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_title(f"sum of $n={n}$", color=INK, fontsize=13)
    axes[0].set_title("$n=1$: flat uniform", color=INK, fontsize=13)
    axes[-1].annotate(r"$\mathcal{N}(0,1)$", xy=(1.3, 0.36), color=ACC, fontsize=12)
    save(fig, "clt_uniforms")


if __name__ == "__main__":
    for f in (gallery_clouds, joint_marginal, conditional_slices, mahalanobis_scene, clt_uniforms):
        f()
    print("done: L13 figures")
