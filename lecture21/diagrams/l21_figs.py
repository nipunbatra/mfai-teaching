#!/usr/bin/env python3
"""Figures for L21 (linear and quadratic programming)."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

INK = "#23373B"
ACC = "#EB811B"
TEAL = "#2C7A7B"
GREEN = "#14B03D"
MUTED = "#6E7F82"
RED = "#D64550"
BLUE = "#2B6CB0"

mpl.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "font.size": 12,
    }
)

OUT = "lecture21/figures"
os.makedirs(OUT, exist_ok=True)
VERTS = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 3]])


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def base_polytope(ax):
    ax.add_patch(Polygon(VERTS, closed=True, facecolor="#E9F4F4", edgecolor=TEAL, lw=2.2))
    ax.scatter(VERTS[:, 0], VERTS[:, 1], s=35, color=INK, zorder=4)
    ax.set(xlim=(-0.25, 3.25), ylim=(-0.25, 3.55), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.14)


def lp_vertices():
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    base_polytope(ax)
    x = np.linspace(-0.2, 3.2, 200)
    for value in [3, 6, 9, 10]:
        y = (value - 3 * x) / 2
        ax.plot(x, y, color=ACC, lw=1.2, alpha=0.7)
        idx = np.argmin(np.abs(y - 3.35))
        if -0.1 < x[idx] < 3.1:
            ax.text(x[idx], y[idx], f"{value}", color=ACC, fontsize=9)
    values = 3 * VERTS[:, 0] + 2 * VERTS[:, 1]
    for (vx, vy), value in zip(VERTS, values):
        ax.text(vx + 0.06, vy + 0.07, f"{value:g}", fontsize=9, color=MUTED)
    ax.scatter([2], [2], s=85, color=RED, zorder=6)
    ax.annotate(
        r"optimum $(2,2)$, value $10$",
        (2, 2),
        xytext=(2.18, 2.62),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10,
    )
    ax.text(0.3, 1.55, "feasible polytope", color=TEAL)
    ax.set_title(r"maximize $3x_1+2x_2$")
    save(fig, "lp_vertices")


def qp_solution():
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    base_polytope(ax)
    z = np.array([3.0, 1.4])
    xx, yy = np.meshgrid(np.linspace(-0.2, 3.2, 220), np.linspace(-0.2, 3.5, 220))
    dist = 0.5 * ((xx - z[0]) ** 2 + (yy - z[1]) ** 2)
    ax.contour(xx, yy, dist, levels=[0.1, 0.5, 1.0, 2.0, 3.0], colors=ACC, linewidths=1.1)
    optimum = np.array([2.0, 1.4])
    ax.scatter([z[0]], [z[1]], s=65, color=BLUE, zorder=5)
    ax.scatter([optimum[0]], [optimum[1]], s=80, color=RED, zorder=6)
    ax.plot([z[0], optimum[0]], [z[1], optimum[1]], color=RED, ls="--")
    ax.text(2.82, 1.52, r"$z$", color=BLUE, fontsize=11)
    ax.annotate(
        r"QP optimum $(2,1.4)$",
        optimum,
        xytext=(0.75, 0.72),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10,
    )
    ax.set_title(r"minimize $\frac{1}{2}\|x-z\|_2^2$ over the same polytope")
    save(fig, "qp_solution")


def portfolio_frontier():
    w = np.linspace(0, 1, 240)
    mu_a, mu_b = 0.06, 0.14
    sig_a, sig_b, rho = 0.08, 0.22, 0.20
    ret = w * mu_a + (1 - w) * mu_b
    var = (w * sig_a) ** 2 + ((1 - w) * sig_b) ** 2 + 2 * w * (1 - w) * rho * sig_a * sig_b
    risk = np.sqrt(var)
    idx_min = np.argmin(risk)
    target = 0.10
    idx_target = np.argmin(np.abs(ret - target))

    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.plot(risk, ret, color=TEAL)
    ax.scatter(risk[::24], ret[::24], c=w[::24], cmap="viridis", s=26)
    ax.scatter([risk[idx_min]], [ret[idx_min]], color=RED, s=75, zorder=5)
    ax.scatter([risk[idx_target]], [ret[idx_target]], color=ACC, s=75, zorder=5)
    ax.annotate("minimum variance", (risk[idx_min], ret[idx_min]), xytext=(0.13, 0.073),
                arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=10)
    ax.annotate("10% target return", (risk[idx_target], ret[idx_target]), xytext=(0.145, 0.108),
                arrowprops={"arrowstyle": "->", "color": ACC}, color=ACC, fontsize=10)
    ax.set(xlabel="portfolio risk (standard deviation)", ylabel="expected return")
    ax.grid(alpha=0.16)
    save(fig, "portfolio_frontier")


def solver_paths():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharex=True, sharey=True)
    for ax in axes:
        base_polytope(ax)
        ax.scatter([2], [2], s=75, color=RED, zorder=6)
    simplex = np.array([[0, 0], [0, 3], [1, 3], [2, 2]])
    interior = np.array([[0.65, 1.15], [1.10, 1.55], [1.48, 1.78], [1.73, 1.91], [1.91, 1.98]])
    axes[0].plot(simplex[:, 0], simplex[:, 1], "o-", color=BLUE)
    axes[1].plot(interior[:, 0], interior[:, 1], "o-", color=ACC)
    axes[0].set_title("simplex: move along vertices")
    axes[1].set_title("interior point: move through the interior")
    fig.tight_layout()
    save(fig, "solver_paths")


if __name__ == "__main__":
    lp_vertices()
    qp_solution()
    portfolio_frontier()
    solver_paths()
    print(f"wrote four figure pairs to {OUT}")
