#!/usr/bin/env python3
"""Figures for L19 (Lagrange multipliers). Run from repo root."""

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
        "axes.linewidth": 1.0,
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.solid_capstyle": "round",
    }
)

OUT = "lecture19/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def constrained_optimum():
    x = np.linspace(-0.2, 1.2, 240)
    y = np.linspace(-0.2, 1.2, 240)
    xx, yy = np.meshgrid(x, y)
    zz = xx * yy

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    levels = [0.04, 0.10, 0.16, 0.25, 0.32]
    cs = ax.contour(xx, yy, zz, levels=levels, colors=TEAL, linewidths=1.15)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    t = np.linspace(0, 1, 100)
    ax.plot(t, 1 - t, color=ACC, lw=3.2)
    ax.text(0.06, 0.83, r"$x+y=1$", color=ACC, fontsize=11, rotation=-45)
    ax.scatter([0.5], [0.5], s=75, color=RED, zorder=5)
    ax.annotate(
        r"maximum $(0.5,0.5)$" + "\n" + r"$xy=0.25$",
        (0.5, 0.5),
        xytext=(0.68, 0.74),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.5},
        color=RED,
        fontsize=11,
    )
    ax.scatter([0, 1], [1, 0], s=32, color=INK, zorder=5)
    ax.text(0.035, 1.045, r"$xy=0$", fontsize=9, color=MUTED)
    ax.text(1.005, 0.065, r"$xy=0$", fontsize=9, color=MUTED)
    ax.set(xlim=(-0.12, 1.12), ylim=(-0.12, 1.12), xlabel="$x$", ylabel="$y$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.16)
    save(fig, "budget_contours")


def gradient_alignment():
    x = np.linspace(0.0, 1.0, 220)
    y = np.linspace(0.0, 1.0, 220)
    xx, yy = np.meshgrid(x, y)
    zz = xx * yy

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.contour(xx, yy, zz, levels=[0.05, 0.10, 0.15, 0.20, 0.25], colors=TEAL, linewidths=1.2)
    t = np.linspace(0, 1, 100)
    ax.plot(t, 1 - t, color=ACC, lw=3.0)
    p = np.array([0.5, 0.5])
    ax.scatter(*p, s=70, color=RED, zorder=6)
    ax.arrow(*p, 0.23, 0.23, width=0.007, head_width=0.045, color=BLUE, length_includes_head=True)
    ax.arrow(*p, 0.36, 0.36, width=0.006, head_width=0.045, color=GREEN, length_includes_head=True)
    ax.arrow(*p, -0.22, 0.22, width=0.005, head_width=0.04, color=MUTED, length_includes_head=True)
    ax.text(0.72, 0.66, r"$\nabla f=(0.5,0.5)$", color=BLUE, fontsize=10)
    ax.text(0.73, 0.88, r"$\nabla h=(1,1)$", color=GREEN, fontsize=10)
    ax.text(0.18, 0.75, r"feasible tangent $(1,-1)$", color=MUTED, fontsize=9)
    ax.text(0.55, 0.44, r"$\nabla f=0.5\,\nabla h$", color=RED, fontsize=11)
    ax.set(xlim=(0, 1.05), ylim=(0, 1.05), xlabel="$x$", ylabel="$y$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.16)
    save(fig, "gradient_alignment")


def shadow_price():
    b = np.linspace(0, 2, 240)
    value = b**2 / 4
    b0 = 1.0
    v0 = 0.25
    slope = 0.5
    tangent = v0 + slope * (b - b0)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(b, value, color=TEAL, label=r"optimal value $V(b)=b^2/4$")
    ax.plot(b, tangent, color=ACC, ls="--", label=r"tangent at $b=1$: slope $0.5$")
    ax.scatter([b0], [v0], s=65, color=RED, zorder=5)
    ax.annotate(
        r"$V'(1)=0.5=\lambda^*$",
        (b0, v0),
        xytext=(1.15, 0.12),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.4},
        color=RED,
        fontsize=11,
    )
    ax.set(xlabel="available budget $b$", ylabel="best attainable product")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    save(fig, "shadow_price")


def simplex():
    h = np.sqrt(3) / 2
    verts = np.array([[0, 0], [1, 0], [0.5, h]])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.add_patch(Polygon(verts, closed=True, facecolor="#E9F4F4", edgecolor=TEAL, lw=2.0))
    uniform = verts.mean(axis=0)
    ax.scatter([uniform[0]], [uniform[1]], s=80, color=RED, zorder=5)
    ax.text(uniform[0], uniform[1] - 0.075, r"uniform: $(1/3,1/3,1/3)$", color=RED, fontsize=11, ha="center")
    labels = [r"$(1,0,0)$", r"$(0,1,0)$", r"$(0,0,1)$"]
    offsets = [(-0.07, -0.08), (0.01, -0.08), (-0.06, 0.035)]
    for (vx, vy), label, (dx, dy) in zip(verts, labels, offsets):
        ax.scatter([vx], [vy], s=42, color=INK)
        ax.text(vx + dx, vy + dy, label, fontsize=10, color=INK)
    for t in (0.25, 0.5, 0.75):
        p1 = (1 - t) * verts[0] + t * verts[2]
        p2 = (1 - t) * verts[1] + t * verts[2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=MUTED, lw=0.7, alpha=0.45)
    ax.text(0.5, -0.15, r"$p_1+p_2+p_3=1$, with every $p_i\geq0$", ha="center", fontsize=11)
    ax.text(0.5, 0.36, "maximum entropy", ha="center", color=RED, fontsize=10)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.2, h + 0.13)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "probability_simplex")


if __name__ == "__main__":
    constrained_optimum()
    gradient_alignment()
    shadow_price()
    simplex()
    print(f"wrote four figure pairs to {OUT}")
