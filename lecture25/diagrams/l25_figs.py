#!/usr/bin/env python3
"""Figures for L25 (Markov chains and PageRank)."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle

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

OUT = "lecture25/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def arrow(ax, start, end, label, rad=0.0, color=MUTED, text_shift=(0, 0)):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            connectionstyle=f"arc3,rad={rad}", lw=1.8, color=color)
    ax.add_patch(patch)
    mid = ((start[0] + end[0]) / 2 + text_shift[0], (start[1] + end[1]) / 2 + text_shift[1])
    ax.text(*mid, label, fontsize=10, color=color, ha="center", va="center")


def weather_chain():
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    sunny, rainy = (0.30, 0.40), (0.70, 0.40)
    for pos, label, color in [(sunny, "sunny", ACC), (rainy, "rainy", BLUE)]:
        ax.add_patch(Circle(pos, 0.105, facecolor="white", edgecolor=color, lw=2.2))
        ax.text(*pos, label, ha="center", va="center", color=color, fontsize=13)
    arrow(ax, (0.41, 0.46), (0.59, 0.46), "0.20", rad=0.25, color=RED, text_shift=(0, 0.11))
    arrow(ax, (0.59, 0.34), (0.41, 0.34), "0.40", rad=0.25, color=GREEN, text_shift=(0, -0.11))
    # self-loops: wide arcs above each node, labels next to the arc apex
    arrow(ax, (0.255, 0.50), (0.345, 0.50), "", rad=-1.7, color=ACC)
    ax.text(0.30, 0.75, "0.80", color=ACC, fontsize=11, ha="center")
    arrow(ax, (0.655, 0.50), (0.745, 0.50), "", rad=-1.7, color=BLUE)
    ax.text(0.70, 0.75, "0.60", color=BLUE, fontsize=11, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save(fig, "weather_chain")


def weather_convergence():
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    dist = np.array([1.0, 0.0])
    history = [dist.copy()]
    for _ in range(8):
        dist = dist @ P
        history.append(dist.copy())
    history = np.array(history)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(history[:, 0], "o-", color=ACC, label="sunny probability")
    ax.plot(history[:, 1], "o-", color=BLUE, label="rainy probability")
    ax.axhline(2 / 3, color=ACC, ls="--", alpha=0.7)
    ax.axhline(1 / 3, color=BLUE, ls="--", alpha=0.7)
    ax.text(6.3, 0.70, r"$2/3$", color=ACC)
    ax.text(6.3, 0.37, r"$1/3$", color=BLUE)
    ax.set(xlabel="day $t$", ylabel="state probability", xlim=(0, 8), ylim=(0, 1.04))
    ax.grid(alpha=0.16)
    ax.legend(frameon=False)
    save(fig, "weather_convergence")


def chain_failures():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    # reducible: two absorbing groups
    positions = [(0.2, 0.5), (0.42, 0.5), (0.67, 0.5), (0.88, 0.5)]
    for i, pos in enumerate(positions):
        axes[0].add_patch(Circle(pos, 0.06, facecolor="white", edgecolor=TEAL, lw=1.8))
        axes[0].text(*pos, str(i + 1), ha="center", va="center")
    dx = 0.04
    arrow(axes[0], (positions[0][0] + dx, 0.545), (positions[1][0] - dx, 0.545), "", rad=0.25, color=MUTED)
    arrow(axes[0], (positions[1][0] - dx, 0.455), (positions[0][0] + dx, 0.455), "", rad=0.25, color=MUTED)
    arrow(axes[0], (positions[2][0] + dx, 0.545), (positions[3][0] - dx, 0.545), "", rad=0.25, color=MUTED)
    arrow(axes[0], (positions[3][0] - dx, 0.455), (positions[2][0] + dx, 0.455), "", rad=0.25, color=MUTED)
    axes[0].set_title("reducible: two closed groups")
    axes[0].text(0.5, 0.18, "long-run behaviour depends on the starting group", ha="center", color=RED)
    # periodic two-state alternator
    left, right = (0.3, 0.5), (0.7, 0.5)
    for pos, label in [(left, "A"), (right, "B")]:
        axes[1].add_patch(Circle(pos, 0.08, facecolor="white", edgecolor=TEAL, lw=1.8))
        axes[1].text(*pos, label, ha="center", va="center")
    arrow(axes[1], (0.38, 0.55), (0.62, 0.55), "1", rad=0.22, color=ACC, text_shift=(0, 0.08))
    arrow(axes[1], (0.62, 0.45), (0.38, 0.45), "1", rad=0.22, color=BLUE, text_shift=(0, -0.08))
    axes[1].set_title("periodic: alternate forever")
    axes[1].text(0.5, 0.18, "$[1,0]$ and $[0,1]$ keep swapping", ha="center", color=RED)
    fig.tight_layout()
    save(fig, "chain_failures")


def pagerank():
    # L5's four-page web: A->B,C  B->A,C  C->A,D  D->A (row-stochastic S).
    labels = ["A", "B", "C", "D"]
    S = np.array([[0, 0.5, 0.5, 0],
                  [0.5, 0, 0.5, 0],
                  [0.5, 0, 0, 0.5],
                  [1.0, 0, 0, 0]])
    alpha = 0.85
    P = alpha * S + (1 - alpha) * np.ones((4, 4)) / 4
    rank = np.ones(4) / 4
    hist = [rank.copy()]
    for _ in range(30):
        rank = rank @ P
        hist.append(rank.copy())
    hist = np.array(hist)
    undamped = np.array([8, 4, 6, 3]) / 21

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = [ACC, TEAL, BLUE, RED]
    for i, label in enumerate(labels):
        ax.plot(hist[:, i], "o-", ms=3.5, color=colors[i], label=f"{label}: {rank[i]:.3f}")
        ax.axhline(undamped[i], color=colors[i], ls=":", lw=1.3, alpha=0.6)
    ax.text(14.7, 0.45, r"dotted: undamped $(8,4,6,3)/21$ from L5",
            ha="right", fontsize=10, color=MUTED)
    ax.set(xlabel="power-iteration step", ylabel="PageRank", xlim=(0, 15), ylim=(0, 0.48))
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=10, title=r"damped, $\alpha=0.85$",
              title_fontsize=10, loc="lower center", ncol=4)
    fig.tight_layout()
    save(fig, "pagerank")
    print("PageRank damped:", dict(zip(labels, rank.round(6))))


if __name__ == "__main__":
    weather_chain()
    weather_convergence()
    chain_failures()
    pagerank()
    print("wrote four figure pairs to", OUT)
