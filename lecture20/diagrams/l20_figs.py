#!/usr/bin/env python3
"""Figures for L20 (duality and KKT). Run from the repository root."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

OUT = "lecture20/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def active_inactive():
    x = np.linspace(-0.6, 2.5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    cases = [
        ((x - 0.2) ** 2, 0.2, "inactive", GREEN),
        ((x - 2.0) ** 2, 1.0, "active", RED),
    ]
    for ax, (f, optimum, state, color) in zip(axes, cases):
        ax.axvspan(-0.6, 1.0, color="#E9F4F4", alpha=0.95)
        ax.axvline(1.0, color=ACC, ls="--", lw=2)
        ax.plot(x, f, color=TEAL)
        value = np.interp(optimum, x, f)
        ax.scatter([optimum], [value], s=65, color=color, zorder=5)
        ax.annotate(
            f"{state} constraint\n$x^*={optimum:g}$",
            (optimum, value),
            xytext=(optimum - 0.45, 1.25 if state == "inactive" else 1.7),
            arrowprops={"arrowstyle": "->", "color": color},
            color=color,
            fontsize=10,
        )
        ax.set_xlim(-0.6, 2.5)
        ax.set_ylim(-0.15, 3.1)
        ax.set_xlabel("$x$")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("objective")
    axes[0].set_title(r"$min,(x-0.2)^2$ subject to $x\leq1$")
    axes[1].set_title(r"$min,(x-2)^2$ subject to $x\leq1$")
    axes[0].text(-0.5, 2.75, "feasible", color=TEAL)
    axes[1].text(-0.5, 2.75, "feasible", color=TEAL)
    fig.tight_layout()
    save(fig, "active_inactive")


def dual_bound():
    lam = np.linspace(0, 5, 400)
    q = lam - lam**2 / 4
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.axhline(1, color=INK, lw=1.8, label=r"primal optimum $p^*=1$")
    ax.fill_between(lam, q, 1, where=q <= 1, color="#F6E9D9", alpha=0.8)
    ax.plot(lam, q, color=TEAL, label=r"dual bound $q(\lambda)=\lambda-\lambda^2/4$")
    ax.scatter([2], [1], s=75, color=RED, zorder=5)
    ax.annotate(
        r"best bound: $\lambda^*=2$, $q=1$",
        (2, 1),
        xytext=(2.55, 0.55),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10,
    )
    ax.text(4.1, 0.15, "duality gap", color=ACC, fontsize=10)
    ax.set(xlabel=r"multiplier $\lambda\geq0$", ylabel="value", xlim=(0, 5), ylim=(-1.4, 1.3))
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "dual_bound")


def complementary_slackness():
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    ax.plot([0, 2.8], [0, 0], color=TEAL, lw=5)
    ax.plot([0, 0], [0, 2.8], color=ACC, lw=5)
    ax.scatter([2.0, 0], [0, 2.0], s=80, color=[GREEN, RED], zorder=5)
    ax.text(1.35, 0.24, r"inactive: slack $>0$, $\lambda=0$", color=GREEN)
    ax.text(0.16, 2.15, r"active: slack $=0$, $\lambda\geq0$", color=RED)
    ax.text(1.15, 1.45, "forbidden by\n$\lambda\,s=0$", color=MUTED, ha="center")
    ax.set(xlabel=r"slack $s=-g(x)\geq0$", ylabel=r"price $\lambda\geq0$")
    ax.set_xlim(-0.15, 3)
    ax.set_ylim(-0.15, 3)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    ax.grid(alpha=0.12)
    save(fig, "complementary_slackness")


def water_filling():
    a = np.array([3.0, 1.0, 0.5])
    nu = 0.75
    x = np.maximum(a - nu, 0)
    labels = ["channel 1", "channel 2", "channel 3"]
    pos = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(pos - 0.18, a, width=0.36, color=MUTED, alpha=0.5, label=r"preferred $a_i$")
    ax.bar(pos + 0.18, x, width=0.36, color=TEAL, label=r"allocation $x_i^*$")
    ax.axhline(nu, color=ACC, ls="--", label=r"threshold $\nu=0.75$")
    for i, value in enumerate(x):
        ax.text(i + 0.18, value + 0.08, f"{value:g}", ha="center", color=TEAL, fontsize=10)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("amount")
    ax.set_ylim(0, 3.5)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    save(fig, "water_filling")


if __name__ == "__main__":
    active_inactive()
    dual_bound()
    complementary_slackness()
    water_filling()
    print(f"wrote four figure pairs to {OUT}")
