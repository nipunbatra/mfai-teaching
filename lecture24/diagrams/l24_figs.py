#!/usr/bin/env python3
"""Figures for L24 (cross-entropy and KL divergence)."""

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

OUT = "lecture24/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def entropy(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(p[p > 0] * np.log2(p[p > 0]))


def cross_entropy(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return -np.sum(p * np.log2(q))


def wrong_model():
    labels = ["sun", "cloud", "rain"]
    p = np.array([0.75, 0.20, 0.05])
    q = np.array([0.45, 0.35, 0.20])
    pos = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    axes[0].bar(pos - 0.18, p, width=0.36, color=TEAL, label="true $p$")
    axes[0].bar(pos + 0.18, q, width=0.36, color=ACC, label="model $q$")
    axes[0].set_xticks(pos, labels)
    axes[0].set_ylabel("probability")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.15)
    lengths = -np.log2(q)
    axes[1].bar(labels, lengths, color=ACC)
    for i, value in enumerate(lengths):
        axes[1].text(i, value + 0.08, f"{value:.2f}", ha="center", fontsize=10)
    ce = cross_entropy(p, q)
    axes[1].set_ylabel(r"model length $-\log_2 q(x)$")
    axes[1].set_ylim(0, max(lengths) + 0.8)
    axes[1].text(0.05, 0.92, f"average under $p$: {ce:.3f} bits", transform=axes[1].transAxes, color=RED)
    axes[1].grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "wrong_weather_model")


def decomposition_bars():
    p = np.array([0.75, 0.20, 0.05])
    qs = [p, np.array([0.60, 0.28, 0.12]), np.array([0.45, 0.35, 0.20])]
    names = ["$q=p$", "nearby model", "forecast $q$"]
    h = entropy(p)
    ce = np.array([cross_entropy(p, q) for q in qs])
    kl = ce - h
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(names, [h] * 3, color=TEAL, label="$H(p)$")
    ax.bar(names, kl, bottom=[h] * 3, color=ACC, label=r"$D_{KL}(p\|q)$")
    for i, value in enumerate(ce):
        ax.text(i, value + 0.04, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("expected bits per symbol")
    ax.set_ylim(0, max(ce) + 0.35)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "cross_entropy_decomposition")


def bernoulli_nll():
    theta = np.linspace(0.01, 0.99, 500)
    p_hat = 0.8
    nll = -(p_hat * np.log(theta) + (1 - p_hat) * np.log(1 - theta))
    min_value = -(p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(theta, nll, color=TEAL)
    ax.scatter([p_hat], [min_value], s=75, color=RED, zorder=5)
    ax.annotate(r"minimum at $\theta=8/10=0.8$", (p_hat, min_value), xytext=(0.43, 1.25),
                arrowprops={"arrowstyle": "->", "color": RED}, color=RED)
    ax.axhline(min_value, color=ACC, ls="--", label=r"empirical entropy $H(0.8)$")
    ax.set(xlabel=r"model probability $\theta$", ylabel="mean negative log-likelihood (nats)",
           xlim=(0, 1), ylim=(0.45, 2.2))
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "bernoulli_cross_entropy")


def kl_directions():
    x = np.linspace(-5, 5, 900)

    def normal(x, mean, sd):
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)

    p = 0.5 * normal(x, -2, 0.6) + 0.5 * normal(x, 2, 0.6)
    q_cover = normal(x, 0, np.sqrt(4.36))
    q_mode = normal(x, -2, 0.6)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharex=True, sharey=True)
    for ax in axes:
        ax.plot(x, p, color=INK, label="target $p$")
        ax.grid(alpha=0.14)
        ax.set_xlabel("$x$")
    axes[0].plot(x, q_cover, color=TEAL, label="Gaussian $q$")
    axes[0].set_title(r"minimize $D_{KL}(p\|q)$: cover mass")
    axes[1].plot(x, q_mode, color=ACC, label="Gaussian $q$")
    axes[1].set_title(r"minimize $D_{KL}(q\|p)$: choose a mode")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "kl_directions")


if __name__ == "__main__":
    wrong_model()
    decomposition_bars()
    bernoulli_nll()
    kl_directions()
    print("wrote four figure pairs to", OUT)
