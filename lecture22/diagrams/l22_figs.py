#!/usr/bin/env python3
"""Figures for L22 (surprise and entropy)."""

import collections
import math
import os
import string

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

OUT = "lecture22/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def surprise_curve():
    p = np.geomspace(0.01, 1, 400)
    info = -np.log2(p)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(p, info, color=TEAL)
    probs = np.array([1, 0.5, 0.25, 0.125, 0.01])
    bits = -np.log2(probs)
    ax.scatter(probs, bits, color=RED, s=55, zorder=5)
    for x, y in zip(probs[:-1], bits[:-1]):
        value = abs(y)  # avoid "-0" for p=1
        unit = "bit" if value == 1 else "bits"
        ax.annotate(f"p={x:g}: {value:g} {unit}", (x, y), xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.annotate("rare event", (0.01, bits[-1]), xytext=(0.12, 6.0),
                arrowprops={"arrowstyle": "->", "color": RED}, color=RED)
    ax.set(xlabel="event probability $p$", ylabel=r"surprise $-\log_2 p$ (bits)", xlim=(0, 1.02), ylim=(-0.1, 7))
    ax.grid(alpha=0.16)
    save(fig, "surprise_curve")


def binary_entropy():
    p = np.linspace(0.001, 0.999, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(p, h, color=TEAL)
    pts = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    vals = -(pts * np.log2(pts) + (1 - pts) * np.log2(1 - pts))
    ax.scatter(pts, vals, color=[MUTED, ACC, RED, ACC, MUTED], s=55, zorder=5)
    ax.annotate("maximum uncertainty\n$p=0.5$, $H=1$ bit", (0.5, 1), xytext=(0.27, 0.52),
                arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=10)
    ax.text(0.03, 0.06, "almost certain", color=MUTED)
    ax.text(0.68, 0.06, "almost certain", color=MUTED)
    ax.set(xlabel=r"coin bias $p=P(X=1)$", ylabel=r"entropy $H_2(p)$ (bits)", xlim=(0, 1), ylim=(0, 1.08))
    ax.grid(alpha=0.16)
    save(fig, "binary_entropy")


def question_tree():
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    levels = [
        [(0.5, 0.94, "1–8")],
        [(0.25, 0.69, "1–4"), (0.75, 0.69, "5–8")],
        [(0.125, 0.44, "1–2"), (0.375, 0.44, "3–4"), (0.625, 0.44, "5–6"), (0.875, 0.44, "7–8")],
        [(i / 8 + 1 / 16, 0.19, str(i + 1)) for i in range(8)],
    ]
    for level_index in range(3):
        for parent_index, (px, py, _) in enumerate(levels[level_index]):
            for child_index in (2 * parent_index, 2 * parent_index + 1):
                cx, cy, _ = levels[level_index + 1][child_index]
                ax.plot([px, cx], [py - 0.035, cy + 0.035], color=MUTED, lw=1.4)
    for level in levels:
        for x, y, label in level:
            ax.text(x, y, label, ha="center", va="center", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": TEAL})
    ax.text(0.5, 1.04, "three balanced yes/no questions identify one of eight equally likely outcomes",
            ha="center", color=INK)
    ax.text(0.08, 0.81, "no", color=BLUE)
    ax.text(0.82, 0.81, "yes", color=BLUE)
    ax.text(0.5, 0.04, r"$\log_2 8=3$ bits", ha="center", color=RED, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    save(fig, "question_tree")


def course_letter_entropy():
    with open("lecture-plan-detailed.md", encoding="utf-8") as handle:
        text = handle.read().lower()
    counts = collections.Counter(ch for ch in text if ch in string.ascii_lowercase)
    total = sum(counts.values())
    letters = list(string.ascii_lowercase)
    probs = np.array([counts[ch] / total for ch in letters])
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    order = np.argsort(probs)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(np.arange(26), probs[order], color=TEAL)
    ax.set_xticks(np.arange(26), np.array(letters)[order])
    ax.set_ylabel("frequency")
    ax.set_xlabel("letters, sorted by frequency")
    ax.grid(axis="y", alpha=0.16)
    ax.text(18.0, probs.max() * 0.82, f"plug-in entropy\n{entropy:.2f} bits/letter", color=RED,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": RED})
    ax.set_title("letters in this course's detailed lecture plan")
    save(fig, "course_letter_entropy")
    print(f"course-plan letter entropy: {entropy:.4f} bits/letter over {total} letters")


if __name__ == "__main__":
    surprise_curve()
    binary_entropy()
    question_tree()
    course_letter_entropy()
    print(f"wrote four figure pairs to {OUT}")
