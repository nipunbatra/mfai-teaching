#!/usr/bin/env python3
"""Figures for L23 (prefix codes and Huffman coding)."""

import collections
import heapq
import itertools
import math
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

OUT = "lecture23/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def prefix_tree():
    nodes = {
        "": (0.0, 0.5),
        "0": (1.0, 0.10),
        "1": (1.0, 0.70),
        "10": (2.0, 0.43),
        "11": (2.0, 0.82),
        "100": (3.0, 0.30),
        "101": (3.0, 0.56),
        "110": (3.0, 0.72),
        "111": (3.0, 0.94),
    }
    labels = {"": "start", "0": "A", "100": "E", "101": "C", "110": "D", "111": "B"}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for path, (x, y) in nodes.items():
        if path:
            parent = path[:-1]
            px, py = nodes[parent]
            ax.plot([px, x], [py, y], color=MUTED, lw=1.6)
            ax.text((px + x) / 2, (py + y) / 2 + 0.035, path[-1], color=BLUE, fontsize=10)
    for path, (x, y) in nodes.items():
        is_leaf = path in labels and path != ""
        color = RED if is_leaf else TEAL
        ax.scatter([x], [y], s=80 if is_leaf else 45, color=color, zorder=4)
        if path in labels:
            code = path if path else ""
            text = labels[path] if path == "" else f"{labels[path]} = {code}"
            ax.text(x + 0.08, y, text, va="center", fontsize=11, color=color)
    ax.text(1.55, 0.05, "no symbol is an ancestor of another symbol", color=INK, ha="center")
    ax.set_xlim(-0.2, 3.7)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    save(fig, "prefix_tree")


def ideal_lengths():
    p = np.geomspace(0.02, 1, 500)
    ideal = -np.log2(p)
    integer = np.ceil(ideal)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(p, ideal, color=TEAL, label=r"ideal $-\log_2 p$")
    ax.step(p, integer, where="post", color=ACC, label=r"integer length $\lceil-\log_2p\rceil$")
    ax.fill_between(p, ideal, integer, color="#F6E9D9", alpha=0.7)
    ax.set(xlabel="symbol probability $p$", ylabel="code length (bits)", xlim=(0, 1), ylim=(-0.1, 6.2))
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "ideal_lengths")


class Node:
    def __init__(self, weight, symbol=None, left=None, right=None):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right


def build_huffman(counts):
    serial = itertools.count()
    heap = [(weight, next(serial), Node(weight, symbol=symbol)) for symbol, weight in sorted(counts.items())]
    heapq.heapify(heap)
    merges = []
    while len(heap) > 1:
        wa, _, a = heapq.heappop(heap)
        wb, _, b = heapq.heappop(heap)
        parent = Node(wa + wb, left=a, right=b)
        merges.append((wa, wb, wa + wb))
        heapq.heappush(heap, (parent.weight, next(serial), parent))
    return heap[0][2], merges


def make_codes(node, prefix="", out=None):
    if out is None:
        out = {}
    if node.symbol is not None:
        out[node.symbol] = prefix or "0"
    else:
        make_codes(node.left, prefix + "0", out)
        make_codes(node.right, prefix + "1", out)
    return out


def leaf_order(node):
    if node.symbol is not None:
        return [node.symbol]
    return leaf_order(node.left) + leaf_order(node.right)


def huffman_tree():
    counts = collections.Counter("MATHEMATICS")
    root, _ = build_huffman(counts)
    codes = make_codes(root)
    leaves = leaf_order(root)
    y_for = {symbol: i for i, symbol in enumerate(leaves)}
    positions = {}

    def place(node, depth=0):
        if node.symbol is not None:
            y = y_for[node.symbol]
        else:
            yl = place(node.left, depth + 1)
            yr = place(node.right, depth + 1)
            y = (yl + yr) / 2
        positions[id(node)] = (depth, y)
        return y

    place(root)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    def draw(node):
        x, y = positions[id(node)]
        if node.symbol is not None:
            ax.scatter([x], [y], s=70, color=RED, zorder=4)
            ax.text(x + 0.10, y, f"{node.symbol}: {node.weight}  →  {codes[node.symbol]}", va="center", fontsize=10)
            return
        ax.scatter([x], [y], s=42, color=TEAL, zorder=4)
        ax.text(x - 0.08, y + 0.18, str(node.weight), color=TEAL, fontsize=9)
        for bit, child in [("0", node.left), ("1", node.right)]:
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y, cy], color=MUTED, lw=1.4)
            ax.text((x + cx) / 2, (y + cy) / 2 + 0.10, bit, color=BLUE, fontsize=9)
            draw(child)

    draw(root)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title('Huffman tree for "MATHEMATICS" (11 letters)')
    ax.axis("off")
    save(fig, "huffman_mathematics")

    total = sum(counts.values())
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
    average = sum((n / total) * len(codes[s]) for s, n in counts.items())
    return counts, codes, entropy, average


def length_comparison():
    weights = {"A": 40, "B": 20, "C": 15, "D": 15, "E": 10}
    root, _ = build_huffman(weights)
    codes = make_codes(root)
    total = sum(weights.values())
    entropy = -sum((n / total) * math.log2(n / total) for n in weights.values())
    average = sum((n / total) * len(codes[s]) for s, n in weights.items())
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    names = ["entropy $H$", "Huffman average $L$", "fixed 3-bit code"]
    values = [entropy, average, 3.0]
    bars = ax.bar(names, values, color=[TEAL, ACC, MUTED])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("bits per letter")
    ax.set_ylim(0, 3.4)
    ax.grid(axis="y", alpha=0.16)
    ax.set_title("source probabilities 0.40, 0.20, 0.15, 0.15, 0.10")
    save(fig, "length_comparison")
    return codes, entropy, average


if __name__ == "__main__":
    prefix_tree()
    ideal_lengths()
    counts, codes, entropy_word, average_word = huffman_tree()
    source_codes, entropy, average = length_comparison()
    print("counts:", dict(sorted(counts.items())))
    print("codes:", dict(sorted(codes.items())))
    print(f"MATHEMATICS entropy={entropy_word:.6f}, average_length={average_word:.6f}")
    print("five-symbol codes:", dict(sorted(source_codes.items())))
    print(f"five-symbol entropy={entropy:.6f}, average_length={average:.6f}")
    print(f"wrote four figure pairs to {OUT}")
