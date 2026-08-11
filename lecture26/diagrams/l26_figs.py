#!/usr/bin/env python3
"""Figures for L26 (learning transition models and course finale)."""

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

OUT = "lecture26/figures"
os.makedirs(OUT, exist_ok=True)
ALPHABET = " " + string.ascii_lowercase


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", transparent=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close(fig)


def clean(text):
    text = text.lower()
    out = []
    previous_space = False
    for ch in text:
        ch = ch if ch in string.ascii_lowercase else " "
        if ch == " ":
            if previous_space:
                continue
            previous_space = True
        else:
            previous_space = False
        out.append(ch)
    return "".join(out).strip()


with open("lecture-plan-detailed.md", encoding="utf-8") as handle:
    TEXT = clean(handle.read())


def bigram_matrix():
    shown = " aeionrstlmc"
    index = {ch: i for i, ch in enumerate(shown)}
    counts = np.zeros((len(shown), len(shown)), dtype=int)
    for a, b in zip(TEXT, TEXT[1:]):
        if a in index and b in index:
            counts[index[a], index[b]] += 1
    view = np.log1p(counts)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(view, cmap="YlGnBu")
    labels = ["space" if ch == " " else ch for ch in shown]
    ax.set_xticks(range(len(shown)), labels, rotation=60)
    ax.set_yticks(range(len(shown)), labels)
    ax.set_xlabel("next character")
    ax.set_ylabel("current character")
    ax.set_title("bigram counts in the course lecture plan (log colour scale)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log(1+\mathrm{count})$")
    save(fig, "bigram_counts")


def next_after_t():
    counts = collections.Counter(b for a, b in zip(TEXT, TEXT[1:]) if a == "t")
    total = sum(counts.values())
    top = counts.most_common(10)
    labels = ["space" if ch == " " else ch for ch, _ in top]
    probs = [n / total for _, n in top]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, probs, color=TEAL)
    for bar, value in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(r"MLE $P(\mathrm{next}\mid t)$")
    ax.set_ylim(0, max(probs) + 0.08)
    ax.grid(axis="y", alpha=0.15)
    ax.set_title(f"{total} observed transitions out of character 't'")
    save(fig, "next_after_t")


def smoothing():
    labels = ["A", "B", "C", "D"]
    counts = np.array([8, 2, 0, 0], dtype=float)
    mle = counts / counts.sum()
    add_one = (counts + 1) / (counts.sum() + len(counts))
    pos = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(pos - 0.18, mle, width=0.36, color=TEAL, label="MLE")
    ax.bar(pos + 0.18, add_one, width=0.36, color=ACC, label="add-one predictive")
    ax.set_xticks(pos, labels)
    ax.set_ylabel("next-state probability")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False)
    ax.set_title("counts $(8,2,0,0)$: smoothing reserves mass for unseen outcomes")
    save(fig, "smoothing")


def train_counts(text, order):
    table = collections.defaultdict(collections.Counter)
    for i in range(order, len(text)):
        context = text[i - order : i] if order else ""
        table[context][text[i]] += 1
    return table


def distribution(table, context, alpha=0.1):
    row = table.get(context, {})
    total = sum(row.values()) + alpha * len(ALPHABET)
    return np.array([(row.get(ch, 0) + alpha) / total for ch in ALPHABET])


def bits_per_character(table, text, order, alpha=0.1):
    losses = []
    for i in range(order, len(text)):
        context = text[i - order : i] if order else ""
        probs = distribution(table, context, alpha)
        losses.append(-math.log2(probs[ALPHABET.index(text[i])]))
    return float(np.mean(losses))


def generate(table, order, n=320, seed=8, alpha=0.001, start="the ", alphabet=ALPHABET):
    rng = np.random.default_rng(seed)
    context = start[-order:] if order else ""
    output = list(start)
    for _ in range(n):
        probs = distribution_over(table, context, alpha, alphabet)
        ch = rng.choice(list(alphabet), p=probs)
        output.append(ch)
        if order:
            context = (context + ch)[-order:]
    return "".join(output)


def distribution_over(table, context, alpha, alphabet):
    row = table.get(context, {})
    total = sum(row.values()) + alpha * len(alphabet)
    return np.array([(row.get(ch, 0) + alpha) / total for ch in alphabet])


def shakespeare_sample(path="input.txt", order=5, n=200, seed=4, alpha=0.001):
    """Order-5 character model on Tiny Shakespeare (karpathy/char-rnn input.txt).

    Raw characters (case, punctuation, newlines) are kept; the initial context is
    the first `order` characters of the corpus. Reproduces the sample on the
    L26 slide "Shakespeare from five-character contexts".
    """
    text = open(path, encoding="utf-8").read()
    alphabet = sorted(set(text))
    table = train_counts(text, order)
    return generate(table, order, n=n, seed=seed, alpha=alpha,
                    start=text[:order], alphabet=alphabet)


def ngram_evaluation():
    split = int(0.8 * len(TEXT))
    train, test = TEXT[:split], TEXT[split:]
    orders = [0, 1, 2, 3]
    bpc = []
    samples = {}
    for order in orders:
        table = train_counts(train, order)
        bpc.append(bits_per_character(table, test, order))
        samples[order] = generate(table, order)
    perplexity = 2 ** np.array(bpc)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(["unigram", "bigram", "trigram", "4-gram"], bpc, color=[MUTED, TEAL, ACC, BLUE])
    for bar, bits, ppl in zip(bars, bpc, perplexity):
        ax.text(bar.get_x() + bar.get_width() / 2, bits + 0.05, f"{bits:.2f} bits\nPPL {ppl:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("held-out bits per character")
    ax.set_ylim(0, max(bpc) + 0.7)
    ax.grid(axis="y", alpha=0.15)
    ax.set_title("character models trained on 80% of the course lecture plan")
    save(fig, "ngram_evaluation")
    print("text characters:", len(TEXT), "train:", len(train), "test:", len(test))
    print("held-out BPC:", dict(zip(orders, np.round(bpc, 4))))
    print("bigram sample:", samples[1])
    print("4-gram sample:", samples[3])


if __name__ == "__main__":
    bigram_matrix()
    next_after_t()
    smoothing()
    ngram_evaluation()
    if os.path.exists("input.txt"):
        print("shakespeare sample:", shakespeare_sample())
    print("wrote four figure pairs to", OUT)
