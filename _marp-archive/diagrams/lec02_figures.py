"""Generate all figures for Lecture 2: Floating Point — How Machines See Numbers.

Style: cream-paper palette matching slides/mfai-theme.css.
Saves PNG to figures/lec02/ and SVG (text kept as text) to figures/lec02/svg/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

PAPER = "#F7F3E9"; INK = "#161513"; MUTED = "#5F5C54"; RULE = "#B5AE9B"
RUST = "#B85A3E"; SAGE = "#5F8573"; SLATE = "#37535F"
WINE = "#8E2A3B"; OCHRE = "#B5945A"; PAPER_ALT = "#EFEADA"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
    "font.size": 13, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",          # keep SVG text as selectable text, not paths
    "savefig.dpi": 200,
})

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "figures" / "lec02"
SVG = PNG / "svg"
PNG.mkdir(parents=True, exist_ok=True)
SVG.mkdir(parents=True, exist_ok=True)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def save(fig, name, tight=True):
    bbox = "tight" if tight else None
    fig.savefig(PNG / f"{name}.png", bbox_inches=bbox)
    fig.savefig(SVG / f"{name}.svg", format="svg", bbox_inches=bbox)
    plt.close(fig)
    print(f"  wrote {name}.png + svg/{name}.svg")


# ---- 1. the float number line: unevenly spaced --------------------------------
def float_number_line():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), layout="constrained",
                                 height_ratios=[1, 1.5])

    # top: a toy float with 3 mantissa bits — every representable value
    vals = [(1 + m / 8) * 2.0 ** e for e in range(-1, 3) for m in range(8)]
    a1.vlines(vals, 0, 0.55, color=SLATE, lw=1.1)
    a1.vlines([0.5, 1, 2, 4], 0, 0.8, color=RUST, lw=2.2)
    for x in [0.5, 1, 2, 4]:
        a1.text(x, 0.95, str(x), ha="center", color=RUST, fontsize=12, fontweight="bold")
    a1.axhline(0, color=MUTED, lw=1)
    for lo, hi, g in [(0.5, 1, "gap 1/16"), (1, 2, "gap 1/8"),
                      (2, 4, "gap 1/4"), (4, 7.5, "gap 1/2")]:
        a1.text((lo + hi) / 2, -0.38, g, ha="center", color=MUTED, fontsize=11)
    a1.set_xlim(0.28, 7.75); a1.set_ylim(-0.65, 1.35)
    a1.axis("off")
    a1.set_title("a toy float (3 mantissa bits): the gap doubles at every power of 2",
                 color=INK, fontsize=14)

    # bottom: actual float32 spacing vs magnitude
    xs = 2.0 ** np.arange(-20, 41)
    gaps = np.spacing(xs.astype(np.float32)).astype(float)
    a2.loglog(xs, gaps, drawstyle="steps-post", color=SLATE, lw=1.8)

    a2.plot([1], [float(np.spacing(np.float32(1.0)))], "o", color=RUST, ms=7, zorder=5)
    a2.annotate("at 1.0 the gap is $2^{-23}\\approx 1.2\\times 10^{-7}$\n(machine epsilon)",
                xy=(1, 1.2e-7), xytext=(1e-5, 3e-3), color=RUST, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.2))
    a2.plot([2.0 ** 24], [2.0], "o", color=WINE, ms=7, zorder=5)
    a2.annotate("at $2^{24}$ the gap is 2:\nx + 1 == x from here on",
                xy=(2.0 ** 24, 2.0), xytext=(2e7, 1e-4), color=WINE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=WINE, lw=1.2))

    a2.set_xlabel("magnitude of x")
    a2.set_ylabel("gap to the next float32")
    a2.set_title("real float32: spacing between consecutive numbers", color=INK, fontsize=14)
    _clean(a2)
    save(fig, "float_number_line")


# ---- 2. float32 bit anatomy, with 6.25 worked in ------------------------------
def float32_anatomy():
    bits = "0" + "10000001" + "10010000000000000000000"
    colors = [SLATE] + [RUST] * 8 + [SAGE] * 23

    fig, ax = plt.subplots(figsize=(13, 3.6), layout="constrained")
    for i, (b, c) in enumerate(zip(bits, colors)):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=c, edgecolor=PAPER, lw=1.5))
        ax.text(i + 0.46, 0.5, b, ha="center", va="center", color=PAPER,
                fontsize=10, fontweight="bold")

    for i, lbl in [(0, "31"), (1, "30"), (8, "23"), (9, "22"), (31, "0")]:
        ax.text(i + 0.46, 1.22, lbl, ha="center", fontsize=9, color=MUTED)

    ax.text(0.46, -0.42, "sign (1 bit)\n0 means positive", ha="center", va="top",
            fontsize=11, color=SLATE)
    ax.text(4.96, -0.42, "exponent (8 bits)\n$10000001_2 = 129 = 2 + 127$", ha="center",
            va="top", fontsize=11, color=RUST)
    ax.text(20.46, -0.42, "mantissa (23 bits)\nthe .1001… after a free leading 1", ha="center",
            va="top", fontsize=11, color=SAGE)

    ax.text(16, -1.55,
            "value $= (-1)^0 \\times 1.1001_2 \\times 2^{129-127} = 1.5625 \\times 4 = 6.25$",
            ha="center", fontsize=14, color=INK)

    ax.set_xlim(-0.4, 32.4); ax.set_ylim(-1.95, 1.75)
    ax.axis("off")
    ax.set_title("float32 anatomy — the 32 bits of 6.25", color=INK, fontsize=15)
    save(fig, "float32_anatomy")


# ---- 3. catastrophic cancellation: quadratic-root relative error ---------------
def cancellation_error():
    bs64 = np.geomspace(20, 2e4, 600)
    bs = bs64.astype(np.float32)
    with np.errstate(all="ignore"):
        disc = np.sqrt(bs * bs - np.float32(4))
        naive = (bs - disc) / np.float32(2)
        stable = np.float32(2) / (bs + disc)
    exact = 2.0 / (bs64 + np.sqrt(bs64 * bs64 - 4.0))
    rel_naive = np.abs(naive.astype(float) - exact) / exact
    rel_stable = np.abs(stable.astype(float) - exact) / exact

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.scatter(bs64, np.maximum(rel_naive, 1e-12), s=7, color=WINE, alpha=0.7,
               label="naive:  $(b-\\sqrt{b^2-4})\\,/\\,2$")
    ax.scatter(bs64, np.maximum(rel_stable, 1e-12), s=7, color=SAGE, alpha=0.7,
               label="stable:  $2\\,/\\,(b+\\sqrt{b^2-4})$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(3e-13, 40)

    ax.axhline(1.19e-7, color=MUTED, ls="--", lw=1)
    ax.text(22, 2.2e-7, "float32 machine epsilon", color=MUTED, fontsize=10, va="bottom")
    ax.axhline(1.0, color=WINE, ls=":", lw=1)
    ax.text(22, 1.7, "100% error — the small root comes back 0.0", color=WINE, fontsize=10)

    ax.set_xlabel("$b$   (smaller root of $x^2 - bx + 1 = 0$, computed in float32)")
    ax.set_ylabel("relative error")
    ax.set_title("catastrophic cancellation: the same root, two algebraically equal formulas",
                 color=INK)
    ax.legend(frameon=False, fontsize=11, loc="center left")
    _clean(ax)
    save(fig, "cancellation_error")


# ---- 4. naive softmax overflows; stable softmax doesn't ------------------------
def softmax_stability():
    a = np.linspace(0, 120, 900)
    with np.errstate(all="ignore"):
        ex = np.exp(a.astype(np.float32)).astype(float)
        ex[np.isinf(ex)] = np.nan
        p_naive = []
        for ai in a:
            z = np.array([ai - 2, ai - 1, ai], dtype=np.float32)
            e = np.exp(z)
            p_naive.append(float(e[-1] / e.sum()))
    p_naive = np.array(p_naive)
    p_stable = 1.0 / (np.exp(-2.0) + np.exp(-1.0) + 1.0)  # 0.66524…
    cutoff = 88.72

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    a1.semilogy(a, ex, color=SLATE, lw=2)
    a1.axhline(3.4028e38, color=WINE, ls="--", lw=1.2)
    a1.text(3, 8e38, "largest float32 $\\approx 3.4\\times10^{38}$", color=WINE, fontsize=11)
    a1.axvline(cutoff, color=WINE, ls=":", lw=1.2)
    a1.axvspan(cutoff, 120, color=WINE, alpha=0.07)
    a1.text(91, 1e10, "exp(a) = inf", color=WINE, fontsize=12)
    a1.set_xlabel("a"); a1.set_ylabel("exp(a) in float32")
    a1.set_title("exp overflows at a ≈ 88.7 — that's all it takes", color=INK)
    _clean(a1)

    a2.plot(a, np.full_like(a, p_stable), color=SAGE, lw=2.6,
            label="stable: subtract max first")
    a2.plot(a, p_naive, color=RUST, lw=2.2, ls="--", label="naive")
    a2.axvspan(cutoff, 120, color=WINE, alpha=0.07)
    a2.text(92, 0.42, "naive\nreturns\nNaN", color=WINE, fontsize=12)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("a"); a2.set_ylabel("largest softmax probability")
    a2.set_title("softmax of z = [a−2, a−1, a] — answer shouldn't depend on a", color=INK)
    a2.legend(frameon=False, fontsize=11, loc="lower left")
    _clean(a2)

    save(fig, "softmax_stability")


# ---- 5. fp32 / fp16 / bf16: bit budgets and the range-precision trade ----------
def fp_formats():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")

    fmts = [("float32", 8, 23), ("float16", 5, 10), ("bfloat16", 8, 7)]
    for y, (name, e, m) in zip([2, 1, 0], fmts):
        a1.barh(y, 1, left=0, color=SLATE, edgecolor=PAPER, height=0.55)
        a1.barh(y, e, left=1, color=RUST, edgecolor=PAPER, height=0.55)
        a1.barh(y, m, left=1 + e, color=SAGE, edgecolor=PAPER, height=0.55)
        a1.text(1 + e / 2, y, str(e), ha="center", va="center", color=PAPER,
                fontsize=11, fontweight="bold")
        a1.text(1 + e + m / 2, y, str(m), ha="center", va="center", color=PAPER,
                fontsize=11, fontweight="bold")
        a1.text(-0.7, y, name, ha="right", va="center", fontsize=13, color=INK)
    a1.set_xlim(-7.5, 33); a1.set_ylim(-0.9, 2.9)
    a1.axis("off")
    a1.legend(handles=[Patch(color=SLATE, label="sign"),
                       Patch(color=RUST, label="exponent (range)"),
                       Patch(color=SAGE, label="mantissa (precision)")],
              frameon=False, fontsize=10, loc="lower center", ncols=3)
    a1.set_title("bit budgets — bfloat16 = float32 with the mantissa chopped", color=INK)

    pts = {"float32": (38.5, 7.2, SLATE, 170),
           "float16": (4.8, 3.3, RUST, 95),
           "bfloat16": (38.5, 2.4, SAGE, 95)}
    for name, (x, y, c, s) in pts.items():
        a2.scatter([x], [y], s=s, color=c, zorder=5)
        dx = -1.5 if name == "float16" else 0
        ha = "right" if name == "float16" else "center"
        dy = 0.75 if name != "bfloat16" else -1.05
        a2.text(x + dx, y + dy, name, ha=ha, fontsize=12, color=c, fontweight="bold")
    a2.annotate("", xy=(36.2, 2.4), xytext=(6.5, 3.1),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4, ls="--"))
    a2.text(21, 3.6, "same 16 bits:\ntrade precision for range", ha="center",
            color=MUTED, fontsize=11, style="italic")
    a2.set_xlim(0, 46); a2.set_ylim(0, 9.2)
    a2.set_xlabel("range: $\\log_{10}$(largest finite value)")
    a2.set_ylabel("precision: decimal digits")
    a2.set_title("why deep learning picked bfloat16", color=INK)
    a2.text(45, 8.6, "float64 sits at (308, 15.9) — far off this chart", ha="right",
            color=MUTED, fontsize=10, style="italic")
    _clean(a2)

    save(fig, "fp_formats")


if __name__ == "__main__":
    print("Generating Lecture 2 figures...")
    float_number_line()
    float32_anatomy()
    cancellation_error()
    softmax_stability()
    fp_formats()
    print("Done: all Lecture 2 figures saved to", PNG)
