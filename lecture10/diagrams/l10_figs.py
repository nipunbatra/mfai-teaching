#!/usr/bin/env python3
"""Figures for L10 (Differentiation on a Computer I). Run from repo root: python3 lecture10/diagrams/l10_figs.py"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture10/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

EPS64 = np.finfo(np.float64).eps          # 2.22e-16
EPS32 = float(np.finfo(np.float32).eps)   # 1.19e-07

def fwd_err64(h):
    """|forward-difference estimate - 1| for f = exp at x = 0, float64."""
    return abs((np.exp(h) - 1.0) / h - 1.0)

def cen_err64(h):
    return abs((np.exp(h) - np.exp(-h)) / (2.0 * h) - 1.0)

def fwd_err32(h):
    h = np.float32(h)
    if h == np.float32(0):
        return 1.0
    est = (np.exp(np.float32(0) + h) - np.exp(np.float32(0))) / h
    return abs(float(est) - 1.0)

CLIP = 1e-18  # log axes cannot show an exact zero


# ── Figure 1 · the hero U-curve (forward difference, float64) ──────────────
def fig_ucurve():
    hs = np.logspace(-17, 0, 400)
    errs = np.array([max(fwd_err64(h), CLIP) for h in hs])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    # the two model terms from the ⭐ derivation, dashed (clipped where irrelevant)
    ht = hs[hs >= 1e-12]
    ax.plot(ht, ht / 2, ls='--', lw=1.7, color=TEAL, label=r'truncation model  $Mh/2$')
    hr = hs[hs <= 1e-4]
    ax.plot(hr, 2 * EPS64 / hr, ls='--', lw=1.7, color=RED, label=r'rounding model  $2\varepsilon/h$')
    # measured error (jagged on the left — that jitter IS rounding noise)
    ax.plot(hs, errs, color=INK, lw=1.9, label='measured error')
    # optimal step + floor
    hstar = 2 * np.sqrt(EPS64)
    ax.axvline(hstar, color=ACC, lw=1.6, ls=':')
    ax.annotate(r'$h^* = 2\sqrt{\varepsilon} \approx 3\times 10^{-8}$',
                xy=(hstar, 6e1), xytext=(1.2e-6, 8e2), color=ACC, fontsize=13.5,
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.3))
    ax.annotate('floor $\\approx 2\\sqrt{\\varepsilon} \\approx 3\\times10^{-8}$\n— half the digits, gone',
                xy=(hstar * 1.6, 2.2e-8), xytext=(2e-4, 2e-12), color=ACC, fontsize=13.5,
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.3))
    ax.text(1e-12, 6e-2, 'rounding dominates\n(slope −1)', color=RED, fontsize=13.5,
            ha='center', va='center')
    ax.text(2.5e-3, 6e-1, 'truncation dominates\n(slope +1)', color=TEAL, fontsize=13.5,
            ha='center', va='center')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-17, 1.5); ax.set_ylim(1e-14, 1e4)
    ax.set_xlabel(r'step size  $h$')
    ax.set_ylabel(r'error of  $(e^h - 1)/h$  vs  $f\,{}^\prime(0)=1$')
    ax.legend(loc='upper left', frameon=False, fontsize=12)
    save(fig, 'l10_ucurve')


# ── Figure 1b · zoom on the valley: the model meets the measurement ────────
def fig_ucurve_zoom():
    hs = np.logspace(-11, -4, 320)
    errs = np.array([max(fwd_err64(h), CLIP) for h in hs])
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(hs, hs / 2, ls='--', lw=1.8, color=TEAL, label=r'truncation model  $Mh/2$')
    ax.plot(hs, 2 * EPS64 / hs, ls='--', lw=1.8, color=RED, label=r'rounding model  $2\varepsilon/h$')
    ax.plot(hs, errs, color=INK, lw=1.9, label='measured error')
    hstar = 2 * np.sqrt(EPS64)
    floor = 2 * np.sqrt(EPS64)
    ax.axvline(hstar, color=ACC, lw=1.6, ls=':')
    ax.axhline(floor, color=ACC, lw=1.2, ls=':')
    ax.annotate(r'models cross at  $h^* = 2\sqrt{\varepsilon} \approx 3.0\times10^{-8}$',
                xy=(hstar, 8e-6), xytext=(2.5e-7, 2.2e-4), color=ACC, fontsize=13.5,
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.3))
    ax.text(2.2e-7, 1.55e-8, r'predicted floor  $2\sqrt{\varepsilon M} \approx 3.0\times10^{-8}$',
            color=ACC, fontsize=13)
    ax.annotate(r'measured best  $\approx 6\times10^{-9}$',
                xy=(6.1e-9, 8e-10), xytext=(1.2e-7, 2.6e-10), color=INK, fontsize=13,
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
    ax.text(3.5e-10, 2.3e-6, 'slope −1', color=RED, fontsize=13, rotation=-38)
    ax.text(3.2e-6, 4.5e-6, 'slope +1', color=TEAL, fontsize=13, rotation=38)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-11, 1e-4); ax.set_ylim(1e-10, 1e-3)
    ax.set_xlabel(r'step size  $h$')
    ax.set_ylabel('error (zoomed to the valley)')
    ax.legend(loc='upper left', frameon=False, fontsize=12)
    save(fig, 'l10_ucurve_zoom')


# ── Figure 2 · forward vs central (float64) ────────────────────────────────
def fig_fwd_central():
    hs = np.logspace(-13, 0, 320)
    ef = np.array([max(fwd_err64(h), CLIP) for h in hs])
    ec = np.array([max(cen_err64(h), CLIP) for h in hs])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(hs, ef, color=TEAL, lw=1.9, label='forward  $(f(x+h)-f(x))/h$')
    ax.plot(hs, ec, color=ACC, lw=1.9, label='central  $(f(x+h)-f(x-h))/2h$')
    # mark each curve's measured minimum
    imf, imc = int(np.argmin(ef)), int(np.argmin(ec))
    ax.plot([hs[imf]], [ef[imf]], 'o', ms=7, color=TEAL)
    ax.plot([hs[imc]], [ec[imc]], 'o', ms=7, color=ACC)
    # floors as dotted reference lines, labelled in the clear bottom-right
    ax.axhline(1.2e-8, color=TEAL, lw=1.2, ls=':')
    ax.text(1.3e-2, 2.6e-8, r'forward floor $\approx 10^{-8}$   at  $h^*\!\approx 3\times10^{-8}$',
            color=TEAL, fontsize=12.5)
    ax.axhline(4e-12, color=ACC, lw=1.2, ls=':')
    ax.text(1.3e-2, 8.5e-12, r'central floor $\approx 10^{-11}$   at  $h^*\!\approx 6\times10^{-6}$',
            color=ACC, fontsize=12.5)
    # slope tags beside the ascending branches
    ax.text(6e-3, 1.3e-1, r'$\propto h$', color=TEAL, fontsize=15)
    ax.text(2.5e-1, 2.5e-4, r'$\propto h^2$', color=ACC, fontsize=15)
    # gradcheck's default eps
    ax.axvline(1e-6, color=BLUE, lw=1.5, ls=':')
    ax.text(1.35e-6, 2.5e0, 'PyTorch gradcheck\ndefault  eps = 1e-6', color=BLUE, fontsize=12.5, ha='left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-13, 1.5); ax.set_ylim(1e-14, 4e1)
    ax.set_xlabel(r'step size  $h$')
    ax.set_ylabel(r'error at  $x=0$,  $f = e^x$')
    ax.legend(loc='upper left', frameon=False, fontsize=12)
    save(fig, 'l10_fwd_central')


# ── Figure 3 · float32 vs float64 (why gradcheck demands double) ───────────
def fig_fp32_fp64():
    hs64 = np.logspace(-17, 0, 400)
    e64 = np.array([max(fwd_err64(h), CLIP) for h in hs64])
    hs32 = np.logspace(-9, 0, 240)
    e32 = np.array([max(fwd_err32(h), CLIP) for h in hs32])
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    ax.plot(hs64, e64, color=TEAL, lw=1.9, label='float64')
    ax.plot(hs32, e32, color=RED, lw=1.9, label='float32')
    ax.annotate('float32 floor $\\approx 2\\times10^{-4}$\n— only 3–4 digits survive',
                xy=(2.4e-4, 1.4e-4), xytext=(6e-4, 1e-8), color=RED, fontsize=13,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.3))
    ax.annotate('float64 floor $\\approx 10^{-8}$\n— 8 digits survive',
                xy=(2.9e-8, 8e-9), xytext=(2e-11, 3e-12), color=TEAL, fontsize=13,
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))
    ax.annotate('$h\\leq 10^{-8}$ in float32:  $1+h$ rounds to $1$\n→ estimate $=0$, error $=100\\%$ (L2!)',
                xy=(2e-9, 1.15), xytext=(2e-16, 6e1), color=RED, fontsize=12.5,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-17, 1.5); ax.set_ylim(1e-14, 1e4)
    ax.set_xlabel(r'step size  $h$')
    ax.set_ylabel(r'forward-difference error,  $f = e^x$ at $x = 0$')
    ax.legend(loc='lower left', frameon=False, fontsize=12)
    save(fig, 'l10_fp32_fp64')


if __name__ == '__main__':
    fig_ucurve()
    fig_ucurve_zoom()
    fig_fwd_central()
    fig_fp32_fp64()
    print('wrote', sorted(os.listdir(OUT)))
