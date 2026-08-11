#!/usr/bin/env python3
"""Figures for L7 (Univariate Calculus & Taylor Series). Run from repo root: python3 lecture7/diagrams/l7_figs.py"""
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
OUT='lecture7/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

# ── the lecture's wiggly loss (same closed form the deck's chalkdust figures use) ──
L   = lambda t: 0.25*t**4 - t**2 + 0.2*np.sin(5*t) + 2
Lp  = lambda t: t**3 - 2*t + np.cos(5*t)
TSTAR, LSTAR, KSTAR = -1.4969867, 0.8279823, 9.3862635   # global min, L(θ*), L''(θ*) (Newton-refined)


def f_loss_zoom():
    """Hook: zoom into a wiggly loss — near the minimum it becomes a parabola, elsewhere a line."""
    windows = [(-2.6, 2.6), (-2.15, -0.85), (TSTAR-0.12, TSTAR+0.12), (0.5-0.06, 0.5+0.06)]
    titles  = ['a loss curve $L(\\theta)$', 'zoom ×4 — one basin',
               'zoom ×22 — at the minimum', 'zoom ×43 — anywhere else']
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.1))
    for ax, (a, b), title in zip(axes, windows, titles):
        t = np.linspace(a, b, 600)
        ax.plot(t, L(t), color=INK)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(a, b)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('$\\theta$', fontsize=11)
    # dashed pure parabola over the minimum window
    t2 = np.linspace(*windows[2], 300)
    axes[2].plot(t2, LSTAR + 0.5*KSTAR*(t2-TSTAR)**2, color=ACC, ls='--', lw=2.0)
    axes[2].plot([TSTAR], [LSTAR], 'o', color=ACC, ms=6)
    axes[2].text(TSTAR, LSTAR+0.062, 'dashed: a pure parabola', color=ACC,
                 fontsize=10.5, ha='center')
    # dashed pure line at the generic point
    t3 = np.linspace(*windows[3], 300)
    axes[3].plot(t3, L(0.5) + Lp(0.5)*(t3-0.5), color=TEAL, ls='--', lw=2.0)
    axes[3].text(0.5, L(0.5)+0.055, 'dashed: a pure line', color=TEAL,
                 fontsize=10.5, ha='center')
    # red zoom boxes: where the next panel lives
    for ax, (a, b) in ((axes[0], windows[1]), (axes[1], windows[2])):
        t = np.linspace(a, b, 200)
        lo, hi = L(t).min(), L(t).max()
        pad = 0.06*(hi-lo)
        ax.add_patch(plt.Rectangle((a, lo-pad), b-a, hi-lo+2*pad,
                                   fill=False, edgecolor=RED, lw=1.6))
    axes[0].set_ylabel('$L(\\theta)$', fontsize=11)
    fig.tight_layout(w_pad=1.4)
    save(fig, 'loss_zoom')


if __name__ == '__main__':
    f_loss_zoom()
    print('done ->', OUT)
