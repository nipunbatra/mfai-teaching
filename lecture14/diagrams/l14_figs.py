#!/usr/bin/env python3
"""Figures for L14 (Maximum Likelihood Estimation). Run from repo root: python3 lecture14/diagrams/l14_figs.py"""
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
OUT='lecture14/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

# ── shared synthetic dataset: y = 0.55 x + 1.2 + N(0, 0.85²) ────────────────
rng = np.random.default_rng(7)
n = 14
x = np.sort(rng.uniform(0.4, 9.6, n))
SIG = 0.85
y = 0.55 * x + 1.2 + rng.normal(0.0, SIG, n)
w, b = np.polyfit(x, y, 1)
xl = np.linspace(-0.2, 10.4, 100)


# ── fig 1 · scatter + fitted line + residual segments ───────────────────────
def fig_residuals():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(xl, w * xl + b, color=ACC, zorder=2)
    hi = list(np.argsort(-np.abs(y - w * x - b))[:3])   # the 3 largest residuals
    for i in range(n):
        yy = w * x[i] + b
        if i in hi:
            ax.plot([x[i], x[i]], [yy, y[i]], color=RED, lw=2.0, ls='--', zorder=1)
        else:
            ax.plot([x[i], x[i]], [yy, y[i]], color=MUTED, lw=1.1, ls='--', alpha=0.65, zorder=1)
    ax.scatter(x, y, s=44, color=INK, zorder=3)
    i = max(hi, key=lambda j: x[j])       # label the right-most big residual
    ax.annotate(r'$r_i = y_i - (w x_i + b)$', xy=(x[i] + 0.12, (y[i] + w * x[i] + b) / 2),
                xytext=(x[i] - 3.6, min(y[i], w * x[i] + b) - 1.4), color=RED, fontsize=13,
                arrowprops=dict(arrowstyle='-', color=RED, lw=1.0))
    ax.annotate(r'$\hat{y} = w x + b$', xy=(8.9, w * 8.9 + b), xytext=(7.6, w * 7.6 + b + 1.7),
                color=ACC, fontsize=13,
                arrowprops=dict(arrowstyle='-', color=ACC, lw=1.0))
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(-0.3, 10.6)
    save(fig, 'linreg_residuals')


# ── fig 2 · the noise model: a Gaussian bell standing on the line at each x ──
def fig_keystone_bells():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(xl, w * xl + b, color=ACC, zorder=2)
    ax.scatter(x, y, s=34, color=INK, alpha=0.5, zorder=3)
    t = np.linspace(-2.5 * SIG, 2.5 * SIG, 90)
    for x0 in (1.8, 4.8, 7.8):
        mu0 = w * x0 + b
        px = x0 + 1.5 * np.exp(-t**2 / (2 * SIG**2))
        ax.plot([x0, x0], [mu0 - 2.5 * SIG, mu0 + 2.5 * SIG], color=MUTED, lw=1.0, ls=':', zorder=1)
        ax.fill_betweenx(mu0 + t, x0, px, color=TEAL, alpha=0.16, zorder=1)
        ax.plot(px, mu0 + t, color=TEAL, lw=1.8, zorder=2)
        ax.scatter([x0], [mu0], s=30, color=ACC, zorder=4)
    ax.annotate(r'$y\,|\,x \sim \mathcal{N}(wx+b,\ \sigma^2)$', xy=(4.8 + 1.3, w * 4.8 + b + 0.55),
                xytext=(1.2, w * 1.2 + b + 3.6), color=TEAL, fontsize=13,
                arrowprops=dict(arrowstyle='-', color=TEAL, lw=1.0))
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_xlim(-0.3, 10.6)
    save(fig, 'mse_keystone_bells')


if __name__ == '__main__':
    fig_residuals()
    fig_keystone_bells()
    print(f'figures written to {OUT}/ (w={w:.3f}, b={b:.3f})')
