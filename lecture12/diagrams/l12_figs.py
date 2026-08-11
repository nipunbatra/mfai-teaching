#!/usr/bin/env python3
"""Figures for L12 (Continuous Distributions). Run from repo root: python3 lecture12/diagrams/l12_figs.py"""
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
OUT='lecture12/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

rng = np.random.default_rng(12)

# ── 1 · histogram → density: 25,000 heights, three bin widths ─────────────
# ES 114's heights dataset is ~N(67.99, 1.90) in inches; synthetic twin here.
MU, SIG = 67.99, 1.90
heights = rng.normal(MU, SIG, 25_000)
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), sharey=True)
for ax, nb in zip(axes, (6, 25, 120)):
    ax.hist(heights, bins=nb, range=(60, 76), density=True, color=TEAL,
            alpha=0.80, edgecolor='white', linewidth=(0.6 if nb <= 25 else 0.0))
    ax.set_title(f'{nb} bins', fontsize=13)
    ax.set_xlabel('height (inches)')
    ax.set_xlim(60, 76)
xs = np.linspace(60, 76, 400)
pdf = np.exp(-((xs - MU) / SIG) ** 2 / 2) / (SIG * np.sqrt(2 * np.pi))
axes[2].plot(xs, pdf, color=ACC, lw=2.6)
axes[2].set_title('120 bins + the limit curve', fontsize=13)
axes[0].set_ylabel('density (per inch)')
fig.tight_layout()
save(fig, 'l12_hist_density')

# ── 2 · inverse-CDF sampling: darts on the u-axis → Exp(1) samples ────────
lam = 1.0
xs = np.linspace(0, 5, 400)
F = 1 - np.exp(-lam * xs)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.5))
ax1.plot(xs, F, color=INK, lw=2.4)
u_darts = np.array([0.08, 0.22, 0.36, 0.50, 0.63, 0.78, 0.90, 0.97])
x_darts = -np.log(1 - u_darts) / lam
for u, x in zip(u_darts, x_darts):
    ax1.plot([0, x], [u, u], ls='--', lw=1.0, color=MUTED)
    ax1.plot([x, x], [0, u], ls='--', lw=1.0, color=MUTED)
    ax1.plot([x], [u], 'o', ms=5.5, color=ACC, zorder=5)
    ax1.plot([0], [u], 'o', ms=4.5, color=BLUE, zorder=5, clip_on=False)
ax1.set_xlim(0, 5); ax1.set_ylim(0, 1.02)
ax1.set_xlabel(r'$x = F^{-1}(u)$'); ax1.set_ylabel(r'$u$')
ax1.set_title('uniform darts on the $u$-axis, pulled through $F^{-1}$', fontsize=12.5)

u = rng.random(10_000)
samples = -np.log(1 - u) / lam
ax2.hist(samples, bins=60, range=(0, 5), density=True, color=TEAL,
         alpha=0.85, edgecolor='none', label='10,000 transformed darts')
ax2.plot(xs, lam * np.exp(-lam * xs), color=ACC, lw=2.6,
         label=r'target pdf  $\lambda e^{-\lambda x}$')
ax2.set_xlim(0, 5)
ax2.set_xlabel(r'$x$'); ax2.set_ylabel('density')
ax2.legend(frameon=False, fontsize=11)
ax2.set_title('the samples wear the target density', fontsize=12.5)
fig.tight_layout()
save(fig, 'l12_inverse_cdf')

print('wrote', sorted(os.listdir(OUT)))
