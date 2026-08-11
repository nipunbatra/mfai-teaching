#!/usr/bin/env python3
"""Figures for L15 (MAP & Conjugate Priors). Run from repo root:
   uv run --no-project --with matplotlib,numpy,scipy python3 lecture15/diagrams/l15_figs.py

Only the multi-panel posterior sequences live here (matplotlib); every
single-curve figure in the deck is computed in-Typst via chalkdust."""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture15/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

# One fixed coin, one fixed flip stream, reused by both figures.
THETA_TRUE = 0.7
rng = np.random.default_rng(7)
flips = (rng.random(1000) < THETA_TRUE).astype(int)
xs = np.linspace(0.001, 0.999, 500)

# ── figure 1 · sequential updating: Beta(2,2) prior, flips stream in ──────────
checkpoints = [1, 5, 20, 100]
a0, b0 = 2, 2
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.0), sharex=True)
for ax, n in zip(axes, checkpoints):
    h = int(flips[:n].sum()); t = n - h
    post = beta_dist(a0 + h, b0 + t)
    ax.plot(xs, beta_dist(a0, b0).pdf(xs), color=TEAL, ls='--', lw=1.8,
            label=r'prior Beta$(2,2)$')
    ax.plot(xs, post.pdf(xs), color=ACC, label='posterior')
    ax.fill_between(xs, post.pdf(xs), color=ACC, alpha=0.14, lw=0)
    ax.axvline(THETA_TRUE, color=INK, ls=':', lw=1.6, label=r'true $\theta = 0.7$')
    ax.set_title(f'$n = {n}$  ·  {h} H, {t} T', fontsize=13)
    ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xlim(0, 1)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'$\theta$')
    print(f'seq n={n}: {h} H {t} T -> posterior Beta({a0+h},{b0+t}), '
          f'mode {(a0+h-1)/(n+a0+b0-2):.3f}')
axes[-1].legend(fontsize=10.5, frameon=False, loc='upper left')  # n=100 panel: top-left is empty
fig.tight_layout()
save(fig, 'seq_updating')

# ── figure 2 · three priors argue, then agree: data swamps the prior ──────────
priors = [(r'optimist Beta$(12,3)$', 12, 3, TEAL),
          (r'skeptic Beta$(3,12)$',  3, 12, BLUE),
          (r'agnostic Beta$(2,2)$',  2,  2, ACC)]
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.0), sharex=True)
panels = [('the three priors', 0), ('after $n = 10$ flips', 10),
          ('after $n = 1000$ flips', 1000)]
for ax, (title, n) in zip(axes, panels):
    h = int(flips[:n].sum()); t = n - h
    for (lbl, a, b, col) in priors:
        ax.plot(xs, beta_dist(a + h, b + t).pdf(xs), color=col, label=lbl)
    if n > 0:
        ax.axvline(THETA_TRUE, color=INK, ls=':', lw=1.6)
        ax.set_title(f'{title}  ·  {h} H, {t} T', fontsize=13)
        print(f'washout n={n}: {h} H {t} T')
    else:
        ax.set_title(title, fontsize=13)
    ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xlim(0, 1)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'$\theta$')
axes[0].legend(fontsize=10.5, frameon=False, loc='upper center')
fig.tight_layout()
save(fig, 'prior_washout')

print('wrote', sorted(os.listdir(OUT)))
