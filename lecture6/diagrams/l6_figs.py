#!/usr/bin/env python3
"""Figures for L6 (SVD & PCA). Run from repo root:
   uv run --no-project --with matplotlib,numpy,scikit-learn python3 lecture6/diagrams/l6_figs.py
"""
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
OUT='lecture6/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

rng = np.random.default_rng(6)

# ════════════════════════════════════════════════════════════════════
# The worked 2x2 of the lecture:  A = [[5, 3], [0, 4]]
#   A^T A = [[25, 15], [15, 25]] -> eigenvalues 40, 10
#   sigma = (2*sqrt(10), sqrt(10)) ~ (6.325, 3.162)
#   V = 45-degree rotation, U = rotation by atan(1/2) ~ 26.57 degrees
# ════════════════════════════════════════════════════════════════════
A2 = np.array([[5.0, 3.0], [0.0, 4.0]])
s1, s2 = 2*np.sqrt(10), np.sqrt(10)
v1 = np.array([1.0, 1.0])/np.sqrt(2);  v2 = np.array([-1.0, 1.0])/np.sqrt(2)
u1 = np.array([2.0, 1.0])/np.sqrt(5);  u2 = np.array([-1.0, 2.0])/np.sqrt(5)
V2 = np.column_stack([v1, v2]); U2 = np.column_stack([u1, u2]); S2 = np.diag([s1, s2])
assert np.allclose(U2 @ S2 @ V2.T, A2)

t = np.linspace(0, 2*np.pi, 361)
circle = np.vstack([np.cos(t), np.sin(t)])                    # (2, 361)

def draw_state(ax, M, lim, title, sub, show_unit=False):
    """Plot M @ unit-circle plus the two tracked vectors M@v1 (ACC), M@v2 (TEAL)."""
    pts = M @ circle
    ax.fill(pts[0], pts[1], color=TEAL, alpha=0.07, zorder=1)
    ax.plot(pts[0], pts[1], color=INK, lw=2.2, zorder=3)
    if show_unit:
        ax.plot(circle[0], circle[1], color=MUTED, lw=1.1, ls=':', zorder=2)
    for vec, col, lbl in ((M @ v1, ACC, None), (M @ v2, TEAL, None)):
        ax.annotate('', xy=vec, xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.4,
                                    mutation_scale=16, shrinkA=0, shrinkB=0))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.5); ax.axvline(0, color=MUTED, lw=0.6, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=15, pad=8)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha='center', va='top',
            fontsize=11.5, color=MUTED)

I2 = np.eye(2)

# ── figure 1: the 4-panel action sequence  x -> V^T x -> S V^T x -> U S V^T x ──
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9))
states = [
    (I2,               r'$x$',                    'unit circle, with $v_1$, $v_2$ marked', 1.6, False),
    (V2.T,             r'$V^{T}x$',               'rotate by $-45°$: circle unchanged,\nmarked vectors now on the axes', 1.6, False),
    (S2 @ V2.T,        r'$\Sigma V^{T}x$',        'stretch axes by $\\sigma_1=6.32$, $\\sigma_2=3.16$', 7.2, True),
    (U2 @ S2 @ V2.T,   r'$U\Sigma V^{T}x = Ax$',  'rotate by $+26.6°$: the ellipse $A$ makes', 7.2, True),
]
for ax, (M, ti, sub, lim, su) in zip(axes, states):
    draw_state(ax, M, lim, ti, sub, show_unit=su)
for i in range(3):
    fig.text(0.255 + 0.245*i, 0.52, '→', fontsize=22, color=MUTED, ha='center', va='center')
fig.subplots_adjust(wspace=0.3, bottom=0.13)
save(fig, 'svd_action')

# ── figures 2-4: pairwise before -> after, one per machine ──
def step_fig(name, M0, M1, t0, t1, sub0, sub1, lim, show_unit=(False, False)):
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.9))
    draw_state(axes[0], M0, lim, t0, sub0, show_unit=show_unit[0])
    draw_state(axes[1], M1, lim, t1, sub1, show_unit=show_unit[1])
    fig.text(0.505, 0.52, '→', fontsize=26, color=ACC, ha='center', va='center')
    fig.subplots_adjust(wspace=0.32, bottom=0.14)
    save(fig, name)

step_fig('svd_step_v', I2, V2.T, r'$x$', r'$V^{T}x$',
         'the circle, with $v_1$ (orange), $v_2$ (teal)',
         'rotated $-45°$ — $v_1, v_2$ land on the axes', 1.6)
step_fig('svd_step_s', V2.T, S2 @ V2.T, r'$V^{T}x$', r'$\Sigma V^{T}x$',
         'the rotated circle (radius 1)',
         'axis 1 stretched $\\times 6.32$, axis 2 $\\times 3.16$', 7.2, show_unit=(False, True))
step_fig('svd_step_u', S2 @ V2.T, U2 @ S2 @ V2.T, r'$\Sigma V^{T}x$', r'$U\Sigma V^{T}x = Ax$',
         'the axis-aligned ellipse',
         'rotated $+26.6°$ — exactly what $A$ does', 7.2, show_unit=(False, True))

# ════════════════════════════════════════════════════════════════════
# Synthetic 256x256 grayscale image (no downloads): a little campus
# skyline at dusk — gradient sky, sun disk, buildings with windows,
# a sloped hill road, and text-like strokes.  Structured enough that
# rank-k reconstructions tell a clean story.
# ════════════════════════════════════════════════════════════════════
n = 256
yy, xx = np.mgrid[0:n, 0:n].astype(float) / (n - 1)     # yy = 0 at top
img = 0.92 - 0.55*yy                                    # dusk sky gradient (rank 1)

# sun disk (soft edge)
r2 = (xx - 0.76)**2 + (yy - 0.18)**2
img += 0.22 * np.clip((0.085**2 - r2) / 0.002, 0, 1)

# gentle hill: darker below a slanted line (a diagonal edge -> slow decay)
hill = yy > (0.86 - 0.18*xx)
img[hill] = 0.30 - 0.10*yy[hill]

# buildings: dark blocks with window grids
buildings = [(0.04, 0.11, 0.42), (0.18, 0.09, 0.55), (0.30, 0.13, 0.35),
             (0.47, 0.08, 0.50), (0.58, 0.12, 0.28), (0.86, 0.10, 0.47)]
for (x0, w, top) in buildings:
    mask = (xx >= x0) & (xx < x0 + w) & (yy >= top) & (yy < 0.88 - 0.18*xx)
    img[mask] = 0.16
    # windows: lit rectangles every few pixels inside the block
    win = mask & (np.floor(xx*n/6) % 2 == 1) & (np.floor(yy*n/7) % 2 == 1) \
               & (xx > x0 + 0.012) & (xx < x0 + w - 0.012) & (yy > top + 0.02)
    img[win] = 0.75

# a slim communication tower with a diagonal guy-wire
tower = (np.abs(xx - 0.72) < 0.006) & (yy > 0.22) & (yy < 0.88 - 0.18*xx)
img[tower] = 0.12
wire = np.abs((yy - 0.24) - 1.9*(xx - 0.72)) < 0.006
wire &= (xx > 0.72) & (xx < 0.995) & (yy < 0.88 - 0.18*xx)
img[wire] = 0.20

# text-like strokes at the bottom (a caption band)
strokes = [(0.06, 0.10), (0.13, 0.19), (0.22, 0.24), (0.28, 0.37), (0.41, 0.45),
           (0.50, 0.58), (0.62, 0.65), (0.70, 0.79), (0.83, 0.90)]
for (a, b) in strokes:
    m = (yy > 0.935) & (yy < 0.955) & (xx > a) & (xx < b)
    img[m] = 0.85
img += 0.012 * rng.standard_normal((n, n))              # faint texture
img = np.clip(img, 0.0, 1.0)

Uim, Sim, Vtim = np.linalg.svd(img, full_matrices=False)
energy = np.cumsum(Sim**2) / np.sum(Sim**2)
fro = np.linalg.norm(img)

def rank_k(k):
    return Uim[:, :k] * Sim[:k] @ Vtim[:k]

ks = (1, 5, 20, 50)

# ── figure 5: hook teaser — original vs rank-20 ──
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title('original — 65,536 numbers', fontsize=14, pad=7)
axes[1].imshow(np.clip(rank_k(20), 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title('rank 20 — 10,260 numbers (16%)', fontsize=14, pad=7)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED); s.set_linewidth(0.8)
fig.subplots_adjust(wspace=0.06)
save(fig, 'teaser_pair')

# ── figure 6: rank-k reconstruction grid ──
def fmt_energy(k):
    e = 100*energy[k-1]
    return f'{e:.2f}%' if e >= 99.9 else (f'{e:.1f}%' if e >= 99 else f'{e:.0f}%')

fig, axes = plt.subplots(1, 5, figsize=(14.6, 3.6))
axes[0].imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title('original (rank 256)', fontsize=13.5, pad=6)
for ax, k in zip(axes[1:], ks):
    ax.imshow(np.clip(rank_k(k), 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'$k={k}$ · {fmt_energy(k)} energy', fontsize=13.5, pad=6)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED); s.set_linewidth(0.8)
fig.subplots_adjust(wspace=0.05)
save(fig, 'compress_grid')

# ── figure 7: singular-value decay + reconstruction-error curve ──
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))
ax = axes[0]
ax.semilogy(np.arange(1, n+1), Sim, color=TEAL, lw=2.2)
ax.scatter(ks, Sim[np.array(ks)-1], color=ACC, s=42, zorder=5)
for k in ks:
    ax.annotate(f'$k={k}$', (k, Sim[k-1]), textcoords='offset points',
                xytext=(7, 6), fontsize=11.5, color=ACC)
ax.set_xlabel('index $i$'); ax.set_ylabel(r'$\sigma_i$  (log scale)')
ax.set_title('singular values fall fast', fontsize=14)

ax = axes[1]
kk = np.arange(0, n+1)
pred = np.sqrt(np.concatenate([[np.sum(Sim**2)], np.sum(Sim**2) - np.cumsum(Sim**2)]).clip(0)) / fro
ax.plot(kk, 100*pred, color=INK, lw=2.2,
        label=r'$\sqrt{\Sigma_{i>k}\,\sigma_i^2}\;/\;\|A\|_F$  (Eckart–Young)')
meas_k = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 180, 256])
meas = [100*np.linalg.norm(img - rank_k(int(k)))/fro for k in meas_k]
ax.scatter(meas_k, meas, color=ACC, s=34, zorder=5, label='measured  $\\|A-A_k\\|_F/\\|A\\|_F$')
ax.set_xlim(0, 130); ax.set_ylim(0, None)
ax.set_xlabel('rank $k$ kept'); ax.set_ylabel('relative error (%)')
ax.set_title('error = the tail of the $\\sigma$s — exactly', fontsize=14)
ax.legend(fontsize=10.5, frameon=False)
fig.subplots_adjust(wspace=0.28)
save(fig, 'sv_decay')

# ════════════════════════════════════════════════════════════════════
# PCA: best-fit axis vs a bad axis (same cloud, two candidate lines)
# ════════════════════════════════════════════════════════════════════
m = 90
Z = rng.standard_normal((m, 2))
Xp = Z @ np.array([[1.9, 0.0], [0.0, 0.62]]) @ np.array(
    [[np.cos(0.5), np.sin(0.5)], [-np.sin(0.5), np.cos(0.5)]])
Xp -= Xp.mean(0)
Cp = Xp.T @ Xp / m
lam, W = np.linalg.eigh(Cp)
w_pc = W[:, -1] * np.sign(W[0, -1])            # principal axis
th_bad = np.arctan2(w_pc[1], w_pc[0]) + np.deg2rad(60)
w_bad = np.array([np.cos(th_bad), np.sin(th_bad)])

def axis_panel(ax, w, title, col):
    proj = Xp @ w
    frac = proj.var() / np.trace(Cp)
    P = np.outer(proj, w)
    for p, q in zip(Xp, P):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.7, alpha=0.55, zorder=1)
    ax.scatter(Xp[:, 0], Xp[:, 1], s=16, color=TEAL, alpha=0.8, zorder=3)
    ax.scatter(P[:, 0], P[:, 1], s=10, color=col, alpha=0.9, zorder=4)
    L = 4.6
    ax.plot([-L*w[0], L*w[0]], [-L*w[1], L*w[1]], color=col, lw=2.4, zorder=2)
    ax.set_xlim(-4.8, 4.8); ax.set_ylim(-3.4, 3.4); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=14, pad=7)
    ax.text(0.5, -0.03, f'spread captured: {100*frac:.0f}% of total variance',
            transform=ax.transAxes, ha='center', va='top', fontsize=12, color=col)
    return frac

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
f_bad = axis_panel(axes[0], w_bad, 'a poorly chosen axis', RED)
f_pc = axis_panel(axes[1], w_pc, 'the principal axis', ACC)
fig.subplots_adjust(wspace=0.16, bottom=0.1)
save(fig, 'pca_bestfit')

# ════════════════════════════════════════════════════════════════════
# PCA on sklearn digits (8x8, ships with sklearn — no download)
# ════════════════════════════════════════════════════════════════════
from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True)                   # (1797, 64), labels 0-9
Xc = Xd - Xd.mean(0)
Ud, Sd, Vtd = np.linalg.svd(Xc, full_matrices=False)
var = Sd**2 / np.sum(Sd**2)
Zd = Xc @ Vtd[:2].T

# ── figure 8: digits in the first two PCs ──
fig, ax = plt.subplots(figsize=(8.6, 5.4))
cmap = plt.get_cmap('tab10')
for d in range(10):
    sel = yd == d
    ax.scatter(Zd[sel, 0], Zd[sel, 1], s=11, color=cmap(d), alpha=0.75,
               label=str(d), linewidths=0)
ax.set_xlabel(f'PC 1  ({100*var[0]:.0f}% of variance)')
ax.set_ylabel(f'PC 2  ({100*var[1]:.0f}% of variance)')
leg = ax.legend(title='digit', ncol=2, fontsize=10.5, title_fontsize=11,
                frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5),
                handletextpad=0.2, columnspacing=0.7, markerscale=1.9)
save(fig, 'digits_pca')

# ── figure 9: one digit, reconstructed from k components ──
idx = int(np.where(yd == 3)[0][0])
ks_d = (1, 4, 8, 16, 32)
fig, axes = plt.subplots(1, 6, figsize=(12.6, 2.6))
axes[0].imshow(Xd[idx].reshape(8, 8), cmap='gray_r', interpolation='nearest')
axes[0].set_title('original (64)', fontsize=13, pad=6)
mu = Xd.mean(0)
for ax, k in zip(axes[1:], ks_d):
    rec = mu + (Xc[idx] @ Vtd[:k].T) @ Vtd[:k]
    ax.imshow(rec.reshape(8, 8), cmap='gray_r', interpolation='nearest')
    ax.set_title(f'$k={k}$', fontsize=13, pad=6)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED); s.set_linewidth(0.8)
fig.subplots_adjust(wspace=0.08)
save(fig, 'digits_recon')

# ════════════════════════════════════════════════════════════════════
# Numbers quoted on slides
# ════════════════════════════════════════════════════════════════════
print('── DECK NUMBERS ──────────────────────────────────────────────')
print(f'2x2 check: sigma = ({s1:.4f}, {s2:.4f});  energy split = '
      f'{s1**2/(s1**2+s2**2):.0%} / {s2**2/(s1**2+s2**2):.0%}')
np_s = np.linalg.svd(A2, compute_uv=False)
print(f'np.linalg.svd sigma = {np_s.round(4)}')
print(f'image: n={n};  storage rank-k = k(2n+1) = 513k')
for k in ks:
    err = np.linalg.norm(img - rank_k(k))/fro
    print(f'  k={k:3d}: numbers={513*k:6d} ({513*k/n**2:5.1%} of {n**2}), '
          f'ratio={n**2/(513*k):5.1f}x, energy={energy[k-1]:6.1%}, rel err={err:5.1%}')
print(f'digits: PC1 var={var[0]:.1%}, PC2 var={var[1]:.1%}, PC1+PC2={var[:2].sum():.1%}')
print(f'digits: 90% of variance needs k={int(np.searchsorted(np.cumsum(var), 0.90)+1)}')
print(f'pca_bestfit: bad axis {f_bad:.0%}, principal axis {f_pc:.0%}')
print(f'digit-recon sample index {idx} (label {yd[idx]})')
print('figures written to', OUT)
