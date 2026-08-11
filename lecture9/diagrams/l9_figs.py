#!/usr/bin/env python3
"""Figures for L9 (Jacobian, Hessian & Multivariate Taylor). Run from repo root: python3 lecture9/diagrams/l9_figs.py"""
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
OUT='lecture9/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

# ── the running R²→R² map: a gentle swirl warp ─────────────────────────────
A_, W_ = 0.6, 1.6
def phi(x, y):
    return x + A_*np.sin(W_*y), y + A_*np.sin(W_*x)
def jac(x, y):
    return np.array([[1.0,               A_*W_*np.cos(W_*y)],
                     [A_*W_*np.cos(W_*x), 1.0]])

ax_pt = np.array([0.5, 0.5])                 # the zoom point a
J = jac(*ax_pt)
print("J at a =", J, " det =", np.linalg.det(J))

# ═══ Figure 1: a vector→vector function warps the plane ══════════════════
def draw_grid(axp, warp):
    ticks = np.arange(-2, 2.01, 0.4)
    s = np.linspace(-2, 2, 220)
    for t in ticks:
        for (xs, ys) in ((np.full_like(s, t), s), (s, np.full_like(s, t))):
            if warp: xs, ys = phi(xs, ys)
            axp.plot(xs, ys, color=MUTED, lw=0.8, alpha=0.55, zorder=1)

def square_path(cx, cy, side, n=60):
    t = np.linspace(0, 1, n)
    e = np.ones_like(t)
    xs = np.concatenate([cx + side*t, cx + side*e, cx + side*(1-t), cx + 0*e])
    ys = np.concatenate([cy + 0*e,    cy + side*t, cy + side*e,     cy + side*(1-t)])
    return xs, ys

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for axp, warp, title in ((axes[0], False, 'input plane'),
                         (axes[1], True,  r'output plane:  $\Phi$ bends the grid')):
    draw_grid(axp, warp)
    xs, ys = square_path(ax_pt[0], ax_pt[1], 0.5)
    if warp: xs, ys = phi(xs, ys)
    axp.fill(xs, ys, color=ACC, alpha=0.30, zorder=3)
    axp.plot(xs, ys, color=ACC, lw=2.2, zorder=4)
    px, py = (phi(*ax_pt) if warp else ax_pt)
    axp.plot([px], [py], 'o', color=INK, ms=6, zorder=5)
    axp.annotate(r'$\mathbf{a}$' if not warp else r'$\Phi(\mathbf{a})$', (px, py),
                 textcoords='offset points', xytext=(-16, -16), fontsize=14, color=INK)
    axp.set_title(title, fontsize=14)
    axp.set_aspect('equal'); axp.set_xlim(-2.35, 2.35); axp.set_ylim(-2.35, 2.35)
    axp.set_xticks([-2, 0, 2]); axp.set_yticks([-2, 0, 2])
fig.text(0.5, 0.5, r'$\Phi$', fontsize=26, color=INK, ha='center', va='center')
fig.text(0.5, 0.40, r'$\longrightarrow$', fontsize=22, color=INK, ha='center', va='center')
fig.tight_layout(w_pad=4.0)
save(fig, 'warp_global')

# ═══ Figure 2: zoom — shrink the square, the parallelogram takes over ═════
# Recenter at Φ(a) and rescale by 1/ε: the TRUE image of the ε-square (ink)
# settles onto ONE fixed parallelogram (orange, edges = columns of J).
eps_list = [0.6, 0.25, 0.08]
uxs, uys = square_path(0.0, 0.0, 1.0, n=240)          # unit square boundary
para = np.array([[0, 0], J[:, 0], J[:, 0] + J[:, 1], J[:, 1], [0, 0]])

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
for k, (axp, eps) in enumerate(zip(axes, eps_list)):
    b = np.array(phi(*ax_pt))
    txs, tys = phi(ax_pt[0] + eps*uxs, ax_pt[1] + eps*uys)
    rxs, rys = (txs - b[0])/eps, (tys - b[1])/eps      # recentered, rescaled
    axp.fill(para[:, 0], para[:, 1], color=ACC, alpha=0.22, zorder=2)
    axp.plot(para[:, 0], para[:, 1], color=ACC, lw=2.0, ls='--', zorder=4)
    axp.plot(rxs, rys, color=INK, lw=2.0, zorder=3)
    axp.annotate('', xytext=(0, 0), xy=J[:, 0],
                 arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=2.4), zorder=6)
    axp.annotate('', xytext=(0, 0), xy=J[:, 1],
                 arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.4), zorder=6)
    axp.plot([0], [0], 'o', color=INK, ms=5, zorder=6)
    if k == 0:
        axp.text(J[0, 0] + 0.06, J[1, 0] - 0.10, r'$J\mathbf{e}_1$', color=TEAL, fontsize=14)
        axp.text(J[0, 1] - 0.42, J[1, 1] + 0.04, r'$J\mathbf{e}_2$', color=BLUE, fontsize=14)
        axp.text(-0.34, -0.30, r'$\Phi(\mathbf{a})$', fontsize=13, color=INK)
    if k == 2:
        axp.text(0.85, -0.28, r'area $= |\det J| \approx$ ' + f'{abs(np.linalg.det(J)):.2f}',
                 color=ACC, fontsize=12.5, ha='center')
    axp.set_title((r'$\varepsilon = $' + f'{eps}') +
                  ('  —  the parallelogram wins' if k == 2 else ''), fontsize=13)
    axp.set_aspect('equal')
    axp.set_xlim(-0.55, 2.05); axp.set_ylim(-0.55, 2.05)
    axp.set_xticks([]); axp.set_yticks([])
fig.text(0.5, 0.015,
         r'image of the $\varepsilon$-square at $\mathbf{a}$ (ink), recentered at $\Phi(\mathbf{a})$ and rescaled by $1/\varepsilon$'
         r'  vs  the parallelogram with edges $J\mathbf{e}_1, J\mathbf{e}_2$ (orange)',
         fontsize=12, color=MUTED, ha='center')
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'jacobian_zoom')

# ═══ Figure 3: the polar map — det J = r is the area dial ═════════════════
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5),
                         gridspec_kw={'width_ratios': [1.0, 1.06]})
dr, dth = 0.25, np.pi/8
cells = [(0.50, np.pi/4, TEAL), (1.50, np.pi/4, ACC)]

# left: (r, theta) space — equal little cells
axp = axes[0]
for r in np.arange(0, 2.001, dr):
    axp.plot([r, r], [0, 2*np.pi], color=MUTED, lw=0.8, alpha=0.55)
for th in np.arange(0, 2*np.pi + 1e-9, dth):
    axp.plot([0, 2], [th, th], color=MUTED, lw=0.8, alpha=0.55)
for (r0, th0, col) in cells:
    axp.fill([r0, r0+dr, r0+dr, r0], [th0, th0, th0+dth, th0+dth],
             color=col, alpha=0.75, zorder=3)
axp.set_title('input $(r, \\theta)$ space:  equal cells', fontsize=13.5)
axp.set_xlabel('$r$'); axp.set_ylabel(r'$\theta$')
axp.set_xlim(-0.06, 2.14); axp.set_ylim(-0.25, 2*np.pi + 0.25)
axp.set_xticks([0, 1, 2])
axp.set_yticks([0, np.pi, 2*np.pi]); axp.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])

# right: (x, y) space — wedge area grows with r
axp = axes[1]
th = np.linspace(0, 2*np.pi, 300)
for r in np.arange(dr, 2.001, dr):
    axp.plot(r*np.cos(th), r*np.sin(th), color=MUTED, lw=0.8, alpha=0.55)
for t0 in np.arange(0, 2*np.pi, dth):
    axp.plot([0, 2*np.cos(t0)], [0, 2*np.sin(t0)], color=MUTED, lw=0.8, alpha=0.55)
offsets = {TEAL: (1.15, -0.55), ACC: (0.75, 0.55)}
for (r0, th0, col) in cells:
    tt = np.linspace(th0, th0+dth, 40)
    xs = np.concatenate([(r0)*np.cos(tt), (r0+dr)*np.cos(tt[::-1])])
    ys = np.concatenate([(r0)*np.sin(tt), (r0+dr)*np.sin(tt[::-1])])
    axp.fill(xs, ys, color=col, alpha=0.85, zorder=3)
    area = dth*dr*(r0 + dr/2)
    rm, thm = r0 + dr/2, th0 + dth/2
    ox, oy = offsets[col]
    axp.annotate(f'area $\\approx$ {area:.3f}',
                 (rm*np.cos(thm), rm*np.sin(thm)),
                 xytext=(rm*np.cos(thm)+ox, rm*np.sin(thm)+oy),
                 fontsize=12.5, color=col,
                 arrowprops=dict(arrowstyle='-', color=col, lw=1.2))
axp.set_title(r'output $(x, y)$ space:  area scales by  $\det J = r$', fontsize=13.5)
axp.set_xlabel('$x$'); axp.set_ylabel('$y$')
axp.set_aspect('equal')
axp.set_xlim(-2.25, 3.05); axp.set_ylim(-2.3, 2.45)
axp.set_xticks([-2, 0, 2]); axp.set_yticks([-2, 0, 2])
fig.tight_layout(w_pad=3.0)
save(fig, 'polar_grid')

print('wrote', sorted(os.listdir(OUT)))
