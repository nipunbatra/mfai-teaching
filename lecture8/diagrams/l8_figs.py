#!/usr/bin/env python3
"""Figures for L8 (Gradients & the Geometry of Surfaces). Run from repo root: python3 lecture8/diagrams/l8_figs.py"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture8/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)

# the deck's diverging ramp: teal -> paper -> orange (matches common/mldiag.typ)
RAMP = mpl.colors.LinearSegmentedColormap.from_list('mfai', [TEAL, '#EFEEEB', ACC])

def clean3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo['grid'].update(color=(0.55, 0.6, 0.6, 0.25), linewidth=0.6)
        axis.line.set_color((0.35, 0.42, 0.44, 0.9))
    ax.tick_params(labelsize=8, pad=-1)

# ── the workhorse bowl of the lecture: f(x, y) = x² + 3y², ∇f = (2x, 6y) ──
f_bowl  = lambda x, y: x**2 + 3*y**2
fx_bowl = lambda x, y: 2*x
fy_bowl = lambda x, y: 6*y


def f_loss_landscape():
    """Hook: the training curve is the altimeter log of a walk on a loss SURFACE."""
    # a two-basin toy loss + plain gradient descent from (1.7, 1.35)
    f  = lambda x, y: (x**2 - 1)**2 + 0.7*y**2
    gx = lambda x, y: 4*x*(x**2 - 1)
    gy = lambda x, y: 1.4*y
    lr, steps = 0.03, 60
    px, py = [1.55, ], [1.3, ]
    for _ in range(steps):
        x, y = px[-1], py[-1]
        px.append(x - lr*gx(x, y)); py.append(y - lr*gy(x, y))
    px, py = np.array(px), np.array(py)
    pz = f(px, py)

    fig = plt.figure(figsize=(12.8, 3.6))
    # (a) the plot everyone stares at
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(np.arange(steps + 1), pz, color=INK, lw=2.2)
    ax0.plot(np.arange(0, steps + 1, 4), pz[::4], 'o', color=INK, ms=3.5)
    ax0.set_xlabel('training step', fontsize=11)
    ax0.set_ylabel(r'loss $\mathcal{L}$', fontsize=11)
    ax0.set_title('the training curve\n(what you are shown)', fontsize=12)
    ax0.tick_params(labelsize=9)
    # (b) the terrain it hides
    X, Y = np.meshgrid(np.linspace(-1.9, 1.9, 160), np.linspace(-1.6, 1.6, 160))
    Z = f(X, Y)
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=RAMP, rstride=2, cstride=2, alpha=0.85,
                     linewidth=0, antialiased=True)
    ax1.plot(px, py, pz + 0.06, color=RED, lw=2.4, zorder=5)
    ax1.plot(px[::4], py[::4], pz[::4] + 0.06, 'o', color=RED, ms=3.5, zorder=6)
    clean3d(ax1)
    ax1.set_xlabel('$w_1$', fontsize=10, labelpad=-4); ax1.set_ylabel('$w_2$', fontsize=10, labelpad=-4)
    ax1.set_zlabel(r'$\mathcal{L}$', fontsize=10, labelpad=-6)
    ax1.set_title('the loss surface\n(where you actually are)', fontsize=12)
    ax1.view_init(elev=38, azim=-64)
    # (c) the map of the walk
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.contour(X, Y, Z, levels=[0.12, 0.35, 0.8, 1.5, 2.5, 4, 6, 8.5], colors=TEAL, linewidths=1.1)
    ax2.plot(px, py, color=RED, lw=2.0)
    ax2.plot(px[::4], py[::4], 'o', color=RED, ms=3.5)
    ax2.plot([1, -1], [0, 0], '*', color=ACC, ms=12)
    ax2.annotate('start', (px[0], py[0]), textcoords='offset points', xytext=(6, 4),
                 fontsize=10, color=RED)
    ax2.set_xlabel('$w_1$', fontsize=11); ax2.set_ylabel('$w_2$', fontsize=11)
    ax2.set_title('the contour map\n(today: learn to read this)', fontsize=12)
    ax2.tick_params(labelsize=9)
    ax2.set_aspect('equal')
    fig.tight_layout(w_pad=2.0)
    save(fig, 'loss_landscape')


def f_surface_to_map():
    """How a contour map is made: slice the surface at constant heights, drop the rings flat."""
    X, Y = np.meshgrid(np.linspace(-3.3, 3.3, 200), np.linspace(-2.0, 2.0, 200))
    Z = f_bowl(X, Y)
    Z[Z > 10.0] = np.nan                     # clean elliptical bowl rim at f = 10
    heights = [2, 4, 6, 8]
    cols = [TEAL, BLUE, ACC, RED]
    angs = [5.1, 0.9, 5.8, 2.0]              # where each K-label sits on its ring

    fig = plt.figure(figsize=(11.6, 4.0))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=RAMP, rstride=3, cstride=3, alpha=0.45, linewidth=0)
    t = np.linspace(0, 2*np.pi, 200)
    for h, c in zip(heights, cols):
        xr, yr = np.sqrt(h)*np.cos(t), np.sqrt(h/3)*np.sin(t)
        ax.plot(xr, yr, h + 0.05, color=c, lw=2.6)           # ring on the surface
        ax.plot(xr, yr, 0.0, color=c, lw=1.2, alpha=0.55)    # its shadow on the floor
    clean3d(ax)
    ax.set_xlabel('$x$', fontsize=10, labelpad=-4); ax.set_ylabel('$y$', fontsize=10, labelpad=-4)
    ax.set_zlabel('$f$', fontsize=10, labelpad=-6)
    ax.set_title('slice at constant heights $f(x,y)=K$', fontsize=12)
    ax.view_init(elev=32, azim=-58)

    ax2 = fig.add_subplot(1, 2, 2)
    for h, c, a in zip(heights, cols, angs):
        ax2.plot(np.sqrt(h)*np.cos(t), np.sqrt(h/3)*np.sin(t), color=c, lw=2.4)
        ax2.annotate(f'$K={h}$', (1.13*np.sqrt(h)*np.cos(a), 1.13*np.sqrt(h/3)*np.sin(a)),
                     fontsize=11, color=c, ha='center', va='center')
    ax2.plot([0], [0], 'o', color=INK, ms=4)
    ax2.set_xlabel('$x$', fontsize=11); ax2.set_ylabel('$y$', fontsize=11)
    ax2.set_title('drop the rings on the floor: the contour map', fontsize=12)
    ax2.set_aspect('equal'); ax2.tick_params(labelsize=9)
    ax2.set_xlim(-3.4, 3.4); ax2.set_ylim(-2.0, 2.0)
    fig.tight_layout(w_pad=2.4)
    save(fig, 'surface_to_map')


def f_slice_partial():
    """Freezing one variable cuts the surface with a plane; the cut is a 1-D curve."""
    X, Y = np.meshgrid(np.linspace(-3, 3, 140), np.linspace(-2, 2, 140))
    Z = f_bowl(X, Y)
    zmax = 16.0
    fig = plt.figure(figsize=(11.6, 4.2))
    for k, (frz, ttl) in enumerate([('y', 'freeze $y=1$: slide along $x$'),
                                    ('x', 'freeze $x=2$: slide along $y$')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        Zc = Z.copy()
        if frz == 'y':
            Zc[Y > 1.0] = np.nan             # saw the mountain open at y = 1
            ax.plot_surface(X, Y, Zc, cmap=RAMP, rstride=3, cstride=3, alpha=0.45,
                            linewidth=0, vmin=0, vmax=zmax + 5)
            xs = np.linspace(-3, 3, 120)
            # the exposed cross-section at y = 1 (a filled cut face)
            verts = [[(x, 1.0, f_bowl(x, 1.0)) for x in xs] + [(3, 1.0, 0.0), (-3, 1.0, 0.0)]]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=mpl.colors.to_rgba(TEAL, 0.22),
                                                 edgecolor='none'))
            ax.plot(xs, np.ones_like(xs), f_bowl(xs, 1.0), color=TEAL, lw=3.2)
            ax.plot([2], [1], [f_bowl(2, 1) + 0.15], 'o', color=RED, ms=6)
        else:
            Zc[X > 2.0] = np.nan             # saw the mountain open at x = 2
            ax.plot_surface(X, Y, Zc, cmap=RAMP, rstride=3, cstride=3, alpha=0.45,
                            linewidth=0, vmin=0, vmax=zmax + 5)
            ys = np.linspace(-2, 2, 120)
            # the exposed cross-section at x = 2
            verts = [[(2.0, y, f_bowl(2.0, y)) for y in ys] + [(2.0, 2, 0.0), (2.0, -2, 0.0)]]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=mpl.colors.to_rgba(ACC, 0.22),
                                                 edgecolor='none'))
            ax.plot(np.full_like(ys, 2.0), ys, f_bowl(2.0, ys), color=ACC, lw=3.2)
            ax.plot([2], [1], [f_bowl(2, 1) + 0.15], 'o', color=RED, ms=6)
        clean3d(ax)
        ax.set_zlim(0, zmax)
        ax.set_xlabel('$x$', fontsize=10, labelpad=-4); ax.set_ylabel('$y$', fontsize=10, labelpad=-4)
        ax.set_zlabel('$f$', fontsize=10, labelpad=-6)
        ax.set_title(ttl, fontsize=12)
        ax.view_init(elev=30, azim=-62)
    fig.tight_layout(w_pad=1.6)
    save(fig, 'slice_partial')


def f_grad_field():
    """The gradient field: quiver arrows of ∇f over the contour map."""
    X, Y = np.meshgrid(np.linspace(-3, 3, 160), np.linspace(-2, 2, 160))
    Z = f_bowl(X, Y)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.contour(X, Y, Z, levels=[0.5, 1.5, 3, 5, 8, 12, 16], colors=TEAL, linewidths=1.1)
    qx, qy = np.meshgrid(np.linspace(-2.7, 2.7, 10), np.linspace(-1.75, 1.75, 8))
    U, V = fx_bowl(qx, qy), fy_bowl(qx, qy)
    ax.quiver(qx, qy, U, V, color=ACC, width=0.0042, scale=90, alpha=0.95)
    ax.plot([0], [0], 'o', color=RED, ms=6)
    ax.annotate(r'$\nabla f = \mathbf{0}$', (0, 0), textcoords='offset points',
                xytext=(8, -14), fontsize=12, color=RED)
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
    ax.set_aspect('equal'); ax.tick_params(labelsize=9)
    save(fig, 'grad_field')


if __name__ == '__main__':
    f_loss_landscape()
    f_surface_to_map()
    f_slice_partial()
    f_grad_field()
    print('done ->', OUT)
