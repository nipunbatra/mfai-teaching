#!/usr/bin/env python3
"""Figures for L4 (Matrices as Linear Maps).
Run from repo root:  uv run --no-project --with matplotlib,numpy python3 lecture4/diagrams/l4_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture4/figures/.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture4/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')


# ---------------------------------------------------------------- the house --
# A recognizable, chiral shape: body+roof, door, chimney.  Chirality (chimney
# on the right side of the roof) makes reflections and rotations unmistakable.
HOUSE = [
    # (vertices, facecolor, alpha)
    (np.array([[0,0],[2,0],[2,1.4],[1,2.2],[0,1.4]]),            ACC,  0.75),  # body+roof
    (np.array([[0.8,0],[1.2,0],[1.2,0.7],[0.8,0.7]]),            'white', 0.95),# door
    (np.array([[1.5,1.8],[1.5,2.15],[1.75,2.15],[1.75,1.6]]),    INK,  0.9),   # chimney
]

def apply(A, pts):
    return pts @ np.asarray(A, dtype=float).T

def draw_grid(ax, A=None, lo=-4, hi=6, color=MUTED, alpha=0.5, lw=0.7, z=1):
    """Integer grid, optionally pushed through the matrix A."""
    A = np.eye(2) if A is None else np.asarray(A, dtype=float)
    for k in range(lo, hi + 1):
        for seg in (np.array([[k, lo], [k, hi]]), np.array([[lo, k], [hi, k]])):
            s = apply(A, seg)
            ax.plot(s[:, 0], s[:, 1], color=color, alpha=alpha, lw=lw, zorder=z)

def draw_axes(ax, lim):
    ax.axhline(0, color=INK, lw=0.9, alpha=0.65, zorder=2)
    ax.axvline(0, color=INK, lw=0.9, alpha=0.65, zorder=2)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    ax.set_aspect('equal'); ax.axis('off')

def draw_house(ax, A=None, s=1.0, t=(0.0, 0.0), z=4):
    A = np.eye(2) if A is None else np.asarray(A, dtype=float)
    for pts, fc, al in HOUSE:
        q = apply(A, pts * s) + np.asarray(t)
        ax.fill(q[:, 0], q[:, 1], facecolor=fc, alpha=al, edgecolor=INK, lw=1.4, zorder=z)

def draw_basis(ax, A=None, labels=('$e_1$', '$e_2$'), z=6, fs=14):
    A = np.eye(2) if A is None else np.asarray(A, dtype=float)
    for v, col, lab, dxy in ((A @ [1, 0], BLUE, labels[0], (0.12, -0.32)),
                             (A @ [0, 1], RED, labels[1], (0.12, 0.12))):
        ax.annotate('', xy=v, xytext=(0, 0), zorder=z,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6, mutation_scale=18))
        ax.text(v[0] + dxy[0], v[1] + dxy[1], lab, color=col, fontsize=fs, weight='bold', zorder=z)

def before_after(A, name, after_title, lim, figsize=(9.6, 4.3), basis=True,
                 after_basis_labels=('$Ae_1$', '$Ae_2$')):
    fig, axs = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    axs[0].set_title('before', fontsize=14, color=MUTED)
    draw_grid(axs[0]); draw_house(axs[0])
    if basis: draw_basis(axs[0])
    draw_axes(axs[0], lim)
    axs[1].set_title(after_title, fontsize=14, color=INK)
    draw_grid(axs[1], alpha=0.15)                       # ghost of the old grid
    draw_grid(axs[1], A=A, color=TEAL, alpha=0.75)      # the moved grid
    draw_house(axs[1], A=A)
    if basis: draw_basis(axs[1], A=A, labels=after_basis_labels)
    draw_axes(axs[1], lim)
    return fig


# ---- 1. the transformation gallery ------------------------------------------
def gallery():
    th = np.deg2rad(30)
    R30 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    fig = before_after(R30, 'tf_rotate', 'after  $R_{30°}$ — rotated', ((-2.6, 3.4), (-0.9, 3.1)))
    save(fig, 'tf_rotate')

    Sc = np.array([[2, 0], [0, 0.5]])
    fig = before_after(Sc, 'tf_scale', 'after  $D$ — stretched ×2, squeezed ×½', ((-1.4, 4.6), (-0.9, 2.6)))
    save(fig, 'tf_scale')

    Sh = np.array([[1, 1], [0, 1]])
    fig = before_after(Sh, 'tf_shear', 'after  $S$ — sheared (top slides right)', ((-1.2, 4.8), (-0.9, 2.6)))
    save(fig, 'tf_shear')

    Fl = np.array([[0, 1], [1, 0]])
    fig = before_after(Fl, 'tf_reflect', 'after  $F$ — mirrored across  $y=x$', ((-1.1, 3.2), (-0.9, 2.9)))
    save(fig, 'tf_reflect')

    # the MCQ mystery: vertical shear (bottom stays, right edge slides up)
    My = np.array([[1, 0], [1, 1]])
    fig = before_after(My, 'mcq_transform', 'after  $M$ = ?', ((-1.2, 3.4), (-0.9, 4.4)))
    save(fig, 'mcq_transform')


# ---- 2. columns are where the basis lands ------------------------------------
def basis_tracking():
    A = np.array([[2, 1], [1, 2]])
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.5), layout='constrained')
    lim = ((-1.4, 4.8), (-1.0, 3.8))
    axs[0].set_title('before', fontsize=14, color=MUTED)
    draw_grid(axs[0]); draw_house(axs[0], s=0.55); draw_basis(axs[0], fs=15)
    draw_axes(axs[0], lim)
    axs[1].set_title('after $A$ — the columns are the landed basis vectors', fontsize=13.5, color=INK)
    draw_grid(axs[1], alpha=0.15); draw_grid(axs[1], A=A, color=TEAL, alpha=0.6)
    draw_house(axs[1], A=A, s=0.55)
    draw_basis(axs[1], A=A, labels=('', ''), fs=13)
    axs[1].text(2.2, 0.62, r'$Ae_1=\binom{2}{1}$ = column 1', color=BLUE, fontsize=13, weight='bold')
    axs[1].text(-1.25, 2.35, r'$Ae_2=\binom{1}{2}$' + '\n= column 2', color=RED, fontsize=13, weight='bold')
    draw_axes(axs[1], lim)
    save(fig, 'basis_tracking')


# ---- 3. composition ----------------------------------------------------------
R90 = np.array([[0, -1], [1, 0]])
SH  = np.array([[1, 1], [0, 1]])

def composition_steps():
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.9), layout='constrained')
    lim = ((-3.6, 2.6), (-0.6, 3.0))
    steps = [(np.eye(2), 'start:  $x$', MUTED),
             (R90, 'after $R$ (rotate 90°):  $Rx$', INK),
             (SH @ R90, 'then $S$ (shear):  $S(Rx) = (SR)\\,x$', INK)]
    for ax, (M, title, tc) in zip(axs, steps):
        ax.set_title(title, fontsize=13.5, color=tc)
        draw_grid(ax, alpha=0.15); draw_grid(ax, A=M, color=TEAL, alpha=0.7)
        draw_house(ax, A=M); draw_axes(ax, lim)
    save(fig, 'composition_steps')

def composition_order():
    fig, axs = plt.subplots(2, 3, figsize=(10.8, 6.6), layout='constrained')
    lim = ((-3.6, 3.2), (-1.2, 3.2))
    rows = [((np.eye(2), R90, SH @ R90), ('start', 'rotate first:  $Rx$', 'then shear:  $(SR)\\,x$')),
            ((np.eye(2), SH, R90 @ SH), ('start', 'shear first:  $Sx$', 'then rotate:  $(RS)\\,x$'))]
    for r, (mats, titles) in enumerate(rows):
        for c, (M, title) in enumerate(zip(mats, titles)):
            ax = axs[r, c]
            final = (c == 2)
            ax.set_title(title, fontsize=13.5, color=(RED if final else INK) if c else MUTED,
                         weight='bold' if final else 'normal')
            draw_grid(ax, alpha=0.12); draw_grid(ax, A=M, color=TEAL, alpha=0.65)
            draw_house(ax, A=M); draw_axes(ax, lim)
    fig.text(0.985, 0.5, 'different endings!', color=RED, fontsize=15, weight='bold',
             rotation=90, ha='center', va='center')
    save(fig, 'composition_order')


# ---- 4. column space: span of the columns ------------------------------------
def column_space():
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.6), layout='constrained')
    lim = ((-4.4, 4.4), (-3.4, 3.4))

    ax = axs[0]
    ax.set_title('independent columns → span is the whole plane', fontsize=13)
    a1, a2 = np.array([2, 1]), np.array([1, 2])
    cs = np.linspace(-2, 2, 13)
    pts = np.array([c1 * a1 + c2 * a2 for c1 in cs for c2 in cs])
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=MUTED, alpha=0.4, zorder=2)
    for v, col, lab in ((a1, BLUE, '$a_1$'), (a2, RED, '$a_2$')):
        ax.annotate('', xy=v, xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.8, mutation_scale=18))
        ax.text(v[0] + 0.15, v[1] + 0.1, lab, color=col, fontsize=15, weight='bold')
    ax.text(-4.1, 2.8, 'every $b$ is reachable', color=GREEN, fontsize=12.5)
    draw_axes(ax, lim)

    ax = axs[1]
    ax.set_title('dependent columns → span is just a line', fontsize=13)
    a1, a2 = np.array([1, 2]), np.array([2, 4])
    t = np.linspace(-1.7, 1.7, 2)
    ax.plot(t * 2.2, t * 4.4, color=TEAL, lw=3, alpha=0.8, zorder=2)
    for v, col, lab, dy in ((a1, BLUE, '$a_1$', 0.0), (a2, RED, '$a_2=2a_1$', -1.1)):
        ax.annotate('', xy=v, xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.8, mutation_scale=18))
        ax.text(v[0] + 0.25, v[1] - 0.15 + dy, lab, color=col, fontsize=14, weight='bold')
    b = np.array([3, 1])
    ax.plot(*b, marker='x', ms=12, mew=3, color=RED)
    ax.text(b[0] - 0.4, b[1] - 0.95, 'unreachable $b$:\n$Ax=b$ has no solution', color=RED, fontsize=12)
    draw_axes(ax, lim)
    save(fig, 'column_space')


# ---- 5. rank collapse --------------------------------------------------------
def rank_collapse():
    C = np.array([[1, 2], [2, 4]])
    rng = np.random.default_rng(4)
    cloud = rng.uniform(-3, 3, size=(90, 2))
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.6), layout='constrained')
    lim = ((-5.5, 5.5), (-4.2, 4.2))

    ax = axs[0]
    ax.set_title('before: the whole plane…', fontsize=13.5, color=MUTED)
    draw_grid(ax); draw_house(ax)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=INK, alpha=0.5, zorder=3)
    nv = np.array([2, -1]) / np.sqrt(5) * 2.4
    ax.annotate('', xy=nv, xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.8, mutation_scale=18))
    ax.text(nv[0] - 1.6, nv[1] - 0.95, 'the doomed direction:\n$C\\binom{2}{-1}=\\binom{0}{0}$',
            color=RED, fontsize=12)
    draw_axes(ax, lim)

    ax = axs[1]
    ax.set_title('after $C$:  …lands on one line', fontsize=13.5)
    draw_grid(ax, alpha=0.12)
    t = np.linspace(-1.15, 1.15, 2)
    ax.plot(t * 5, t * 10, color=TEAL, lw=2.4, alpha=0.9, zorder=2)
    img = apply(C, cloud)
    ax.scatter(img[:, 0], img[:, 1], s=10, color=INK, alpha=0.5, zorder=3)
    hpts = apply(C, np.vstack([p for p, _, _ in HOUSE]) * 0.38)   # squashed house = a stick
    lo_i, hi_i = np.argmin(hpts[:, 0]), np.argmax(hpts[:, 0])
    ax.plot([hpts[lo_i, 0], hpts[hi_i, 0]], [hpts[lo_i, 1], hpts[hi_i, 1]],
            color=ACC, lw=7, alpha=0.95, zorder=5, solid_capstyle='round')
    ax.annotate('the house, squashed flat', xy=(hpts[hi_i, 0] * 0.7, hpts[hi_i, 1] * 0.7),
                xytext=(2.05, 0.6), color=ACC, fontsize=12.5, weight='bold',
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.4))
    ax.text(1.0, -2.9, 'every input lands on\nthe line $y=2x$', color=TEAL, fontsize=12.5)
    draw_axes(ax, lim)
    save(fig, 'rank_collapse')


# ---- 6. invertible or not ----------------------------------------------------
def invert_or_not():
    th = np.deg2rad(30)
    R30 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    C = np.array([[1, 2], [2, 4]])
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.4), layout='constrained')

    ax = axs[0]
    ax.set_title('rotation: nothing lost — undo with $A^{-1}$', fontsize=13, color=INK)
    lim = ((-4.4, 4.0), (-1.3, 3.9))
    draw_grid(ax, alpha=0.15)
    draw_house(ax, s=0.95, t=(-3.3, 0.0))
    draw_house(ax, A=R30, s=0.95, t=(1.3, 0.0))
    ax.annotate('', xy=(1.35, 2.75), xytext=(-1.6, 2.75), zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.2,
                                connectionstyle='arc3,rad=-0.25', mutation_scale=17))
    ax.annotate('', xy=(-1.6, 0.35), xytext=(2.1, 0.30), zorder=6,
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.2,
                                connectionstyle='arc3,rad=-0.25', mutation_scale=17))
    ax.text(-0.30, 3.45, '$A$', color=INK, fontsize=15, weight='bold')
    ax.text(0.05, -0.85, '$A^{-1}$', color=GREEN, fontsize=15, weight='bold')
    draw_axes(ax, lim)

    ax = axs[1]
    ax.set_title('collapse: two inputs, one output — no way back', fontsize=13, color=RED)
    lim = ((-3.8, 5.4), (-1.4, 7.0))
    draw_grid(ax, alpha=0.15, lo=-4, hi=7)
    t = np.linspace(-0.25, 0.85, 2)
    ax.plot(t * 4, t * 8, color=TEAL, lw=2.4, alpha=0.9, zorder=2)
    p, q = np.array([3, 0]), np.array([-1, 2])            # C p = C q = (3, 6)
    y = C @ p
    for v, col, lab in ((p, BLUE, '$p$'), (q, GREEN, '$q$')):
        ax.plot(*v, marker='o', ms=9, color=col, zorder=5)
        ax.text(v[0] + 0.18, v[1] - 0.75, lab, color=col, fontsize=15, weight='bold')
        ax.annotate('', xy=y, xytext=v, zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8,
                                    connectionstyle='arc3,rad=-0.12', mutation_scale=14))
    ax.plot(*y, marker='o', ms=10, color=RED, zorder=6)
    ax.text(y[0] + 0.3, y[1] - 0.25, r'$Cp = Cq = \binom{3}{6}$', color=RED, fontsize=13.5)
    ax.text(-3.5, 5.6, 'which input was it?\nthe answer is gone', color=RED, fontsize=12)
    draw_axes(ax, lim)
    save(fig, 'invert_or_not')


# ---- 7. least squares preview ------------------------------------------------
def least_squares():
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2), layout='constrained')

    ax = axs[0]
    t = np.array([0., 1., 2.]); y = np.array([1., 3., 4.])
    a, b = 7 / 6, 1.5                                    # verified with np.linalg.lstsq
    ts = np.linspace(-0.25, 2.35, 2)
    ax.plot(ts, a + b * ts, color=TEAL, lw=2.6, zorder=3, label=r'best line $\hat y = \frac{7}{6} + \frac{3}{2}t$')
    for ti, yi in zip(t, y):
        ax.plot([ti, ti], [yi, a + b * ti], color=RED, lw=1.8, ls='--', zorder=2)
    ax.scatter(t, y, s=70, color=INK, zorder=4)
    ax.text(1.28, 2.05, 'residuals', color=RED, fontsize=12.5)
    ax.set_title('no line passes through all 3 points', fontsize=13)
    ax.set_xlabel('$t$'); ax.set_ylabel('$y$')
    ax.set_xlim(-0.35, 2.45); ax.set_ylim(0, 5); ax.legend(fontsize=11.5, loc='upper left', frameon=False)
    ax.tick_params(length=3)

    ax = axs[1]
    ax.set_title('the same picture in column-space language', fontsize=13)
    plane = np.array([[-2.6, -0.9], [1.4, -0.9], [2.6, 0.9], [-1.4, 0.9]])
    ax.fill(plane[:, 0], plane[:, 1], facecolor=TEAL, alpha=0.14, edgecolor=TEAL, lw=1.2, zorder=1)
    ax.text(1.15, -0.78, 'col$(A)$ — everything $Ax$ can reach', color=TEAL, fontsize=12)
    bvec = np.array([0.9, 2.3]); proj = np.array([0.9, 0.25])
    ax.annotate('', xy=bvec, xytext=(-1.8, -0.55),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.4, mutation_scale=17))
    ax.annotate('', xy=proj, xytext=(-1.8, -0.55),
                arrowprops=dict(arrowstyle='-|>', color=ACC, lw=2.4, mutation_scale=17))
    ax.plot([bvec[0], proj[0]], [bvec[1], proj[1]], color=RED, lw=1.8, ls='--', zorder=3)
    ax.text(0.95, 2.3, '$b$ (off the plane)', color=INK, fontsize=13)
    ax.text(0.6, -0.25, r'$A\hat{x}$ = closest reachable point', color=ACC, fontsize=12.5)
    ax.text(1.05, 1.15, r'$b - A\hat{x}$', color=RED, fontsize=12.5)
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-1.3, 2.75)
    ax.set_aspect('equal'); ax.axis('off')
    save(fig, 'least_squares')


# ---- 8. image transforms (the anchor) ----------------------------------------
def _sample_image():
    try:
        import matplotlib.cbook as cbook
        with cbook.get_sample_data('grace_hopper.jpg') as f:
            img = plt.imread(f)
        g = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        return g[::2, ::2] / (255.0 if g.max() > 1.5 else 1.0)
    except Exception:
        h = w = 220
        yy, xx = np.mgrid[0:h, 0:w]
        g = 0.92 * np.ones((h, w))
        face = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (w * 0.38) ** 2
        g[face] = 0.55
        for ex in (w * 0.38, w * 0.62):
            g[(xx - ex) ** 2 + (yy - h * 0.42) ** 2 < (w * 0.05) ** 2] = 0.1
        mouth = ((xx - w / 2) ** 2 + (yy - h * 0.48) ** 2 < (w * 0.24) ** 2) & (yy > h * 0.58)
        g[mouth] = 0.1
        return g

def warp(img, A, pad=1.45):
    h, w = img.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    H, W = int(h * pad), int(w * pad)
    oy, ox = (H - 1) / 2, (W - 1) / 2
    Ainv = np.linalg.inv(A)
    ys, xs = np.mgrid[0:H, 0:W]
    X, Y = xs - ox, -(ys - oy)                       # math coords, y up
    src = Ainv @ np.stack([X.ravel(), Y.ravel()])
    sx = np.round(src[0].reshape(H, W) + cx).astype(int)
    sy = np.round(-src[1].reshape(H, W) + cy).astype(int)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.full((H, W), np.nan)
    out[ok] = img[sy[ok], sx[ok]]
    return out

def image_transform():
    img = _sample_image()
    th = np.deg2rad(25)
    mats = [(np.eye(2), 'original image (a grid of pixels)'),
            (np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]),
             'every pixel × a rotation matrix'),
            (np.array([[1, 0.4], [0, 1]]),
             'every pixel × a shear matrix')]
    cmap = plt.cm.gray.copy(); cmap.set_bad(alpha=0.0)
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.0), layout='constrained')
    for ax, (A, title) in zip(axs, mats):
        ax.imshow(np.ma.masked_invalid(warp(img, A)), cmap=cmap, vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_title(title, fontsize=13)
        ax.axis('off')
    save(fig, 'image_transform')


if __name__ == '__main__':
    print('L4 figures →', OUT)
    gallery()
    basis_tracking()
    composition_steps()
    composition_order()
    column_space()
    rank_collapse()
    invert_or_not()
    least_squares()
    image_transform()
    print('done.')
