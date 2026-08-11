#!/usr/bin/env python3
"""Figures for L18 (Second-Order Methods: Newton & Gauss-Newton).
Run from repo root:  uv run --no-project --with matplotlib,numpy,scipy python3 lecture18/diagrams/l18_figs.py
Emits SVG + PNG twins (transparent, metropolis palette) into lecture18/figures/,
and prints the REAL numbers (data, Gauss-Newton iterates, curve_fit answer) that
the deck bakes into its tables. Fixed seed -> reproducible slide numbers.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

INK='#23373B'; ACC='#EB811B'; TEAL='#2C7A7B'; GREEN='#14B03D'; MUTED='#6E7F82'; RED='#D64550'; BLUE='#2B6CB0'
mpl.rcParams.update({
  'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','savefig.transparent':True,
  'font.family':'sans-serif','font.sans-serif':['IBM Plex Sans','DejaVu Sans','Arial'],
  'text.color':INK,'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
  'axes.linewidth':1.0,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,
  'lines.linewidth':2.4,'lines.solid_capstyle':'round',
})
OUT='lecture18/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.svg + .png')

# ────────────────────────────────────────────────────────────────────
# The running example: fit y = a·exp(-b t) to 8 noisy sensor readings.
# ────────────────────────────────────────────────────────────────────
A_TRUE, B_TRUE, SIGMA = 2.5, 0.6, 0.08
rng = np.random.default_rng(18)
t = np.arange(8) * 0.5                          # 0.0, 0.5, ..., 3.5
y = A_TRUE * np.exp(-B_TRUE * t) + SIGMA * rng.standard_normal(8)
y = np.round(y, 3)                              # slide-friendly data (still "real")

def model(tt, a, b):
    return a * np.exp(-b * tt)

def resid(th):
    a, b = th
    return model(t, a, b) - y                   # r_i = f(t_i; θ) - y_i

def jac(th):
    a, b = th
    e = np.exp(-b * t)
    return np.stack([e, -a * t * e], axis=1)    # 8×2: [∂r/∂a, ∂r/∂b]

def loss(th):
    r = resid(th)
    return 0.5 * float(r @ r)

TH0 = np.array([1.0, 0.2])                      # deliberately bad initial guess

# ── Gauss-Newton loop (the deck's iterate table) ──
def gauss_newton(th0, iters=6):
    th, hist = th0.copy(), [th0.copy()]
    for _ in range(iters):
        r, J = resid(th), jac(th)
        delta = np.linalg.solve(J.T @ J, -J.T @ r)
        th = th + delta
        hist.append(th.copy())
    return np.array(hist)

GN = gauss_newton(TH0, iters=6)

# ── plain GD on the same loss (for the landscape contrast) ──
def grad(th):
    r, J = resid(th), jac(th)
    return J.T @ r                              # ∇L = Jᵀr

JTJ0 = jac(TH0).T @ jac(TH0)
lam_max = float(np.linalg.eigvalsh(JTJ0).max())
LR = 0.9 * 2 / lam_max                          # near the stability edge, still slow
def gd_run(th0, lr, steps):
    th, hist = th0.copy(), [th0.copy()]
    for _ in range(steps):
        th = th - lr * grad(th)
        hist.append(th.copy())
    return np.array(hist)

GD_STEPS = 60                                   # what the landscape figure draws
GD = gd_run(TH0, LR, GD_STEPS)

def gd_steps_to(tol, lr, cap=40000):
    Lstar = loss(gauss_newton(TH0, iters=8)[-1])
    th = TH0.copy()
    for k in range(1, cap + 1):
        th = th - lr * grad(th)
        if loss(th) - Lstar < tol:
            return k
    return None

# ── scipy cross-check: what curve_fit (Levenberg–Marquardt) returns ──
popt, pcov = curve_fit(model, t, y, p0=TH0, method='lm')

# ── console dump: every number the deck bakes in ──
np.set_printoptions(precision=4, suppress=True)
print("data t :", list(t))
print("data y :", list(y))
r0, J0 = resid(TH0), jac(TH0)
print("\nθ0 =", TH0, "  loss(θ0) =", f"{loss(TH0):.4f}")
print("r(θ0)  =", r0)
print("J(θ0) rows 0,1,7:", J0[0], J0[1], J0[7])
print("JᵀJ(θ0) =\n", J0.T @ J0)
print("Jᵀr(θ0) =", J0.T @ r0, "  (= ∇L)")
print("δ0 = solve(JᵀJ, -Jᵀr) =", np.linalg.solve(J0.T @ J0, -J0.T @ r0))
print("\nGauss-Newton iterates (a, b, loss):")
for k, th in enumerate(GN):
    print(f"  k={k}:  a={th[0]:.6f}  b={th[1]:.6f}  loss={loss(th):.10f}")
print("\ncurve_fit popt =", popt, " (std errs:", np.sqrt(np.diag(pcov)), ")")
print("JᵀJ eigenvalues at θ0:", np.linalg.eigvalsh(JTJ0), " -> stability edge 2/λmax =", 2/lam_max)
print(f"GD, tuned lr={LR:.4f} (90% of edge): steps to within 1e-6 of L*: {gd_steps_to(1e-6, LR)},"
      f"  1e-10: {gd_steps_to(1e-10, LR)}")
print(f"GD, lr=0.15 (~25% past the edge): steps to 1e-6: {gd_steps_to(1e-6, 0.15, cap=5000)} (never — bounces)")
print(f"GD after {GD_STEPS} plotted steps: θ=", GD[-1], f" loss={loss(GD[-1]):.8f}",
      f"  (GN loss after 5: {loss(GN[5]):.8f})")

# ────────────────────────────────────────────────────────────────────
# Fig 1 · the hook: noisy decay data (+ the fit everyone gets from scipy)
# ────────────────────────────────────────────────────────────────────
tt = np.linspace(0, 3.7, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.scatter(t, y, s=70, color=INK, zorder=5, label='sensor readings')
ax.plot(tt, model(tt, *popt), color=ACC, label=f'curve_fit: $a\\,e^{{-b t}}$')
ax.annotate(f'$a = {popt[0]:.3f}$\n$b = {popt[1]:.3f}$', xy=(2.3, 1.5),
            fontsize=14, color=ACC)
ax.set_xlabel('time $t$ (s)'); ax.set_ylabel('reading $y$')
ax.legend(frameon=False, loc='upper right', fontsize=12)
save(fig, 'decay_data')

# ────────────────────────────────────────────────────────────────────
# Fig 2 · Gauss-Newton fitted-curve evolution: iterations 0, 1, 2, 5
# ────────────────────────────────────────────────────────────────────
picks = [0, 1, 2, 5]
fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.9), sharey=True)
for ax, k in zip(axes, picks):
    a, b = GN[k]
    ax.scatter(t, y, s=34, color=INK, zorder=5)
    ax.plot(tt, model(tt, a, b), color=ACC)
    ax.set_title(f'iteration {k}', fontsize=13)
    ax.text(0.97, 0.80, f'$a={a:.2f}$\n$b={b:.2f}$\n$L={loss(GN[k]):.3f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=11.5, color=MUTED)
    ax.set_xlabel('$t$')
    ax.set_ylim(-0.15, 2.9)
axes[0].set_ylabel('$y$')
save(fig, 'gn_evolution')

# ────────────────────────────────────────────────────────────────────
# Fig 3 · the (a, b) landscape: GN jumps into the bowl, GD crawls
# ────────────────────────────────────────────────────────────────────
aa = np.linspace(0.4, 3.4, 220)
bb = np.linspace(-0.05, 1.15, 220)
AA, BB = np.meshgrid(aa, bb)
LL = np.zeros_like(AA)
for i in range(8):
    LL += 0.5 * (AA * np.exp(-BB * t[i]) - y[i]) ** 2
levels = np.geomspace(LL.min() + 0.02, LL.max(), 12)
fig, ax = plt.subplots(figsize=(6.9, 3.6))
ax.contour(AA, BB, LL, levels=levels, colors=[TEAL], linewidths=1.0, alpha=0.75)
ax.plot(GD[:, 0], GD[:, 1], color=MUTED, lw=1.9,
        label=f'gradient descent · {GD_STEPS} steps ($\\eta$ hand-tuned)')
ax.plot(GN[:, 0], GN[:, 1], color=ACC, lw=2.2, marker='o', ms=5.5, label='Gauss-Newton · 5 steps')
ax.plot(*popt, marker='*', ms=15, color=RED, ls='none', label='minimum', zorder=6)
ax.annotate(r'$\theta_0$', xy=(TH0[0], TH0[1]), xytext=(TH0[0]-0.28, TH0[1]-0.05),
            fontsize=14, color=INK)
ax.set_xlabel('$a$'); ax.set_ylabel('$b$')
ax.legend(frameon=False, fontsize=11.5, loc='upper left')
save(fig, 'gn_landscape')

# ────────────────────────────────────────────────────────────────────
# Fig 4 · the LM trust dial: (JᵀJ + λI)δ = -Jᵀr as λ sweeps GN → GD
# ────────────────────────────────────────────────────────────────────
lams = [0.0, 1.0, 4.0, 20.0, 100.0]
cols = [ACC, '#D9822B', '#B98A45', '#8F8A60', TEAL]
fig, ax = plt.subplots(figsize=(6.9, 3.6))
aa2 = np.linspace(0.55, 2.95, 200)
bb2 = np.linspace(-0.02, 0.95, 200)
AA2, BB2 = np.meshgrid(aa2, bb2)
LL2 = np.zeros_like(AA2)
for i in range(8):
    LL2 += 0.5 * (AA2 * np.exp(-BB2 * t[i]) - y[i]) ** 2
ax.contour(AA2, BB2, LL2, levels=np.geomspace(LL2.min() + 0.02, LL2.max(), 11),
           colors=[MUTED], linewidths=0.8, alpha=0.55)
g0 = J0.T @ r0
label_off = {0.0: (0.04, 0.02), 1.0: (0.05, -0.01), 4.0: (0.06, -0.03)}
# the lambda=20 and lambda=100 steps are tiny and cluster at theta0 — label them
# from open space with thin leader lines instead of stacking text on the arrows
leader_pos = {20.0: (1.60, 0.15), 100.0: (0.58, 0.115)}
for lam, c in zip(lams, cols):
    d = np.linalg.solve(J0.T @ J0 + lam * np.eye(2), -g0)
    label = 'GN step' if lam == 0 else (r'$\lambda=%g$' % lam)
    ax.annotate('', xy=TH0 + d, xytext=TH0,
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4, mutation_scale=16))
    if lam in leader_pos:
        ax.annotate(label, xy=TH0 + d, xytext=leader_pos[lam], fontsize=12, color=c,
                    arrowprops=dict(arrowstyle='-', color=c, lw=0.8, shrinkA=2, shrinkB=3))
    else:
        ax.annotate(label, xy=TH0 + d + np.array(label_off[lam]), fontsize=12, color=c)
gdir = -g0 / np.linalg.norm(g0) * 0.55
ax.annotate('', xy=TH0 + gdir, xytext=TH0,
            arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1.6, ls='--', mutation_scale=13))
ax.annotate(r'$-\nabla L$ direction', xy=TH0 + gdir + np.array([0.05, 0.0]),
            fontsize=12, color=TEAL)
ax.plot(*TH0, 'o', color=INK, ms=7)
ax.annotate(r'$\theta_0$', xy=TH0, xytext=TH0 + np.array([-0.2, -0.02]), fontsize=14, color=INK)
ax.plot(*popt, marker='*', ms=14, color=RED, ls='none', zorder=6)
ax.annotate('min', xy=popt, xytext=popt + np.array([0.05, 0.03]), fontsize=12, color=RED)
ax.set_xlabel('$a$'); ax.set_ylabel('$b$')
save(fig, 'lm_dial')

print("done.")
