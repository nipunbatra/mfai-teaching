// ═══════════════════════════════════════════════════════════════════
//  mldiag — deck-themed instances of the chalkdust packages.
//  The packages live in ~/git/chalkdust (symlinked into @local);
//  this file binds them to the metropolis palette ONCE so a slide is a
//  one-liner:
//
//    #import "../common/mldiag.typ": *
//    #contour(ad.fn2(f), paths: (gd(ad.grad-fn(f), (2, 1)),))   // palette-locked
//    #lines(fn: x => dist.pdf(dist.normal(), x), domain: (-4, 4))
// ═══════════════════════════════════════════════════════════════════

#import "@local/chalkdust-theme:0.1.0": theme
#import "@local/chalkdust-convgrid:0.1.0" as tg   // value grids (matrices as heat-cells)
#import "@local/chalkdust-plot:0.1.0" as mp       // bar & line plots
#import "@local/chalkdust-frame:0.1.0" as md      // pandas-lite: md.frame(csv("x.csv")), md.xy(f, "a", "b")
#import "@local/chalkdust-dist:0.1.0" as dist     // distributions: normal/uniform/beta-like, pdf/nll — the TRUE loss
#import "@local/chalkdust-field:0.1.0" as fld     // 2-D/3-D fields: contour/heatmap/surface of f(x,y)
#import "@local/chalkdust-optim:0.1.0" as opt     // optimizers: opt.gd/momentum/adam(grad, x0) → trajectory
#import "@local/chalkdust-rand:0.1.0" as rnd      // seeded PRNG: rnd.randn(seed, i), rnd.shuffle(seed, arr)
#import "@local/chalkdust-learn:0.1.0" as ml      // in-Typst fitting: linear/logistic regression, PCA, k-means
#import "@local/chalkdust-autodiff:0.1.0" as ad   // reverse-mode autodiff: ad.grad(f, x) exact; ad.grad-fn/fn2 for optim+contour
#import "@local/chalkdust-linalg:0.1.0" as la     // matvec/matmul/solve/inv/det/eig-sym — eigen demos computed in-slide
#import "metropolis.typ": INK, ACC, TEAL, GREEN, BLUE, MUTED, RED

// the deck theme: metropolis palette, teal→paper→orange diverging ramp
#let mfai-theme = theme(
  ink: INK, muted: MUTED, accent: ACC, accent2: TEAL,
  positive: GREEN, negative: RED,
  ramp: (TEAL, rgb("#EFEEEB"), ACC),
  cycle: (INK, GREEN, TEAL, ACC, RED),
)

// ── pre-themed figure one-liners ──
#let grid-map        = tg.grid-map.with(theme: mfai-theme)
#let conv-op         = tg.conv-op.with(theme: mfai-theme)
#let pool-op         = tg.pool-op.with(theme: mfai-theme)
#let attn-matrix     = tg.attn-matrix.with(theme: mfai-theme)
#let receptive-field = tg.receptive-field.with(theme: mfai-theme)
#let patchify        = tg.patchify.with(theme: mfai-theme)

// ── plots (ml-plot), palette-locked ──
#let bars  = mp.bars.with(theme: mfai-theme)
#let lines = mp.lines.with(theme: mfai-theme)

// ── 2-D / 3-D fields (ml-field), palette-locked ──
#let contour = fld.contour.with(theme: mfai-theme)
#let heatmap = fld.heatmap.with(theme: mfai-theme)
#let surface = fld.surface.with(theme: mfai-theme)

// ── optimizers (ml-optim) — pure numerics, no theme; feed the path to contour(paths:) ──
#let minimize = opt.minimize
#let gd       = opt.gd
#let momentum = opt.momentum
#let nesterov = opt.nesterov
#let rmsprop  = opt.rmsprop
#let adam     = opt.adam
#let sgd      = opt.sgd
#let numgrad  = opt.numgrad
#let grad2d   = opt.grad2d   // gradient of a loss f(x,y) → no hand-derived ∇ on the slide

// ── small fitted models (computed in Typst; feed predictions straight to lines) ──
#let linreg-fit     = ml.linreg-fit
#let linreg-predict = ml.linreg-predict

// ── the math helpers (figures + prose can share one computation) ──
#let conv2d        = tg.conv2d
#let conv-out-size = tg.conv-out-size
#let softmax-rows  = tg.softmax-rows
