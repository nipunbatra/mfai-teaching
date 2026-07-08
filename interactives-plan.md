# MFAI Interactives Plan

Interactives live in the separate repo `~/git/interactive` (published at nipunbatra.github.io/interactive-articles) and are linked/embedded from slides and the site — same model as dl-teaching. External explorables (Setosa, Distill, Seeing Theory, Olah) are linked directly from decks and resources.

## Existing — link as-is (from ~/git/interactive, 72 articles)

| Interactive | Used in | Note |
|---|---|---|
| `numerical-tricks` | L2, T1 | log-sum-exp, stable softmax, overflow sliders |
| `autograd` | L10–L11, T5 | autodiff vs finite differences vs symbolic; graph walk |
| `multivariate-normal` | L13, T6 | clouds, marginals, slices |
| `mle-map-coin` | L14–L15, T7 | slide-the-prior coin flips |
| `bayesian-posterior` | L15, T7 | prior × likelihood = posterior, 8 experiments |
| `optimizer-race` | L17, T8 | GD vs momentum vs Adam on 2D landscapes |
| `optimizers-beyond` | L18 | Newton, BFGS on a 2D loss |
| `lr-schedule-visualizer` | L17 ⭐⭐⭐ | pointer only |
| `info-theory` | L22, L24, T12 | entropy/CE/KL/MI recomputed as you drag |
| `prob-programming` | L15 ⭐⭐⭐ | enrichment |

## External — link from decks/resources (verified 2026-07)

| Explorable | Used in |
|---|---|
| [Setosa: eigenvectors & eigenvalues](https://setosa.io/ev/eigenvectors-and-eigenvalues/) | L5 |
| [Setosa: Markov chains](https://setosa.io/ev/markov-chains/) | L25 |
| [matrixmultiplication.xyz](http://matrixmultiplication.xyz/) | L4 |
| [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/) | L17 |
| [Seeing Theory](https://seeing-theory.brown.edu/) | L12, L15 |
| [Olah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | L24 |
| [Distribution Explorer](https://distribution-explorer.github.io/) | L12 |
| [Interactive Linear Algebra (GaTech)](https://textbooks.math.gatech.edu/ila/) | Module 1 reference |

(immersivemath.com was unreachable when checked — re-verify before listing.)

## To build (priority order, in ~/git/interactive)

| P | Interactive | For | Spec |
|---|---|---|---|
| P1 | **float-explorer** | L2, T1 | IEEE-754 bit toggles → value; number-line density view; toy 8-bit format matching T1 worksheet; eps/overflow markers. Fills the one true content gap. |
| P1 | **eigen-explorer** | L5 | drag a vector, see Av; eigen-directions glow when input aligns; 2×2 matrix editable; power-iteration animation mode |
| P2 | **svd-image-compressor** | L6, T3 | upload/choose image, rank-k slider, reconstruction + error curve (Strang demo, interactive) |
| P2 | **huffman-builder** | L23, T12 | type text → letter counts → greedy tree grows step-by-step; bits/char vs entropy meter |
| P3 | **taylor-zoom** | L7 | function picker, expansion point, degree slider; zoom shows local-line story |
| P3 | **lagrange-contours** | L19–L20 | contour + constraint curve; drag along constraint; gradient arrows align at optimum; KKT toggle for inequality |
| P4 | **markov-playground** | L25–L26, T13 | editable transition matrix, animated walker, stationary-distribution bar race; tiny text-training mode (bigram counts fill matrix) |
| P4 | **gaussian-ellipse** | L13 | drag Σ eigenvalues/angle, see cloud + ellipse + conditionals slice |

Build order tracks the semester: P1 needed by week 1–3, P4 by week 12.

## Technical notes

- Same stack as existing articles: standalone HTML/CSS/vanilla-JS, single file per article, no build step.
- Review pipeline: `scripts/review_interactive.py` in dl-teaching points at `~/git/interactive/src/articles` — reuse as-is.
- Each deck links its interactive on the relevant slide AND the site's schedule table gets an Interactive column entry.
