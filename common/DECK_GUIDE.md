# MFAI Deck Guide — the style contract every lecture deck follows

This is the authoring contract for `lectureN/LN-<slug>.typ` decks. It encodes the
dl-teaching Typst system (cloned here) plus MFAI's pedagogy. A deck that follows this
guide is indistinguishable in *system* from every other deck; only the mathematics changes.

## 0. The one-paragraph summary

Touying + metropolis theme via `common/metropolis.typ`. One `== heading` per slide,
`= heading` per section divider, `#pause` for builds (nothing else). Handout compile is
what ships. Figures are computed in-Typst (chalkdust via `common/mldiag.typ`, fletcher)
wherever possible; matplotlib SVG+PNG twins only when Typst can't (3-D, images, bit-level
diagrams). Every lecture: AI hook first, picture → numbers → code → symbols for every
concept, one ⭐ reproducible derivation, MCQ checkpoint pairs, a `#focus-slide` insight
line last. 55–75 handout slides for 80 minutes.

## 1. Files & naming

```
lectureN/LN-<slug>.typ            # the deck (N = 1..26, no zero-padding)
lectureN/diagrams/lN_figs.py      # matplotlib figures (only if needed), run from repo root
lectureN/figures/*.svg + *.png    # committed outputs (SVG+PNG twins, dpi 200, transparent)
slides-pdf/LN.pdf                 # committed handout build
```

Compile (always from repo root; `--root .` is mandatory — figure paths are root-absolute):

```bash
typst compile --root . --input handout=true lectureN/LN-<slug>.typ slides-pdf/LN.pdf   # ships
typst compile --root . lectureN/LN-<slug>.typ /tmp/LN-presentation.pdf                  # to present
typst watch --root . --input handout=true lectureN/LN-<slug>.typ /tmp/LN.pdf            # authoring
./build-slides-pdf.sh                                                                    # all decks
```

## 2. Deck skeleton (copy verbatim, edit bracketed parts)

```typst
// <Lecture title> — Lecture N · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lectureN/LN-<slug>.typ
//   typst compile --root . --input handout=true lectureN/LN-<slug>.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *          // only if the deck uses chalkdust — most do
#show: metropolis-deck.with(
  title: [<Lecture title>],
  subtitle: [<subtitle — the lecture's promise in ≤ 8 words>],
)

#title-slide()

= <Section 1 — the AI hook>
== <first slide> ...

= <Section k>
...

#focus-slide[
  <Insight line from lecture-plan-detailed.md, verbatim.>
  #v(12pt)
  #set text(size: 22pt)
  Next: *<next lecture title>* — <one clause on why it follows>.
]
```

Never call `#slide(...)`, `#only`, `#alternatives`, or `#meanwhile`. `= heading` renders
a section-divider slide; `== heading` starts a slide. `#pause` is the only build primitive
(`#uncover("2-")` allowed for progressively-revealed tables only).

## 3. The pedagogy contract (non-negotiable, from course-design.md)

1. **AI hook opens** — Section 1 is a real AI capability/failure the math explains;
   the deck resolves it (or explicitly defers with a pointer). Use the hook stated in
   `lecture-plan-detailed.md` for this lecture.
2. **Picture → numbers → code → symbols** for every core concept, in that order:
   a visual/geometric slide (#V), a hand-checkable numeric example (2×2 matrices,
   3-state chains, 5-node graphs), ~5 lines of NumPy/PyTorch in a `#codebox`, and only
   then the general definition.
3. **One ⭐ derivation** — the one named in the plan, tagged `#D`, broken over 2–4 slides
   with a `#pause` per step; everything else is plausibility, not proof.
4. **MCQ checkpoints** — 2–3 per deck at natural break points: `== Checkpoint: <topic> #Q`
   with `#mcq(...)`, then `== Answer: <topic> #A` with `#mcq-answer(...)` on the NEXT slide.
5. **Callbacks** — the plan's continuity ledger names what this lecture plants/pays off;
   say it on the slide ("L2 callback: this is why we work in log-space").
6. **Insight line closes** — verbatim from the plan, in the final `#focus-slide`.
7. **⭐⭐⭐ optional material** — tag the heading `#OPT`, place at the deck's end, after
   the summary/checkpoint, before the focus slide.

## 4. Slide rhythm & density

- **55–75 handout slides** (80-minute lecture, ~1/min). Hard floor 50, hard cap 80.
- 5–9 sections (`=`), each 4–12 slides.
- ~1.5–2.5 `#pause` per content slide on average (bullets land one at a time when they
  carry separate ideas; don't pause every line reflexively).
- One idea per slide. If a slide needs >5 bullet lines or >2 display equations, split it.
- Body text stays at default size. Never shrink below 16pt to make content fit — cut instead.
- Captions under figures: `#align(center, text(size: 16pt, fill: MUTED)[...])`.

## 4a. Voice and narrative

Use `~/git/dl-teaching/lecture1` and `lecture2` as the language reference. The
course should still have examples, stories, intuition, and visual continuity;
the story must come from a concrete problem and its calculations.

- Prefer direct titles: **"Gaussian likelihood"**, **"Verify the four XOR
  inputs"**, **"The multiplier is a sensitivity"**, **"Solve the constrained
  example"**.
- Avoid manufactured plot language and recurring characters: no "mystery",
  "crime scene", "hero", "villain", "fate", "confession", "case closed",
  "superpower", or similar narration. Do not label sections "the hook" or
  "the payoff"; show the motivating example and return to it by name.
- A useful physical analogy is welcome when it explains the mathematics
  (rolling ball, contour touching a constraint, water filling an epigraph).
  Do not stack several metaphors around one result.
- Write in plain declarative sentences. Prefer **compute, compare, derive,
  verify, interpret, fit, constrain** over slogans such as "one weird trick",
  "license plate", "magic", or "the whole plot".
- Keep one numerical example alive across several slides. Reuse its symbols,
  colours, and values so that the derivation feels cumulative rather than
  episodic.
- `#result[...]` states a mathematical conclusion students can reuse. It is not
  a tagline.
- Introduce notation only when the current example needs it, following the
  example → calculation → general statement rhythm used in *Mathematics for
  Machine Learning*.
- Callbacks should be factual ("L9: positive Hessian eigenvalues imply local
  convexity"), not narrative debt ("the promise returns", "the cameo pays
  off").

## 5. Chips, callouts, boxes — when to use what

| Element | Use |
|---|---|
| `== Title #V` | slide whose payload is a figure/visual |
| `== Title #Q` / `#A` | MCQ checkpoint pair (adjacent slides) |
| `== Title #D` | a step of the ⭐ derivation |
| `== Title #I` | slide pointing at an interactive |
| `== Title #OPT` | ⭐⭐⭐ optional slide |
| `#result[...]` | THE takeaway of a section — the boxed line students photograph. ≤ 2 lines. |
| `#notebox[...]` | side-fact, convention, honest footnote |
| `#alertbox[...]` | pitfall/failure ("this is where training NaNs") |
| `#interbox(link-to: "https://…")[...]` | interactive pointer with link chip |
| `#codebox[```python ...```]` | code ≤ 12 lines; `#codebox(size: 13pt)` only when unavoidable |
| `#two(a, b)` | side-by-side (figure + bullets, before/after); `r: (55%, 45%)` to skew |

Interactives live at `https://nipunbatra.github.io/interactive-articles/` — define
`#let IA = "https://nipunbatra.github.io/interactive-articles/"` at the top when used, and
check `interactives-plan.md` for which articles map to this lecture.

## 6. Figures — the decision ladder

**(a) chalkdust (preferred — computed in the deck, palette-locked, vector):**

```typst
// function plots (single or multi-series)
#align(center, lines(fn: x => calc.exp(-x*x/2)/calc.sqrt(2*calc.pi), domain: (-4, 4),
  size: (95mm, 42mm), x-label: $x$, y-label: $p(x)$))
// several series: lines((xs1, ys1), (xs2, ys2), labels: (...), colors: (TEAL, ACC))

// distributions (dist.normal/uniform/exponential/laplace/bernoulli/categorical/gaussian-2d)
#align(center, lines(fn: x => dist.pdf(dist.normal(mu: 1.0, sigma: 0.5), x), domain: (-1, 3)))

// contour of any f(x,y), with optimizer trajectories driven by EXACT autodiff gradients
#let f = ad.expr("x^2 + 10*y^2", ("x", "y"))
#align(center, contour(ad.fn2(f), xlim: (-2.6, 2.6), ylim: (-0.9, 0.9),
  samples: 56, levels: 9, size: (105mm, 36mm), color: TEAL,
  paths: (gd(ad.grad-fn(f), (-2.3, 0.75), lr: 0.06, steps: 24),),
  marks: ((0, 0, [min], RED),)))
// optimizers: gd/momentum/nesterov/rmsprop/adam/sgd(grad, x0, lr:, steps:, ...)

// bar charts (probability distributions, letter counts, softmax w/ temperature)
#align(center, bars((0.5, 0.3, 0.2), labels: ($a$, $b$, $c$), annotate: true))

// in-slide numerics so prose and figure share one computation
#let (vals, vecs) = la.eig-sym(((2.0, 1.0), (1.0, 2.0)))   // tuple! vals desc-sorted
// la.matvec/matmul/solve/inv/det/transpose/dot/norm; ml.pca(data, k: 2); ml.linreg-fit
// rnd.randn(seed, i) for reproducible "random" data
```

**(b) fletcher** (imported via metropolis.typ: `diagram, node, edge`) — graphs and
block diagrams: computation graphs, Markov state diagrams, pipelines. See
`neuron-diagram`/`mlp-diagram` in `common/metropolis.typ` for the house style
(spacing ~20mm, node-stroke 0.9pt + INK, edges "-|>" with 0.7pt + MUTED).

**(c) matplotlib** (`lectureN/diagrams/lN_figs.py`) — only for: 3-D surfaces, image-based
demos (SVD compression, eigenfaces), dense annotated diagrams (IEEE-754 bit fields),
anything raster. Script must be runnable from repo root, self-contained, and use this
exact preamble:

```python
#!/usr/bin/env python3
"""Figures for LN (<title>). Run from repo root: python3 lectureN/diagrams/lN_figs.py"""
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
OUT='lectureN/figures'; os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', transparent=True)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', transparent=True, dpi=200)
    plt.close(fig)
```

Run it with (system python3 has no matplotlib):

```bash
uv run --no-project --with matplotlib,numpy python3 lectureN/diagrams/lN_figs.py
```

Reference from the deck with a **root-absolute** path and the SVG name (the `.png` twin
is loaded automatically): `#fig("/lectureN/figures/name.svg", w: 62%)`.

**Never ASCII art. Never a figure path that doesn't exist** (compile catches it — always
compile after adding figures).

## 7. Math conventions

- Typst math only (no LaTeX): `$bold(x)$`, `$norm(x)$`, `$EE[X]$`, `$cal(L)$`,
  `$nabla f$`, `$x_(t+1)$`, `$hat(theta)$`, `$underbrace(..., "label")$`, `$mat(1, 2; 3, 4)$`.
- Display math on its own line: `$ theta_(t+1) = theta_t - eta nabla cal(L)(theta_t) $`
  (auto left-aligned by the theme).
- Color inside math sparingly to bind symbol ↔ picture:
  `$#text(fill: ACC)[$eta$]$` — accent the ONE symbol the slide is about.
- Vectors bold (`bold(x)`), matrices caps (`A`), scalars lowercase, data-count $n$,
  dimension $d$, parameters $theta$, learning rate $eta$, loss $cal(L)$.

## 8. Quality gates before a deck is "done"

```bash
typst compile --root . --input handout=true lectureN/LN-<slug>.typ slides-pdf/LN.pdf
python3 scripts/audit_typst_slides.py slides-pdf/LN.pdf --max-raster-images <n_mpl_figs>
```

- Compile is clean (no missing fonts/images, no warnings).
- Handout page count 50–80.
- Audit passes: no clipped text, fonts embedded, raster count = the matplotlib figures
  you deliberately shipped (chalkdust/fletcher figures are vector and don't count).
- Every beat in the lecture's `lecture-plan-detailed.md` entry is on some slide; the ⭐
  derivation, insight line, reading, and AI hook all present.
- Spot-render 3–4 pages (`pdftoppm -png -r 72 -f K -l K slides-pdf/LN.pdf /tmp/pg`) and
  look: nothing overflows, figures readable at 72 dpi, palette consistent.
