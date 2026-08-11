// The Optimization Landscape — Lecture 16 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture16/L16-optimization-landscape.typ
//   typst compile --root . --input handout=true lecture16/L16-optimization-landscape.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [The Optimization Landscape],
  subtitle: [When local information tells the global truth],
)

// ── shared numerics (computed at compile time; slides display these, never typed numbers) ──
#let fmt(x, d: 4) = {
  let r = calc.round(x + 0.0, digits: d)
  if r == 0.0 { r = 0.0 }                       // normalize -0.0 for display
  let s = str(r)
  if d == 0 { return s }
  if s.contains(".") {
    let p = s.split(".")
    let frac = p.at(1)
    while frac.len() < d { frac = frac + "0" }
    p.at(0) + "." + frac
  } else {
    let frac = ""
    while frac.len() < d { frac = frac + "0" }
    s + "." + frac
  }
}
// build mat() from code: a `;` straight after a #fmt(..) interpolation would be
// swallowed as a code-statement terminator and flatten the matrix to one row
#let m2(M, d: 4) = math.mat(
  (fmt(M.at(0).at(0), d: d), fmt(M.at(0).at(1), d: d)),
  (fmt(M.at(1).at(0), d: d), fmt(M.at(1).at(1), d: d)),
)
#let v2(v, d: 4) = $vec(#fmt(v.at(0), d: d), #fmt(v.at(1), d: d))$
#let amin = math.op("arg min", limits: true)

// ── sample a function into a series (mix curves, segments and dots in one plot) ──
#let pts(f, a, b, n: 96) = range(n + 1).map(i => { let x = a + i * (b - a) / n; (x, f(x)) })

// ── the deck's two mascots ──────────────────────────────────────────
// convex mascot: the course's favourite bowl (L5's matrix, L9's Hessian, L13's Σ)
#let BOWL = ((2.0, 1.0), (1.0, 2.0))
#let SADL = ((2.0, 0.0), (0.0, -2.0))
#let DOMEM = ((-2.0, -1.0), (-1.0, -2.0))
#let qform(A) = (x, y) => 0.5 * (A.at(0).at(0)*x*x + 2.0*A.at(0).at(1)*x*y + A.at(1).at(1)*y*y)
#let (lamB, _) = la.eig-sym(BOWL)
#let (lamS, _) = la.eig-sym(SADL)
#let (lamD, _) = la.eig-sym(DOMEM)

// non-convex mascot: the tilted double well  w(x) = (x²−1)² + x/2
// (note for later: it is the PRODUCT of two convex parabolas (x−1)²(x+1)², plus a tilt)
#let dw(x) = { let s = x*x - 1.0; s*s + 0.5*x }
#let dwp(x) = 4.0*x*x*x - 4.0*x + 0.5
#let dwpp(x) = 12.0*x*x - 4.0
// the slide computes its own stationary points: Newton on w′ from three starts
#let refine(x0) = { let x = x0; for i in range(16) { x = x - dwp(x) / dwpp(x) }; x }
#let XG = refine(-1.0)     // global minimum  ≈ −1.06
#let XM = refine(0.1)      // local maximum   ≈  0.13
#let XL = refine(0.9)      // local minimum   ≈  0.93
// its 2-D sibling (adds a clean y-bowl), and a non-convex comparison
#let R2(x, y) = dw(x) + 1.2*y*y
#let rug(x, y) = dw(x) + 1.2*y*y + 0.38*calc.sin(3.1*x)*calc.sin(2.3*y) + 0.22*calc.sin(2.1*y + 0.8)

// ── the worked least-squares problem: 3 points, 2 parameters ──
#let XD = ((1.0, 0.0), (1.0, 1.0), (1.0, 2.0))     // rows (1, xᵢ) for xᵢ = 0, 1, 2
#let YD = (1.0, 2.0, 4.0)
#let XtX = la.matmul(la.transpose(XD), XD)          // = ((3,3),(3,5))
#let Xty = la.matvec(la.transpose(XD), YD)          // = (7, 10)
#let THS = la.solve(XtX, Xty)                       // θ* = (5/6, 3/2)
#let (lamLS, _) = la.eig-sym(XtX)
#let lsf(b, w) = XD.zip(YD).map(p => { let r = p.at(0).at(0)*b + p.at(0).at(1)*w - p.at(1); r*r }).sum()

// ── small function library for the galleries ──
#let sq(x) = x * x
#let cube(x) = x * x * x
#let softp(x) = calc.ln(1.0 + calc.exp(x))

// ── tiny vector-drawing helpers for the convex-set panels (pure Typst, no raster) ──
#let dotm(x, y, col: INK) = place(dx: x - 1.2mm, dy: y - 1.2mm, circle(radius: 1.2mm, fill: col, stroke: none))
#let seg(x1, y1, x2, y2, col: INK) = place(line(start: (x1, y1), end: (x2, y2), stroke: 1.4pt + col))
#let panel(body) = box(width: 50mm, height: 32mm, body)

#title-slide()

// ═══════════════════════════ 1 · motivating comparison ═══════════════════════════
= Two optimization problems

== Linear and nonlinear model fitting

Both of these minimize a loss — L14 taught us that's what "fitting" is:

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [*fit a linear model*], [*train a deep network*]),
  [the code], [`np.linalg.lstsq(X, y)`], [loop of `loss.backward(); opt.step()`],
  [knobs to tune], [none], [learning rate, schedule, warmup, init, clipping, batch size, …],
  [rerun on the same data], [same answer, every time], [depends on the seed],
  [failure modes], [—], [divergence, plateaus, `nan` at step 617 (L2!)],
)

#pause
Both tasks *minimize a function of the parameters*, but the two objectives have different geometry.

#pause
#notebox[*Question.* Which differences come from the algorithm, and which come from the function being minimized?]

== Convex and non-convex objectives #V

The difference is the *shape of the surface* each one walks on:

#two(
  align(center, surface(qform(BOWL), xlim: (-1.8, 1.8), ylim: (-1.8, 1.8), samples: 24,
    cell: 2.6mm, height: 30mm, title: [linear model's loss: *one* bowl])),
  align(center, surface(rug, xlim: (-1.7, 1.7), ylim: (-1.4, 1.4), samples: 30,
    cell: 2.4mm, height: 30mm, title: [deep net's loss (cartoon): *many* dents])),
)

#pause
- Left: the objective has one basin; every local minimum is global
#pause
- Right: the objective can have several valleys, ridges, and saddle points

#pause
#result[For a convex objective, every local minimum is a global minimum.]

== From estimation to optimization

L14 expressed parameter fitting as minimization:

$ hat(theta) = amin_theta underbrace("NLL"(theta), "MSE, BCE, CE, …") quad quad #text(size: 17pt, fill: MUTED)[L14: "Fitting = minimizing NLL. Module 4 is the art of walking downhill."] $

#pause
Module 4 studies this minimization problem in six stages:

#align(center, diagram(
  spacing: (13mm, 8mm), node-stroke: 0.9pt,
  {
    node((0,0), align(center, text(size: 14pt)[*L16* \ objective geometry]), shape: fletcher.shapes.rect, fill: ACC.lighten(86%), stroke: ACC, inset: 6pt)
    node((1,0), align(center, text(size: 14pt)[*L17* \ gradient descent]), shape: fletcher.shapes.rect, fill: white, stroke: INK, inset: 6pt)
    node((2,0), align(center, text(size: 14pt)[*L18* \ Newton methods]), shape: fletcher.shapes.rect, fill: white, stroke: INK, inset: 6pt)
    node((3,0), align(center, text(size: 14pt)[*L19–20* \ constraints and KKT]), shape: fletcher.shapes.rect, fill: white, stroke: INK, inset: 6pt)
    node((4,0), align(center, text(size: 14pt)[*L21* \ recognize LP/QP]), shape: fletcher.shapes.rect, fill: white, stroke: INK, inset: 6pt)
    edge((0,0), (1,0), "-|>", stroke: 0.7pt + MUTED)
    edge((1,0), (2,0), "-|>", stroke: 0.7pt + MUTED)
    edge((2,0), (3,0), "-|>", stroke: 0.7pt + MUTED)
    edge((3,0), (4,0), "-|>", stroke: 0.7pt + MUTED)
  }
))

#pause
This lecture develops the geometry used by the remaining optimization lectures.

== Learning outcomes

By the end of this lecture you will be able to:

+ Use the optimizer's vocabulary: *objective, minimizer, stationary point, local vs global*.
+ Classify flat spots by the Hessian's *eigenvalue signs* — L9's gallery, now on the job.
+ Test a *set* for convexity with segments, and a *function* with chords.
+ Certify convexity fast: $f'' >= 0$ / $H succ.eq 0$, plus the *sum, max, affine-composition* rules.
+ Prove ⭐ that every local minimum of a convex function is global.
+ Prove *least squares is convex* via $H = 2X^top X succ.eq 0$ (⭐) — and say what deep nets give up.

// ═══════════════════════════ 2 · the predicament ═══════════════════════════
= Local information available to an optimizer

== The problem, stated once for the whole module

$ min_(theta in RR^d) f(theta) $

- $f$: the *objective* — for us, almost always a loss / NLL (L14)
#pause
- a *minimizer* $theta^star$: an input where the smallest value is reached — $theta^star = amin_theta f(theta)$; the *minimum* $f(theta^star)$ is the value there
#pause
- maximizing is the same craft upside-down: $max f = -min(-f)$ — L14 already negated log-likelihoods for exactly this reason, so *everything from now on is a minimization*

#pause
#result[The optimization problem is $min_theta f(theta)$; the following lectures differ in how they use information about $f$.]

== Function values, gradients, and Hessians #V

#align(center, lines(
  (
    pts(dw, -1.65, 1.65),
    pts(x => dw(0.55) + dwp(0.55) * (x - 0.55), 0.28, 0.82, n: 1),
    pts(dw, 0.38, 0.72),
    ((0.55, dw(0.55)),),
  ),
  colors: (MUTED.lighten(45%), INK, ACC, INK),
  dashes: ("dotted", "dashed", "solid", "solid"),
  markers: (false, false, false, true),
  annotations: (
    (0.62, 1.75, text(size: 14pt, fill: INK)[you are here: $f(theta)$, $nabla f(theta)$]),
    (-0.95, -1.05, text(size: 14pt, fill: MUTED)[the valley you *cannot see*]),
  ),
  size: (128mm, 47mm), x-label: $theta$, y-label: $f(theta)$,
))

#pause
At a current parameter value $theta$, an optimizer can evaluate three kinds of local information:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*probe*], [*what it is in ML*], [*cost*]),
  [$f(theta)$], [run the model, read the loss], [cheap],
  [$nabla f(theta)$], [backprop (L8–L11)], [cheap — one backward pass],
  [$H(theta)$], [curvature matrix (L9)], [$d^2$ entries — rarely affordable],
)

== Why not just look everywhere?

Because $theta$ lives in $RR^d$, and $d$ is not 2. Grid-search with a modest 10 values per coordinate:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*model*], [*$d$*], [*grid evaluations $10^d$*]),
  [line $w x + b$], [2], [$10^2 = 100$ — trivial],
  [tiny MLP], [$10^3$], [$10^1000$ — no.],
  [GPT-class model], [$10^11$], [$10^(10^11)$ — the observable universe has $tilde 10^80$ atoms],
)

#pause
#result[Exhaustive search is impossible at ML scale; practical methods use local evaluations of $f$, $nabla f$, and sometimes $H$.]

#pause
#notebox[Local derivatives describe a neighbourhood. Convexity is the additional global property that makes those local statements sufficient.]

== Taylor models use local information

L9 built "the formula every optimizer lives on" — today it starts living:

$ f(theta + bold(delta)) thin approx thin underbrace(f(theta), "probe 1") + underbrace(nabla f(theta)^top bold(delta), "probe 2: the local plane") + underbrace(1/2 bold(delta)^top H(theta) thin bold(delta), "probe 3: the local bowl or saddle") $

#pause
- L17 trusts the first two terms for one small step — *gradient descent*
#pause
- L18 trusts all three and jumps — *Newton's method*
#pause
- Today's question is prior to both: when do *local* conclusions ("it's flat here", "it curves up here") say anything about the *global* landscape?

// ═══════════════════════════ 3 · stationary points ═══════════════════════════
= Stationary points: where the search can stop

== Stationary points

While $nabla f(theta) eq.not bold(0)$, the Taylor plane is tilted; a sufficiently small step against the gradient lowers $f$ (L17 makes this precise).

#pause
$ "it can only park where" quad nabla f(theta) = bold(0) quad "— a" bold("stationary point") $

#pause
- Also called a *critical point*; the first-order model is flat there
#pause
- But *flat is not the same as bottom* — L7 met smiles, frowns and $x^3$'s fake-out in 1-D

#pause
#notebox[The gradient test finds candidate optima. The Hessian distinguishes minima, maxima, and saddle points.]

== Classifying stationary points in one dimension #V

#align(center, grid(columns: 3, column-gutter: 9mm, row-gutter: 3mm, align: center,
  lines(fn: sq, domain: (-1.4, 1.4), size: (52mm, 31mm), markers: false,
    points: ((0.0, 0.0, [min]),)),
  lines(fn: x => -sq(x), domain: (-1.4, 1.4), size: (52mm, 31mm), markers: false,
    points: ((0.0, 0.0, [max]),)),
  lines(fn: cube, domain: (-1.2, 1.2), size: (52mm, 31mm), markers: false,
    points: ((0.0, 0.0, [neither!]),)),
  text(size: 15pt)[$f'' = 2 > 0$ — smile],
  text(size: 15pt)[$f'' = -2 < 0$ — frown],
  text(size: 15pt)[$f'' = 6x = 0$ — inconclusive],
))

#pause
All three have $f'(0) = 0$. The *second* derivative tells them apart — except when it's $0$, where the test goes silent ($x^3$: neither a min nor a max).

== Hessian eigenvalues classify the local shape #V

#let gal-s(A, name) = surface(qform(A), xlim: (-1.6, 1.6), ylim: (-1.6, 1.6), samples: 20, cell: 2.1mm, height: 22mm, title: name)
#align(center, grid(columns: 3, column-gutter: 12mm, row-gutter: 3mm, align: center,
  gal-s(BOWL, [*bowl* — minimum]),
  gal-s(DOMEM, [*dome* — maximum]),
  gal-s(SADL, [*saddle* — neither]),
  text(size: 15pt)[$lambda = #fmt(lamB.at(0), d: 0), #fmt(lamB.at(1), d: 0)$ — all $> 0$],
  text(size: 15pt)[$lambda = #fmt(lamD.at(0), d: 0), #fmt(lamD.at(1), d: 0)$ — all $< 0$],
  text(size: 15pt)[$lambda = #fmt(lamS.at(0), d: 0), #fmt(lamS.at(1), d: 0)$ — mixed signs],
))

#pause
Curvature became a matrix in L9; its *eigenvalue signs* are the shape. (Eigenvalues above: computed live by the in-slide solver.)

#pause
#result[At a stationary point: $H$'s eigenvalue signs *name the flat spot* — all $+$ min, all $-$ max, mixed $=$ saddle.]

== The Hessian-eigenvalue test

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*eigenvalues of $H$ at $nabla f = bold(0)$*], [*classification*], [*descent implication*]),
  [all $lambda_i > 0$ (PD)], [local *minimum*], [park — locally, done],
  [all $lambda_i < 0$ (ND)], [local *maximum*], [any direction escapes],
  [mixed signs (indefinite)], [*saddle*], [escape along a $lambda < 0$ eigenvector],
  [some $lambda_i = 0$], [*test is silent*], [look beyond second order ($x^3$, troughs)],
)

#pause
Hand tests for $2 times 2$ still apply (L9): $det H < 0 arrow.r$ saddle; $det H > 0$ — read the trace.

#pause
#alertbox[This test is local. Positive Hessian eigenvalues certify a local minimum, not a global one.]

== Worked example: two stationary points

$ f(x, y) = x^3 - 3x + y^2 quad quad nabla f = vec(3x^2 - 3, 2y) = bold(0) quad arrow.r.double quad (x, y) = (plus.minus 1, 0) $

#pause
Hessian: $H = mat(6x, 0; 0, 2)$ — depends on where you stand (curvature is local):

#pause
- At $(1, 0)$: $H = mat(6, 0; 0, 2)$, $lambda = 6, 2$ — both positive $arrow.r$ *local minimum*
#pause
- At $(-1, 0)$: $H = mat(-6, 0; 0, 2)$, $lambda = -6, 2$ — mixed $arrow.r$ *saddle*

#pause
And since $f(x, 0) = x^3 - 3x arrow.r -infinity$ as $x arrow.r -infinity$: that "local minimum" is *not* global — nothing is. Flat-spot analysis alone can never tell you this.

== Classify both points in NumPy

#codebox[```python
import numpy as np

def hess(x, y):                       # for f = x**3 - 3*x + y**2
    return np.array([[6*x, 0.0], [0.0, 2.0]])

for p in [(1, 0), (-1, 0)]:
    lam = np.linalg.eigvalsh(hess(*p))          # symmetric -> real, sorted
    kind = "min" if lam[0] > 0 else ("max" if lam[-1] < 0 else "saddle")
    print(p, lam, kind)
# (1, 0)  [2. 6.]  min
# (-1, 0) [-6. 2.] saddle
```]

#pause
`eigvalsh` is the same symmetric-eigenvalue calculation used in L9; here the signs determine the local classification.

== A local minimum need not be global #V

Consider the non-convex function $w(x) = (x^2 - 1)^2 + x\/2$. Every marked value below is computed in-slide using Newton's method on $w' = 0$:

#align(center, lines(
  (pts(dw, -1.62, 1.62),),
  colors: (TEAL,), markers: false,
  points: (
    (XG, dw(XG), [global min $(#fmt(XG, d: 2), #fmt(dw(XG), d: 2))$]),
    (XL, dw(XL), [local min $(#fmt(XL, d: 2), #fmt(dw(XL), d: 2))$]),
    (XM, dw(XM), [local max]),
  ),
  size: (122mm, 45mm), x-label: $theta$, y-label: $w(theta)$,
))

#pause
- *Local minimum*: no better point in some ball around it. *Global minimum*: no better point anywhere.
#pause
- A walker started at $theta = 1.5$ parks at $#fmt(XL, d: 2)$ — flat, curving up, *fully convinced* — and is wrong by $#fmt(dw(XL) - dw(XG), d: 2)$ in loss.

#pause
#alertbox[The values of $f$, $nabla f$, and $H$ at $#fmt(XL, d: 2)$ do not reveal the lower minimum at $#fmt(XG, d: 2)$.]

== Checkpoint: what do local derivatives certify? #Q

#mcq([An optimizer stops at $hat(theta)$ on a smooth, *unknown* $f: RR^2 arrow.r RR$, with $nabla f(hat(theta)) = bold(0)$ and $H(hat(theta))$ having eigenvalues $2$ and $5$. What is *certain*?],
  [$hat(theta)$ is a local minimum — and that is all we can certify],
  [$hat(theta)$ is the global minimum],
  [$hat(theta)$ is a saddle point],
  [$f$ is convex],
)

== Answer: local derivatives certify a local minimum #A

#mcq-answer("A", [a local minimum — and nothing more],
  [Both eigenvalues positive $arrow.r$ PD $arrow.r$ a genuine local min (L9). But *global* (B) is a claim about all of $RR^2$, and local probes at one point can't reach that far — our two-valley mascot just demonstrated it. (D) reverses today's logic: curvature at *one* point never certifies convexity — $x^3$ has $f''(1) = 6 > 0$ yet isn't convex. Turning "local min" into "global min" needs a *property of $f$*, not a better probe. Enter convexity.])

// ═══════════════════════════ 4 · convex sets ═══════════════════════════
= Convex sets: segments that never leave

== The fix must be a property of shapes

What we want is a *certificate*:

$ underbrace("local conditions", nabla f = bold(0) ", " H succ.eq 0) + underbrace("a property of" f "as a whole", "?") quad arrow.r.double quad "global claim" $

#pause
That property is *convexity*, and it comes in two pieces, buildable in one lecture:

#pause
- *convex sets* — regions where straight segments never leave (this section)
#pause
- *convex functions* — surfaces whose chords never dip below the graph (next)

#pause
#notebox[Sets first because functions need somewhere to live: the chord test only makes sense if the segment between two inputs stays inside the domain.]

== The segment test

#two(r: (58%, 42%),
  [
    A set $C subset.eq RR^d$ is *convex* if for all $bold(x), bold(y) in C$ and every $t in [0, 1]$:

    $ t bold(x) + (1 - t) bold(y) in C $

    #pause
    - $t bold(x) + (1-t) bold(y)$ is a point *on the segment* from $bold(y)$ to $bold(x)$ — $t$ is the slider
    #pause
    - Convex = "walk between any two members and you never step outside"
  ],
  [
    #panel({
      place(dx: 12mm, dy: 3mm, circle(radius: 13mm, fill: TEAL.lighten(82%), stroke: 1.2pt + TEAL))
      seg(17mm, 11mm, 33mm, 22mm)
      dotm(17mm, 11mm); dotm(33mm, 22mm)
      place(dx: 12.5mm, dy: 6mm, text(size: 13pt)[$bold(x)$])
      place(dx: 34.5mm, dy: 20.5mm, text(size: 13pt)[$bold(y)$])
    })
    #v(1mm)
    #panel({
      place(dx: 0mm, dy: 0mm, curve(fill: RED.lighten(90%), stroke: 1.2pt + RED,
        curve.move((26mm, 2mm)),
        curve.quad((2mm, 16mm), (26mm, 30mm)),
        curve.quad((18mm, 16mm), (26mm, 2mm)),
        curve.close()))
      seg(23mm, 5.5mm, 23mm, 26.5mm, col: RED.darken(15%))
      dotm(23mm, 5.5mm); dotm(23mm, 26.5mm)
      place(dx: 25mm, dy: 13mm, text(size: 12.5pt, fill: RED.darken(10%))[escapes!])
    })
  ],
)

#pause
#result[One definition, one picture: *no shortcut between members ever leaves the club*.]

== Numbers: walking the segment

Take $bold(x) = (4, 1)$ and $bold(y) = (0, 3)$. The mix $t bold(x) + (1 - t) bold(y)$, by hand:

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([$t$], [$0$], [$1\/4$], [$1\/2$], [$3\/4$], [$1$]),
  [mix], [$(0, 3)$], [$(1, 2.5)$], [$(2, 2)$], [$(3, 1.5)$], [$(4, 1)$],
)

#pause
- $t = 0$ sits at $bold(y)$, $t = 1$ at $bold(x)$; in between, a straight walk — each coordinate is mixed independently
#pause
- The same expression with weights $(t, 1-t)$ will reappear within minutes as the *chord* of a function — same algebra, one dimension up

#pause
#notebox[Vocabulary you'll meet in Boyd: a *convex combination* is a mix with nonnegative weights summing to 1 — segments are the 2-point case.]

== Examples of convex and non-convex sets #V

#align(center, grid(columns: 3, column-gutter: 8mm, row-gutter: 1.5mm, align: center,
  panel({
    place(dx: 12mm, dy: 3mm, circle(radius: 13mm, fill: TEAL.lighten(82%), stroke: 1.2pt + TEAL))
    seg(18mm, 11mm, 32mm, 23mm)
    dotm(18mm, 11mm); dotm(32mm, 23mm)
  }),
  panel({
    place(dx: 7mm, dy: 4mm, rect(width: 36mm, height: 24mm, radius: 5mm, fill: TEAL.lighten(82%), stroke: 1.2pt + TEAL))
    seg(13mm, 9mm, 38mm, 24mm)
    dotm(13mm, 9mm); dotm(38mm, 24mm)
  }),
  panel({
    place(polygon(fill: TEAL.lighten(82%), stroke: 1.2pt + TEAL,
      (10mm, 10mm), (28mm, 3mm), (46mm, 13mm), (40mm, 30mm), (14mm, 29mm)))
    seg(16mm, 13mm, 40mm, 24mm)
    dotm(16mm, 13mm); dotm(40mm, 24mm)
  }),
  text(size: 15pt, fill: GREEN.darken(15%))[disk ✓], text(size: 15pt, fill: GREEN.darken(15%))[box ✓], text(size: 15pt, fill: GREEN.darken(15%))[polytope ✓],
  panel({
    place(polygon(fill: RED.lighten(90%), stroke: 1.2pt + RED,
      (10mm, 4mm), (26mm, 4mm), (26mm, 15mm), (19mm, 15mm), (19mm, 30mm), (10mm, 30mm)))
    seg(24mm, 9mm, 14mm, 27mm, col: RED.darken(15%))
    dotm(24mm, 9mm); dotm(14mm, 27mm)
  }),
  panel({
    place(dx: 0mm, dy: 0mm, curve(fill: RED.lighten(90%), stroke: 1.2pt + RED,
      curve.move((26mm, 2mm)),
      curve.quad((2mm, 16mm), (26mm, 30mm)),
      curve.quad((18mm, 16mm), (26mm, 2mm)),
      curve.close()))
    seg(23mm, 5.5mm, 23mm, 26.5mm, col: RED.darken(15%))
    dotm(23mm, 5.5mm); dotm(23mm, 26.5mm)
  }),
  panel({
    place(dx: 5mm, dy: 9mm, circle(radius: 8mm, fill: RED.lighten(90%), stroke: 1.2pt + RED))
    place(dx: 31mm, dy: 9mm, circle(radius: 8mm, fill: RED.lighten(90%), stroke: 1.2pt + RED))
    seg(13mm, 17mm, 39mm, 17mm, col: RED.darken(15%))
    dotm(13mm, 17mm); dotm(39mm, 17mm)
  }),
  text(size: 15pt, fill: RED)[a notch ✗], text(size: 15pt, fill: RED)[a crescent ✗], text(size: 15pt, fill: RED)[two islands ✗],
))

#pause
One escaped segment is enough to convict: convexity is a *for-all* claim, non-convexity needs one witness.

== The convex sets ML lives on

#table(
  columns: (1fr, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*set*], [*where you meet it*], [*convex?*]),
  [$RR^d$ — all of parameter space], [unconstrained training], [✓],
  [box $[0, 1]^d$], [valid pixel intensities], [✓],
  [half-space ${bold(x) : bold(a)^top bold(x) <= b}$], [one side of a flat wall], [✓],
  [norm ball ${theta : norm(theta) <= r}$ (L3)], [weight constraints, regularization], [✓],
  [simplex ${bold(p) : p_i >= 0, sum_i p_i = 1}$], [probability vectors — softmax outputs live here], [✓ (L19 optimizes over it)],
  [sphere ${bold(x) : norm(bold(x)) = 1}$ — the *skin* only], [normalized embeddings], [#text(fill: RED)[✗]],
)

#pause
The sphere fails beautifully: mix two antipodes at $t = 1\/2$ and land at $bold(0)$ — the center, off the skin. *Ball convex, shell not.*

#pause
#notebox[*Intersections preserve convexity* — clip convex sets together and the result stays convex (each fence is respected). L21's LP feasible regions are exactly such stacks of half-spaces. Unions, as the two islands showed, do not.]

// ═══════════════════════════ 5 · convex functions ═══════════════════════════
= Convex functions: the chord test

== Lift the segment to the graph #V

A function is *convex* when every chord — the segment joining two points *on the graph* — sits on or above the graph:

#align(center, lines(
  (
    pts(sq, -2.1, 3.1),
    ((-1.0, 1.0), (3.0, 9.0)),
  ),
  colors: (TEAL, ACC),
  markers: (false, true),
  points: ((1.0, 1.0, [graph: $1$]), (1.0, 5.0, [chord: $5$]),),
  size: (118mm, 46mm), x-label: $x$, y-label: $f(x)$,
))

#pause
$ f(underbrace(t x_1 + (1-t) x_2, "mix the inputs")) <= underbrace(t f(x_1) + (1-t) f(x_2), "mix the outputs = chord height") quad forall x_1, x_2, thin t in [0, 1] $

#pause
#result[Convex: *the graph hangs below its chords.* (Domain must be a convex set — now you know why sets came first.)]

== Reading the inequality with numbers

$f(x) = x^2$, endpoints $x_1 = -1$, $x_2 = 3$. Mixed input: $t x_1 + (1-t) x_2 = 3 - 4t$.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([$t$], [$0$], [$1\/4$], [$1\/2$], [$3\/4$], [$1$]),
  [mixed input $3 - 4t$], [$3$], [$2$], [$1$], [$0$], [$-1$],
  [graph $f("mix") = (3-4t)^2$], [$9$], [$4$], [$1$], [$0$], [$1$],
  [chord $t f(-1) + (1-t) f(3) = 9 - 8t$], [$9$], [$7$], [$5$], [$3$], [$1$],
)

#pause
Graph $<=$ chord in every column — with equality exactly at the endpoints. That is the definition, felt in integers.

#pause
#notebox[*Mnemonic:* conve*x* like a cu*p* — no. Try this instead: convex holds water ($union$-shaped), concave is a cave ($inter$-shaped). Crude, universal, effective.]

== Two checks that need no calculus

*$f(x) = x^2$, by pure algebra* (the chord inequality, expanded):

$ "LHS" - "RHS" = (t x_1 + (1-t) x_2)^2 - t x_1^2 - (1-t) x_2^2 = -t(1-t)(x_1 - x_2)^2 <= 0 quad checkmark $

#pause
($t(1-t) >= 0$ on $[0,1]$; a square is $>= 0$. Multiply out once at home — three lines.)

#pause
*$f(x) = abs(x)$, by the triangle inequality* (L3):

$ abs(t x_1 + (1-t) x_2) <= abs(t x_1) + abs((1-t) x_2) = t abs(x_1) + (1-t) abs(x_2) quad checkmark $

#pause
#result[The chord definition never mentions derivatives — it happily certifies the *kinked* $abs(x)$, where $f''(0)$ does not even exist.]

== The 1-D gallery, with verdicts #V

#align(center, grid(columns: 4, column-gutter: 7mm, row-gutter: 2mm, align: center,
  lines(fn: sq, domain: (-2.0, 2.0), size: (44mm, 30mm), markers: false),
  lines(fn: x => calc.abs(x), domain: (-2.0, 2.0), size: (44mm, 30mm), markers: false),
  lines(fn: x => calc.exp(x), domain: (-2.0, 2.0), size: (44mm, 30mm), markers: false),
  lines(fn: x => calc.ln(x), domain: (0.09, 4.0), size: (44mm, 30mm), markers: false),
  text(size: 15pt)[$x^2$ — convex on $RR$],
  text(size: 15pt)[$abs(x)$ — convex on $RR$],
  text(size: 15pt)[$e^x$ — convex on $RR$],
  text(size: 15pt)[$ln x$ — *concave* on $(0, infinity)$],
))

#pause
- $e^x$: convex even though it never turns around — convexity is about *bending up*, not about having a minimum
#pause
- $ln x$: bends *down* everywhere — concave; and $-ln x$ is convex (the shape of every NLL's building block, L14)

== $x^3$ · the function that teaches domains #V

#two(
  align(center, lines(
    (pts(cube, 0.0, 1.55), ((0.25, cube(0.25)), (1.4, cube(1.4)))),
    colors: (TEAL, GREEN), markers: (false, true),
    size: (60mm, 40mm), x-label: $x$, title: [on $[0, infinity)$: convex ✓],
  )),
  align(center, lines(
    (pts(cube, -1.55, 1.55), ((-1.3, cube(-1.3)), (1.0, cube(1.0)))),
    colors: (TEAL, RED), markers: (false, true),
    size: (60mm, 40mm), x-label: $x$, title: [on $RR$: the chord *crosses* ✗],
  )),
)

#pause
- On $[0, infinity)$ every chord clears the graph; on $(-infinity, 0]$ it's concave; on all of $RR$, neither — the right chord dips below *and* rises above
#pause
- "Is $f$ convex?" is ill-posed until you say *on which (convex) set* — same lesson the optimizers will learn about where they are allowed to search

== Concave functions reverse the inequality

$f$ is *concave* $arrow.l.r.double$ $-f$ is convex $arrow.l.r.double$ chords hang *below* the graph.

#pause
Module 3 already used concavity:

$ underbrace(max_theta thin ell(theta), "maximize log-likelihood (L14)") quad = quad underbrace(min_theta thin "NLL"(theta), "minimize a" #h(3pt) bold("convex") #h(3pt) "loss — often") $

#pause
- The coin's $ell(theta) = 7 ln theta + 3 ln(1 - theta)$: sum of concaves — concave; its NLL is convex
#pause
- Gaussian $ell(mu)$: a downward parabola in $mu$ (L14 showed it) — concave; MSE convex

#pause
#result[Maximizing a concave likelihood is equivalent to minimizing its convex negative log-likelihood.]

== One idea, two costumes: the epigraph

The two halves of today are the same object. The *epigraph* of $f$ is everything on or above the graph:

#two(r: (46%, 54%),
  align(center, box(width: 64mm, height: 42mm, {
    place(dx: 0mm, dy: 0mm, curve(fill: TEAL.lighten(86%), stroke: none,
      curve.move((6mm, 9mm)),
      curve.quad((32mm, 65mm), (58mm, 9mm)),
      curve.line((58mm, 3mm)),
      curve.line((6mm, 3mm)),
      curve.close()))
    place(dx: 0mm, dy: 0mm, curve(stroke: 2pt + TEAL,
      curve.move((6mm, 9mm)),
      curve.quad((32mm, 65mm), (58mm, 9mm))))
    seg(17mm, 19.5mm, 51mm, 15mm)
    dotm(17mm, 19.5mm); dotm(51mm, 15mm)
    place(dx: 27mm, dy: 6mm, text(size: 14pt, fill: TEAL.darken(25%))[epi $f$])
  })),
  [
    $ "epi" f = {(bold(x), v) : v >= f(bold(x))} $
    #pause
    - $f$ is a convex *function* $arrow.l.r.double$ $"epi" f$ is a convex *set*
    #pause
    - A chord's endpoints are in the epigraph, so the whole chord must be — and "chord stays in epi $f$" *is* "chord above graph"
  ],
)

#pause
#notebox[This is Boyd's favourite bridge (Ch 3.1.7) — one segment test runs the entire theory. Nothing new to memorize.]

// ═══════════════════════════ 6 · the toolkit ═══════════════════════════
= Recognizing convexity without chords

== Chords are the definition — calculus is the shortcut

Checking chords for every pair is direct but slow. For a twice-differentiable function on an open interval, curvature settles it:

$ f "convex" quad arrow.l.r.double quad f''(x) >= 0 "everywhere on the interval" $

#pause
The gallery, re-certified in one line each:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*$f$*], [*$f''$*], [*classification*]),
  [$x^2$], [$2 > 0$], [convex on $RR$ (and *strictly* — no flat stretches)],
  [$e^x$], [$e^x > 0$], [convex on $RR$],
  [$ln x$], [$-1\/x^2 < 0$], [concave on $(0, infinity)$],
  [$x^3$], [$6x$ — changes sign], [convex on $[0, infinity)$ only],
  [$abs(x)$], [undefined at $0$], [test silent — the chord already convicted ✓ convex],
)

== In $d$ dimensions: the Hessian test

$ f "convex on" C quad arrow.l.r.double quad H(theta) succ.eq 0 quad "for every" theta in C $

#pause
Worked: $f(theta_1, theta_2) = theta_1^2 + theta_2^2$.

$ H = mat(2, 0; 0, 2) quad "everywhere" quad arrow.r.double quad lambda = 2, 2 > 0 quad arrow.r.double quad "convex (strictly)" checkmark $

#pause
And the whole quadratic family at once: $q(theta) = 1/2 theta^top A theta$ has $H = A$ everywhere (L9's ⭐) — so *$q$ convex $arrow.l.r.double$ $A succ.eq 0$*: bowls and troughs yes, saddles and domes no. The shape gallery was a convexity gallery all along.

#pause
#alertbox[Assumptions: $C$ is open and convex and $f$ is twice continuously differentiable on $C$. One PD Hessian at one point proves nothing ($x^3$ at $x = 1$, again).]

== The composition rules: convexity is stackable

Convex pieces snap together — for $f, g$ convex:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*rule*], [*why (one breath)*]),
  [$alpha f$ for $alpha >= 0$], [scaling both sides of the chord inequality preserves it],
  [$f + g$], [add the two chord inequalities],
  [$f(A bold(x) + bold(b))$ — *affine into convex*], [an affine map sends segments to segments, then $f$'s chords take over],
  [$max(f, g)$], [the upper envelope: each chord clears whichever graph is on top],
)

#pause
#alertbox[Products and general compositions do not preserve convexity. For example, $(x-1)^2 dot (x+1)^2 = (x^2 - 1)^2$: two convex parabolas multiply to form a non-convex double well.]

== Assemble the objectives you'll actually meet

#table(
  columns: (1fr, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*objective*], [*build*], [*convex?*]),
  [$norm(X theta - bold(y))^2$ — least squares], [affine $theta arrow.bar X theta - bold(y)$ into $norm(dot)^2$], [✓ (⭐ proof next section)],
  [$norm(X theta - bold(y))^2 + lambda norm(theta)^2$ — ridge], [sum of two convex terms], [✓],
  [$norm(X theta - bold(y))^2 + lambda sum_i abs(theta_i)$ — lasso], [sum; $abs(dot)$ of an affine map], [✓],
  [$max(0, 1 - z(theta))$, $z$ affine — hinge], [max of affine and constant], [✓ (SVMs, ML course)],
  [deep net loss], [compositions upon compositions], [#text(fill: RED)[✗ in general]],
)

#pause
Adding a *convex* penalty with a nonnegative coefficient preserves convexity. A non-convex regularizer can destroy it.

#pause
#result[You will almost never chord-check a real objective. You will *assemble* it from certified parts — this table is how convexity scales.]

== Log-sum-exp is convex

L2 derived $"LSE"(bold(x)) = log sum_i e^(x_i)$ for numerical survival. It is also *convex* — here in 1-D against the max it approximates:

#align(center, lines(
  fn: (x => calc.max(x, 0.0), softp),
  domain: (-3.6, 3.6),
  colors: (MUTED, ACC), dashes: ("dashed", "solid"), markers: false,
  labels: ($max(x, 0)$, $log(e^0 + e^x)$),
  size: (100mm, 37mm), x-label: $x$,
))

#pause
- A *smooth max*: hugs $max(x, 0)$, never kinks — max's convexity survives the smoothing
#pause
- Why convex: its Hessian is $H = "diag"(bold(p)) - bold(p) bold(p)^top$ with $bold(p) = "softmax"(bold(x))$, and $bold(v)^top H bold(v) = "Var"_(i tilde bold(p))[v_i] >= 0$ — *a variance* (L12) cannot be negative

#pause
#notebox[Consequence you'll cash in ML: logistic-regression training minimizes $"LSE" - "affine"$, so it is convex — that's why it "just works" like least squares does.]

== Checkpoint: certify or convict #Q

#mcq([Which one is *not* convex on all of $RR^2$, $theta = (theta_1, theta_2)$?],
  [$norm(X theta - bold(y))^2 + 5 norm(theta)^2$],
  [$max(theta_1 + theta_2, thin e^(theta_1))$],
  [$log(e^(theta_1) + e^(theta_2))$],
  [$theta_1^2 - theta_2^2$],
)

== Answer: certify or convict #A

#mcq-answer("D", [$theta_1^2 - theta_2^2$ — a saddle, not a bowl],
  [Its Hessian is $mat(2, 0; 0, -2)$ everywhere — mixed eigenvalue signs, indefinite, the gallery's saddle: not convex (and not concave). The rest assemble from certified parts: (A) sum of convex ✓; (B) max of an affine and a convex ✓; (C) log-sum-exp ✓. Note how you convicted D and certified A–C *without one chord* — the toolkit is the point.])

// ═══════════════════════════ 7 · local and global minima ═══════════════════════════
= Local and global minima of convex functions

== ⭐ Every local minimum of a convex function is global #D

#result[If $f$ is convex, *every local minimum is a global minimum.*]

#pause
Proof by picture first — suppose a convex $f$ had a local min $hat(theta)$ *and* a strictly better point $bold(z)$:

#align(center, lines(
  (
    pts(dw, -1.3, 1.35),
    ((XL, dw(XL)), (XG, dw(XG))),
  ),
  colors: (TEAL, ACC), markers: (false, true),
  annotations: ((0.15, 0.28, text(size: 14pt, fill: ACC.darken(10%))[the chord *dips below* $f(hat(theta))$ immediately]),),
  size: (112mm, 42mm), x-label: $theta$,
  points: ((XL, dw(XL), [$hat(theta)$]), (XG, dw(XG), [$bold(z)$])),
))

#pause
The chord from $(hat(theta), f(hat(theta)))$ to $(bold(z), f(bold(z)))$ slopes downward. Convexity keeps the graph below that chord, so points arbitrarily close to $hat(theta)$ have lower values. This contradicts the assumption that $hat(theta)$ is a local minimum.

== ⭐ The same picture, in four lines of algebra #D

*Setup.* $f$ convex; $hat(theta)$ a local min; suppose some $bold(z)$ has $f(bold(z)) < f(hat(theta))$.

#pause
*Walk toward $bold(z)$.* For $t in (0, 1]$, the mix $theta_t = (1-t) hat(theta) + t bold(z)$ lies on the segment. Convexity:

$ f(theta_t) <= (1-t) f(hat(theta)) + t f(bold(z)) $

#pause
*Use the bad news.* Since $f(bold(z)) < f(hat(theta))$:

$ f(theta_t) < (1-t) f(hat(theta)) + t f(hat(theta)) = f(hat(theta)) $

#pause
*Contradiction.* As $t arrow.r 0$, $theta_t$ enters *every* ball around $hat(theta)$ — yet each $theta_t$ is strictly lower. No ball certifies $hat(theta)$ as a local min. $qed$

#pause
#result[For a convex function, a local minimum cannot coexist with a point having a lower objective value.]

== ⭐ Least squares is convex #D

The objective L4 set up, L6 solved at scale, L14 crowned as Gaussian NLL:

$ f(theta) = norm(X theta - bold(y))^2 = (X theta - bold(y))^top (X theta - bold(y)) = theta^top X^top X theta - 2 bold(y)^top X theta + bold(y)^top bold(y) $

#pause
Differentiate with L9's two earned rules ($nabla thin theta^top A theta = 2 A theta$ for symmetric $A$; $nabla thin bold(b)^top theta = bold(b)$):

$ nabla f(theta) = 2 X^top X theta - 2 X^top bold(y) quad quad quad H = nabla(nabla f) = 2 X^top X quad "— constant: the same bowl everywhere" $

#pause
Convexity now hangs on one question: is $X^top X$ *positive semi-definite for every possible data matrix* $X$?

== ⭐ The one-line certificate: $X^top X succ.eq 0$, always #D

Take any direction $bold(v) in RR^d$ and test the quadratic form (L9's definition of PSD):

$ bold(v)^top (X^top X) thin bold(v) = (X bold(v))^top (X bold(v)) = norm(X bold(v))^2 >= 0 quad checkmark $

#pause
#result[A squared length cannot be negative. Hence $H = 2 X^top X succ.eq 0$ for any data, so least squares is convex; every stationary point, when one exists, is a global minimizer.]

#pause
On our tiny dataset $x = (0, 1, 2)$, $y = (1, 2, 4)$ — everything below computed live in-slide:

$ X^top X = #m2(XtX, d: 0), quad lambda = #fmt(lamLS.at(0), d: 2), #fmt(lamLS.at(1), d: 2) > 0, quad nabla f = bold(0) arrow.r.double theta^star = #v2(THS, d: 2) $

#pause
Both eigenvalues are strictly positive, so this objective is strictly convex and $theta^star$ ($hat(y) = #fmt(THS.at(1), d: 1) x + #fmt(THS.at(0), d: 2)$) is its unique minimizer.

== The certified bowl, seen from above #V

$f(b, w) = sum_i (b + w x_i - y_i)^2$ over the whole parameter plane — one basin, one bottom:

#align(center, contour(
  (b, w) => lsf(b, w),
  xlim: (-1.3, 3.0), ylim: (0.0, 3.0), samples: 56,
  levels: (0.3, 0.8, 2.0, 4.0, 8.0, 14.0, 22.0, 32.0),
  size: (108mm, 40mm), color: TEAL,
  marks: ((THS.at(0), THS.at(1), [$theta^star = (#fmt(THS.at(0), d: 2), #fmt(THS.at(1), d: 1))$], RED),),
  x-label: $b$, y-label: $w$,
))

#pause
- Elongated, tilted — but *one* valley: every downhill path drains to $theta^star$ (L17 walks it; the tilt and stretch will matter there)
#pause
- When columns of $X$ are linearly dependent (L4's rank!), an eigenvalue of $X^top X$ hits $0$: the bowl grows a *flat trough* of equally-good minimizers — still convex, still all-global, just no longer unique

== Compare the two fitting problems

*Why does the linear model "just work"?*

#pause
- Least squares, ridge, lasso, and logistic regression have convex objectives under their usual formulations: local minima are global. An algorithm still needs suitable step sizes or other convergence conditions.
#pause
- A deep network is *compositions of compositions* — the one operation the rule table refuses to bless. Its landscape genuinely holds valleys, ridges, saddles, plateaus
#pause
- Deep-network training therefore lacks the same global certificate and is more sensitive to initialization and optimization choices.

#pause
#result[Convexity gives a global guarantee from local optimality. Non-convex objectives do not provide that guarantee.]

// ═══════════════════════════ 8 · non-convex reality ═══════════════════════════
= Non-convex objectives

== Several features of a non-convex objective #V

Our mascot's 2-D sibling, $R(x, y) = (x^2 - 1)^2 + x\/2 + 1.2 y^2$ — all three flavors of flat spot in one place (marks placed by the in-slide Newton solver):

#align(center, contour(
  R2,
  xlim: (-1.75, 1.75), ylim: (-1.05, 1.05), samples: 56,
  levels: (-0.35, -0.1, 0.2, 0.55, 1.0, 1.6, 2.4, 3.4),
  size: (112mm, 40mm), color: TEAL,
  marks: (
    (XG, 0.0, [global min], GREEN),
    (XL, 0.0, [local min], ACC),
    (XM, 0.0, [saddle], RED),
  ),
  x-label: $x$, y-label: $y$,
))

#pause
- The 1-D local *max* at $#fmt(XM, d: 2)$ became a 2-D *saddle* (down along $x$, up along $y$) — extra dimensions turn maxima into saddles
#pause
- Two basins, one ridge between: where you land is decided by where you start

== A counting heuristic for curvature signs

Toy model only: suppose the $d$ Hessian eigenvalue signs were independent fair coin flips. Then the chance of all-positive curvature would be $2^(-d)$:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*$d$*], [*$P("all" lambda > 0) approx 2^(-d)$*], [*meaning*]),
  [$2$], [$0.25$], [minima common in toy problems],
  [$20$], [$approx 10^(-6)$], [minima already rare],
  [$10^6$], [$approx 10^(-301030)$], [all-positive is negligible under this toy model],
)

#pause
- A saddle has a $lambda < 0$ eigen-direction that provides a local descent direction; saddles often slow first-order methods without being local minima
#pause
- Real Hessian eigenvalues are neither independent nor fair coin flips, so this table is not a probability model for neural-network stationary points. It only illustrates why mixed signs have many combinatorial patterns in high dimension.

== Plateaus: small gradients away from an optimum

#align(center, lines(
  (pts(x => 1.0 - calc.exp(-x * x), -3.9, 3.9, n: 120),),
  colors: (TEAL,), markers: false,
  points: ((0.0, 0.0, [true min]), (3.0, 1.0 - calc.exp(-9.0), [$f' approx 0.0007$ — "flat", says the probe]),),
  size: (108mm, 36mm), x-label: $theta$, y-label: $f(theta)$,
))

#pause
- On a *plateau*, $nabla f approx bold(0)$ with no minimum anywhere near: steps shrink to a crawl — the failure mode behind "training is stuck" (and behind saturated activations, as the DL course will show)
#pause
- Neural-network training can nevertheless find useful parameters. Explaining when and why requires assumptions beyond this lecture; initialization, normalization, and architecture all affect the landscape and optimization dynamics.

#pause
#notebox[Convex objectives support global optimality certificates. For non-convex objectives, local derivative tests remain local.]

== Lecture 16 — summary

- *One problem shape*: $min_theta f(theta)$ — evaluated through $f$, $nabla f$, and sometimes $H$ at the current $theta$.
- *Stationary points* $nabla f = bold(0)$: classified locally by the signs of $H$'s eigenvalues (L9) — minimum, maximum, or saddle.
- *Convex set*: segments never leave. *Convex function*: chords never dip below the graph; epigraph ties the two.
- *Certify fast*: $f'' >= 0$ / $H succ.eq 0$ everywhere; or assemble from rules — scale, sum, affine composition, max; log-sum-exp is convex (a variance says so).
- ⭐ *Convex $arrow.r.double$ local = global* — the four-line chord contradiction.
- ⭐ *Least squares is convex*: $H = 2 X^top X$ and $bold(v)^top X^top X bold(v) = norm(X bold(v))^2 >= 0$ — for any data.
- *Without convexity*: saddle points and plateaus can occur, and local optimality need not imply global optimality.

#pause
#notebox[*Read before L17* — MML §7.3 to 7.3.1 (convexity, ~4 pages, our notation) · Boyd & Vandenberghe, _Convex Optimization_ (free PDF): *skim* §2.1–2.3 for set pictures and §3.1.1–3.1.5 + 3.2 for the function tests and operation rules — read figures and boxed statements, skip proofs on a first pass. The Distill momentum article waits for L17.]

== Practice problems

Try on paper; anything numeric can be checked with five lines of NumPy.

+ Chord-check $f(x) = abs(2x - 1)$: prove convexity from the definition (triangle inequality + the affine trick).
+ Find *all* stationary points of $f(x, y) = x^4 + y^4 - 4x y$ and classify each with the Hessian. Which are global? (Careful: there's a tie.)
+ Is $g(x) = x e^x$ convex on $RR$? On which interval is it convex? ($g'' = (x + 2) e^x$.)
+ Ridge: show $H = 2 X^top X + 2 lambda I$ and explain why $lambda > 0$ makes $H$ positive *definite* even when $X^top X$ has a zero eigenvalue. What does that do to the trough of equally-good minimizers?
+ Prove or refute with a picture: (a) the intersection of two convex sets is convex; (b) the union is.
+ For $d = 2$, evaluate log-sum-exp's Hessian $"diag"(bold(p)) - bold(p) bold(p)^top$ at $bold(x) = (0, 0)$, find the eigenvector with $lambda = 0$, and connect it to L2's subtract-the-max trick.

== ⭐⭐⭐ Jensen's inequality: chords, industrialized #OPT

Run the chord inequality with $n$ points instead of 2 (induction), weights $t_i >= 0$, $sum t_i = 1$:

$ f(sum_i t_i x_i) <= sum_i t_i f(x_i) quad quad "and with" t_i = "probabilities:" quad f(EE[X]) <= EE[f(X)] $

#pause
- For *concave* $f$, flip: $EE[log X] <= log EE[X]$ — the single most-used inequality in probabilistic ML
#pause
- Module 5 runs on it: entropy bounds, $"KL" >= 0$, the ELBO — all one chord, industrialized (L22–L24)

#pause
#notebox[Weights on the simplex — the convex set from today's gallery. The definitions keep meeting.]

== ⭐⭐⭐ LSE's flat valley is L2's shift-invariance #OPT

Log-sum-exp is convex but not *strictly*: its bowl has one perfectly flat rail.

#align(center, lines(
  fn: (t => calc.ln(calc.exp(t) + calc.exp(-t)), t => t + calc.ln(2.0)),
  domain: (-2.4, 2.4),
  colors: (TEAL, ACC), markers: false,
  labels: ([across the valley: strictly convex], [along $(1,1)$: affine — flat rail]),
  size: (102mm, 36mm), x-label: $t$,
))

#pause
- Along $bold(1) = (1, dots, 1)$: $"LSE"(bold(x) + c bold(1)) = "LSE"(bold(x)) + c$ — affine, zero curvature; indeed $H bold(1) = ("diag"(bold(p)) - bold(p) bold(p)^top) bold(1) = bold(p) - bold(p) = bold(0)$
#pause
- That flat direction *is* L2's softmax shift-invariance — subtract-the-max just slides along the rail where nothing changes. Geometry and numerics, one fact.

#focus-slide[
  For a convex objective, every local minimum is global.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Gradient Descent* — now we actually walk downhill: trust the linear model for one small step, and repeat.
]
