// Second-Order Methods: Newton & Gauss-Newton — Lecture 18 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture18/L18-second-order.typ
//   typst compile --root . --input handout=true lecture18/L18-second-order.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Second-Order Methods],
  subtitle: [Newton, Gauss-Newton, and Levenberg–Marquardt],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

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
#let v2(v, d: 4) = $vec(#fmt(v.at(0), d: d), #fmt(v.at(1), d: d))$
#let sci(v) = {                                  // 2.3 × 10⁻¹⁰ style, for tiny errors
  let ex = calc.floor(calc.log(v))
  let m = v / calc.pow(10.0, ex)
  $#fmt(m, d: 1) times 10^(#ex)$
}

// ── the 1-D running function: f(x) = x − ln x, minimum at x* = 1 ──
// Newton simplifies to x⁺ = 2x − x² (so e⁺ = e², exactly — digits double).
#let f1(x) = x - calc.ln(x)
#let x0-1d = 0.5
#let f0-1d = f1(x0-1d)                           // 1.1931
#let fp-1d = 1.0 - 1.0 / x0-1d                   // −1
#let fpp-1d = 1.0 / (x0-1d * x0-1d)              // 4
#let xN-1d = x0-1d - fp-1d / fpp-1d              // 0.75 — the parabola's vertex
#let q1d(x) = f0-1d + fp-1d * (x - x0-1d) + 0.5 * fpp-1d * calc.pow(x - x0-1d, 2)
#let newton-hist = {                             // (k, x_k, |x_k − 1|) — the digits-double table
  let x = x0-1d
  let rows = ()
  for k in range(6) {
    rows.push((k, x, calc.abs(x - 1.0)))
    x = 2.0 * x - x * x
  }
  rows
}
#let gd-hist = {                                 // GD on the same f, η = 0.3 — for the race plot
  let x = x0-1d
  let pts = ()
  for k in range(31) {
    pts.push((k, calc.abs(x - 1.0)))
    x = x - 0.3 * (1.0 - 1.0 / x)
  }
  pts
}

// ── the κ = 50 quadratic bowl: GD crawls, Newton jumps (computed via la.solve) ──
#let bowl50 = ad.expr("0.5*x^2 + 25*y^2", ("x", "y"))
#let Hq = ((1.0, 0.0), (0.0, 50.0))
#let q0 = (-2.2, 0.65)
#let q-newton = (q0, la.vsub(q0, la.solve(Hq, ad.grad(bowl50, q0))))

// ── the saddle: GD escapes, Newton dives in (same machinery) ──
#let sad = ad.expr("0.5*x^2 - 0.5*y^2", ("x", "y"))
#let Hs = ((1.0, 0.0), (0.0, -1.0))
#let s0 = (1.7, 0.14)
#let s-newton = (s0, la.vsub(s0, la.solve(Hs, ad.grad(sad, s0))))

// ── L9's bowl, cross-examined live: Newton step at (1,1) on x² + xy + y² ──
#let A9 = ((2.0, 1.0), (1.0, 2.0))
#let qf9 = ad.expr("x^2 + x*y + y^2", ("x", "y"))
#let g9 = ad.grad(qf9, (1.0, 1.0))               // (3, 3)
#let d9 = la.scale(la.solve(A9, g9), -1.0)       // (−1, −1)

// ── the double well: negative curvature flips Newton uphill ──
#let dw(x) = calc.pow(x, 4) / 4 - x * x / 2
#let xd = 0.3
#let fd = dw(xd)
#let fpd = calc.pow(xd, 3) - xd                  // −0.273
#let fppd = 3 * xd * xd - 1.0                    // −0.73 < 0!
#let qdw(x) = fd + fpd * (x - xd) + 0.5 * fppd * calc.pow(x - xd, 2)
#let xt = xd - fpd / fppd                        // −0.074 — Newton aims at the hill
#let qt = qdw(xt)

#title-slide()

// ═══════════════════ 1 · motivating example ═══════════════════
= Nonlinear least squares in `scipy`

== A sensor, a decay, eight points #V

You're calibrating a sensor. After a pulse, its reading decays like $a thin e^(-b t)$ — and the datasheet needs the decay rate $b$. You log eight noisy readings:

#fig("/lecture18/figures/decay_data.svg", w: 56%)

#pause
Two unknowns $(a, b)$, eight equations that do not agree exactly. This is a nonlinear least-squares problem.

== Everyone calls the same one-liner

#codebox[```python
from scipy.optimize import curve_fit

def model(t, a, b):
    return a * np.exp(-b * t)

popt, pcov = curve_fit(model, t, y, p0=(1.0, 0.2))
# popt = [2.4274, 0.5628]
```]

#pause
No learning rate is supplied by the user; the solver adapts its step internally and returns the fitted parameters.

#pause
#notebox[With no bounds supplied, `curve_fit` defaults to `method='lm'`. We will derive the core damped Gauss-Newton idea and then distinguish it from MINPACK's full implementation.]

== From a linear model to a quadratic model

L17's gradient descent trusted a *line* — and paid for it in steps, with the condition number $kappa$ setting the price.

#pause
#result[When the Hessian is positive definite, the quadratic local model has a unique minimizer. Solving for it gives the Newton step.]

#pause
- *Newton*: model the loss with L9's quadratic Taylor approximation and solve for the model's minimum.
#pause
- *Gauss-Newton*: when the loss is a *sum of squares*, get the curvature almost free.
#pause
- *Levenberg–Marquardt*: damp the Gauss-Newton system when its local model is unreliable.

== Learning outcomes

By the end of this lecture you will be able to:

+ Derive ⭐ the Newton step $bold(delta) = -H^(-1) nabla f$ by minimizing the quadratic model.
+ Explain why exact Newton finishes a positive-definite quadratic in one step in exact arithmetic.
+ Run Newton by hand in 1-D and recognize *quadratic convergence* (digits double).
+ Price a Newton step ($d^2$ memory, $d^3$ solve) and name its failure modes (saddles, ascent).
+ Set up *nonlinear least squares*: residual vector $bold(r)$, its Jacobian $J$, $nabla cal(L) = J^top bold(r)$.
+ Derive the *Gauss-Newton* system $(J^top J) bold(delta) = -J^top bold(r)$ and interpret LM's damping parameter $lambda$.
+ Say precisely what `scipy.optimize.curve_fit` runs.

// ═══════════════════ 2 · a parabola instead of a line ═══════════════════
= A parabola instead of a line

== L17 in one slide: the line and its price

Gradient descent came from trusting the *first-order* Taylor model for one small step:

$ f(bold(x) + bold(delta)) approx f(bold(x)) + nabla f(bold(x))^top bold(delta) quad arrow.r.double quad bold(delta) = -eta thin nabla f(bold(x)) $

#pause
- The learning rate $eta$ = how far you trust a line that is only true at a point.
#pause
- On a bowl with condition number $kappa$: the safe $eta$ is set by the *steepest* wall, so the *flattest* direction crawls — the step budget grows like $kappa$. That was the main result from L17.

#pause
#notebox[A line has no bottom. However far you trust it, it only ever says "downhill is that way" — never "stop *here*".]

== A positive-definite quadratic model has a minimizer

L9 built the *second-order* Taylor model and promised it would run Module 4:

$ f(bold(x) + bold(delta)) thin approx thin underbrace(f(bold(x)), "height") + underbrace(nabla f(bold(x))^top bold(delta), "tilt") + underbrace(1/2 bold(delta)^top H(bold(x)) thin bold(delta), "bend") $

#pause
- GD kept height + tilt. Today we keep the *bend* too.
#pause
- If $H$ is positive definite (L9: all eigenvalues $> 0$), this model is a *bowl* — and a bowl has a *bottom you can name*.

#pause
#result[When the quadratic model is convex, solve directly for its minimizer.]

== Picture first: two models, two policies #V

Same function, same point, both local stories drawn at $x_k = #fmt(x0-1d, d: 1)$ (here $f(x) = x - ln x$):

#align(center, lines(
  fn: (f1, x => f0-1d + fp-1d * (x - x0-1d), q1d),
  domain: (0.2, 1.6), samples: 120, markers: false,
  colors: (INK, MUTED, ACC), dashes: ("solid", "dashed", "dashed"),
  labels: ([true $f$], [line — GD's model], [parabola — Newton's]),
  points: ((x0-1d, f0-1d, [$x_k$]), (xN-1d, q1d(xN-1d), [$x_(k+1)$])),
  vlines: ((1.0, [true min]),),
  size: (108mm, 47mm), x-label: $x$,
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[the line never bottoms out — *you* must choose a step; the parabola has a vertex — *it* chooses the step for you])

== The two policies, side by side

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [*gradient descent (L17)*], [*Newton (today)*]),
  [local model], [line: $f + nabla f^top bold(delta)$], [bowl: $f + nabla f^top bold(delta) + 1/2 bold(delta)^top H bold(delta)$],
  [the step], [$-eta thin nabla f$ — direction from the model, *length from you*], [solve for the model's minimum — *no $eta$ supplied*],
  [uses], [slopes only], [slopes *and* curvature],
)

#pause
One question remains: *where exactly is the bowl's bottom?* That formula is today's ⭐.

// ═══════════════════ 3 · the Newton step ═══════════════════
= The Newton step

== ⭐ Step 1 — in 1-D: minimize the parabola #D

The model around $x_k$, as a function of the step $delta$:

$ q(delta) = f(x_k) + f'(x_k) thin delta + 1/2 f''(x_k) thin delta^2 $

#pause
A parabola in $delta$ — minimize it like any 1-D function (set $(dif q) / (dif delta) = 0$, L7):

$ f'(x_k) + f''(x_k) thin delta = 0 quad arrow.r.double quad delta^* = -(f'(x_k)) / (f''(x_k)) $

#pause
*Numbers* — $f(x) = x - ln x$ at $x_k = 0.5$: $f' = 1 - 1/0.5 = -1$, $f'' = 1/0.5^2 = 4$:

$ delta^* = -(-1)/4 = 0.25 quad arrow.r.double quad x_(k+1) = 0.75 quad "— the vertex in the picture, exactly" $

== ⭐ Step 2 — in $n$-D: minimize the quadratic model #D

$ q(bold(delta)) = f(bold(x)_k) + nabla f^top bold(delta) + 1/2 bold(delta)^top H bold(delta) $

#pause
Differentiate with respect to $bold(delta)$ using L9's two gradient rules ($nabla (bold(g)^top bold(delta)) = bold(g)$, and the ⭐ rule $nabla (1/2 bold(delta)^top H bold(delta)) = H bold(delta)$ for symmetric $H$):

$ nabla_bold(delta) thin q = nabla f + H bold(delta) = bold(0) $

#pause
#result[*The Newton system:* solve $H bold(delta) = -nabla f$. If $H$ is invertible, this is $bold(delta)_"Newton" = -H^(-1) nabla f$.]

#pause
This stationary point is the model's *minimum* only when $H$ is positive definite. Indefinite or singular Hessians require safeguards.

== Read the step like an engineer

$ underbrace(bold(delta) = -eta thin nabla f, "GD") quad quad "vs" quad quad underbrace(bold(delta) = -H^(-1) thin nabla f, "Newton") $

#pause
- When $H$ is positive definite, Newton can be read as a gradient step preconditioned by $H^(-1)$.
#pause
- In the eigenbasis of a positive-definite $H$, direction $i$ is scaled by $1 slash lambda_i$: steep directions receive smaller components and flat directions larger ones.
#pause
- In practice you never form $H^(-1)$: *solve* $H bold(delta) = -nabla f$ (Module 1 discipline — solving is cheaper and more stable than inverting).

== Numbers: one Newton step on L9's bowl

$f(x, y) = x^2 + x y + y^2$ at $(1, 1)$ — the running function of L9:

$ nabla f = vec(2x + y, x + 2y) = vec(3, 3), quad quad H = mat(2, 1; 1, 2) "(everywhere)" $

#pause
Solve $H bold(delta) = -nabla f$ by hand (2×2 elimination):

$ mat(2, 1; 1, 2) bold(delta) = vec(-3, -3) quad arrow.r.double quad bold(delta) = vec(-1, -1) quad arrow.r.double quad bold(x)_1 = (0, 0) $

#pause
And $nabla f(0,0) = bold(0)$ — the *exact minimum, in one step*. (Chalkdust's solver agrees, live: $bold(delta) = #v2(d9, d: 0)$.) Not a coincidence — the next slide shows it holds for every quadratic.

== One Newton step solves any quadratic #D

Take *any* quadratic $f(bold(x)) = 1/2 bold(x)^top A bold(x) + bold(b)^top bold(x) + c$ with $A$ positive definite. Then (L9 rules):

$ nabla f = A bold(x) + bold(b), quad quad H = A quad "— the same matrix at every point" $

#pause
So from *any* starting point $bold(x)_0$:

$ bold(x)_1 = bold(x)_0 - A^(-1)(A bold(x)_0 + bold(b)) = bold(x)_0 - bold(x)_0 - A^(-1) bold(b) = -A^(-1) bold(b) $

#pause
— which is exactly where $nabla f = bold(0)$: *the* minimum.

#pause
#result[For a positive-definite quadratic, exact Newton reaches the minimizer in one step in exact arithmetic. Conditioning still affects the numerical stability of the linear solve.]

== Compare gradient descent and Newton on $kappa=50$ #V

Use the same start on $f = 1/2(x^2 + 50 y^2)$; both trajectories are computed in the slide:

#align(center, stack(dir: ltr, spacing: 8mm,
  contour(ad.fn2(bowl50), xlim: (-2.6, 2.6), ylim: (-0.9, 0.9),
    samples: 56, levels: 9, size: (72mm, 27mm), color: TEAL,
    paths: (gd(ad.grad-fn(bowl50), q0, lr: 0.036, steps: 40),),
    marks: ((0, 0, [min], RED),), title: [gradient descent · 40 steps]),
  contour(ad.fn2(bowl50), xlim: (-2.6, 2.6), ylim: (-0.9, 0.9),
    samples: 56, levels: 9, size: (72mm, 27mm), color: TEAL,
    paths: ((), (), q-newton),
    marks: ((0, 0, [min], RED),), title: [Newton · 1 step]),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[GD: zigzags die out, then a long crawl — 40 steps and still $tilde 23%$ of the $x$-gap left. Newton: $H bold(delta) = -nabla f$, solved once.])

#pause
#notebox[For an exactly represented positive-definite quadratic, Newton still needs one iteration as $kappa$ grows. In floating-point arithmetic, an ill-conditioned Newton system can lose accuracy.]

== Code: Newton in five lines

#codebox[```python
def newton(grad, hess, x, steps=10):
    for _ in range(steps):
        x = x - np.linalg.solve(hess(x), grad(x))   # solve, never invert
    return x

A = np.array([[1., 0.], [0., 50.]])                  # the kappa = 50 bowl
newton(lambda x: A @ x, lambda x: A, np.array([-2.2, 0.65]), steps=1)
# array([0., 0.])          <- one step, exact
```]

#pause
Compare L17: no `lr=`, no schedule, no divergence watch. The price appears in Section 5 — first, run it on functions that *aren't* parabolas.

== Checkpoint: Newton on a quadratic #Q

#mcq([Newton's method is applied to $f(x) = 3(x - 2)^2$ starting at $x_0 = 10$. Where is $x_1$?],
  [$x_1 = 2$],
  [$x_1 = 6$ — halfway],
  [$x_1 = 4$],
  [Cannot say — depends on the learning rate],
)

== Answer: one step reaches the quadratic minimum #A

#mcq-answer("A", [$x_1 = 2$, the exact minimum],
  [$f$ is quadratic, so the second-order model equals $f$: $f'(10) = 6(10 - 2) = 48$, $f'' = 6$, $delta = -48/6 = -8$, and $x_1 = 2$. Newton has no learning rate; the curvature $f''$ scales the gradient.])

// ═══════════════════ 4 · beyond quadratics ═══════════════════
= Newton beyond quadratics

== For a non-quadratic objective, rebuild the model

For a non-quadratic objective, the model is local, so recompute it after every step:

$ bold(x)_(k+1) = bold(x)_k - H(bold(x)_k)^(-1) nabla f(bold(x)_k) quad quad "(1-D: " x_(k+1) = x_k - f'(x_k) slash f''(x_k) ")" $

#pause
Each step reaches the stationary point of the current model, not necessarily the function's minimum; then the model is rebuilt.

#pause
*Our 1-D running example* $f(x) = x - ln x$ (min at $x^* = 1$): the update simplifies beautifully —

$ x_(k+1) = x_k - (1 - 1\/x_k) / (1\/x_k^2) = x_k - (x_k^2 - x_k) = 2 x_k - x_k^2 $

No calculus left at run time. Let's run it.

== You've met Newton before

Newton's method was born for *root finding* — solve $g(x) = 0$ by following the tangent:

$ x_(k+1) = x_k - g(x_k) / (g'(x_k)) quad quad "(Newton–Raphson, the school version:" thin sqrt(2) "via" thin x_(k+1) = (x_k + 2\/x_k)\/2 ")" $

#pause
Minimizing $f$ means solving $nabla f = bold(0)$. Substitute $g = f'$:

$ x_(k+1) = x_k - (f'(x_k)) / (f''(x_k)) quad "— the same formula we just derived from the parabola" $

#pause
#notebox[Two equivalent readings: solve for the parabola's vertex, or apply Newton root-finding to the gradient. A zero of $nabla f$ need not be a minimum.]

== Watch the digits double

$x_(k+1) = 2 x_k - x_k^2$ from $x_0 = 0.5$, run live in the slide (target $x^* = 1$):

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*k*], [*$x_k$*], [*error $|x_k - 1|$*], [*correct digits ($-log_10$)*]),
  ..newton-hist.map(((k, xv, e)) => (
    [$#k$], [#fmt(xv, d: 10)], [#sci(e)], [#fmt(-calc.log(e), d: 1)]
  )).flatten()
)

#pause
The correct-digit count $0.3 arrow.r 0.6 arrow.r 1.2 arrow.r 2.4 arrow.r 4.8 arrow.r 9.6$ — *doubling every step*.

#pause
#result[Here it's exact: $x_(k+1) - 1 = 2x_k - x_k^2 - 1 = -(x_k - 1)^2$, so $e_(k+1) = e_k^2$. The error *squares*.]

== Convergence on a log scale #V

Same function, same start: GD ($eta = 0.3$) vs Newton, error per iteration — log scale:

#align(center, lines((gd-hist, newton-hist.map(((k, xv, e)) => (k, e))),
  log-y: true, legend: "tr",
  labels: ([gradient descent], [Newton]),
  colors: (TEAL, ACC), markers: true,
  size: (105mm, 47mm), x-label: [iteration], y-label: [error],
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[GD is a *straight line* on a log scale — a constant factor ($0.7$) per step. Newton's slope steepens every step: the error is squaring. $9.6$ correct digits by $k = 5$ vs $5.3$ by $k = 30$.])

== Local quadratic convergence

Near a non-degenerate minimizer, if the Hessian is Lipschitz continuous and each Newton system is solved accurately:

$ norm(bold(e)_(k+1)) <= C norm(bold(e)_k)^2 quad quad "— \"quadratic convergence\": digits double per step" $

#pause
- GD has a linear rate on a strongly convex quadratic. Newton has a local quadratic rate; the constant $C$ and the size of the convergence neighbourhood still depend on the problem's derivatives and conditioning.
#pause
- The qualification is *near*: far from the minimum, the quadratic approximation may be inaccurate.

#pause
#alertbox[Newton converges rapidly when initialized near a suitable minimum. Far away, the step can increase the objective or converge to another stationary point.]

// ═══════════════════ 5 · cost and failure modes ═══════════════════
= Computational cost and failure modes

== Computational cost of one Newton step

For $d$ parameters, per iteration:

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [*needs*], [*memory*], [*arithmetic*]),
  [gradient descent], [$nabla f$], [$O(d)$], [$O(d)$ once $nabla f$ is in hand],
  [Newton], [$nabla f$, $H$, a solve], [$O(d^2)$], [$O(d^3)$ for the solve],
)

#pause
At $d = 10^6$, a dense $H$ has $10^12$ entries $approx$ *4 TB* in float32, before any workspace. A generic dense factorization has cubic arithmetic cost and is impractical at this scale.

#pause
#result[Dense Newton steps can be effective at small dimension, but quadratic storage and cubic generic solves rule them out at million-parameter scale.]

== Newton can converge to a saddle point #V

Newton chases $nabla f = bold(0)$ — and (L16) a saddle *is* a $nabla f = bold(0)$ point. On $f = 1/2(x^2 - y^2)$:

#align(center, stack(dir: ltr, spacing: 10mm,
  contour(ad.fn2(sad), xlim: (-2.4, 2.4), ylim: (-2.4, 2.4),
    samples: 52, levels: 10, size: (46mm, 40mm), color: TEAL,
    paths: (gd(ad.grad-fn(sad), s0, lr: 0.11, steps: 26),),
    marks: ((0, 0, [saddle], RED),), title: [gradient descent · drifts off]),
  contour(ad.fn2(sad), xlim: (-2.4, 2.4), ylim: (-2.4, 2.4),
    samples: 52, levels: 10, size: (46mm, 40mm), color: TEAL,
    paths: ((), (), s-newton),
    marks: ((0, 0, [saddle], RED),), title: [Newton · dives straight in]),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[GD moves along the negative-curvature direction; the raw Newton step solves for the saddle's stationary point and reaches it in one step])

== An indefinite Hessian can produce an ascent direction #V

On a double well, stand at $x_k = #fmt(xd, d: 1)$ where $f'' = #fmt(fppd, d: 2) < 0$ — the parabola *frowns*:

#align(center, lines(
  fn: (dw, qdw),
  domain: (-1.45, 1.45), samples: 140, markers: false,
  colors: (INK, RED), dashes: ("solid", "dashed"),
  labels: ([$f$ — two valleys, one hill], [Newton's model at $x_k$]),
  points: ((xd, fd, [$x_k$]), (xt, qt, [peak])),
  vlines: ((-1.0, none), (1.0, [the real minima]),),
  size: (105mm, 44mm), x-label: $x$,
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[a frowning parabola's stationary point is its *peak*: the "Newton step" here moves *left, toward the hilltop at* $x = 0$ — and iterating converges to that local maximum])

#pause
#alertbox[When $H$ is not positive definite, $-H^(-1) nabla f$ need not point downhill. Newton without a safety check is only trustworthy where the world is bowl-shaped.]

== Damping, line search, and trust regions

Common safeguards modify or restrict the raw Newton step:

- *Damping*: solve $(H + tau I) thin bold(delta) = -nabla f$. Eigenvalues shift by $tau$; the model becomes positive definite only when $tau > -lambda_"min"(H)$ (strictly, with margin).
#pause
- *Line search*: take the Newton *direction*, but test the length before committing.
#pause
- *Trust region*: only believe the model inside a radius; grow or shrink the radius by results.

#pause
#notebox[The same damping idea appears next in nonlinear least squares, with an implementation-specific trust-region rule for choosing the parameter.]

// ═══════════════════ 6 · nonlinear least squares ═══════════════════
= Sums of squares: nonlinear least squares

== Back to the sensor: the loss is a sum of squares

Our eight readings, and the model $hat(y)(t) = a thin e^(-b t)$ with $bold(theta) = (a, b)$:

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 6.5pt,
  table.header([$t_i$], [0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]),
  [$y_i$], [2.465], [1.762], [1.426], [0.928], [0.914], [0.632], [0.385], [0.352],
)

#pause
Per point, a *residual* — the model's miss; the loss adds the squares (L14: this is the Gaussian NLL):

$ r_i (bold(theta)) = a thin e^(-b t_i) - y_i, quad quad cal(L)(bold(theta)) = 1/2 sum_(i=1)^8 r_i (bold(theta))^2 = 1/2 norm(bold(r)(bold(theta)))^2 $

#pause
#notebox[The $1/2$ is cosmetics: it cancels the 2 that differentiation drops. This *shape* — squared residuals — is everywhere: calibration, GPS, camera bundle adjustment, regression.]

== The residual vector and its Jacobian

$bold(r): RR^2 -> RR^8$ — two parameters in, eight misses out. Its derivative is a *Jacobian* (L9): one row per residual, one column per parameter, $8 times 2$:

$ J_(i 1) = (partial r_i) / (partial a) = e^(-b t_i), quad quad J_(i 2) = (partial r_i) / (partial b) = -a thin t_i thin e^(-b t_i) $

#pause
At the bad first guess $bold(theta)_0 = (1.0, 0.2)$, three of the eight rows:

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*row*], [$t_i$], [$partial r_i slash partial a$], [$partial r_i slash partial b$]),
  [$i = 1$], [0.0], [1.000], [0.000],
  [$i = 2$], [0.5], [0.905], [$-0.452$],
  [$i = 8$], [3.5], [0.497], [$-1.738$],
)

#pause
Late-time rows barely feel $a$ but feel $b$ strongly — the Jacobian *is* the sensitivity table.

== The gradient in terms of the Jacobian

Chain rule on $cal(L) = 1/2 bold(r)^top bold(r)$, shapes doing the bookkeeping (L9's habit):

$ nabla cal(L) = underbrace(J^top, 2 times 8) underbrace(bold(r), 8 times 1) quad in RR^2 $

#pause
At $bold(theta)_0 = (1.0, 0.2)$, with the measured residuals: $nabla cal(L) = J^top bold(r) = (-2.894, thin 0.937)$.

#pause
- Gradient descent is ready to go: $bold(theta) arrow.l bold(theta) - eta thin J^top bold(r)$. L17 machinery, nothing new.
#pause
- But Newton wants $H$ — second derivatives of all eight $r_i$: expensive. *The sum-of-squares structure supplies the curvature far more cheaply* — that is Gauss-Newton, next.

// ═══════════════════ 7 · Gauss-Newton ═══════════════════
= Gauss-Newton: curvature almost for free

== The idea: linearize the residuals, not the loss

Replace each residual by its L9 linear approximation at the current $bold(theta)$:

$ bold(r)(bold(theta) + bold(delta)) approx bold(r) + J bold(delta) quad arrow.r.double quad cal(L)(bold(theta) + bold(delta)) approx 1/2 norm(bold(r) + J bold(delta))^2 $

#pause
Look at what the right side *is*: a *linear least-squares problem in $bold(delta)$* — the one problem Module 1 taught us to solve exactly (and L16 proved convex).

#pause
#result[A line inside a square is a parabola: linearizing $bold(r)$ hands us a quadratic model of $cal(L)$ — without ever touching a second derivative.]

== The Gauss-Newton step

Minimize $1/2 norm(bold(r) + J bold(delta))^2$ over $bold(delta)$: its gradient is $J^top (bold(r) + J bold(delta)) = bold(0)$ — the *normal equations* of the linearized fit:

#pause
#result[*Gauss-Newton:* solve the linear least-squares problem $min_bold(delta) norm(bold(r) + J bold(delta))^2$. If $J$ has full column rank, its normal equations are $(J^top J) bold(delta) = -J^top bold(r)$.]

#pause
Pattern-match against Newton's $H bold(delta) = -nabla f$:

- right side: $J^top bold(r) = nabla cal(L)$ ✓ (last slide) — same gradient,
#pause
- left side: $J^top J$ *standing where the Hessian stands*. Is it the Hessian? Almost —

== Relation to Newton's Hessian

Differentiate $nabla cal(L) = J^top bold(r)$ once more (product rule — each factor depends on $bold(theta)$):

$ H = J^top J + sum_(i=1)^n r_i thin nabla^2 r_i $

#pause
Gauss-Newton keeps the first term, *drops the second*. When is that fair?

- $r_i approx 0$ near a good fit — the dropped term is weighted by the *misses*;
#pause
- or $r_i$ nearly linear ($nabla^2 r_i approx 0$) — nothing to drop.

#pause
#result[$J^top J$ uses first derivatives only and is always PSD. It is positive definite, and the Gauss-Newton step unique, when $J$ has full column rank.]

== Numbers: the first step from $bold(theta)_0 = (1.0, 0.2)$

Assemble the 2×2 system from the real data (all eight rows summed — check any entry by hand):

$ J^top J = mat(4.403, -5.488; -5.488, 11.938), quad quad J^top bold(r) = vec(-2.894, 0.937) $

#pause
$ mat(4.403, -5.488; -5.488, 11.938) bold(delta) = vec(2.894, -0.937) quad arrow.r.double quad bold(delta) = vec(1.310, 0.524) quad arrow.r.double quad bold(theta)_1 = (2.310, thin 0.724) $

#pause
From $(1.0, 0.2)$ straight to $(2.31, 0.72)$ — one solve of a 2×2 system.

#pause
#notebox[$b$ overshoots the fitted value $0.563$ on the first step because the residual linearization is local. Recompute it and continue.]

== The run: five solves and the loss stops changing

The whole optimization, on the real data (this table is the figure script's actual output):

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*k*], [$a_k$], [$b_k$], [*loss $cal(L)(bold(theta)_k)$*]),
  [0], [1.000000], [0.200000], [1.69626671],
  [1], [2.309903], [#text(fill: RED)[0.723655]], [0.20392293],
  [2], [2.418643], [0.525822], [0.03006333],
  [3], [2.425973], [0.560925], [0.02159785],
  [4], [2.427402], [0.562773], [0.02158193],
  [5], [*2.427436*], [*0.562798*], [*0.02158193*],
)

#pause
$b$ overshoots and then settles as each residual linearization is recomputed. By $k = 5$ the loss has stopped changing at the displayed precision; this example alone does not establish a general quadratic rate for Gauss-Newton.

== The fit, evolving #V

#fig("/lecture18/figures/gn_evolution.svg", w: 98%)

#pause
The first solve moves close to the data; by iteration 2 the remaining error is comparable to the observation noise.

== Parameter trajectory in the $(a, b)$ plane #V

#fig("/lecture18/figures/gn_landscape.svg", w: 62%)

#pause
#align(center, text(size: 16pt, fill: MUTED)[gradient descent zigzags into the curved valley — and its $eta$ was hand-tuned *using $J^top J$'s eigenvalues*; 25% more $eta$ and it never converges. Gauss-Newton: two visible hops, three invisible ones.])

== Code: Gauss-Newton in eight lines

#codebox[```python
theta = np.array([1.0, 0.2])                 # (a, b) — the bad guess
for _ in range(6):
    a, b = theta
    r = a * np.exp(-b * t) - y               # residuals        (8,)
    J = np.stack([np.exp(-b * t),            # dr/da            (8, 2)
                  -a * t * np.exp(-b * t)], axis=1)
    delta = np.linalg.lstsq(J, -r, rcond=None)[0]   # avoids forming J^T J
    theta = theta + delta
# theta = [2.427437, 0.562798]
```]

#pause
Residuals, Jacobian, one small solve — no Hessian code anywhere, and no learning rate.

// ═══════════════════ 8 · Levenberg–Marquardt ═══════════════════
= Levenberg–Marquardt damping

== When the Gauss-Newton model is unreliable

Gauss-Newton can fail when its local residual model is inaccurate or poorly conditioned:

- Far from the fit, the linearization lies — our very first step overshot $b$ by 29%.
#pause
- If the data barely constrains a parameter, $J^top J$ is *nearly singular* (L6: tiny singular values) — the solve amplifies noise into a *wild* step.
#pause
- Nothing stops a step from *increasing* the loss — the update never checks it.

#pause
Section 5 already prescribed the fix: *damp* the system — add $lambda I$ before solving.

== The damping parameter $lambda$

#result[*Levenberg damping:* $(J^top J + lambda I) thin bold(delta) = -J^top bold(r)$. Practical LM implementations often use a scaled diagonal and choose $lambda$ through a trust-region calculation.]

#pause
The two limiting cases explain the effect of $lambda$:

- $lambda -> 0$: the damping vanishes — *pure Gauss-Newton*, full quadratic confidence.
#pause
- $lambda$ large: $lambda I$ dominates $J^top J$, so $bold(delta) approx -1/lambda J^top bold(r) = -1/lambda nabla cal(L)$ — a *small gradient-descent step*.

#pause
In this simplified system, one scalar interpolates between the Gauss-Newton step and a small negative-gradient step.

== How $lambda$ changes the step #V

All steps solved from the same $bold(theta)_0$ on the real problem — only $lambda$ moves:

#fig("/lecture18/figures/lm_dial.svg", w: 60%)

#pause
#align(center, text(size: 16pt, fill: MUTED)[as $lambda$ increases, the step shrinks and rotates from the Gauss-Newton step toward the $-nabla cal(L)$ direction])

== Adapt $lambda$ using the observed loss

A simple classroom controller adjusts $lambda$ after evaluating a trial step:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*trial step…*], [*action*], [*meaning*]),
  [lowered the loss], [accept; decrease $lambda$], [allow a step closer to Gauss-Newton],
  [raised the loss], [reject; increase $lambda$], [shrink and rotate the step toward the gradient],
)

#pause
A real LM implementation compares predicted and observed reduction and updates its trust parameter accordingly. Its convergence rate depends on the residuals, Jacobian rank, and local regularity.

#pause
#notebox[This is Section 5's trust-region idea compressed into a scalar: $lambda$ small = big trust region, $lambda$ large = tiny one.]

== Reproduce `curve_fit` on the sensor data

For this unconstrained problem, `curve_fit(..., method='lm')` calls SciPy's `leastsq` wrapper around MINPACK. With no `jac` supplied, MINPACK's `lmdif` estimates $J$ by forward differences (L10) and computes a scaled trust-region Levenberg–Marquardt step using QR-based linear algebra:

#codebox[```python
popt, pcov = curve_fit(model, t, y, p0=(1.0, 0.2))
# popt      = [2.427437, 0.562798]      <- curve_fit (LM)
# our GN k=5: [2.427436, 0.562798]      <- this lecture, 8 lines
```]

#pause
The hand-coded Gauss-Newton iteration and MINPACK reach the same fitted parameters to six decimals on this well-started example.

#pause
#result[`curve_fit(method='lm')` uses MINPACK's modified Levenberg–Marquardt algorithm; it is more sophisticated than the multiply/divide-$lambda$ teaching rule above.]

== Interactive: beyond gradient descent #I

#interbox(link-to: IA + "optimizers-beyond")[
  Compare Newton and BFGS with gradient descent on 2-D objectives; inspect the quadratic, saddle, and poorly initialized cases, then add damping.
]

#pause
Try the ill-conditioned bowl first: it is this lecture's $kappa = 50$ picture, live.

== Checkpoint: large Levenberg–Marquardt damping #Q

#mcq([In the simplified $(J^top J + lambda I)$ damping system, increasing $lambda$ makes the step approach…],
  [the pure Gauss-Newton step],
  [a tiny step along $-nabla cal(L)$],
  [the exact Newton step (with the true Hessian)],
  [a step of fixed length in a random descent direction],
)

== Answer: the step approaches scaled gradient descent #A

#mcq-answer("B", [a tiny step along $-nabla cal(L)$],
  [For large $lambda$, $(J^top J + lambda I) approx lambda I$, so $bold(delta) approx -1/lambda J^top bold(r) = -1/lambda nabla cal(L)$. The direction approaches the negative gradient and the length shrinks as $1 slash lambda$. A diagonally scaled LM implementation approaches a correspondingly scaled gradient direction.])

// ═══════════════════ 9 · summary & what's next ═══════════════════
= Summary & what's next

== Lecture 18 — summary

- *Newton* ⭐: minimize the L9 quadratic model → solve $H bold(delta) = -nabla f$.
- *Quadratics*: one exact-arithmetic Newton step reaches the minimizer from anywhere.
- *Local rate*: nonsingular, Lipschitz Hessian ⇒ the error squares — digits double.
- *Cost and failure modes*: $d^2$ memory, $O(d^3)$ solve; indefinite $H$ needs safeguards.
- *Least squares* $cal(L) = 1/2 norm(bold(r))^2$: $nabla cal(L) = J^top bold(r)$, $H = J^top J + sum_i r_i nabla^2 r_i$.
- *Gauss-Newton*: drop $sum_i r_i nabla^2 r_i$ — $(J^top J) bold(delta) = -J^top bold(r)$; five solves fit our sensor.
- *Levenberg–Marquardt*: damp with an adaptive $lambda$; `curve_fit` runs MINPACK's LM.

#pause
#notebox[*Read before L19* — MML §7.1.3 · Solomon, _Numerical Algorithms_, Ch. 9 (nonlinear least squares & Gauss-Newton). *T9*: Newton and Gauss-Newton by hand and in NumPy, on this lecture's data.]

== Practice problems

Try on paper; verify against `np.linalg.solve` and `curve_fit`.

+ Run one Newton step on $f(x) = x^2 - 4x + 3$ from $x_0 = 0$. Why is one step enough — and what does GD with $eta = 0.1$ leave undone after 10 steps?
+ On $f(x, y) = x^2 + x y + y^2$ from $(3, -1)$: solve $H bold(delta) = -nabla f$ by hand and verify you land at $(0, 0)$.
+ For $f(x) = x - ln x$, show the Newton update is $x_(k+1) = 2x_k - x_k^2$ and that $e_(k+1) = e_k^2$ exactly. How many steps from $x_0 = 0.5$ to 300 correct digits?
+ Run Newton on the double well $f(x) = x^4\/4 - x^2\/2$ from $x_0 = 0.3$ for three steps. Where is it converging, and what is the sign of $f''$ along the way?
+ For the model $hat(y) = a sin(b t)$: write $r_i$ and both columns of $J$, then the Gauss-Newton system at a given $(a, b)$.
+ In LM, show $bold(delta)(lambda) -> -1/lambda nabla cal(L)$ as $lambda -> infinity$. Which *two* properties of the step does $lambda$ therefore control simultaneously?

== ⭐⭐⭐ Why deep learning still runs first-order #OPT

If Newton is this good, why is every LLM ($d approx 10^9$) trained with Adam-flavored GD (L17 ⭐⭐⭐)?

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*per step*], [*GD/Adam*], [*Newton*]),
  [memory], [$O(d)$], [$O(d^2)$: $tilde 4 times 10^9$ GB],
  [arithmetic], [$O(d)$ past the gradient], [$O(d^3)$: $10^27$ FLOPs — *years*],
  [minibatches], [cheap unbiased gradients], [noisy curvature estimates],
  [indefinite curvature], [may escape strict saddles], [attracted to saddle points],
)

#pause
#notebox[Large-scale alternatives: *L-BFGS* (built from gradient history), Hessian-vector products $H bold(v)$ at near-gradient cost, structured approximations (K-FAC, Shampoo).]

#focus-slide[
  Newton doesn't step downhill — it teleports to the bottom of its local bowl.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Constrained Optimization: Lagrange Multipliers* — so far the search was free; real problems have fences.
]
