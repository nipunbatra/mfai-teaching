// Gradient Descent — Lecture 17 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture17/L17-gradient-descent.typ
//   typst compile --root . --input handout=true lecture17/L17-gradient-descent.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Gradient Descent],
  subtitle: [Trust the line, one small step at a time],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── in-deck numerics: every trajectory below is COMPUTED, not drawn ──────────
// GD on f(x) = (a/2) x² from x0 — the iterate sequence ((t, x_t), …)
#let gd-seq(a, eta, x0, steps, clip: none) = {
  let x = x0
  let pts = ((0, x),)
  for t in range(steps) {
    x = (1 - eta * a) * x
    let xr = if clip == none { x } else { calc.max(-clip, calc.min(clip, x)) }
    pts.push((t + 1, xr))
  }
  pts
}
// the same walk as (x_t, f(x_t)) dots on the bowl (for the four-panel figure)
#let gd-bowl(a, eta, x0, steps, clip: 2.7) = {
  let x = x0
  let pts = ((x, 0.5 * a * x * x),)
  for t in range(steps) {
    x = (1 - eta * a) * x
    let xc = calc.max(-clip, calc.min(clip, x))
    pts.push((xc, 0.5 * a * xc * xc))
  }
  pts
}
// loss_t sequence — exactly L1's loop (record the loss, then step)
#let gd-loss(a, eta, x0, steps) = {
  let x = x0
  let pts = ()
  for t in range(steps + 1) {
    pts.push((t, 0.5 * a * x * x))
    x = (1 - eta * a) * x
  }
  pts
}
// a sampled bowl curve (x, (a/2)x²) for the four-panel backdrops
#let bowl-curve(a, lo, hi) = range(0, 61).map(i => {
  let x = lo + (hi - lo) * i / 60
  (x, 0.5 * a * x * x)
})

#title-slide()

= Two quadratic objectives, one learning rate

== The same learning rate can converge or diverge #V

Gradient descent uses the same code and learning rate $eta = 0.8$ on two quadratic objectives:

#fig("/lecture1/figures/lr_two_problems.svg", w: 78%)

#pause
It converges slowly on one and diverges on the other. This lecture explains both outcomes with exact calculations.

== Why one learning rate behaves differently

Start by testing whether $eta = 0.8$ is simply too large or too small:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Possible explanation*], [*What it would predict*], [*Observation*]),
  [$eta$ is too *big*], [both problems overshoot and explode], [#text(fill: RED)[✗ A crawls]],
  [$eta$ is too *small*], [both problems crawl], [#text(fill: RED)[✗ B explodes]],
)

#pause
The same number cannot be simultaneously too big and too small…

#pause
#result[…unless "big" and "small" are measured *relative to something in the problem*. Naming that something is today's whole job.]

== Everything we need is already on the table

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*from*], [*what it gave us*]),
  [L7], [the local line: $f(x_0 + h) approx f(x_0) + f'(x_0) h$ — "L17 will quote today's line"],
  [L8], [the direction: $-nabla f$ is steepest descent (Cauchy–Schwarz) — we even ran a small GD],
  [L16], [the license: on convex bowls, local information is globally trustworthy],
)

#pause
- L8 settled *which way* to step. Nobody has yet said *how far*.

#pause
- Today: derive the algorithm, then analyze how $eta$ interacts with curvature.

== Learning outcomes

By the end of this lecture you will be able to:

+ *Derive* gradient descent from the first-order Taylor approximation developed in L7.
+ Implement the *four-line algorithm* and read $eta$ as a trust radius.
+ Predict *crawl / converge / oscillate / diverge* from $eta$ and the curvature — the wall at $eta = 2\/a$.
+ Run the *exact analysis on quadratics*: contraction factor $1 - eta a$, per-axis rates.
+ Compute a *condition number* $kappa$ and explain the zig-zag pathology it causes.
+ Explain why *feature scaling* speeds up training, and how *momentum* fights ravines.

= Deriving gradient descent from Taylor's approximation

== The first-order Taylor model #V

At $x = 1$ on $f(x) = x^2$, the derivative defines the local linear approximation.

#let ftr(x) = x * x
#let elltr(x) = 2 * x - 1
#align(center, lines(fn: (ftr, elltr), domain: (-1.5, 1.9), samples: 90,
  colors: (INK, ACC), dashes: ("solid", "dashed"), markers: false,
  labels: ([$f$], [the line]),
  points: ((1, 1), (0.6, 0.36), (-1.2, 1.44)),
  size: (96mm, 46mm), x-label: $x$,
))
#align(center, text(size: 16pt, fill: MUTED)[dots: start at $x = 1$ · a small step to $x = 0.6$ · a big step to $x = -1.2$])

#pause
A small step decreases the true function. A large step can increase it even though the linear approximation continues downward. The approximation is local.

== The approximation is accurate only for small steps

L7's local replacement, in today's many-variable clothes (L9): for a step $Delta$ from $theta$,

$ f(theta + Delta) approx f(theta) + nabla f(theta)^top Delta $

#pause
- The error of this replacement grows like $norm(Delta)^2$ (the second-order Taylor term) — *double* your step, the line lies *four times* harder.

#pause
- The linear model is useful only over a neighbourhood in which the neglected curvature term remains small.

#pause
#notebox[This is the spine of the lecture: *trust the linear model for one small step, then rebuild the line and repeat*. Everything else is bookkeeping about $r$.]

== Which direction? L8 already answered

Inside the trust radius, pick the step the *line* likes best:

$ min_(norm(Delta) <= r) space nabla f(theta)^top Delta $

#pause
L8 callback — by Cauchy–Schwarz the dot product is most negative when $Delta$ points *opposite* the gradient:

$ Delta^star = -r space (nabla f(theta)) / norm(nabla f(theta)) $

#pause
- Direction: $-nabla f$, dead against the gradient — *steepest descent*, exactly L8's compass.
#pause
- New today: the length $r$ — the part L8 left open.

== The update rule

Fold the radius and the gradient's length into one knob, $#text(fill: ACC)[$eta$] > 0$:

#result[$ theta_(t+1) = theta_t - #text(fill: ACC)[$eta$] thin nabla f(theta_t) $]

#pause
- $#text(fill: ACC)[$eta$]$ is the *learning rate*: it scales the gradient, so the step length is $eta norm(nabla f(theta_t))$.
#pause
- Rebuild the line at the new point, step again. That is the entire method.

#pause
#notebox[This is the application anticipated in L7: the update uses today's local linear approximation to choose the next point.]

== A sufficiently small step decreases the objective

For differentiable $f$, substitute $Delta = -eta nabla f$ into the first-order expansion:

$ f(theta - eta nabla f) = f(theta) - eta norm(nabla f(theta))^2 + o(eta) $

#pause
- If $nabla f(theta) != bold(0)$, the negative linear term dominates for sufficiently small positive $eta$, so the true objective decreases.

#pause
- Near a stationary point $nabla f -> 0$, so the step norm $eta norm(nabla f)$ shrinks.

#pause
#alertbox[The decrease above is a local first-order prediction. A step that is too large can increase the true objective, as in the previous figure.]

== The whole algorithm is four lines

#codebox[```python
theta = theta0                      # 1. start somewhere
for t in range(T):                  # 2. repeat
    g = grad(L, theta)              # 3. which way is uphill  (L8)
    theta = theta - lr * g          # 4. small trusted step downhill
```]

#pause
- Stop when $norm(g)$ is tiny, the loss plateaus, or the budget $T$ runs out. A small gradient alone does not distinguish a minimum from a saddle or plateau.
#pause
- This loop and its stochastic variants train many models, from linear regression to neural networks.

#pause
#result[One hyperparameter, no memory, only first derivatives. The rest of today: what that one hyperparameter does.]

== Numbers first: halving your way down

$f(x) = x^2$ (curvature $f'' = a = 2$), learning rate $eta = 0.25$, start $x_0 = 2$:

$ x_1 = 2 - 0.25 dot (2 dot 2) = 1, quad x_2 = 1 - 0.25 dot 2 = 0.5, quad x_3 = 0.25, quad dots $

#pause
- Every step lands exactly *halfway* to the minimum: $x_(t+1) = x_t - 0.25 dot 2 x_t = 0.5 thin x_t$.

#pause
- The distance to the minimum shrinks by the *same factor $0.5$ every step* — a clean geometric decay.

#pause
#notebox[The next section derives this per-step factor exactly.]

== On a round bowl, GD heads straight to the minimum #V

$f(x, y) = x^2 + y^2$ — same curvature in every direction:

#let bowl-round = ad.expr("x^2 + y^2", ("x", "y"))
#align(center, contour(ad.fn2(bowl-round), xlim: (-2.4, 2.4), ylim: (-1.8, 1.8),
  samples: 50, levels: 8, size: (76mm, 57mm), color: TEAL,
  paths: (gd(ad.grad-fn(bowl-round), (-1.9, 1.4), lr: 0.12, steps: 12),),
  marks: ((0, 0, [min], RED),), x-label: $x$, y-label: $y$,
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[round contours ⟹ $-nabla f$ points *straight at the minimum*; steps shrink as $norm(nabla f) -> 0$])

#pause
This round bowl is the well-conditioned reference case.

== What $eta$ really is

#pause
- *Too small*: the guaranteed first-order decrease per step is much smaller than it could be.

#pause
- *Too large*: the neglected curvature dominates, and the true $f$ may rise.

#pause
- *Just right*: the largest step the line can still honor.

#pause
#notebox[How large is that, exactly? It must depend on how fast the line goes stale — the *curvature*. That is a number we can compute. Next section: compute it.]

= Choosing the learning rate

== Four behaviours as $eta$ changes #V

GD on $f(x) = 1/2 x^2$ (curvature $a = 1$), $x_0 = 2.4$, twelve steps — four choices of $eta$:

#let lr-cfgs = (
  ([$eta = 0.05$ · crawls], 0.05, TEAL),
  ([$eta = 0.8$ · converges], 0.8, GREEN),
  ([$eta = 1.7$ · oscillates in], 1.7, ACC),
  ([$eta = 2.15$ · diverges], 2.15, RED),
)
#align(center, stack(dir: ltr, spacing: 4mm,
  ..lr-cfgs.map(cf => lines((bowl-curve(1.0, -2.8, 2.8), gd-bowl(1.0, cf.at(1), 2.4, 12)),
    colors: (INK, cf.at(2)), markers: (false, true), y-ticks: false,
    size: (37mm, 30mm), title: cf.at(0))),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[every dot is a real iterate, computed by this slide — crawl · converge · overshoot-but-converge · explode])

== The same four walks, in time #V

Plot the iterate $x_t$ itself against the step count:

#align(center, lines(
  (gd-seq(1.0, 0.05, 2.4, 12), gd-seq(1.0, 0.8, 2.4, 12),
   gd-seq(1.0, 1.7, 2.4, 12), gd-seq(1.0, 2.15, 2.4, 12, clip: 5.2)),
  colors: (TEAL, GREEN, ACC, RED),
  hlines: ((0,),),
  x-label: [step], y-label: [iterate], size: (102mm, 46mm),
))
#align(center, text(size: 15pt, fill: MUTED)[colors as before: #text(fill: TEAL)[$eta = 0.05$] · #text(fill: GREEN)[$eta = 0.8$] · #text(fill: ACC)[$eta = 1.7$] · #text(fill: RED)[$eta = 2.15$]])

#pause
- #text(fill: TEAL)[crawl] and #text(fill: GREEN)[converge] stay on one side; #text(fill: ACC)[overshoot] *crosses the minimum every step* and still converges; #text(fill: RED)[diverge] crosses and *grows* (clipped at $plus.minus 5.2$).

== The wall sits at exactly $eta = 2 slash a$

On a bowl of curvature $a$, the step from $x$ has length $eta a |x|$ — against a distance-to-minimum of $|x|$:

#pause
- $eta a < 1$: land *short* of the minimum — same side, monotone approach.
#pause
- $1 < eta a < 2$: land *past* the minimum but *closer* than you started — a damped oscillation.
#pause
- $eta a > 2$: land past it and *farther out* — each step amplifies the error. Divergence.

#pause
#result[The speed limit is $#text(fill: ACC)[$eta < 2 slash a$]$ — two divided by the curvature.]

#pause
#notebox[If $f$ has an $L$-Lipschitz gradient, then $0 < eta < 2\/L$ guarantees a decrease at every non-stationary iterate. Convergence needs additional conditions such as a lower-bounded objective; convexity gives stronger conclusions.]

== What you'll actually see: the training curve #V

In $10^6$ dimensions you can't watch $x_t$ — only the *loss*. Same four runs, log scale (L2 taught us to love logs):

#align(center, lines(
  (gd-loss(1.0, 0.05, 2.4, 12), gd-loss(1.0, 0.8, 2.4, 12),
   gd-loss(1.0, 1.7, 2.4, 12), gd-loss(1.0, 2.15, 2.4, 12)),
  colors: (TEAL, GREEN, ACC, RED), log-y: true, markers: true,
  x-label: [step], y-label: [loss], size: (98mm, 46mm),
))
#align(center, text(size: 15pt, fill: MUTED)[#text(fill: TEAL)[crawl] · #text(fill: GREEN)[fast] · #text(fill: ACC)[oscillating] · #text(fill: RED)[diverging]])

#pause
#align(center, text(size: 16pt, fill: MUTED)[geometric decay = *straight line* on a log axis; the slope is the per-step factor — divergence slopes *up*])

#pause
You will diagnose learning rates from curves like these for the rest of your career.

== Interactive: vary the learning rate #I

#interbox(link-to: IA + "optimizer-race")[
  Drop a start point on a 2-D loss surface and sweep the learning rate live: watch the same four behaviours — crawl, converge, overshoot, explode — appear as you cross the $2\/a$ wall.
]

#pause
Find the wall experimentally on the bowl, then check it against the curvature readout.

== Checkpoint: the knife edge #Q

#mcq(
  [Run GD on $f(x) = (x - 3)^2$ — curvature $a = f'' = 2$ — with $eta = 1$, from $x_0 = 0$. What happens?],
  [Converges to $x = 3$ in one step],
  [Diverges to infinity],
  [Bounces between $0$ and $6$ forever],
  [Converges slowly, like $0.5^t$],
)

== Answer: the knife edge #A

#mcq-answer([C], [bounces between $0$ and $6$ forever],
  [The update is $x arrow.l x - 2(x - 3) = 6 - x$: each step *reflects* through the minimum. $eta = 1$ is exactly the wall $2\/a = 1$ — the error factor is $-1$: never shrinks, never grows.])

#pause
$ x: quad 0 arrow.r 6 arrow.r 0 arrow.r 6 arrow.r dots.c quad #text(fill: MUTED)[— a perpetual, loss-preserving cycle around $x^* = 3$] $

= Exact analysis on quadratics

== Why quadratics give a local model

Near a minimum $theta^*$ of a smooth loss, use the Taylor model from L9:

$ f(theta) approx f(theta^*) + 1/2 space f''(theta^*) (theta - theta^*)^2 $

#pause
- Near a non-degenerate smooth local minimum, the second-order Taylor model is approximately quadratic, with curvature $a = f''(theta^*)$ in one dimension.

#pause
- And on a quadratic we don't have to approximate anything: GD can be solved *exactly*, in one line.

#pause
#result[Exact answers on positive-definite quadratics describe local gradient-descent behavior near a smooth strict minimum.]

== ⭐ One line of algebra #D

Take $f(x) = a/2 space x^2$ with curvature $a > 0$, so $f'(x) = a x$. Apply the update:

#pause
$ x_(t+1) = x_t - eta space a x_t = underbrace((1 - eta a), "one number") space x_t $

#pause
Unroll it — the whole future in closed form:

$ x_t = (1 - eta a)^t space x_0 $

#pause
#result[GD on a quadratic is *repeated multiplication by the constant* $1 - eta a$. Nothing else happens.]

#pause
Check against the halving demo: $a = 2$, $eta = 0.25$ ⟹ $1 - eta a = 0.5$. ✓

== ⭐ The contraction factor decides everything #D

Give $c = 1 - #text(fill: ACC)[$eta$] a$ its name: the *contraction factor*. Error after $t$ steps: $|c|^t |x_0|$; loss shrinks by $c^2$ per step.

#pause
#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*$eta$ range*], [*factor $c$*], [*behavior*]),
  [$0 < eta < 1\/a$], [$0 < c < 1$], [monotone convergence (crawl if $c approx 1$)],
  [$eta = 1\/a$], [$c = 0$], [#text(fill: GREEN)[the minimum in *one step*]],
  [$1\/a < eta < 2\/a$], [$-1 < c < 0$], [overshoots each step, still converges],
  [$eta = 2\/a$], [$c = -1$], [bounces forever (the knife edge)],
  [$eta > 2\/a$], [$c < -1$], [#text(fill: RED)[diverges]],
)

#pause
#result[Convergence $arrow.l.r.double$ $|1 - eta a| < 1$ $arrow.l.r.double$ $0 < eta < 2\/a$. The best $eta$ is $1\/a$ — *one over the curvature*.]

== The factor, as a picture #V

$|1 - eta a|$ against $eta$ (drawn for $a = 1$):

#align(center, lines(fn: e => calc.abs(1 - e), domain: (0, 2.5), samples: 120,
  colors: (INK,), markers: false,
  hlines: ((1, none, RED),),
  vlines: ((1.0, [$eta^* = 1\/a$], GREEN), (2.0, [the wall $2\/a$], RED)),
  annotations: ((0.5, 0.14, [monotone]), (1.5, 0.14, [oscillating]), (0.5, 1.28, text(fill: RED)[divergence])),
  x-label: $eta$, y-label: [$|1 - eta a|$], size: (96mm, 46mm),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[a V with its tip at $eta = 1\/a$: undershoot on the left branch, overshoot on the right, cross $|c| = 1$ and the iterates diverge])

== Apply the contraction factor to both objectives #D

The opening two objectives were $f(x) = a/2 x^2$ with different curvatures and the same $eta = 0.8$:

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [*problem A* · $a = 0.02$], [*problem B* · $a = 3.2$]),
  [factor $c = 1 - 0.8 a$], [$0.984$], [#text(fill: RED)[$-1.56$]],
  [$|c|$ vs $1$], [just below — barely contracts], [above — *expands*],
  [after 30 steps], [$0.984^30 = 0.62$ of the error left #h(2pt) ⟹ #h(2pt) loss $times 0.38$], [error $times 1.56^30$; loss $times 2.43^30 approx 4 times 10^11$],
)

#pause
- L1's figure annotated problem A with "still 38% of the starting loss" — that number is $0.984^60$. *Predicted by one line of algebra.*

#pause
- For problem B, $eta = 0.8$ sits beyond its wall $2\/3.2 = 0.625$, so divergence was guaranteed.

== Compute both trajectories #V

The L1 figure again — except now *this slide computes it* from $x_(t+1) = (1 - eta a) x_t$:

#align(center, stack(dir: ltr, spacing: 9mm,
  lines((gd-loss(0.02, 0.8, 1.0, 30),), colors: (TEAL,), log-y: true,
    size: (62mm, 42mm), x-label: [step], y-label: [loss],
    title: [A · $a = 0.02$ · $c = 0.984$]),
  lines((gd-loss(3.2, 0.8, 1.0, 30),), colors: (RED,), log-y: true,
    size: (62mm, 42mm), x-label: [step],
    title: [B · $a = 3.2$ · $c = -1.56$]),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[straight lines on the log axis — geometric decay and geometric explosion, slopes $2 log |c|$])

#pause
#result[The same $eta$ is small relative to curvature $0.02$ and too large relative to curvature $3.2$.]

== ⭐ One learning rate must serve every eigendirection #D

The deeper trouble: one problem can contain *both curvatures at once*. Take the diagonal bowl

$ f(x, y) = 1/2 (a x^2 + b y^2), quad nabla f = (a x, thin b y), quad a "flat," b "steep" $

#pause
The update never mixes the coordinates — GD *decouples* into two 1-D stories with one shared $eta$:

$ x_t = (1 - eta a)^t x_0, quad quad y_t = (1 - eta b)^t y_0 $

#pause
- Stability is hostage to the *steep* axis: need $eta < 2\/b$.
#pause
- Speed is hostage to the *flat* axis: its factor $1 - eta a$ then sits near $1$.

#pause
#notebox[L5 callback: Hessian eigenvectors rotate any bowl into this diagonal form — $a$ and $b$ are its extreme eigenvalues, so the diagonal case is general.]

== Watch both factors at once #V

$f(x, y) = 1/2 (x^2 + 9 y^2)$, $eta = 0.2$: per-axis factors $1 - 0.2 dot 1 = #text(fill: TEAL)[$0.8$]$ and $1 - 0.2 dot 9 = #text(fill: RED)[$-0.8$]$.

#let bowl-9 = ad.expr("0.5*x^2 + 4.5*y^2", ("x", "y"))
#align(center, contour(ad.fn2(bowl-9), xlim: (-2.6, 2.6), ylim: (-1.1, 1.1),
  samples: 56, levels: 9, size: (95mm, 40mm), color: TEAL,
  paths: (gd(ad.grad-fn(bowl-9), (-2.2, 0.8), lr: 0.2, steps: 16),),
  marks: ((0, 0, [min], RED),), x-label: $x$, y-label: $y$,
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[$y$ flips sign every step (factor $-0.8$) while $x$ shrinks steadily (factor $+0.8$) — both visible in one real trajectory])

== No $eta$ serves two curvatures

Try to pick one $eta$ for A's and B's curvatures *simultaneously* ($a = 0.02$, $b = 3.2$):

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*$eta$*], [*flat axis ($a = 0.02$)*], [*steep axis ($b = 3.2$)*]),
  [$0.8$ (L1's)], [$c = 0.984$ — crawls], [#text(fill: RED)[$c = -1.56$ — explodes]],
  [$0.625 = 2\/b$], [$c = 0.9875$ — crawls], [#text(fill: RED)[$c = -1$ — bounces forever]],
  [$0.5$], [$c = 0.99$ — crawls *worse*], [$c = -0.6$ — fine],
  [$0.3125 = 1\/b$], [$c = 0.99375$ — crawls worse still], [#text(fill: GREEN)[$c = 0$ — one step!]],
  [$50 = 1\/a$], [#text(fill: GREEN)[$c = 0$ — one step!]], [#text(fill: RED)[$c = -159$ — catastrophic]],
)

#pause
#result[Every row sacrifices an axis. When curvatures are far apart, *no learning rate is good* — that situation has a name.]

== The condition number controls the convergence rate #D

$ kappa = a_"max" / a_"min" quad #text(fill: MUTED)[$0 < a_"min" <= a_"max"$ — for the opening SPD quadratic: $3.2 \/ 0.02 = 160$] $

#pause
Best compromise: $eta^* = 2\/(a_"max" + a_"min")$ balances the two factors at

$ |c| = (kappa - 1) / (kappa + 1) approx 1 - 2/kappa $

#pause
#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([$kappa$], [$1$], [$10$], [#text(fill: ACC)[$160$]], [$10^4$]),
  [best factor], [$0$], [$0.818$], [#text(fill: ACC)[$0.9876$]], [$0.9998$],
  [steps to cut error $100 times$], [$1$], [$approx 23$], [#text(fill: ACC)[$approx 369$]], [$approx 23#h(0pt),000$],
)

#pause
#result[For this quadratic family, the number of gradient-descent steps grows in proportion to the condition number $kappa$.]

= The ill-conditioned bowl

== Both curvatures in one problem #V

Put the opening example's two curvatures into one objective: $f(x, y) = 1/2 (0.02 x^2 + 3.2 y^2)$, $kappa = 160$, and run GD with a safe $eta = 0.55$:

#let bowl-m2 = ad.expr("0.01*x^2 + 1.6*y^2", ("x", "y"))
#align(center, contour(ad.fn2(bowl-m2), xlim: (-2.6, 2.6), ylim: (-0.9, 0.9),
  samples: 60, levels: 9, size: (104mm, 36mm), color: TEAL,
  paths: (gd(ad.grad-fn(bowl-m2), (-2.3, 0.75), lr: 0.55, steps: 36),),
  marks: ((0, 0, [min], RED),), x-label: $x$, y-label: $y$,
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[the infamous *zig-zag*: oscillate across the ravine (factor $-0.76$), crawl along it (factor $0.989$)])

#pause
#alertbox[In a local quadratic model, Hessian eigenvalues play the role of directional curvatures. A wide positive eigenvalue range produces the same slow/oscillatory tradeoff; zero or negative eigenvalues require separate analysis.]

== Why zig-zag? L8 told us

#pause
- L8 ⭐: every GD step is *perpendicular to the contour* it starts on.

#pause
- In a ravine, contours are stretched ellipses — perpendicular-to-the-contour points *across* the valley, hardly *along* it.

#pause
- The steep axis pins $eta$ below its wall $2\/b$, so the along-valley axis moves at factor $1 - eta a approx 1 - 2\/kappa$: a crawl.

#pause
#result[Zig-zag is not a bug in the code — it is steepest descent doing exactly what "steepest" means, on a bowl where *steepest ≠ toward the minimum*.]

== Where do squashed bowls come from? Your features

Fit a price from two features using the loss $norm(X bold(w) - bold(y))^2/(2n)$. Its Hessian is $X^top X/n$; the diagonal entry for weight $w_j$ is $"mean"(x_j^2)$:

#pause
#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*feature*], [*values*], [*curvature $"mean"(x_j^2)$*]),
  [house area (m²)], [$80, 120, 160$], [$46400\/3 approx 15#h(0pt),467$],
  [rooms], [$2, 3, 4$], [$29\/3 approx 9.67$],
)

$ "diagonal scale ratio" = (46400\/3) / (29\/3) = 1600 $

#pause
#result[Units can create a large coordinate-scale mismatch. Here there is a second problem: area $= 40 times$ rooms, so the columns are perfectly dependent and the true Hessian condition number is infinite.]

== The fix costs one line: standardize

#codebox[```python
X = (X - X.mean(axis=0)) / X.std(axis=0)   # each column: mean 0, std 1
```]

#pause
- Every column now has $"mean"(x_j^2) = 1$, removing conditioning caused purely by unequal units. Correlated columns can still make $X^top X$ ill-conditioned, so standardization does *not* guarantee $kappa approx 1$.
#pause
- In this example the two standardized columns remain identical, so the Hessian is still singular. Standardization fixes scale, not collinearity.

#pause
- Standardization often improves gradient-based optimization by rescaling coordinate directions. It is a simple diagonal preconditioner.

#pause
#notebox[Deep networks cannot pre-scale all internal activations by hand; BatchNorm and LayerNorm provide related internal normalization mechanisms. Their optimization effects are studied in ES 667.]

== Checkpoint: units and curvature #Q

#mcq(
  [You re-measure the "area" feature in *centimeters²* instead of m² — every value grows $times 10^4$. In the least-squares loss, the curvature along that weight's axis becomes…],
  [unchanged — units can't matter],
  [$times 10^4$],
  [$times 10^8$],
  [$div 10^4$],
)

== Answer: units and curvature #A

#mcq-answer([C], [$times 10^8$],
  [Curvature along $w_j$ is $"mean"(x_j^2)$ — the feature scale enters *squared*: $(10^4)^2 = 10^8$. One careless unit choice, eight orders of magnitude of conditioning damage (and the safe $eta$ shrinks by $10^8$ to match).])

#pause
#notebox[Same lesson from the other side: the *optimum weight* shrinks by $10^4$ — the minimum didn't move in "price space", but the bowl got squashed $10^8$-fold along that axis.]

= Momentum: memory against the ravine

== A marble doesn't zig-zag

Roll a heavy marble down the ravine instead of teleporting a point:

#pause
- The gradient acts as a *force*; the marble accumulates *velocity*.

#pause
- Across the valley the force flips direction every instant — the pushes *cancel* in the velocity.

#pause
- Along the valley the force always agrees — the pushes *add up*, and the marble accelerates.

#pause
#result[Same information ($nabla f$), one new state variable (velocity) — and the ravine's two axes get *opposite* treatment. This is the *heavy-ball method*.]

== The update rule: two lines now

$ v_(t+1) = #text(fill: ACC)[$beta$] thin v_t + nabla f(theta_t), quad quad theta_(t+1) = theta_t - eta thin v_(t+1) $

#pause
#codebox[```python
v = 0
for t in range(T):
    g = grad(L, theta)
    v = beta * v + g            # remember where we've been going
    theta = theta - lr * v      # step along the memory, not the gradient
```]

#pause
- $#text(fill: ACC)[$beta$] in [0, 1)$ is the *momentum coefficient* — how much velocity survives each step. Default: $0.9$.

== Unroll it: a moving average of gradients

Feed the recursion into itself:

$ v_t = g_t + beta g_(t-1) + beta^2 g_(t-2) + beta^3 g_(t-3) + dots.c $

#pause
- A *geometrically-weighted memory* of every gradient so far: recent ones loud, old ones fading by $beta$ each step.

#pause
- GD is the $beta = 0$ special case: memory of exactly one gradient.

#pause
#notebox[You have met this object: it is a geometric series in disguise — the same $1 + beta + beta^2 + dots.c = 1\/(1 - beta)$ that priced L2's float gaps and L15's discounted pseudo-counts.]

== Numbers: cancel vs accumulate

$beta = 0.9$. Compare the two axes of the ravine, in steady state:

#pause
*Along the valley* — gradients keep agreeing, say $g = -0.4$ every step:

$ v arrow.r -0.4 (1 + beta + beta^2 + dots.c) = (-0.4)/(1 - 0.9) = -4 quad #text(fill: GREEN)[— amplified $times 10$] $

#pause
*Across the valley* — gradients alternate $+2, -2, +2, dots$:

$ v arrow.r 2 (1 - beta + beta^2 - dots.c) = 2/(1 + 0.9) approx 1.05 quad #text(fill: TEAL)[— damped to half] $

#pause
#result[Momentum is a *direction filter*: consistent signals gain $1\/(1 - beta)$, flip-flopping signals shrink to $1\/(1 + beta)$.]

== How much memory? $1 slash (1 - beta)$ steps

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*$beta$*], [*effective memory*], [*character*]),
  [$0.5$], [$approx 2$ steps], [barely more than GD],
  [$0.9$], [$approx 10$ steps], [the workhorse default],
  [$0.99$], [$approx 100$ steps], [a supertanker — smooth, slow to turn],
)

#pause
- Along consistent directions the *effective* step is $approx eta\/(1 - beta)$ — budget $eta$ accordingly (that's why the momentum run below uses a smaller $eta$).

#pause
#alertbox[Too much memory overshoots: the marble sails past the valley's end and *rings* back and forth. If the loss oscillates on a slow timescale, suspect $beta$, not $eta$.]

== The ravine, rerun #V

Same bowl ($kappa = 160$), same start, same 36 steps:

#let rav-fig(t, p) = contour(ad.fn2(bowl-m2), xlim: (-2.6, 2.6), ylim: (-0.9, 0.9),
  samples: 60, levels: 9, size: (74mm, 26mm), color: TEAL,
  paths: (p,), marks: ((0, 0, [min], RED),), title: t)
#align(center, stack(dir: ltr, spacing: 8mm,
  rav-fig([plain GD · $eta = 0.55$],
    gd(ad.grad-fn(bowl-m2), (-2.3, 0.75), lr: 0.55, steps: 36)),
  rav-fig([momentum · $eta = 0.21$, $beta = 0.9$],
    momentum(ad.grad-fn(bowl-m2), (-2.3, 0.75), lr: 0.21, beta: 0.9, steps: 36)),
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[momentum rings across the valley at first, but the along-valley coordinate — GD's crawl — actually *arrives*])

#pause
#result[Momentum doesn't shrink $kappa$ — it changes how much $kappa$ *hurts*.]

== Interactive: the momentum map #I

#interbox(link-to: "https://distill.pub/2017/momentum/")[
  Distill, *"Why Momentum Really Works"* — this week's assigned reading, and the most beautiful interactive article in optimization. Drag $eta$ and $beta$ across the 2-D stability map; watch the ravine's per-axis rates respond exactly as today's algebra predicts.
]

#pause
#notebox[As you read, test yourself: the article's "effective step size" $eta\/(1 - beta)$, its condition-number plots, its ringing regime — every one of them appeared on a slide today.]

= Summary & what's next

== The whole lecture in one table

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*symptom (loss curve)*], [*diagnosis*], [*medicine*]),
  [barely-tilted straight decay], [$eta a_"min" << 1$: crawling], [raise $eta$ toward the wall],
  [loss rises geometrically], [$eta > 2\/a_"max"$: past the wall], [cut $eta$ (the classic $div 10$)],
  [fast drop, then endless crawl], [$kappa >> 1$: ravine], [*standardize features*; add *momentum*],
  [slow large-period ringing], [$beta$ too high: supertanker], [lower $beta$ before touching $eta$],
)

#pause
#result[Every row follows from $x_(t+1) = (1 - eta a) x_t$ in a different region of $(eta, a, kappa)$ space.]

== Lecture 17 — summary

- *GD derived* ⭐: one step on L7's local line — $theta_(t+1) = theta_t - eta nabla f$; direction from L8.
- *Learning-rate regimes*: crawl / converge / oscillate / diverge; the wall is $eta = 2\/a$.
- *Quadratic local model* ⭐: iterates are $(1 - eta a)^t x_0$; per-axis factors on positive-definite bowls; best 1-D rate at $eta = 1\/a$.
- *Conditioning*: $kappa = a_"max"\/a_"min"$; best factor $(kappa - 1)\/(kappa + 1)$; steps scale with $kappa$.
- *Preconditioning*: feature scale enters the Hessian squared; standardizing fixes scale only.
- *Momentum*: $v = beta v + g$; consistent directions $times 1\/(1 - beta)$, flip-flops damped — ravines tamed, not cured.

#pause
#notebox[*Read before L18*: MML §7.1 (gradient descent) · Distill, *Why Momentum Really Works* (assigned). *T8* this week: GD convergence on quadratics, condition numbers, and the optimizer race.]

== Practice problems

Try on paper; verify in the T8 notebook.

+ For $f(x) = 2x^2$ with $eta = 0.4$: the contraction factor and first three iterates from $x_0 = 1$. Converging?
+ On $f(x) = 5/2 x^2$: which $eta$ converge, which lands in one step, and what happens at exactly $eta = 0.4$?
+ On $f(x, y) = 1/2 (x^2 + 16 y^2)$ with the best single $eta$: per-axis factors, and the steps to shrink the slow-axis error $1000 times$.
+ Uncorrelated features with standard deviations $1$ and $50$ (means $0$): estimate $kappa$ before and after standardization. What if they are nearly collinear?
+ Momentum with $beta = 0.95$: the effective memory length, and the limiting $|v|$ under alternating gradients $plus.minus g$.
+ Reproduce the opening pair in NumPy ($a = 0.02, 3.2$, $eta = 0.8$, 30 steps); find the largest $eta$ that converges on both.

== ⭐⭐⭐ SGD exists #OPT

A finite-sum loss is $cal(L)(theta) = 1/n sum_i ell_i(theta)$. If index $i$ is sampled uniformly, then $EE_i[nabla ell_i(theta)] = nabla cal(L)(theta)$. One-example SGD uses this unbiased but noisy estimate and usually costs much less per update than a full-data gradient.

#let bowl-sgd = ad.expr("x^2 + y^2", ("x", "y"))
#align(center, contour(ad.fn2(bowl-sgd), xlim: (-2.4, 2.4), ylim: (-1.6, 1.6),
  samples: 46, levels: 8, size: (66mm, 44mm), color: TEAL,
  paths: (sgd(ad.grad-fn(bowl-sgd), (-1.9, 1.2), lr: 0.11, noise: 1.6, seed: 5, steps: 34),),
  marks: ((0, 0, [min], RED),),
))

#pause
#notebox[Minibatches, learning-rate schedules, and the adaptive per-coordinate rates of RMSProp and Adam belong to ES 335 and ES 667. The contraction-factor analysis here is their quadratic baseline.]

#focus-slide[
  The learning rate is how far you trust a linear approximation.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Newton & Gauss–Newton* — what if you trusted a *quadratic* model instead?
]
