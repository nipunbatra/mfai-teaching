// Jacobian, Hessian & Multivariate Taylor — Lecture 9 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture9/L9-jacobian-hessian.typ
//   typst compile --root . --input handout=true lecture9/L9-jacobian-hessian.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Jacobian, Hessian & Multivariate Taylor],
  subtitle: [When derivatives become matrices],
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
#let v2(v, d: 4) = $vec(#fmt(v.at(0), d: d), #fmt(v.at(1), d: d))$

// the lecture's recurring matrices
#let A1 = ((2.0, 1.0), (1.0, 2.0))     // the bowl — L5's matrix, and H of x² + xy + y²
#let A2 = ((2.0, 0.0), (0.0, -2.0))    // the saddle
#let A3 = ((2.0, 0.0), (0.0, 0.0))     // the trough
#let (lam1, _) = la.eig-sym(A1)
#let (lam2, _) = la.eig-sym(A2)
#let (lam3, _) = la.eig-sym(A3)

// quadratic form q(x) = ½ xᵀAx as a plottable function (A symmetric)
#let qform(A) = (x, y) => 0.5 * (A.at(0).at(0)*x*x + 2.0*A.at(0).at(1)*x*y + A.at(1).at(1)*y*y)

// live autodiff cross-examination of the ⭐ rule: ∇(xᵀA₁x) at (1,1)
#let qf1 = ad.expr("2*x*x + 2*x*y + 2*y*y", ("x", "y"))   // xᵀA₁x for A₁ = [2 1; 1 2]
#let gcheck = ad.grad(qf1, (1.0, 1.0))                     // should equal 2·A₁·(1,1) = (6, 6)
#let gclosed = la.scale(la.matvec(A1, (1.0, 1.0)), 2.0)

// ── the derivative zoo — one orienting table, revisited through the deck ──
// stage 1: everything above today greyed-in, today's rows still "?"
// stage 2: the Jacobian row lands   ·   stage 3: the Hessian row lands too
#let zoo(stage) = {
  let J-row = if stage >= 2 {
    ([$f: RR^n -> RR^m$], text(fill: ACC, weight: 600)[$J(bold(x))$], [matrix, $m times n$], [the local linear map (*today*)])
  } else {
    ([$f: RR^n -> RR^m$], text(fill: MUTED)[*?*], text(fill: MUTED)[*?*], text(fill: MUTED)[today])
  }
  let H-row = if stage >= 3 {
    ([$f: RR^n -> RR$, order 2], text(fill: ACC, weight: 600)[$H(bold(x))$], [matrix, $n times n$, symmetric], [curvature, in every direction (*today*)])
  } else {
    ([$f: RR^n -> RR$, order 2], text(fill: MUTED)[*?*], text(fill: MUTED)[*?*], text(fill: MUTED)[today])
  }
  table(
    columns: (auto, auto, auto, 1fr),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 8pt,
    table.header([*function*], [*derivative*], [*shape*], [*what it is*]),
    [$f: RR -> RR$], [$f'(x)$], [a number], [slope of the local line (L7)],
    [$f: RR^n -> RR$], [$nabla f(bold(x))$], [vector, $n$], [all partials; points straight uphill (L8)],
    ..J-row,
    ..H-row,
  )
}

#title-slide()

// ═══════════════════ 1 · the hook ═══════════════════
= The hook: a derivative with a million inputs

== The function AI actually trains #V

A neural network with its loss attached is *one function*: parameters in, a single number out.

#align(center, scale(80%, reflow: true, mlp-diagram((4, 5, 5, 1),
  labels: ([$theta in RR^d$], [layer], [layer], [$cal(L) in RR$]))))

#pause
For real models $d approx 10^6$ to $10^12$. Training (L17) needs this function's *derivative*.

#pause
*But what single object is "the derivative" of $RR^(10^6) -> RR$?* And between the layers sit maps $RR^n -> RR^m$ — vectors in, *vectors out*. L7's one number cannot be the answer.

== Today's promise — and today's safety rail

The derivative grows up in two steps:

- *vector in, vector out* $arrow.r$ the derivative is a matrix: the *Jacobian*;
#pause
- *curvature* of a surface $arrow.r$ also a matrix: the *Hessian* — and its eigenvalue signs (L5!) say whether you're in a bowl, on a saddle, or in a trough.

#pause
#notebox[Fair warning: this is the *heaviest-notation lecture of the course* — so we run it with a safety valve. Every idea happens on $RR^2 -> RR^2$, small enough to check by hand. The general case is always the *same picture, more rows*.]

#pause
By the end, "Jacobian" and "Hessian" will be *pictures*, not notation: a parallelogram, and a bowl.

== The derivative zoo — today's map #V

One table to navigate the whole lecture. Two rows you own; two we fill today:

#zoo(1)

#pause
Same idea every row: *the derivative is the best local linear story* — only its container grows.

== Learning outcomes

By the end of this lecture you will be able to:

+ Compute a 2×2 *Jacobian*, read it as a local linear map, and read $abs(det J)$ as area scaling.
+ Assemble a *Hessian* and explain entry $(i, j)$ as "how slope $i$ responds to nudge $j$".
+ Classify $bold(x)^top A bold(x)$ as bowl / saddle / trough from *eigenvalue signs*, with 2×2 hand tests.
+ Derive ⭐ $nabla(bold(x)^top A bold(x)) = (A + A^top) bold(x)$.
+ Write the *second-order Taylor expansion* — the formula every optimizer lives on.
+ Keep shapes straight: gradient-as-column, $J$ is $m times n$, chain rule = matrix product.

// ═══════════════════ 2 · the Jacobian ═══════════════════
= Vector in, vector out: the Jacobian

== A function that moves points #V

$Phi: RR^2 -> RR^2$ eats a point, returns a point. Feed it a whole grid and watch space bend:

#fig("/lecture9/figures/warp_global.svg", w: 58%)

#pause
#align(center, text(size: 16pt, fill: MUTED)[$Phi(x, y) = (x + 0.6 sin 1.6y, #h(6pt) y + 0.6 sin 1.6x)$ — and every network layer is one of these (L4)])

#pause
Globally: curvy. But the little orange square just got *leaned and stretched*. Hold that thought.

== The one case we already own: a linear map

For a *linear* map (L4: "a matrix is a verb") the nudge story is exact, everywhere, with no zooming:

$ F(bold(x)) = A bold(x) = mat(2, 1; 3, -2) bold(x) quad arrow.r.double quad F(bold(a) + bold(delta)) = F(bold(a)) + A bold(delta) $

#pause
Nudge the input by $bold(delta)$ — the output moves by $A bold(delta)$. *The matrix $A$ is the derivative*, at every point.

#pause
#result[Target for today: for a curvy $Phi$, find the matrix that plays the role of $A$ *near one point*.]

== Numbers: nudge the polar map, watch both outputs

Take the most useful curvy $RR^2 -> RR^2$ map there is — polar coordinates — at $(r, theta) = (2, pi/3)$:

$ Phi(r, theta) = (underbrace(r cos theta, x), thin underbrace(r sin theta, y)), quad quad Phi(2, pi/3) = (1.000, thin 1.732) $

#pause
Nudge each input by $0.01$, one at a time (L7's nudge machine, run twice per output):

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*nudge*], [*$Delta x$, per unit nudge*], [*$Delta y$, per unit nudge*]),
  [$r: 2 -> 2.01$], [$+0.00500 slash 0.01 = 0.500$], [$+0.00866 slash 0.01 = 0.866$],
  [$theta: pi/3 -> pi/3 + 0.01$], [$-0.01737 slash 0.01 = -1.737$], [$+0.00991 slash 0.01 = 0.991$],
)

#pause
Two inputs × two outputs = *four* sensitivities. One number was never going to be enough — but four numbers have an obvious home.

== Pack the four sensitivities: the Jacobian

House rule: *one row per output, one column per input.* Each entry is a partial derivative (L8):

$ J = mat((partial x)/(partial r), (partial x)/(partial theta); (partial y)/(partial r), (partial y)/(partial theta)) = mat(cos theta, -r sin theta; sin theta, r cos theta) quad arrow.r.double_((r, theta) = (2, pi\/3)) quad mat(0.500, -1.732; 0.866, 1.000) $

#pause
Exactly the four measured ratios (the $-1.737$ and $0.991$ were drifting here as the nudge shrank).

#pause
- Row 1 = $nabla x^top$: how output $x$ feels each input — a gradient lying on its side.
#pause
- Sanity check on the linear map: every partial of $A bold(x)$ is a constant, and $J = A$. ✓

== The picture: little squares become parallelograms #V

True image of a tiny square (ink) vs the parallelogram $J$ makes of it (orange) — the mismatch *dies faster than $epsilon$*:

#fig("/lecture9/figures/jacobian_zoom.svg", w: 64%)

#pause
#result[$ Phi(bold(a) + bold(delta)) approx Phi(bold(a)) + J(bold(a)) thin bold(delta) $ The Jacobian is the linear map that impersonates $Phi$ near $bold(a)$.]

== $det J$: the area dial

L4: a determinant is *the area-scaling factor of a linear map* — so $abs(det J)$ is $Phi$'s local area dial:

#fig("/lecture9/figures/polar_grid.svg", w: 58%)

#pause
For polar: $det J = r cos^2 theta + r sin^2 theta = r$ — the orange cell ($r = 1.5$) gets $tilde 3 times$ the teal cell's area ($r = 0.5$). At $r = 0$: $det J = 0$, the whole $theta$-axis is *crushed to a point* (L4's null space, gone local).

== Code: PyTorch measures the same matrix

#codebox[```python
import torch
from torch.autograd.functional import jacobian

def polar(p):
    r, th = p
    return torch.stack([r * torch.cos(th), r * torch.sin(th)])

J = jacobian(polar, torch.tensor([2.0, torch.pi / 3]))
# tensor([[ 0.5000, -1.7321],
#         [ 0.8660,  1.0000]])
torch.det(J)          # tensor(2.0000)  = r  ✓
```]

#pause
Our four hand-measured ratios, to four decimals — and the area dial reads $r = 2$, as derived.

#pause
#notebox[*Planted for L12:* when probability densities change variables, they must pay exactly this $abs(det J)$ area toll. Same dial, new customer.]

== The general case: same picture, more rows

For $f: RR^n -> RR^m$ — nothing new, just a bigger crate ($m$ outputs stacked, $n$ inputs across):

$ J(bold(x)) = mat((partial f_1)/(partial x_1), dots.c, (partial f_1)/(partial x_n); dots.v, dots.down, dots.v; (partial f_m)/(partial x_1), dots.c, (partial f_m)/(partial x_n)) in RR^(m times n), quad quad J_(i j) = "how output" i "feels input" j $

#pause
The zoo collapses out of this one object:

- $m = 1$: a single row — the gradient lying on its side, $J = nabla f^top$ (L8).
#pause
- $m = n = 1$: a $1 times 1$ matrix — L7's lone number $f'(x)$.

#pause
#result[There is only *one* derivative idea. The Jacobian is its general form; slope and gradient are its small print.]

== Chain rule: composition becomes multiplication

L7's chain rule multiplied slopes. Jacobians *are* the local linear maps — and composing linear maps is matrix multiplication (L4):

$ J_(g compose f)(bold(a)) = underbrace(J_g (f(bold(a))), k times m) dot underbrace(J_f (bold(a)), m times n) quad in RR^(k times n) $

#pause
Square through $f$ #sym.arrow.r parallelogram; that parallelogram through $g$ #sym.arrow.r another parallelogram. Multiply the matrices, compose the leans.

#pause
And the shapes *must click*: $(k times m)(m times n) = (k times n)$ — your free error-detector for every derivation.

#pause
#notebox[*Planted for L10–L11:* a network is $f_L compose dots.c compose f_1$, so its derivative is a *product of Jacobians*. Backprop will be nothing but a clever order for multiplying them out.]

== The zoo, second visit #V

#zoo(2)

#pause
One row left. It needs a different question: not "vector out?" but *"what happened to curvature?"*

== Checkpoint: shapes first, always #Q

#mcq([$f(x, y, z) = (x^2 + y^2, thin y - z)$ maps $RR^3 -> RR^2$. What is the shape of its Jacobian — and is $det J$ available to talk about areas?],
  [$3 times 2$, and yes],
  [$2 times 3$, and no],
  [$2 times 3$, and yes],
  [$3 times 3$, and yes],
)

== Answer: shapes first, always #A

#mcq-answer("B", [$J$ is $2 times 3$, and $det J$ does not exist],
  [One row per *output* (2), one column per *input* (3): $J = mat(2x, 2y, 0; 0, 1, -1)$, a $2 times 3$ crate. Determinants belong to *square* matrices only — a map $RR^3 -> RR^2$ flattens 3-D volumes onto a plane, and "how much did the volume scale?" stops being a fair question. The area dial needs $n = m$.])

// ═══════════════════ 3 · the Hessian ═══════════════════
= Curvature becomes a matrix: the Hessian

== In one dimension, curvature was one number

L7's second act: at a flat point, the sign of $f''$ told the whole story — smile $arrow.r$ minimum, frown $arrow.r$ maximum.

#pause
Now stand on a 2-D surface $f: RR^2 -> RR$ and walk in different directions:

#pause
- along one street the ground may *curve up* like a smile,
- along the cross-street it may curve up harder, or not at all — *or curve down*.

#pause
One number per direction, infinitely many directions. Sounds hopeless — but so did "one slope per direction" before L8 packed every directional slope into *one vector* $nabla f$.

#pause
Same trick again, one storey up.

== Numbers: the gradient is itself a map we know

Take $f(x, y) = x^2 + x y + y^2$ (keep it in view — it returns all lecture). Its gradient:

$ nabla f(x, y) = vec(2x + y, x + 2y) $

#pause
Look at what that *is*: vector in, vector out — $nabla f: RR^2 -> RR^2$. We built the perfect tool for those *ten minutes ago*.

#pause
Take the Jacobian *of the gradient*:

$ J_(nabla f) = mat((partial)/(partial x)(2x + y), (partial)/(partial y)(2x + y); (partial)/(partial x)(x + 2y), (partial)/(partial y)(x + 2y)) = mat(2, 1; 1, 2) $

#pause
That matrix has a name — and for this $f$, it is *L5's favorite matrix*. Not a coincidence we'll waste.

== The Hessian, defined

$ H(bold(x)) = J_(nabla f)(bold(x)), quad quad H_(i j) = (partial^2 f)/(partial x_i partial x_j) = "how slope" i "responds to a nudge of" x_j $

#pause
- Diagonal $H_(i i)$: L7's old curvature along axis $i$ — walk east, does the eastward slope grow?
#pause
- Off-diagonal $H_(i j)$: the *twist* — walk north, does the *eastward* slope change?

#pause
And a gift: for any twice-continuously-differentiable $f$, mixed partials agree — $H_(i j) = H_(j i)$ (Schwarz's theorem, stated with a straight face like L5's spectral theorem).

#pause
#result[The Hessian is *symmetric* — so the spectral theorem applies: real eigenvalues, perpendicular eigenvectors. L5's whole toolkit just walked in the door.]

== Test drive: the curvature you feel along a street

Walking from $bold(a)$ in unit direction $bold(u)$, the ground under you is the 1-D function $g(t) = f(bold(a) + t bold(u))$, and (chain rule, twice):

$ g''(0) = bold(u)^top H bold(u) $

#pause
For our $f$ with $H = mat(2, 1; 1, 2)$, feel three streets:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*direction $bold(u)$*], [*$bold(u)^top H bold(u)$*], [*the street feels like*]),
  [$(1, 0)$], [$2$], [a smile],
  [$(1, 1) slash sqrt(2)$], [$(2 + 1 + 1 + 2) slash 2 = 3$], [the *steepest* smile],
  [$(1, -1) slash sqrt(2)$], [$(2 - 1 - 1 + 2) slash 2 = 1$], [the gentlest smile],
)

#pause
$3$ and $1$ are $H$'s *eigenvalues*, felt along its *eigenvectors* (L5: $lambda = 3, 1$ at $(1,1), (1,-1)$). The extreme curvatures of a surface are an eigen-problem.

== Same picture, more rows: a 3-variable Hessian

$f(x, y, z) = x^2 + y^2 + x y z$. First the gradient, then differentiate *it*, coordinate by coordinate:

$ nabla f = vec(2x + y z, 2y + x z, x y) quad arrow.r.double quad H = mat(2, z, y; z, 2, x; y, x, 0) quad arrow.r.double_((2, thin 1, thin 3)) quad mat(2, 3, 1; 3, 2, 2; 1, 2, 0) $

#pause
Symmetric — check. Entries depend on where you stand — curvature is *local*, computed per point.

#pause
#notebox[Read $H_(1 2) = z$ out loud: "how the $x$-slope responds to nudging $y$." At $(2, 1, 3)$: nudge $y$ by $0.01$ and the $x$-slope $2x + y z$ climbs by $0.03$. Every entry of every Hessian is a sentence like this.]

== Code: the Hessian in one call

#codebox[```python
from torch.autograd.functional import hessian

f = lambda v: v[0]**2 + v[0]*v[1] + v[1]**2
hessian(f, torch.tensor([1.0, 1.0]))
# tensor([[2., 1.],
#         [1., 2.]])

g = lambda v: v[0]**2 + v[1]**2 + v[0]*v[1]*v[2]
hessian(g, torch.tensor([2.0, 1.0, 3.0]))
# tensor([[2., 3., 1.], [3., 2., 2.], [1., 2., 0.]])
```]

#pause
Under the hood it does what we did: differentiate the gradient — the Jacobian of $nabla f$.

// ═══════════════════ 4 · quadratic forms ═══════════════════
= The shape gallery: quadratic forms

== The pure-curvature family

Which functions have the *same* Hessian everywhere — curvature standing still so we can stare at it? Quadratics. For a symmetric $A$, multiplied out once by hand on 2×2:

$ q(bold(x)) = 1/2 bold(x)^top A bold(x) = 1/2 (x_1, x_2) mat(a, b; b, c) vec(x_1, x_2) = 1/2 (a x_1^2 + 2b x_1 x_2 + c x_2^2) quad "— a number" $

#pause
For $A = mat(2, 1; 1, 2)$: $q = x^2 + x y + y^2$ — *our running $f$*. Claim (⭐ proof in the next section): $nabla q = A bold(x)$ and $H = A$, at every point.

#pause
#result[A quadratic form is a Hessian standing still. Classify its shapes, and you've classified *all* local curvature.]

== The gallery: bowls, saddles, troughs #V

#let gal-s(A, name) = surface(qform(A), xlim: (-2, 2), ylim: (-2, 2), samples: 20, cell: 2.0mm, height: 21mm, title: name)
#let gal-c(A) = contour(qform(A), xlim: (-2, 2), ylim: (-2, 2), samples: 48, levels: 8, size: (52mm, 33mm), color: TEAL)
#align(center, grid(columns: 3, column-gutter: 11mm, row-gutter: 3.5mm, align: center,
  gal-s(A1, [*bowl* $A = mat(delim: "(", 2, 1; 1, 2)$]),
  gal-s(A2, [*saddle* $A = mat(delim: "(", 2, 0; 0, -2)$]),
  gal-s(A3, [*trough* $A = mat(delim: "(", 2, 0; 0, 0)$]),
  gal-c(A1), gal-c(A2), gal-c(A3),
  text(size: 15pt)[$lambda = #fmt(lam1.at(0), d: 0), #fmt(lam1.at(1), d: 0)$ — both $> 0$],
  text(size: 15pt)[$lambda = #fmt(lam2.at(0), d: 0), #fmt(lam2.at(1), d: 0)$ — mixed signs],
  text(size: 15pt)[$lambda = #fmt(lam3.at(0), d: 0), #fmt(lam3.at(1), d: 0)$ — a zero],
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[surfaces and their contour maps, computed live from $q(bold(x)) = 1/2 bold(x)^top A bold(x)$; eigenvalues by the in-slide solver — ellipses, hyperbolas, parallel lines])

== Eigenvalues read the shape — L5, cashed in

$A$ is symmetric, so L5 fires: $A = Q Lambda Q^top$ — "a perpendicular grid of dials". In eigen-coordinates $bold(t) = Q^top bold(x)$ the cross-term dies; the dials are *curvatures*:

$ q = 1/2 (lambda_1 t_1^2 + lambda_2 t_2^2 + dots.c + lambda_n t_n^2) quad "— one independent parabola per eigen-direction" $

#pause
#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*eigenvalue signs*], [*shape*], [*at a critical point this means*]),
  [all $lambda_i > 0$], [bowl], [strict local minimum — every street smiles],
  [all $lambda_i >= 0$, some $= 0$], [trough], [a flat valley floor along the $lambda = 0$ directions],
  [mixed signs], [saddle], [up some streets, down others — not a min!],
  [all $lambda_i < 0$], [dome], [local maximum],
)

#pause
#notebox[*Planted for L13:* the bowl's ellipse contours (axes = eigenvectors) are exactly a Gaussian's, with $A = Sigma^(-1)$.]

== The official words: definiteness

The vocabulary every ML paper uses for "which shape is it":

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*$A$ is…*], [*definition: for all $bold(x) eq.not bold(0)$*], [*equivalently*]),
  [positive definite (PD)], [$bold(x)^top A bold(x) > 0$], [all $lambda_i > 0$],
  [positive semi-definite (PSD)], [$bold(x)^top A bold(x) >= 0$], [all $lambda_i >= 0$],
  [negative definite (ND)], [$bold(x)^top A bold(x) < 0$], [all $lambda_i < 0$],
  [indefinite], [both signs occur], [mixed-sign $lambda_i$],
)

#pause
Why the two columns agree: in eigen-coordinates $bold(x)^top A bold(x) = sum_i lambda_i t_i^2$ — a sum of parabolas can only be always-positive if every dial $lambda_i$ is.

#pause
#result[*Positive definite = the surface is a bowl.* That one sentence decodes half the theorems you will ever meet about optimization and covariance.]

== Hand tests for 2×2 — ten-second shape reading

L5's sanity checks were $"tr" A = sum lambda_i$ and $det A = product lambda_i$. For symmetric 2×2 they *decide the shape*:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*test*], [*verdict*]),
  [$det A < 0$], [$lambda$'s have opposite signs — *saddle*, done (a negative det always betrays a flip: L4, L5)],
  [$det A > 0$], [same signs — read $"tr" A$: positive $arrow.r$ *bowl*, negative $arrow.r$ *dome*],
  [$det A = 0$], [some $lambda = 0$ — *semi-definite*: trough (or flat)],
)

#pause
For $n times n$, the computer's test is eigenvalues; the classical hand test is *Sylvester's criterion* (all leading principal minors $> 0$ $arrow.l.r.double$ PD — stated for your shelf, not proved).

#pause
#alertbox[Positive *entries* do not make a matrix positive *definite*: $mat(1, 2; 2, 1)$ is all-positive yet $det = -3 < 0$ — a saddle. Definiteness lives in the eigenvalues, not the ink.]

== Numbers: classify three matrices

#let lam-of(M) = {
  let (l, _) = la.eig-sym(M)
  $#fmt(l.at(0), d: 0), #fmt(l.at(1), d: 0)$
}
#table(
  columns: (auto, auto, auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*$A$*], [*tr*], [*det*], [*verdict by hand*], [*$lambda$ (live)*]),
  [$mat(2, 1; 1, 2)$], [$4$], [$3$], [det $> 0$, tr $> 0$ — *bowl* (PD)], [#lam-of(A1)],
  [$mat(1, 2; 2, 1)$], [$2$], [$-3$], [det $< 0$ — *saddle* (indefinite)], [#lam-of(((1.0, 2.0), (2.0, 1.0)))],
  [$mat(1, 1; 1, 1)$], [$2$], [$0$], [det $= 0$, tr $> 0$ — *trough* (PSD)], [#lam-of(((1.0, 1.0), (1.0, 1.0)))],
)

#pause
The $lambda$ column is computed by the in-slide eigensolver at compile time — trace and det agree with $sum lambda_i$ and $product lambda_i$ on every row (L5's exam armor, still working).

== Code: three lines settle any matrix

#codebox[```python
import numpy as np

for A in ([[2, 1], [1, 2]], [[1, 2], [2, 1]], [[1, 1], [1, 1]]):
    lam = np.linalg.eigvalsh(A)      # symmetric -> real, sorted
    print(lam)
# [1. 3.]     all > 0        -> bowl    (PD)
# [-1. 3.]    mixed signs    -> saddle  (indefinite)
# [0. 2.]     a zero, rest + -> trough  (PSD)
```]

#pause
`eigvalsh` — the `h` means *Hermitian/symmetric*: it promises real eigenvalues because L5's spectral theorem promises them first.

== Checkpoint: read the shape #Q

#mcq([At a critical point ($nabla f = bold(0)$), the Hessian is $H = mat(4, -2; -2, 1)$. The surface there is a…],
  [bowl — strict local minimum],
  [saddle],
  [trough — a flat direction],
  [dome — local maximum],
)

== Answer: read the shape #A

#mcq-answer("C", [trough — a flat valley direction],
  [$det H = 4 - 4 = 0$, so a $lambda$ is exactly $0$; $"tr" H = 5 > 0$ pins the other at $+5$ (indeed $lambda^2 - 5lambda + 0 = 0 arrow.r.double lambda = 5, 0$). Positive curvature along one eigen-street, *perfectly flat* along the other ($(1, 2)$, where $H bold(u) = bold(0)$) — a PSD trough, not a strict minimum. Beware option A: $det = 0$ is precisely where "bowl" claims go to die.])

// ═══════════════════ 5 · the ⭐ derivation ═══════════════════
= The one derivation: the gradient of a quadratic form

== Why this gradient, of all gradients

$bold(x)^top A bold(x)$ is the most-differentiated expression in machine learning:

- least squares (L4's preview): $norm(X theta - y)^2$ expands into one;
#pause
- the Gaussian's exponent (L13) *is* one; the L2 regularizer $theta^top theta$ is one with $A = I$;
#pause
- every second-order Taylor model (ten minutes from now) ends in one.

#pause
So we earn its gradient honestly — today's ⭐, the difficulty valve fully open:

#notebox[*Plan:* brute-force the 2×2 case symbol by symbol (no matrix calculus allowed), *then* watch the general answer repack itself. One derivation, everything else today is plausibility.]

== ⭐ Step 1: multiply the 2×2 out, no shortcuts #D

Let $A = mat(a, b; c, d)$ (deliberately *not* symmetric) and $bold(x) = vec(x_1, x_2)$:

#pause
$ bold(x)^top A bold(x) = (x_1, x_2) vec(a x_1 + b x_2, c x_1 + d x_2) $

#pause
$ = a x_1^2 + b x_1 x_2 + c x_1 x_2 + d x_2^2 = a x_1^2 + (b + c) x_1 x_2 + d x_2^2 $

#pause
A plain two-variable polynomial — L8 can differentiate this in its sleep. Note who survived: $b$ and $c$ only ever appear as the *sum* $b + c$.

== ⭐ Step 2: two honest partial derivatives #D

$ (partial)/(partial x_1) [a x_1^2 + (b + c) x_1 x_2 + d x_2^2] = 2a x_1 + (b + c) x_2 $

#pause
$ (partial)/(partial x_2) [a x_1^2 + (b + c) x_1 x_2 + d x_2^2] = (b + c) x_1 + 2d x_2 $

#pause
Stack them into the gradient (our column convention):

$ nabla (bold(x)^top A bold(x)) = vec(2a x_1 + (b + c) x_2, (b + c) x_1 + 2d x_2) $

== ⭐ Step 3: recognize the matrix hiding in it #D

$ vec(2a x_1 + (b+c) x_2, (b+c) x_1 + 2d x_2) = mat(2a, b + c; b + c, 2d) vec(x_1, x_2) = [mat(a, b; c, d) + mat(a, c; b, d)] bold(x) $

#pause
#result[$ nabla (bold(x)^top A bold(x)) = (A + A^top) bold(x) quad quad arrow.r.double quad "symmetric" A: quad nabla = 2 A bold(x) $]

#pause
Same argument in $RR^n$, one line: $bold(x)^top A bold(x) = sum_(i, j) A_(i j) x_i x_j$, and $partial slash partial x_k$ harvests row $k$ ($sum_j A_(k j) x_j$) plus column $k$ ($sum_i A_(i k) x_i$) — that is $(A bold(x))_k + (A^top bold(x))_k$. Same picture, more rows.

#pause
*Corollary (one more derivative):* $H = J_(nabla) = A + A^top$ — for symmetric $A$ and $q = 1/2 bold(x)^top A bold(x)$: $nabla q = A bold(x)$, $H = A$. The gallery's caption is now a theorem.

== The rule, cross-examined

The slide checks itself. Chalkdust's autodiff evaluates $nabla (bold(x)^top A bold(x))$ for $A = mat(2, 1; 1, 2)$ at $(1, 1)$ — reverse mode, no symbols:

$ "autodiff:" nabla = #v2(gcheck, d: 4) quad quad "our rule:" 2 A vec(1, 1) = #v2(gclosed, d: 4) quad #text(fill: GREEN)[— identical ✓] $

#pause
Two rules you now own for free, by choosing $A$:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*expression*], [*gradient*], [*why*]),
  [$norm(bold(x))^2 = bold(x)^top I bold(x)$], [$2 bold(x)$], [$A = I$ is symmetric],
  [$bold(b)^top bold(x)$ (linear)], [$bold(b)$], [the warm-up act: no square, constant slope],
)

#pause
#notebox[These two + today's ⭐ generate most of the Matrix Cookbook's first page. The rest is looked up, not re-derived — more on that in the bookkeeping section.]

// ═══════════════════ 6 · multivariate Taylor ═══════════════════
= Taylor, one degree at a time

== L7's ladder, one more time

L7 replaced a function near $x_0$ by polynomials, one degree at a time:

$ f(x_0 + h) approx f(x_0) + f'(x_0) thin h + 1/2 f''(x_0) thin h^2 $

#pause
Every ingredient just grew up. Upgrade each term with today's zoo:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*1-D ingredient*], [*$n$-D upgrade*], [*the term becomes*]),
  [step $h$], [step vector $bold(delta)$], [],
  [$f'(x_0) thin h$], [$nabla f$ (L8)], [$nabla f(bold(x))^top bold(delta)$ — a dot product],
  [$1/2 f''(x_0) thin h^2$], [$H$ (today)], [$1/2 bold(delta)^top H(bold(x)) bold(delta)$ — a quadratic form],
)

#pause
Slope became a vector, curvature became a matrix — and $h^2$ became "sandwich $H$ between two $bold(delta)$'s".

== The formula every optimizer lives on

$ f(bold(x) + bold(delta)) thin approx thin underbrace(f(bold(x)), "height") + underbrace(nabla f(bold(x))^top bold(delta), "tilt: the local plane") + underbrace(1/2 bold(delta)^top H(bold(x)) thin bold(delta), "bend: the local bowl or saddle") $

#pause
Read it as a story: stand at $bold(x)$; the surface near you is *a height, plus a tilt, plus a pure-curvature shape from the gallery* — and you can now classify that shape by eigenvalue signs.

#pause
#result[L16–L18 run on this one line: gradient descent trusts the tilt, Newton trusts the bend. Today it just has to become *yours*.]

== Numbers: predict a function you've never met

$f(x, y) = x^2 y$ at $bold(a) = (1, 1)$, step $bold(delta) = (0.1, 0.1)$. Harvest the three ingredients:

$ f(bold(a)) = 1, quad quad nabla f = vec(2x y, x^2) = vec(2, 1), quad quad H = mat(2y, 2x; 2x, 0) = mat(2, 2; 2, 0) $

#pause
Assemble, term by term:

$ f(bold(a) + bold(delta)) approx 1 + underbrace((2, 1) dot (0.1, 0.1), 0.3) + underbrace(1/2 (0.1, 0.1) mat(2, 2; 2, 0) vec(0.1, 0.1), 1/2 (0.06) = 0.03) = 1.33 $

#pause
Truth: $f(1.1, 1.1) = 1.1^2 times 1.1 = 1.331$. Off by $0.001$ — exactly the dropped third-order term.

#pause
Note the Hessian is *indefinite* here ($det = -4 < 0$: saddle curvature) — Taylor doesn't care; it reports the local shape, whatever it is.

== The three stories, along one street #V

Walk from $bold(a) = (1,1)$ along $bold(delta) = (t, t)$: the true $f$ becomes $g(t) = (1 + t)^3$, and the Taylor terms become L7-style line and parabola — all three plotted live:

#align(center, lines(
  fn: (t => calc.pow(1.0 + t, 3.0), t => 1.0 + 3.0 * t, t => 1.0 + 3.0 * t + 3.0 * t * t),
  domain: (-1.35, 1.05), samples: 140, markers: false,
  colors: (INK, MUTED, ACC), dashes: ("solid", "dashed", "dashed"),
  labels: ([true $f$ along the street], [order 1: height + tilt], [order 2: + the bowl term]),
  points: ((0.0, 1.0, [$bold(a)$]),),
  size: (102mm, 46mm), x-label: $t$,
))

#pause
Near $bold(a)$: the parabola hugs the truth visibly longer than the line — the multivariate story, sliced back down to L7's picture.

== Code: the whole expansion in five lines

#codebox[```python
from torch.autograd.functional import jacobian, hessian

f = lambda v: v[0]**2 * v[1]
a = torch.tensor([1.0, 1.0]); d = torch.tensor([0.1, 0.1])

g, H = jacobian(f, a), hessian(f, a)
pred2 = f(a) + g @ d + 0.5 * d @ H @ d
# pred2 = 1.3300     true f(a + d) = 1.3310    gap = 3rd order
```]

#pause
Five lines you will reuse in T5, and conceptually in every optimizer of Module 4.

== A promissory note: where this formula gets cashed

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*lecture*], [*what it does with today's expansion*]),
  [L16], [*convexity*: $H$ PSD everywhere — the surface is one global bowl, no bad valleys],
  [L17], [*gradient descent*: trust the tilt, step downhill; the learning rate is literally "how far you trust the linear term" — and $H$'s eigenvalues decide its fate (L5's warning)],
  [L18], [*Newton's method*: trust the bowl — minimize the quadratic model, jump $bold(delta)^* = -H^(-1) nabla f$],
)

#pause
#result[Every optimizer is a policy about *which Taylor term to trust, and how far*.]

// ═══════════════════ 7 · bookkeeping & take-home ═══════════════════
= Bookkeeping & the take-home

== Our contract: the gradient is a column

Matrix calculus has a dirty secret: two bookkeeping schools. *Numerator layout* writes the derivative of a scalar as a row; *denominator layout* as a column. Both are fine; mixing them mid-derivation is how homework dies.

#pause
This course signs one contract and sticks to it (MML's):

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*object*], [*shape*], [*note*]),
  [$bold(x)$, $bold(delta)$, $nabla f$], [column, $n$], [*gradient-as-column* — same shape as $bold(x)$, so $bold(x) - eta nabla f$ type-checks (L17)],
  [$J$ of $f: RR^n -> RR^m$], [$m times n$], [rows = outputs; for $m = 1$, $J = nabla f^top$ (a row)],
  [$H$], [$n times n$], [symmetric; $H = J_(nabla f)$],
)

#pause
#alertbox[When you open any other book or blog, *find its convention first* — a transposed-looking formula is usually the other school, not an error.]

== Shape-checking is a proof technique

The chain rule *forces* every shape, so let the shapes catch your bugs:

$ underbrace(J_(g compose f), k times n) = underbrace(J_g, k times m) underbrace(J_f, m times n), quad quad underbrace(nabla f(bold(x))^top bold(delta), (1 times n)(n times 1) = "scalar"), quad quad underbrace(bold(delta)^top, 1 times n) underbrace(H, n times n) underbrace(bold(delta), n times 1) = "scalar" $

#pause
If a derivation produces a shape mismatch, it is wrong — no further reading required. That five-second check replaces half of matrix-calculus memorization.

#pause
#notebox[*References for life:* the #link("https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf")[Matrix Cookbook] (look identities up — nobody re-derives them), #link("https://explained.ai/matrix-calculus/")[Parr & Howard, _The Matrix Calculus You Need for Deep Learning_] (the gentle full story), MML §5.4–5.7 (our text, same conventions).]

== The hook, resolved

A neural network is a chain $cal(L) = f_L compose dots.c compose f_1 (theta)$. What is its derivative? Now we can *name every piece*:

+ each layer $f_i: RR^n -> RR^m$ has a *Jacobian* — a lean-and-stretch parallelogram map;
#pause
+ the chain rule multiplies them: $J = J_L dots.c J_1$;
#pause
+ the last output is the scalar loss, so the whole product collapses to *one row* — the gradient $nabla cal(L)^top$, L8's uphill vector, for a million inputs;
#pause
+ the loss's curvature is a *Hessian* — but at $d = 10^6$ it has $10^12$ entries ($approx 4$ TB in float32): real optimizers never store it, they respect it (L17–L18).

#pause
#result[Promise kept: the Jacobian is a parallelogram, the Hessian is a bowl — and L10–L11 build the machine that multiplies those Jacobians for you.]

== The zoo, complete #V

#zoo(3)

#pause
One idea, four containers: *the derivative is the best local linear (then quadratic) story* — the container just matches the function's shape.

== Lecture 9 — summary

- *Jacobian*: $m times n$ crate of partials; the local linear map — $Phi(bold(a) + bold(delta)) approx Phi(bold(a)) + J bold(delta)$; little squares $arrow.r$ parallelograms; $abs(det J)$ = the area dial (polar: $det J = r$).
- *Chain rule* = matrix multiplication of Jacobians; shapes must click ($(k times m)(m times n)$).
- *Hessian* = Jacobian of the gradient; symmetric; entry $(i, j)$ = "how slope $i$ feels nudge $j$"; $bold(u)^top H bold(u)$ = curvature along $bold(u)$, extremes at eigenvectors.
- *Quadratic forms* $1/2 bold(x)^top A bold(x)$: bowls (all $lambda > 0$), saddles (mixed), troughs (a zero $lambda$); 2×2 hand test via det and trace; ⭐ $nabla(bold(x)^top A bold(x)) = (A + A^top) bold(x)$.
- *Taylor, order 2*: $f + nabla f^top bold(delta) + 1/2 bold(delta)^top H bold(delta)$ — height, tilt, bend; the optimizer's home.

#pause
#notebox[*Read before L10:* MML §5.4–5.7; skim #link("https://explained.ai/matrix-calculus/")[explained.ai/matrix-calculus]; keep the Matrix Cookbook bookmarked as a reference, not bedtime reading. *T5* (after L11) drills Jacobians and the full backward pass.]

== Practice problems

Try on paper; verify with `torch.autograd.functional` as in today's code slides.

+ $F(x, y) = (x^2 y, thin x + y^3)$: compute $J$ at $(1, 1)$ and $det J$. Is $F$ locally inflating or crushing areas there — by what factor?
+ For $Phi(x, y) = (e^x cos y, thin e^x sin y)$: show $det J = e^(2x)$. Why can this map never crush area to zero?
+ $f(x, y) = x^3 + y^3 - 3x y$: find both critical points, compute $H$ at each, and classify (one saddle, one bowl).
+ $f(x, y) = x y$ has *no* squared term anywhere. Compute $H$, its eigenvalues, and the shape at $(0,0)$.
+ Apply the ⭐ rule to the non-symmetric $A = mat(1, 4; 0, 1)$: write $nabla(bold(x)^top A bold(x))$, then verify by expanding $bold(x)^top A bold(x)$ by hand.
+ Second-order Taylor of $f(x, y) = e^x sin y$ at $(0, pi/2)$: show $nabla f = (1, 0)$, $H = mat(1, 0; 0, -1)$, predict $f(0.1, pi/2 + 0.1)$, and compare with the true value $1.0997$.

== ⭐⭐⭐ Beyond order two: derivatives become tensors #OPT

Why does everyone stop at the Hessian? Keep differentiating and count indices:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*order*], [*object*], [*entries at $d = 10^6$*]),
  [1], [$nabla f$ — vector], [$10^6$ (4 MB)],
  [2], [$H$ — matrix], [$10^12$ (4 TB)],
  [3], [$partial^3 f$ — a 3-index *tensor*], [$10^18$ (4 EB — exabytes)],
)

#pause
Storage explodes by $d$ per order while (L7's lesson) each Taylor term buys less — so ML lives on order 1, consults order 2, and never writes order 3 down.

#pause
#notebox[Escape hatch, teased for L11 ⭐⭐⭐: you can compute *Hessian-vector products* $H bold(v)$ without ever materializing $H$ — differentiate $nabla f^top bold(v)$ once more. Curvature information at gradient prices.]

#focus-slide[
  Curvature is a matrix, and its sign is the shape of the bowl.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Differentiation on a Computer* — the computer takes over differentiation: finite differences pick a fight with floating point (L2 returns), and forward mode wins it.
]
