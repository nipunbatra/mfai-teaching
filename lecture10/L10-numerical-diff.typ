// Differentiation on a Computer I: Finite Differences → Forward Mode — Lecture 10 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture10/L10-numerical-diff.typ
//   typst compile --root . --input handout=true lecture10/L10-numerical-diff.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Differentiation on a Computer I],
  subtitle: [Finite differences and forward-mode automatic differentiation],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the lecture's running expression: f(x, y) = (x + y) · x² at (3, 1) ──
// The SAME 5-node graph L11 walks backward and T5 drills — here the tangents
// flow FORWARD. Values in teal, tangents in blue (L11 paints adjoints orange).
#let g5f(vals: (:), tans: (:), hi: ()) = {
  let P  = (x: (0, 0.1), y: (0, 1.55), b: (1.15, -0.72), a: (1.15, 0.85), f: (2.3, 0.1))
  let TT = (x: $x$, y: $y$, b: $b = x^2$, a: $a = x + y$, f: $f = a dot b$)
  let CC = (x: INK, y: INK, b: TEAL, a: TEAL, f: ACC)
  align(center, diagram(spacing: (25mm, 7.5mm), {
    for (u, v) in (("x", "b"), ("x", "a"), ("y", "a"), ("b", "f"), ("a", "f")) {
      edge(P.at(u), P.at(v), "-|>", stroke: 0.8pt + MUTED)
    }
    for (k, pos) in P {
      let c = CC.at(k)
      let ishi = k in hi
      let rows = (text(size: 13pt, weight: 700, fill: INK, TT.at(k)),)
      if k in vals { rows.push(text(size: 11.5pt, weight: 500, fill: TEAL, vals.at(k))) }
      if k in tans { rows.push(text(size: 11.5pt, weight: 700, fill: BLUE, tans.at(k))) }
      node(pos, align(center, stack(spacing: 3pt, ..rows)),
        shape: fletcher.shapes.rect,
        fill: if ishi { BLUE.lighten(88%) } else if c == INK { white } else { c.lighten(91%) },
        stroke: if ishi { 1.5pt + BLUE } else { 0.9pt + c },
        corner-radius: 3pt, inset: 5.5pt)
    }
  }))
}

// three roads out of "f as code" — the Baydin trichotomy as a picture
#let road(name, tag, col) = align(left, stack(spacing: 4pt,
  text(size: 14.5pt, weight: 700, fill: col, name),
  text(size: 12pt, fill: INK, tag)))
#let trichotomy = align(center, diagram(spacing: (34mm, 4.5mm), {
  node((0, 1), align(center, stack(spacing: 4pt,
      text(size: 15pt, weight: 700, fill: INK)[$f$ as code],
      text(size: 12pt, fill: MUTED)[wanted: $f'$])),
    shape: fletcher.shapes.rect, fill: white, stroke: 1pt + INK, corner-radius: 3pt, inset: 8pt)
  node((1, 0), road([1 · symbolic], [do the algebra, output a formula], TEAL),
    shape: fletcher.shapes.rect, fill: TEAL.lighten(93%), stroke: 0.9pt + TEAL, corner-radius: 3pt, inset: 7.5pt)
  node((1, 1), road([2 · numeric], [nudge $x$, measure the slope], RED),
    shape: fletcher.shapes.rect, fill: RED.lighten(93%), stroke: 0.9pt + RED, corner-radius: 3pt, inset: 7.5pt)
  node((1, 2), road([3 · automatic], [run $f$ on numbers that carry slopes], ACC),
    shape: fletcher.shapes.rect, fill: ACC.lighten(91%), stroke: 0.9pt + ACC, corner-radius: 3pt, inset: 7.5pt)
  for r in (0, 1, 2) { edge((0, 1), (1, r), "-|>", stroke: 0.8pt + MUTED) }
}))

// funnel: a million inputs, one output — the shape that decides everything
#let funnel = align(center, diagram(spacing: (19mm, 6.5mm), {
  let ins  = ((0, 0), (0, 1), (0, 2))
  let mids = ((1, 0.5), (1, 1.5))
  for (i, p) in ins.enumerate() {
    node(p, text(size: 13pt, ($theta_1$, $theta_2$, $theta_3$).at(i)), radius: 4.6mm,
      fill: white, stroke: 0.9pt + INK)
  }
  node((0, 2.9), text(size: 11.5pt, fill: MUTED)[⋮ #h(2pt) $10^9$ of these], stroke: none)
  for m in mids { node(m, none, radius: 3.2mm, fill: TEAL.lighten(88%), stroke: 0.9pt + TEAL) }
  node((2, 1), text(size: 13pt, [loss]), radius: 5.8mm, fill: ACC.lighten(88%), stroke: 1.4pt + ACC)
  for p in ins { for m in mids { edge(p, m, "-|>", stroke: 0.7pt + MUTED) } }
  for m in mids { edge(m, (2, 1), "-|>", stroke: 0.7pt + MUTED) }
}))

#title-slide()

= How PyTorch checks analytical gradients

== The bug that does not crash

You extend PyTorch with a custom operation — a fused kernel, a new layer. Autograd asks *you* to supply its derivative.

#pause
Get it slightly wrong, and *nothing crashes*:

- the loss still goes down (mostly) — wrong gradients are often "downhill-ish"
#pause
- the model trains, ships… and is quietly worse than it should be
#pause

#alertbox[A wrong gradient may produce no exception or NaN. A numerical gradient check gives an independent comparison before the operation is used in training.]

== A numerical check: `torch.autograd.gradcheck`

Every serious framework ships a gradient checker. PyTorch's:

#codebox[```python
import torch
from torch.autograd import gradcheck

def my_op(x):                     # your shiny new operation
    return torch.sin(x) * x**2

x = torch.randn(4, dtype=torch.float64, requires_grad=True)
print(gradcheck(my_op, (x,)))     # True — backward() is correct
```]

#pause
And what does it compare `backward()` against? *This*:

$ f'(x) approx (f(x + h) - f(x - h)) / (2h) quad quad "— a slope, measured with a tiny nudge" $

== Three design questions in that one call

PyTorch compares analytical gradients with finite-difference secants. Three implementation choices need explanation:

#pause
+ *Accuracy* — the numerical comparison is an approximation. How large can its error be?
#pause
+ *Numerical choices* — the default tolerances are designed for float64, `eps = 1e-6`, and the formula is symmetric. We will derive why.
#pause
+ *Cost* — if nudging is useful for checking, why is it not used to train?

#pause
#notebox[All three fall out of one lecture. The answers involve L2's machine epsilon, L7's Taylor remainder — and a number system where $epsilon^2 = 0$.]

== Two ways to compute a derivative

#result[Finite differences approximate a derivative and incur truncation and rounding error. Automatic differentiation applies the chain rule to the executed program, without choosing a step size.]

#pause
Two parts:

- *Finite differences*: truncation decreases with $h$, while rounding error increases. Their sum produces a U-shaped error curve and an optimal scale for $h$ ⭐.
#pause
- *Forward-mode AD*: dual numbers propagate a value and tangent through each primitive, so there is no finite-difference truncation or cancellation.

== Learning outcomes

By the end of this lecture you will be able to:

+ Estimate derivatives with *forward and central differences* and predict their error orders.
+ Explain the tradeoff between *truncation and rounding error*, and read an error-vs-$h$ *U-curve*.
+ Derive the *optimal step* $h^* approx sqrt(epsilon)$ and the "half your digits" floor. ⭐
+ Explain why finite differences *cannot scale* to $10^6$ parameters.
+ Compute with *dual numbers* $a + b epsilon$, $epsilon^2 = 0$ — and show that they apply derivative rules exactly to the represented program (up to ordinary floating-point rounding).
+ Run *forward-mode autodiff* by hand: a (value, tangent) trace, one pass per input.

= Three approaches to a derivative

== Symbolic, numerical, and automatic differentiation #V

#trichotomy

#pause
#v(4pt)
This trichotomy — *symbolic / numeric / automatic* — organizes everything about differentiation on machines (it's the framing of the Baydin et al. survey in your reading).

#pause
The first two methods follow directly from calculus. Automatic differentiation will be derived from computation graphs in L11.

== Symbolic differentiation: do the algebra

What you did in L7, mechanized (SymPy, Mathematica): apply the rules, output a *formula*.

$ f(x) = x^2 sin(x) quad arrow.r quad f'(x) = 2x sin(x) + x^2 cos(x) quad #text(fill: GREEN)[— exact ✓] $

#pause
Exact, human-readable… and it needs $f$ to *be* a formula. Your training loss — Python loops, branches, lookups — has no closed form to hand over.

#pause
And formulas have a darker problem. Watch what happens when we *compose* — the one thing deep learning does all day.

== Symbolic differentiation has a swelling problem

Iterate a humble map (composition = depth, exactly like stacked layers): $ell_1 = x$, then $ell_(k+1) = 4 ell_k (1 - ell_k)$:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Depth*], [*The formula*]),
  [$ell_2$], [$4x(1 - x)$],
  [$ell_3$], [$16x(1 - x)(1 - 2x)^2$],
  [$ell_4$], [$64x(1 - x)(1 - 2x)^2 (1 - 8x + 8x^2)^2$],
)

#pause
Now differentiate $ell_4$ — four compositions deep — and expand:

$ #text(size: 16pt)[$dif ell_4 \/ dif x = -131072 x^7 + 458752 x^6 - 638976 x^5 + 450560 x^4 - 168960 x^3 + 32256 x^2 - 2688 x + 64$] $

#pause
#result[*Expression swell*: each level of composition multiplies the derivative's size. A 100-layer network's symbolic gradient would be astronomical.]

== Numerical differentiation: measure a secant slope

No algebra at all — *nudge and divide*:

#codebox[```python
def derivative(f, x, h=1e-8):
    return (f(x + h) - f(x)) / h    # works on ANY f — even a black box
```]

#pause
- Three lines. No formula needed — loops, branches, whole programs welcome.
#pause
- The result is approximate. Its error depends on both $h$ and the floating-point format.

#pause
#notebox[The default `h=1e-8` is not universally optimal. The ⭐ derivation will show how a useful scale follows from curvature and machine precision.]

== Automatic differentiation: propagate derivatives

*Automatic differentiation (AD)*: run the *program* of $f$, but on values that carry their own derivatives — every `+` and `*` updates both.

#pause
- No finite-difference approximation: it differentiates the sequence of implemented primitives; floating-point evaluation still rounds normally
#pause
- *Formula-free* like numeric — works on code, never builds an expression
#pause
- Cost: a small constant times the cost of running $f$ itself

#pause
PyTorch's `autograd` is one implementation. Today we build forward mode by hand; L11 develops reverse mode and backpropagation.

#pause
First we quantify the error and cost of finite differences.

= Forward differences: two sources of error

== L7's limit, with the limit amputated #V

L7 _defined_ the derivative as a limit of secant slopes. A computer can't take limits — it can only *stop early*:

$ f'(x) = lim_(h arrow.r 0) (f(x + h) - f(x)) / h quad quad arrow.r.double quad quad D_h f(x) := (f(x + h) - f(x)) / h $

#align(center, lines(
  fn: (x => calc.exp(x), x => 1 + x, x => 1 + 1.7183 * x),
  domain: (-1.6, 1.6),
  labels: ([$e^x$], [tangent · slope 1], [secant · slope 1.72]),
  colors: (INK, GREEN, ACC),
  dashes: ("solid", "dashed", "solid"),
  markers: false,
  size: (118mm, 44mm),
  x-label: $x$,
))

#align(center, text(size: 16pt, fill: MUTED)[$f = e^x$ at $x = 0$, true slope $1$: the secant with $h = 1$ overshoots to $1.72$. Shrink $h$…])

== First contact: it kind of works

Test on $f = e^x$ at $x = 0$, where the truth is exactly $f'(0) = e^0 = 1$:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*$h$*], [*$D_h f(0)$ (float64)*], [*error*]),
  [$10^(-1)$], [1.0517091808…], [5.2 × 10⁻²],
  [$10^(-2)$], [1.0050167084…], [5.0 × 10⁻³],
  [$10^(-3)$], [1.0005001667…], [5.0 × 10⁻⁴],
)

#pause
Divide $h$ by 10 → error divides by 10. Extrapolate the pattern: $h = 10^(-16)$ should give *16 correct digits*.

#pause
And look closer: the error isn't just _order_ $h$ — it is almost exactly $h\/2$. That "2" is not a coincidence. Taylor knows why.

== Truncation error from Taylor's theorem

*L7 callback* — Taylor's remainder form is an _equality_ (some $xi$ between $x$ and $x + h$): $f(x + h) = f(x) + h f'(x) + (h^2\/2) f''(xi)$. Rearrange the forward difference:

$ D_h f(x) = f'(x) + underbrace(h / 2 f''(xi), #text(fill: TEAL)[truncation error — $O(h)$]) $

#pause
- For $e^x$ at $0$: $f'' approx 1$ → error $approx h\/2$ — exactly the table's $5 times 10^(-3)$ at $h = 10^(-2)$ ✓
#pause
- Truncation $prop h$: you stopped Taylor early and pay the curvature you ignored. Cure: *shrink $h$*.

#pause
So drive $h arrow.r 10^(-16)$ and collect 16 digits… right? You spent L2 learning why not.

== Subtraction cancels leading digits

*L2 callback: catastrophic cancellation* — subtracting near-equals promotes rounding garbage to the front. And $f(x + h) - f(x)$ is a subtraction of near-equals *by design*.

Take $h = 10^(-8)$. What float64 actually stores and computes:

#let mono(c, s, w: 400) = text(font: "IBM Plex Mono", size: 14.5pt, fill: c, weight: w, s)
#codebox[
  #grid(columns: (auto, auto, auto, 1fr), column-gutter: 0pt, row-gutter: 6pt, align: left,
    mono(INK, "e^h stored  =  "), mono(MUTED, "1.00000000"), mono(INK, "99999999392252903…", w: 700), [#h(8pt)#text(size: 13.5pt, fill: RED)[← last digits: rounding noise]],
    mono(INK, "1.0         =  "), mono(MUTED, "1.00000000"), mono(MUTED, "00000000000000000"), [],
    mono(INK, "difference  =  "), mono(RED, "0.00000000", w: 700), mono(INK, "99999999392252903…", w: 700), [#h(8pt)#text(size: 13.5pt, fill: RED)[← 8 leading digits annihilated]],
  )
]

#pause
Both inputs carried \~16 good digits. Their difference carries *\~8* — then `/ h` scales the surviving noise up by $10^8$.

#pause
#result[Started with 16 digits. Kept 8. The estimate: $0.99999999#text(fill: RED)[39…]$ — noise wearing a derivative costume.]

== A rounding-error model grows as $epsilon\/h$

L2's rounding rule: each stored value is off by a relative $delta$, $|delta| lt.eq epsilon$ — recall $epsilon_64 approx 2.2 times 10^(-16)$, $epsilon_32 approx 1.2 times 10^(-7)$.

#pause
So each evaluation of $f$ is really $f dot (1 + delta)$ — wrong by up to $epsilon |f|$. The numerator inherits up to $2 epsilon |f|$ of garbage, and dividing by $h$ *amplifies* it:

$ "rounding error of " D_h f quad approx quad (2 epsilon |f(x)|) / h quad #text(fill: RED)[— grows as $h$ shrinks!] $

#pause
- Truncation shrinks with $h$; rounding error grows with $1\/h$.
#pause

#result[For large $h$, truncation dominates. For small $h$, rounding dominates. Between them is a minimum-error scale.]

== Measured error across step sizes

$f = e^x$, $x = 0$, float64 — every digit below is real:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 6pt,
  table.header([*$h$*], [*$D_h f(0)$*], [*error*]),
  [$10^(-4)$], [1.0000500017…], [5 × 10⁻⁵],
  [$10^(-8)$], [#text(fill: GREEN)[*0.9999999939…*]], [#text(fill: GREEN)[*6 × 10⁻⁹* — the best it ever gets]],
  [$10^(-12)$], [1.0000889006…], [9 × 10⁻⁵ — worse again!],
  [$10^(-15)$], [1.1102230246…], [1 × 10⁻¹ — that's $epsilon_64$'s face],
  [$10^(-16)$], [#text(fill: RED)[*0.0000000000*]], [#text(fill: RED)[*100%*]],
)

#pause
#alertbox[The last row is pure L2: $1 + 10^(-16)$ *rounds to 1* — the nudge fell into the float gap, the numerator is exactly $0$, and the estimator reports that the derivative of _everything_ is zero. With total confidence.]

== The U-curve #V

#fig("/lecture10/figures/l10_ucurve.svg", w: 84%)

#align(center, text(size: 16pt, fill: MUTED)[Error vs $h$ for $D_h e^x$ at $x = 0$ in float64: truncation dominates on the right and rounding on the left.])

== How to read a U-curve

- *Right wall, slope $+1$*: truncation, $prop h$ — smooth, predictable, Taylor's receipt
#pause
- *Left wall, slope $-1$*: rounding, $prop epsilon\/h$ — jagged, because rounding noise is erratic, not smooth
#pause
- *The floor* near $h approx 10^(-8)$: error $approx 10^(-8)$ — the best this method will _ever_ do in float64

#pause
#result[There is no winning $h$ — only a least-bad one. Every finite-difference scheme has a floor it cannot go below.]

#pause
We now derive the location and scale of the minimum.

= ⭐ The best possible step

== ⭐ Step 1: model the total error #D

Total error = truncation + rounding. Bound $|f''| lt.eq M$ near $x$ (curvature scale, L7), take $|f(x)| approx 1$ for clean bookkeeping (our $e^0$ case exactly):

$ E(h) = underbrace(M / 2 h, #text(fill: TEAL)[truncation — grows with $h$]) + underbrace((2 epsilon) / h, #text(fill: RED)[rounding — grows with $1\/h$]) $

#pause
- One term climbs, one falls: $E(h)$ is U-shaped — we *modeled* the measured curve
#pause
- Those are the two dashed lines drawn on the U-curve figure — this is a model you can check

#pause
Minimizing a function of one variable… we happen to own a lecture on that (L7).

== ⭐ Step 2: minimize — with L7's own calculus #D

Set the derivative of the error (the derivative of the error of a derivative!) to zero:

$ E'(h) = M / 2 - (2 epsilon) / h^2 = 0 quad arrow.r.double quad h^* = 2 sqrt(epsilon / M) $

#pause
At the optimum, plug $h^*$ back in — the two modeled contributions are equal:

$ E(h^*) = sqrt(epsilon M) + sqrt(epsilon M) = 2 sqrt(epsilon M) $

#pause
#result[$h^* = 2 sqrt(epsilon \/ M) approx sqrt(epsilon)$, #h(8pt) best error $approx sqrt(epsilon)$ — for $M tilde 1$, *both* are "root machine epsilon".]

#pause
Float64: $h^* = 2 sqrt(2.2 times 10^(-16)) approx 3 times 10^(-8)$, error floor $approx 3 times 10^(-8)$.

== ⭐ Step 3: verify against the machine #D #V

Zoom into the U-curve's valley and lay the model on top of the measurement:

#fig("/lecture10/figures/l10_ucurve_zoom.svg", w: 67%)

#pause
The terms cross near $h^* approx 3 times 10^(-8)$. The predicted worst-case floor is $3 times 10^(-8)$, the measured best error is $6 times 10^(-9)$, and the two asymptotic slopes are $plus.minus 1$.

== The precision floor: roughly half the digits

Relative error $sqrt(epsilon)$ means: *of the digits your float carries, forward differences keep half.*

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Format*], [*$epsilon$ (L2's table)*], [*digits carried*], [*digits surviving $sqrt(epsilon)$*]),
  [float64], [2.2 × 10⁻¹⁶], [\~16], [#text(fill: GREEN)[\~8]],
  [float32], [1.2 × 10⁻⁷], [\~7], [#text(fill: RED)[\~3.5]],
  [float16], [9.8 × 10⁻⁴], [\~3], [#text(fill: RED)[\~1.5 — noise]],
)

#pause
With default settings, a float32 numerical check may not distinguish a correct gradient from an error around $10^(-3)$. Gradient-checking tools therefore recommend double precision and warn that lower-precision inputs may fail their default tolerances.

#pause
#notebox[Rule of thumb worth memorizing: finite differences cost you *half your precision*. Exactness isn't on the menu — at any $h$.]

== Same experiment, half the bits #V

#fig("/lecture10/figures/l10_fp32_fp64.svg", w: 72%)

#pause
Same $f$, same code — float32's floor sits roughly *ten thousand times higher*, and below $h = 10^(-8)$ its estimate is exactly $0$ because `1.0 + 1e-8 == 1.0` in float32.

= Central differences: a better secant

== Symmetrize the secant #V

Don't lean on one side — straddle the point: sample at $x - h$ *and* $x + h$.

$ C_h f(x) := (f(x + h) - f(x - h)) / (2h) $

#align(center, lines(
  fn: (x => calc.exp(x), x => 1 + x, x => 1.5431 + 1.1752 * x),
  domain: (-1.6, 1.6),
  labels: ([$e^x$], [tangent · slope 1], [central secant · slope 1.18]),
  colors: (INK, GREEN, ACC),
  dashes: ("solid", "dashed", "solid"),
  markers: false,
  size: (114mm, 44mm),
  x-label: $x$,
))

#align(center, text(size: 16pt, fill: MUTED)[Same generous $h = 1$: the one-sided secant said 1.72; the symmetric one says 1.18 — already near-parallel to the tangent.])

== Why symmetry buys a whole order

Write Taylor *both ways* and subtract — watch the even terms die:

$ f(x + h) = f + h f' + h^2 / 2 f'' + h^3 / 6 f''' + dots quad quad f(x - h) = f - h f' + h^2 / 2 f'' - h^3 / 6 f''' + dots $

#pause
$ C_h f(x) = f'(x) + underbrace(h^2 / 6 f'''(xi), #text(fill: TEAL)[truncation now $O(h^2)$]) quad quad #text(fill: MUTED)[— the $f''$ term cancelled *exactly*] $

#pause
Halve $h$ → *quarter* the truncation error. At $h = 10^(-2)$: forward is off by $5 times 10^(-3)$, central by $1.7 times 10^(-5)$.

#pause
#result[Same two evaluations as forward (reused smartly), *one order better*. This is why every serious gradient checker is central.]

== Central differences: the same tradeoff at lower error #V

#fig("/lecture10/figures/l10_fwd_central.svg", w: 63%)

#pause
⭐ redo (practice): minimize $E(h) = M h^2\/6 + epsilon\/h$ → $h^* prop epsilon^(1\/3) approx 6 times 10^(-6)$, floor $prop epsilon^(2\/3) approx 10^(-11)$.

#pause
#result[For float64 central differences, the balance between truncation and rounding error places the optimal $h$ near $10^(-6)$.]

== Checkpoint: choosing the step size #Q

#mcq([Central differences, float64, well-behaved $f$. You shrink $h$ from $10^(-5)$ to $10^(-7)$ and the error gets *worse*. What happened?],
  [Truncation error grew — $h$ is still too big],
  [You crossed $h^*$: rounding's $epsilon\/h$ term now dominates],
  [Central differences are only valid for $h > 10^(-5)$],
  [The function must not be differentiable at $x$],
)

== Answer: choosing the step size #A

#mcq-answer([B], [past $h^*$, rounding dominates],
  [For float64 central differences, $h^* approx 6 times 10^(-6)$. Below it, truncation continues to shrink as $h^2$, but cancellation error grows as $epsilon\/h$. Therefore smaller $h$ is not always more accurate.])

= The wall: one evaluation per input

== A gradient is $n$ separate questions

For $f: RR^n arrow.r RR$, the gradient wants $partial f \/ partial theta_i$ for *every* $i$ — and a nudge can only ask about *one input at a time*:

#codebox[```python
def grad_fd(f, theta, h=1e-6):
    g = np.zeros_like(theta)
    for i in range(len(theta)):          # one loop turn PER parameter
        e = np.zeros_like(theta); e[i] = h
        g[i] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g                             # cost: 2n evaluations of f
```]

#pause
Nudging $theta_3$ tells you *nothing* about $theta_7$. There is no bulk discount.

== Function-evaluation cost grows with input dimension #V

#align(center, bars(
  (2, 3, 4, 5, 6, 7, 8, 9),
  labels: ([1], [2], [3], [4], [5], [6], [7], [8]),
  annotate: true,
  size: (116mm, 44mm),
  title: [forward-difference evaluations of $f$ for one gradient, vs number of inputs $n$],
))

#pause
The staircase is the point: $n + 1$ evaluations (forward), $2n$ (central) — *linear in the number of parameters*, every single gradient step.

== Now scale it to a real model

One forward pass of a big model ≈ 60 ms. One finite-difference gradient:

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Model*], [*Parameters $n$*], [*Time for ONE gradient*]),
  [linear regression], [10], [\~0.7 s — fine],
  [small MLP], [10⁵], [1.7 hours],
  [GPT-3], [1.75 × 10¹¹], [#text(fill: RED)[*\~330 years*]],
)

#pause
#alertbox[Training needs *millions* of gradient steps. Finite differences don't need a better $h$ — they need a different job.]

#pause
Finite differences are suitable for occasional checks on small tensors, but not for every training step of a large model.

== Comparison after two approaches

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Route*], [*Accuracy*], [*Cost for $nabla f$, $f: RR^n arrow.r RR$*]),
  [symbolic], [exact], [#text(fill: RED)[expression swell; needs a formula]],
  [finite differences], [#text(fill: RED)[$approx$, floor $sqrt(epsilon)$ — half your digits]], [#text(fill: RED)[$n + 1$ evaluations]],
  [*wanted*], [#text(fill: GREEN)[exact]], [#text(fill: GREEN)[\~cost of running $f$, no formula]],
)

#pause
The remaining goal is a formula-free derivative at roughly the cost of evaluating $f$. Dual numbers provide the construction.

= Dual numbers: an infinitesimal you can compute with

== Remove the quadratic term algebraically

Stare at why truncation existed at all:

$ f(x + h) = f(x) + h f'(x) + underbrace(h^2 / 2 f'' + dots, "everything we DON'T want") $

#pause
The unwanted terms all carry $h^2$ or higher. Finite differences make $h^2$ _small_ and pay $sqrt(epsilon)$ for it.

#pause
We introduce a formal unit with $epsilon^2 = 0$ but $epsilon eq.not 0$. No real number has this property; it defines a different algebra.

#pause
#result[Dual numbers extend real arithmetic with a formal unit $epsilon$ satisfying $epsilon^2 = 0$.]

== Meet the dual numbers: decree $epsilon^2 = 0$

A *dual number* is $a + b epsilon$ — carry the pair $(a, b)$, compute with ordinary algebra plus one new rule, $epsilon^2 = 0$:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Operation*], [*Result — just expand, then apply $epsilon^2 = 0$*]),
  [$(a + b epsilon) plus.minus (c + d epsilon)$], [$(a plus.minus c) + (b plus.minus d) epsilon$],
  [$(a + b epsilon)(c + d epsilon)$], [$a c + (a d + b c) epsilon + underbrace(b d epsilon^2, = thin 0) = a c + (a d + b c) epsilon$],
)

#pause
Look hard at the product's $epsilon$-part: $a d + b c$ — if $b, d$ were "derivatives of $a, c$"… that is the *product rule*, appearing uninvited.

#pause
#notebox[Two epsilons in this lecture, by historical accident: *machine epsilon* $epsilon_64$ (a float's gap, L2) and this *dual unit* $epsilon$ (an algebraic symbol). Unrelated beasts. From here on, $epsilon$ means the dual unit.]

== Example: $(3 + epsilon)^2$

Feed $x = 3 + epsilon$ — "3, plus a formal nudge" — through plain squaring:

$ (3 + epsilon)^2 = 9 + 6 epsilon + epsilon^2 = 9 + 6 epsilon $

#pause
Read the pair: value $9 = 3^2$ ✓ #h(6pt) and $epsilon$-coefficient $6 = 2 dot 3 = (dif / (dif x)) x^2 |_(x=3)$ ✓

#pause
Once more: $(3 + epsilon)^3 = 27 + 27 epsilon$ — and indeed $(x^3)' = 3 x^2 = 27$ at $x = 3$. ✓

#pause
#result[Push $x + epsilon$ through the arithmetic and read the derivative from the $epsilon$-coefficient. There is no finite step or subtraction of nearby values.]

== Why the $epsilon$-slot always holds the derivative

Not a coincidence — it's Taylor, with the tail *annihilated by algebra*. For smooth $f$, substitute a dual $t = b epsilon$ into the expansion:

$ f(a + b epsilon) = f(a) + f'(a) b epsilon + f''(a) / 2 b^2 underbrace(epsilon^2, 0) + dots = f(a) + f'(a) thin b thin epsilon $

#pause
Every term we fought as "truncation error" contains $epsilon^2$ — and $epsilon^2$ *is zero*. Nothing is truncated; nothing is left to truncate.

#pause
#result[Finite differences make the Taylor tail *small* and pay $sqrt(epsilon_64)$. Dual numbers make it *zero* — by decree.]

#pause
Thus the finite-difference error sources are absent: there is no truncated finite step and no subtraction of nearby values.

== Every primitive learns one dual rule

$f(a + b epsilon) = f(a) + f'(a) b epsilon$ hands each primitive its own one-liner:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Primitive*], [*Dual rule*], [*…which is*]),
  [$u + v$], [$(a + c) + (b + d) epsilon$], [sum rule],
  [$u dot v$], [$a c + (a d + b c) epsilon$], [product rule],
  [$e^u$], [$e^a + e^a b epsilon$], [chain rule for $exp$],
  [$sin u$], [$sin a + (cos a) b epsilon$], [chain rule for $sin$],
  [$1 \/ u$], [$1 \/ a - (b \/ a^2) epsilon$], [reciprocal rule (practice)],
)

#pause
A dozen rules cover a numerical library. In code this is *operator overloading*: teach `+` and `*` to a new class, and every program built from them differentiates itself.

== A `Dual` class in ten lines

#codebox[```python
class Dual:
    def __init__(self, v, d):
        self.v, self.d = v, d                  # value, derivative
    def __add__(self, o):
        return Dual(self.v + o.v, self.d + o.d)
    def __mul__(self, o):
        return Dual(self.v * o.v,
                    self.v * o.d + self.d * o.v)   # product rule

x = Dual(3.0, 1.0)          # x = 3 + ε   (derivative of x wrt x is 1)
print((x * x).d)            # 6.0   — exact d/dx x² at 3
print((x * x * x).d)        # 27.0  — exact d/dx x³ at 3
```]

#pause
Every `*` applies the product rule to the tangent component. No step size is present in this code.

= Forward mode: derivatives that ride along

== The same recipe, now carrying two columns

*Forward-mode AD* = evaluate $f$ step by step, dual all the way. Take $f(x, y) = (x + y) dot x^2$ at $(3, 1)$ — for $partial f\/partial x$, *seed* $dot(x) = 1, dot(y) = 0$:

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Step*], [*Value*], [*Tangent rule*], [*Tangent*]),
  [$x$ (leaf)], [$3$], [seed], [$dot(x) = bold(1)$],
  [$y$ (leaf)], [$1$], [seed], [$dot(y) = 0$],
  uncover("2-")[$a = x + y$], uncover("2-")[$4$], uncover("2-")[$dot(a) = dot(x) + dot(y)$], uncover("2-")[$1$],
  uncover("2-")[$b = x^2$], uncover("2-")[$9$], uncover("2-")[$dot(b) = 2 x dot(x)$], uncover("2-")[$6$],
  uncover("3-")[$f = a dot b$], uncover("3-")[$36$], uncover("3-")[$dot(f) = dot(a) b + a dot(b)$], uncover("3-")[$9 + 24 = bold(33)$],
)

#uncover("3-")[#result[$f = 36$, #h(6pt) $partial f \/ partial x = 33$ — one sweep, carrying value and tangent.]]

== Tangents ride the same arrows #V

#g5f(
  vals: (x: [$= 3$], y: [$= 1$], a: [$= 4$], b: [$= 9$], f: [$= 36$]),
  tans: (x: $dot(x) = 1$, y: $dot(y) = 0$, a: $dot(a) = 1$, b: $dot(b) = 6$, f: $dot(f) = 33$),
)

#pause
- Values (teal) flow left → right; tangents (blue) flow *with* them, through the *same* graph, in the *same* pass
#pause
- Hold this picture: L11 will send a different quantity through these arrows — *backward*

== Where 33 came from: two routes, added

$x$ reaches $f$ two ways — through $a$ and through $b$ — and the tangent of the product added one contribution per route:

$ dot(f) = underbrace(dot(a) b, "via " a: 1 dot 9 = 9) + underbrace(a dot(b), "via " b: 4 dot 6 = 24) = 33 $

#pause
Sanity check by L8's rules, on the closed form $f = x^3 + x^2 y$:

$ (partial f) / (partial x) = 3 x^2 + 2 x y = 27 + 6 = 33 quad checkmark $

#pause
#notebox[File away "*sum over routes*" — it is the multivariate chain rule wearing graph clothes, and it is the exact spot where L11's backward pass will plug in.]

== The second input needs a second pass

$dot(f)$ answered *one* question: sensitivity to $x$. For $partial f \/ partial y$, re-run with the other one-hot seed $dot(x) = 0, dot(y) = 1$:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Step*], [*Value*], [*Tangent (new seed)*]),
  [$a = x + y$], [$4$], [$dot(a) = 0 + 1 = 1$],
  [$b = x^2$], [$9$], [$dot(b) = 2 dot 3 dot 0 = 0$],
  [$f = a dot b$], [$36$], [$dot(f) = 1 dot 9 + 4 dot 0 = bold(9)$],
)

#pause
#result[$nabla f(3, 1) = (33, 9)$ in *two* passes: one seed direction per input.]

#pause
Keep $(33, 9)$ warm — L11 re-derives this exact pair by a very different route.

== The code agrees, digit for digit

The ten-line `Dual` class, on the two-variable $f$ — same seeds, same numbers:

#codebox[```python
def f(x, y):
    return (x + y) * x * x      # ordinary code — no rewrite needed

print(f(Dual(3.0, 1.0), Dual(1.0, 0.0)).d)   # 33.0   ∂f/∂x  (seed x)
print(f(Dual(3.0, 0.0), Dual(1.0, 1.0)).d)   # 9.0    ∂f/∂y  (seed y)
```]

#pause
No `h` anywhere in this program — so nothing to tune, no U-curve, no floor. The trichotomy's third road is *real*.

#pause
#notebox[For comparison, central differences at a near-optimal $h$ return approximately $32.99999999$. `Dual` returns the floating-point result of the derivative arithmetic, nearest to $33$ in this example.]

== Forward mode in the wild

Real frameworks ship exactly this trick (as "jvp" — more in a moment):

#codebox[```python
import jax
f = lambda x, y: (x + y) * x**2
val, dfdx = jax.jvp(f, (3.0, 1.0), (1.0, 0.0))   # value 36.0, tangent 33.0
```]

#pause
- The seed needn't be one-hot: seeding $(v_1, v_2)$ computes $v_1 partial_x f + v_2 partial_y f = nabla f dot bold(v)$ — a *directional derivative* (L8 callback!)
#pause
- For vector-valued $f$, that dot product becomes $J bold(v)$ (L9's Jacobian): a *Jacobian–vector product*, JVP — computed without ever building $J$

== Forward mode requires one pass per input direction

The finite-difference approximation is gone. Now count passes for the function shape used in ML:

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Function shape*], [*Forward-mode passes*], [*Preferred mode*]),
  [$f: RR arrow.r RR^m$ (one knob, many outputs)], [#text(fill: GREEN)[*1*]], [#text(fill: GREEN)[perfect]],
  [$f: RR^n arrow.r RR$ (a *loss*: $n = 10^9$ knobs, one number)], [#text(fill: RED)[*$n$* — one per seed]], [#text(fill: RED)[330 years again]],
)

#pause
One seed vector is propagated per pass. Computing every coordinate of a scalar-output gradient still requires $n$ forward-mode passes.

#pause
#alertbox[Forward mode avoids finite-difference error, but a full gradient of $f: RR^n arrow.r RR$ still needs one pass per input direction.]

== Checkpoint: dual arithmetic #Q

#mcq([In dual arithmetic, $(2 + epsilon)^3$ evaluates to…],
  [$8$],
  [$8 + 3 epsilon$],
  [$8 + 12 epsilon$],
  [$8 + 12 epsilon + 6 epsilon^2$],
)

== Answer: dual arithmetic #A

#mcq-answer([C], [$8 + 12 epsilon$],
  [$(2 + epsilon)^2 = 4 + 4 epsilon$; then $(4 + 4 epsilon)(2 + epsilon) = 8 + 4 epsilon + 8 epsilon = 8 + 12 epsilon$. The $epsilon$-slot holds $12 = 3 dot 2^2$ — exactly $(x^3)'$ at $x = 2$. Option D forgot the decree: any $epsilon^2$ term is *already* zero, not "small".])

= From forward mode to reverse mode

== `gradcheck`, re-read with today's eyes

#codebox[```python
gradcheck(my_op, (x,))    # float64 recommended; default eps=1e-6; central formula
```]

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7.5pt,
  table.header([*Design choice*], [*Which is exactly…*]),
  [central, not forward], [$O(h^2)$ truncation — one order better, same evaluations],
  [float64 recommended], [float32's finite-difference floor is often too high for the default tolerances],
  [eps = 1e-6], [central's float64 sweet spot $h^* tilde epsilon_64^(1\/3)$ — the U-curve's flat bottom],
  [tests, not training], [$2n$ evaluations: affordable once on a tiny operation, too costly for each training step],
)

#pause
#result[Gradient checking compares an analytical/autodiff gradient with an independent finite-difference approximation.]

== Method comparison

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Method*], [*Assessment*], [*Cost for $nabla f$, $f: RR^n arrow.r RR$*]),
  [symbolic], [exact, but expressions explode], [—],
  [finite differences], [truncation vs rounding — both lose], [$n$ evals, *approximate*],
  [forward-mode AD], [#text(fill: GREEN)[no finite-difference approximation; carries (value, derivative) pairs]], [#text(fill: RED)[$n$ *passes* — one per input]],
  [#text(fill: MUTED)[reverse-mode AD]], [#text(fill: MUTED)[L11]], [#text(fill: MUTED)[?]],
)

#pause
Forward mode is efficient for few-input, many-output functions. For a many-input, scalar loss, reverse mode has the favorable pass count.

== Look at the shape of ML's function

#funnel

#pause
- Forward mode seeds an *input* → the wide end: $10^9$ seeds, $10^9$ passes
#pause
- But the narrow end has exactly *one* node — the loss…
#pause

#result[Reverse mode seeds the scalar output and propagates adjoints backward through the graph, producing all input partial derivatives in one reverse sweep. L11 derives it.]

== Interactive: three roads on one screen #I

#interbox(link-to: IA + "autograd")[
  The *autograd* article puts symbolic, finite-difference and autodiff methods side by side on a live computation graph. Rebuild today's $f = (x + y) dot x^2$, vary $h$ to see the U-curve, then switch to the graph walk and recover $33$ and $9$ without an $h$ parameter.
]

#pause
#notebox[Ten minutes before the next lecture: in the same article, press "reverse" on the graph walk and watch *which direction* the numbers travel. You'll have guessed most of L11.]

== Practice problems

Try on paper; verify in the T5 notebook (it pairs with L10–L11).

+ Estimate $dif / (dif x) sin x$ at $0$ with $h = 0.1$, forward and central. Both beat their advertised orders — which Taylor term vanished?
+ ⭐ redo for central: minimize $E(h) = M h^2 \/ 6 + epsilon \/ h$; check $h^*$ and floor against the figure.
+ Your GPU is float16 ($epsilon approx 10^(-3)$). Can *any* $h$ pass a $10^(-5)$-tolerance gradient check? Use the $sqrt(epsilon)$ rule.
+ Find $c + d epsilon$ with $(a + b epsilon)(c + d epsilon) = 1$; match your $d$ to $(1\/x)'$.
+ Trace forward mode on $g(x, y) = x y + x$ at $(2, 3)$, both seeds. Check: $partial g\/partial x = y + 1$, $partial g\/partial y = x$.
+ $n = 10^7$ parameters, 10 ms per pass: time one central-difference gradient, then one forward-mode gradient. (Both answers should make you want L11.)

== ⭐⭐⭐ Complex-step differentiation #OPT

Complex-step differentiation provides a related way to avoid subtractive cancellation:

$ f(x + i h) = f(x) + i h f'(x) - h^2 / 2 f''(x) - i h^3 / 6 f''' + dots quad arrow.r.double quad f'(x) = ("Im" f(x + i h)) / h + O(h^2) $

#pause
The imaginary part starts at *zero* — extracting it subtracts nothing. No cancellation → no left wall → $h$ can be absurdly small:

#codebox[```python
>>> np.imag(np.exp(0 + 1e-200j)) / 1e-200
1.0                                # exact — with h = 10⁻²⁰⁰ (!)
```]

#pause
$i^2 = -1$ only leaks into the *real* part at $O(h^2)$ — harmless. Dual numbers go one step further: $epsilon^2 = 0$ kills the leak entirely. (Squire & Trapp, 1998 — beloved in engineering codes that predate autodiff.)

== Lecture 10 — summary

- *Three approaches*: symbolic (may swell) · numerical (approximate) · automatic (chain rule on code).
- *Finite-difference error*: truncation $M h\/2$ (L7) vs rounding $2 epsilon\/h$ (L2) — the U-curve and its floor.
- ⭐ $h^* approx sqrt(epsilon)$, best error $approx sqrt(epsilon)$ — *half your digits*; central improves to $epsilon^(2\/3)$. Hence `gradcheck`: float64 · central · `eps = 1e-6`.
- *The wall*: one nudge per input → $n + 1$ evals per gradient — \~330 years/step for GPT-3.
- *Forward mode*: $epsilon^2 = 0$ → $f(a + b epsilon) = f(a) + f'(a) b epsilon$ for the lifted primitives; sweep (value, tangent) once per *input* seed — $nabla f(3, 1) = (33, 9)$ in two passes.

#pause
#notebox[*Read before L11* — MML §5.6 (start); Baydin et al., _AD in ML: a Survey_, §2–3 (skim — today's trichotomy is their map). *T5* drills the dual trace, then L11's backward walk on the *same* graph.]

#focus-slide[
  Finite differences balance truncation against floating-point rounding.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Reverse Mode & Backprop* — reverse the arrows: one pass for *all* inputs.
]
