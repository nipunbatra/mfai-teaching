// Univariate Calculus & Taylor Series — Lecture 7 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture7/L7-calculus-taylor.typ
//   typst compile --root . --input handout=true lecture7/L7-calculus-taylor.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Univariate Calculus & Taylor Series],
  subtitle: [Every smooth function is locally a line],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the lecture's wiggly loss (same closed form as diagrams/l7_figs.py) ──
// L(θ) = 0.25 θ⁴ − θ² + 0.2 sin 5θ + 2; Newton-refined constants:
#let wig(t) = 0.25 * calc.pow(t, 4) - t * t + 0.2 * calc.sin(5 * t) + 2
#let TSTAR = -1.4969867          // global minimizer θ*
#let LSTAR = 0.8279823           // L(θ*)
#let KSTAR = 9.3862635           // L''(θ*)

// ── Taylor partial sums used by several figures ──
#let cosT2(x) = 1 - x * x / 2
#let cosT4(x) = cosT2(x) + calc.pow(x, 4) / 24
#let cosT6(x) = cosT4(x) - calc.pow(x, 6) / 720
#let logS(n) = x => range(1, n + 1).map(k => calc.pow(-1, k + 1) * calc.pow(x, k) / k).sum()

// ── the zoom-triptych function: f(x) = sin 2x + x/2 at x₀ = 0.4 ──
#let zf(x) = calc.sin(2 * x) + 0.5 * x
#let ZF0 = 0.9173560909          // f(0.4)
#let ZS0 = 1.8934134186          // f'(0.4) = 2 cos 0.8 + 0.5
#let ztan(x) = ZF0 + ZS0 * (x - 0.4)

// ── the numeric chain of the lecture, drawn once (fletcher, house style) ──
#let chain-diagram(ml: false) = align(center, diagram(
  spacing: (31mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    let (a, b, c) = if ml { ($w$, $hat(y)$, $cal(L)$) } else { ($x$, $u$, $y$) }
    let (l1, l2, l3) = if ml {
      (([$hat(y) = f(w)$], [model]), ([$cal(L) = ell(hat(y))$], [loss]),
       [$(dif cal(L))/(dif w) = (dif cal(L))/(dif hat(y)) dot (dif hat(y))/(dif w)$])
    } else {
      (([$u = 3x + 1$], [ratio $3$]), ([$y = u^2$], [ratio $14$]), [end to end: $3 times 14 = 42$])
    }
    let two-line(top, bot) = align(center, stack(spacing: 3pt,
      text(size: 13pt, fill: TEAL, top), text(size: 12pt, fill: MUTED, bot)))
    node((0, 0), a, radius: 6mm)
    node((1, 0), b, radius: 6mm, stroke: 0.9pt + TEAL)
    node((2, 0), c, radius: 6mm, stroke: 0.9pt + ACC)
    edge((0, 0), (1, 0), two-line(l1.at(0), l1.at(1)), "-|>", stroke: 1.1pt + TEAL, label-sep: 6pt)
    edge((1, 0), (2, 0), two-line(l2.at(0), l2.at(1)), "-|>", stroke: 1.1pt + TEAL, label-sep: 6pt)
    edge((2, 0), (0, 0), "-|>", bend: 30deg, stroke: 1.25pt + ACC,
      label: text(size: 13pt, fill: ACC, l3), label-sep: 8pt)
  }))

#title-slide()

= The hook: every loss curve is secretly a parabola

== Zoom in on any loss curve #V

#fig("/lecture7/figures/loss_zoom.svg", w: 98%)

#pause
Wiggly from far away. Zoom at the minimum: a *parabola* appears. Zoom anywhere else: a *line*. Try any smooth loss — same two shapes, every time.

#pause
#notebox[*Guess before we continue:* why a _line_ almost everywhere, but a _parabola_ exactly at the bottom? The answer is one formula — you'll derive it today.]

== Today's one idea

#result[A derivative is the *best local linear replacement* of a function. Taylor series is the systematic upgrade to *polynomial* replacements.]

#pause
Why ML cares: training never sees the whole loss — only a *tiny neighbourhood* of the current weights. Local replacements are the *only* map it has.

#pause
Three questions, in order:
+ What is the best local *line*? → the derivative
+ Can we do better than a line? → Taylor's parabola, cubic, …
+ Where does the local story *break*? → convergence radius

== Learning outcomes

By the end of this lecture you will be able to:

+ Read $f'(x_0)$ as a *sensitivity*: $Delta"output"$ per $Delta"input"$ — and estimate it from a numeric table.
+ Use the *local line* $f(x_0 + h) approx f(x_0) + f'(x_0) thin h$ to predict function values.
+ Apply the *chain rule* as ratio bookkeeping through a composition.
+ Classify flat points with the *second derivative* — curvature and its sign.
+ Build the *Taylor polynomial* of a function and bound its error (⭐ $cos x$ at $0$).
+ Explain the *convergence radius* — why a Taylor series can fail far from its base point.

= A derivative is a sensitivity

== Nudge the input, watch the output

Take $f(x) = x^2$ at $x_0 = 3$, so $f(3) = 9$. Nudge $x$ and *measure the response*:

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*nudge* $h$], [$f(3 + h)$], [*response* $Delta f$], [*ratio* $Delta f slash h$]),
  [$1$], [$16$], [$7$], [$7$],
  [$0.1$], [$9.61$], [$0.61$], [$6.1$],
  [$0.01$], [$9.0601$], [$0.0601$], [$6.01$],
  [$0.001$], [$9.006001$], [$0.006001$], [$6.001$],
)

#pause
The response shrinks with the nudge — but the *ratio* stabilizes: $7, 6.1, 6.01, 6.001, dots$

#pause
#result[The ratio is heading somewhere: *6*. That destination is the derivative $f'(3)$.]

== The nudge machine, in code

#codebox[```python
f = lambda x: x**2
for h in [1, 0.1, 0.01, 0.001]:
    print(h, (f(3 + h) - f(3)) / h)
# 1       7.0
# 0.1     6.100000000000012
# 0.01    6.009999999999849
# 0.001   6.000999999999479
```]

#pause
*L2 callback* — see the garbage digits (`…012`, `…849`)? That's floating-point rounding leaking into $Delta f slash h$. Shrink $h$ too far and the noise *wins*.

#pause
#notebox[The full war — truncation error vs rounding error in numerical derivatives — is L10's opening act. Today we take the *exact* route instead: limits.]

== The picture: secants tilting into the tangent #V

// computed in-deck: chords of x² through (3, 9), steepness 6 + h
#let par-pts = range(0, 61).map(i => { let x = 0.8 + i * (5.4 - 0.8) / 60; (x, x * x) })
#align(center, lines(
  (par-pts, ((3.0, 9.0), (5.0, 25.0)), ((3.0, 9.0), (4.0, 16.0)), ((1.4, -0.6), (4.7, 19.2))),
  labels: ([$f(x) = x^2$], [secant $h = 2$ · slope $8$], [secant $h = 1$ · slope $7$], [tangent · slope $6$]),
  colors: (INK, MUTED, TEAL, ACC),
  markers: (false, true, true, false),
  points: ((3.0, 9.0, [$x_0 = 3$]),),
  size: (98mm, 50mm), x-label: $x$,
))

#pause
Each row of the table is a *secant line* through $(3, 9)$ and $(3 + h, f(3 + h))$. As $h -> 0$ the secants tilt into a limiting line: the *tangent*.

== Only now, the symbols

The number the ratios approach gets a name:

$ f'(x_0) = lim_(h -> 0) (f(x_0 + h) - f(x_0)) / h $

#pause
- The fraction is the *table*: response over nudge — an average rate across $[x_0, x_0 + h]$.
#pause
- The limit is the *stabilized ratio*: the instantaneous rate _at_ $x_0$.

#pause
#notebox[Notation zoo, all the same object: $f'(x_0)$, #h(6pt) $(dif f)/(dif x) (x_0)$, #h(6pt) $dot(x)$ (physics). From L8 on, $partial$ joins the zoo.]

== Read derivatives as sensitivities

$f'(x_0)$ answers one question: *if the input moves a little, how hard does the output move?*

#table(
  columns: (1fr, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*function*], [$f'$], [*meaning*]),
  [position $arrow.r$ time], [$approx 20$ m/s], [speed],
  [revenue $arrow.r$ price], [$-300$ ₹/₹], [raise price ₹1, lose ₹300],
  [loss $arrow.r$ one weight], [$-0.8$], [raise $w$ by $epsilon$, loss drops $0.8 epsilon$],
)

#pause
The last row is the *only number training ever consults* about a weight — millions of times per run (L17).

#pause
#result[$f'(x_0) approx (Delta "output") / (Delta "input")$ — a plain ratio of small changes. Everything else in this course is bookkeeping around that ratio.]

== Interactive: watch the limit happen #I

#interbox(link-to: IA + "derivative-tangent")[
  Drag the base point along $x^2$, $sin x$, or $x^3 - x$; shrink the secant step $h$ and watch the average slope converge onto the derivative — then zoom until the curve straightens into its tangent line.
]

#pause
The two sliders are exactly today's two sections: *secant $arrow.r$ tangent* (the limit) and *zoom $arrow.r$ straight* (the local line).

= A derivative is a local line

== Zoom in: every smooth curve straightens #V

$f(x) = sin 2x + x slash 2$ near $x_0 = 0.4$ — same curve, same dashed tangent, three window widths:

#align(center, stack(dir: ltr, spacing: 5mm,
  lines(fn: (zf, ztan), domain: (-1.6, 2.4), colors: (INK, ACC), dashes: ("solid", "dashed"),
    markers: false, y-ticks: false, size: (36mm, 32mm), title: [width $4$]),
  lines(fn: (zf, ztan), domain: (-0.1, 0.9), colors: (INK, ACC), dashes: ("solid", "dashed"),
    markers: false, y-ticks: false, size: (36mm, 32mm), title: [width $1$ · zoom $times 4$]),
  lines(fn: (zf, ztan), domain: (0.28, 0.52), colors: (INK, ACC), dashes: ("solid", "dashed"),
    markers: false, y-ticks: false, size: (36mm, 32mm), title: [width $0.24$ · zoom $times 17$]),
))

#pause
At zoom $times 17$ the curve and its tangent are *indistinguishable*. "Smooth" _means_ this: zoom far enough and there is only one line left.

== The local replacement formula

Rearrange the derivative's definition: for small $h$, the ratio $approx f'(x_0)$, so

$ f(x_0 + h) thin approx thin underbrace(f(x_0), "where you are") + underbrace(f'(x_0), "sensitivity") dot h $

#pause
This is a *trade*: swap a function you can't handle for the line you can.

#pause
#result[Near $x_0$, *replace* $f$ by the line through $(x_0, f(x_0))$ with slope $f'(x_0)$. That replacement is what "derivative" is _for_.]

== Test drive: $sqrt(4.1)$ with no calculator

$f(x) = sqrt(x)$, base point $x_0 = 4$ (where life is easy): $f(4) = 2$, $f'(x) = 1 slash (2 sqrt(x))$, so $f'(4) = 0.25$.

#pause
$ sqrt(4.1) approx 2 + 0.25 times 0.1 = 2.025 $

#pause
Truth: $sqrt(4.1) = 2.02485dots$ — the line is off by $0.00015$, about *0.008%*, for zero effort.

#pause
#notebox[This is how calculators, game engines, and hand computers before them evaluate functions: keep a few exact anchor points, *replace by lines (or short polynomials) in between*.]

== What "best line" means

Many lines pass through $(x_0, f(x_0))$. The tangent is the one whose error *dies faster than $h$*. For $f(x) = x^2$ at $3$, the tangent's error is exactly $h^2$:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([$h$], [error $= f(3 + h) - (9 + 6h)$], [error$thin slash thin h$]),
  [$0.1$], [$0.01$], [$0.1$],
  [$0.01$], [$0.0001$], [$0.01$],
  [$0.001$], [$0.000001$], [$0.001$],
)

#pause
Shrink $h$ by $10 times$ #sym.arrow.r error shrinks $100 times$; the *relative* error vanishes.

#pause
#result[Pick any other slope $m eq.not 6$: the error stays $approx (6 - m) h$ — proportional to $h$, never negligible. *Only the tangent's error is superlinear.* That is the sense in which it is "best".]

== Checkpoint: predict with a derivative #Q

#mcq([A sensor function satisfies $f(2) = 5$ and $f'(2) = -3$. Best quick estimate of $f(2.1)$?],
  [$4.7$],
  [$5.3$],
  [$4.97$],
  [$2.0$],
)

== Answer: predict with a derivative #A

#mcq-answer([A], [$f(2.1) approx 4.7$],
  [Local line: $f(2 + 0.1) approx f(2) + f'(2) times 0.1 = 5 + (-3)(0.1) = 4.7$. B flips the sign of the sensitivity; C uses $h = 0.01$; D subtracts $f'$ itself instead of $f' dot h$ — a units crime: sensitivity must be _scaled by the nudge_.])

= Rules, and the chain that runs ML

== The rules you already know

School calculus, on one slide — this course *refreshes*, it does not reteach:

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  [$ (x^n)' = n x^(n-1) $], [$ (e^x)' = e^x $], [$ (ln x)' = 1 slash x $],
  [$ (sin x)' = cos x $], [$ (cos x)' = -sin x $], [$ (a f + b g)' = a f' + b g' $],
  [$ (f g)' = f' g + f g' $], [$ (f slash g)' = (f' g - f g') / g^2 $], [$ (f(g(x)))' = f'(g(x)) thin g'(x) $],
)

#pause
Only the last cell carries today's weight. Every loss you will ever differentiate is a *composition* — so the chain rule gets its own treatment.

== The chain rule is ratio bookkeeping

$y = (3x + 1)^2$ — a two-stage pipeline $x -> u = 3x + 1 -> y = u^2$. Nudge $x = 2$ by $Delta x = 0.01$ and *track the nudge through the pipeline*:

#pause
#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*stage*], [*value*], [*nudge arriving*], [*local ratio*]),
  [$x$], [$2 -> 2.01$], [$Delta x = 0.01$], [—],
  [$u = 3x + 1$], [$7 -> 7.03$], [$Delta u = 0.03$], [$Delta u slash Delta x = 3$],
  [$y = u^2$], [$49 -> 49.4209$], [$Delta y = 0.4209$], [$Delta y slash Delta u = 14.03$],
)

#pause
End-to-end sensitivity: $Delta y slash Delta x = 42.09 approx 3 times 14.03$. Each stage *multiplies the nudge by its local ratio* — the bookkeeping is just: multiply the ratios.

== The chain rule, in symbols

#chain-diagram()

$ (dif y)/(dif x) = (dif y)/(dif u) dot (dif u)/(dif x) quad quad "here:" quad 2u dot 3 = 2(7) dot 3 = 42 $

#pause
In Leibniz notation the $dif u$'s *visually cancel* — that's the ratio bookkeeping made typographical.

#pause
#alertbox[Evaluate the outer ratio *at the inner value*: $dif y slash dif u = 2u$ is read at $u = 7$, not at $x = 2$. Forgetting this is the classic chain-rule bug.]

== Why ML is one long chain

#chain-diagram(ml: true)

#pause
A real network is this picture *a hundred stages deep*: weights $->$ layer $->$ layer $-> dots ->$ loss. Training needs $dif cal(L) slash dif w$ for *every* weight.

#pause
#result[Same rule, industrial scale: multiply local ratios along the pipeline. Doing that bookkeeping _efficiently_ for millions of weights is backpropagation — L10–L11.]

= Flat points and curvature

== Where the local line goes flat

If the best local line has *slope zero*, nudging cannot help you (to first order): $x_0$ is a *critical point*, candidate max or min. Worked: $f(x) = x^3 - 3x$:

$ f'(x) = 3x^2 - 3 = 0 quad arrow.r.double quad x = -1 "or" x = 1 $

#align(center, lines(fn: x => calc.pow(x, 3) - 3 * x, domain: (-2.5, 2.5),
  markers: false, colors: (INK,),
  points: ((-1.0, 2.0, [flat: max]), (1.0, -2.0, [flat: min])),
  size: (92mm, 42mm), x-label: $x$, y-label: $f(x)$))

#pause
Both flat points have $f' = 0$ — the first derivative *cannot tell them apart*. We need the next order of information.

== The second derivative is curvature

$f''$ is the derivative _of the slope_: is the local line *tilting up or down* as you pass through?

#align(center, stack(dir: ltr, spacing: 7mm,
  lines(fn: x => x * x, domain: (-1.4, 1.4), markers: false, colors: (GREEN,),
    y-ticks: false, size: (33mm, 29mm), title: [$f'' > 0$ · smile]),
  lines(fn: x => -x * x, domain: (-1.4, 1.4), markers: false, colors: (RED,),
    y-ticks: false, size: (33mm, 29mm), title: [$f'' < 0$ · frown]),
  lines(fn: x => calc.pow(x, 3), domain: (-1.4, 1.4), markers: false, colors: (BLUE,),
    y-ticks: false, size: (33mm, 29mm), title: [$f'' = 0$ · undecided]),
))

#pause
Sign gallery: *smile* holds water (slope increasing), *frown* spills it (slope decreasing), zero curvature at a point settles nothing by itself.

== The second-derivative test

At a critical point ($f'(x_0) = 0$), read the curvature:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*curvature*], [*local shape*], [*verdict*]),
  [$f''(x_0) > 0$], [smile — curve sits _above_ its flat tangent], [local *min*],
  [$f''(x_0) < 0$], [frown — curve sits _below_ its flat tangent], [local *max*],
  [$f''(x_0) = 0$], [test is silent], [look further],
)

#pause
Our worked example: $f'' (x)= 6x$, so $f''(-1) = -6 < 0$ (max ✓) and $f''(1) = 6 > 0$ (min ✓) — matching the picture.

#pause
#notebox[Honest footnote: $f(x) = x^4$ has $f'(0) = f''(0) = 0$, yet $0$ is a perfectly good minimum. "Silent" means _silent_, not "no extremum". (L9 upgrades this test to matrices — same fine print.)]

== Checkpoint: classify the flat point #Q

#mcq([Training halts where $cal(L)'(theta_0) = 0$ and $cal(L)''(theta_0) = 8$. What does the loss look like locally?],
  [a frown — local maximum],
  [a smile: $cal(L)(theta) approx cal(L)(theta_0) + 4 (theta - theta_0)^2$],
  [a smile: $cal(L)(theta) approx cal(L)(theta_0) + 8 (theta - theta_0)^2$],
  [flat in every direction — cannot say more],
)

== Answer: classify the flat point #A

#mcq-answer([B], [local min, $cal(L)(theta_0) + 4(theta - theta_0)^2$],
  [$cal(L)'' > 0$ #sym.arrow.r smile #sym.arrow.r local minimum. The parabola's coefficient is $1/2 cal(L)''(theta_0) = 4$, not $8$ — that factor $1/2$ is Taylor's, and you are about to see exactly where it comes from.])

= Taylor: upgrading the line

== The line is order one — keep going #V

The tangent matches $f$'s *value* and *slope* at $x_0$. Next upgrade: also match the *curvature*. $f = e^x$ at $0$:

#align(center, lines(
  fn: (x => calc.exp(x), x => 1 + x, x => 1 + x + x * x / 2),
  domain: (-2, 1.6), samples: 90, markers: false,
  colors: (INK, TEAL, ACC), dashes: ("solid", "dashed", "dashed"),
  labels: ([$e^x$], [$1 + x$], [$1 + x + x^2 slash 2$]),
  size: (96mm, 48mm), x-label: $x$,
))

#pause
The parabola hugs the curve visibly longer than the line. At $x = 1$: truth $2.718$, line $2$, parabola $2.5$ — each order *halves the miss and more*.

== The matching game

*Order-$n$ Taylor polynomial*: the polynomial that agrees with $f$ at $x_0$ in value and the first $n$ derivatives — $n + 1$ conditions, $n + 1$ coefficients, one winner.

#pause
Why the coefficient of $(x - x_0)^k$ is $f^((k))(x_0) slash k!$: differentiate $(x - x_0)^k$ exactly $k$ times and the powers rain down:

$ (dif^k) / (dif x^k) thin (x - x_0)^k = k dot (k - 1) dot dots dot 1 = k! quad arrow.r.double quad "coefficient" = f^((k))(x_0) / k! $

#pause
The $k!$ is a *repair factor*: it cancels the rain so the $k$-th derivatives match exactly.

#pause
#notebox[Your checkpoint's $1/2$ was this factor at $k = 2$: coefficient $= f''(x_0) slash 2! = f'' slash 2$.]

== The Taylor formula

$ f(x_0 + h) thin approx thin f(x_0) + f'(x_0) thin h + (f''(x_0)) / 2! thin h^2 + (f'''(x_0)) / 3! thin h^3 + dots + (f^((n))(x_0)) / n! thin h^n $

#pause
- Order $0$: a constant (stay where you are) · order $1$: the local line · order $2$: the local parabola…
#pause
- Each term is a *correction* the previous order couldn't see.

#pause
#result[Taylor's promise: near $x_0$, any smooth function can be *replaced by a polynomial* — and polynomials we can evaluate, minimize, and reason about.]

== ⭐ Second-order Taylor of $cos x$, step 1: harvest #D

Everything Taylor needs about $cos$ at $x_0 = 0$ is three numbers. Harvest them:

#pause
$ f(x) = cos x quad arrow.r quad f(0) = 1 $

#pause
$ f'(x) = -sin x quad arrow.r quad f'(0) = 0 $

#pause
$ f''(x) = -cos x quad arrow.r quad f''(0) = -1 $

#pause
#notebox[Keep harvesting and a pattern locks in: $f''' (0)= sin 0 = 0$, $f^((4))(0) = cos 0 = 1$, then the cycle $1, 0, -1, 0$ repeats forever. Every _odd_ derivative dies at $0$ — remember this.]

== ⭐ Step 2: assemble, then test on numbers #D

Drop the three harvested numbers into the formula:

$ cos x thin approx thin 1 + 0 dot x + (-1) / 2 thin x^2 quad = quad 1 - x^2 / 2 $

#pause
Test at $x = 0.5$, where we can check everything by hand:

$ 1 - (0.25) / 2 = 0.875 quad "vs" quad cos(0.5) = 0.87758dots quad arrow.r quad "error" = 0.00258 $

#pause
Now the suspicious part. The *next* Taylor term would be $x^4 slash 24$, and at $x = 0.5$ that term equals $(0.5)^4 slash 24 = 0.00260$.

#pause
#result[The error *is* the first term we dropped, almost exactly. That is Taylor's error story in one number.]

== ⭐ Step 3: the error, visualized #D #V

$abs(cos x - (1 - x^2 slash 2))$ against the dropped term $x^4 slash 24$ — log scale, six decades:

#align(center, lines(
  fn: (x => calc.abs(calc.cos(x) - (1 - x * x / 2)), x => calc.pow(x, 4) / 24),
  domain: (0.1, 3), samples: 120, markers: false, log-y: true,
  colors: (ACC, MUTED), dashes: ("solid", "dashed"),
  points: ((0.5, 0.00258, [$x = 0.5$]), (1.0, 0.0403, [$x = 1$])),
  size: (100mm, 46mm), x-label: $x$,
))
#align(center, text(size: 16pt)[#text(fill: ACC)[error of $1 - x^2 slash 2$] · #text(fill: MUTED)[dashed: $x^4 slash 24$] — glued together until $x approx 1.5$])

#pause
#result[Quartic error: halve the distance $x$ #sym.arrow.r the error falls $2^4 = 16 times$. Second-order replacements are _absurdly_ accurate close in.]

== The full build-up: $cos x$, order by order #V

#align(center, lines(
  fn: (x => calc.cos(x), cosT2, cosT4, cosT6),
  domain: (-calc.pi, calc.pi), samples: 140, markers: false,
  colors: (INK, TEAL, GREEN, ACC),
  hlines: ((1.0, [orders 0 & 1: the flat line up here], MUTED),),
  size: (118mm, 52mm), x-label: $x$,
))
#align(center, text(size: 16pt)[#text(fill: INK)[$cos x$] · #text(fill: MUTED)[orders 0 & 1] · #text(fill: TEAL)[order 2] · #text(fill: GREEN)[order 4] · #text(fill: ACC)[order 6]])

#pause
Orders $0$ and $1$ *coincide* — the slope at $0$ is zero, so the best line is flat (every odd derivative died in step 1).

#pause
Each even order hugs the curve further out: the replacement's *trust region grows* with the order.

== The build-up in code

#codebox[```python
from math import cos, factorial
def T(x, n):                       # order-n Taylor of cos at 0
    return sum((-1)**k * x**(2*k) / factorial(2*k)
               for k in range(n // 2 + 1))
for n in [0, 2, 4, 6]:
    print(n, T(1.0, n), abs(cos(1.0) - T(1.0, n)))
# 0  1.0                 0.4597
# 2  0.5                 0.0403
# 4  0.5416666666666666  0.00136
# 6  0.5402777777777777  2.45e-05
```]

#pause
At $x = 1$, every two orders buy roughly a factor *30* of accuracy. Six terms of grade-school arithmetic already give $cos(1)$ to $4$ digits — this loop _is_ how libraries compute $cos$.

== Checkpoint: how fast does the error die? #Q

#mcq([The order-2 error of $cos$ at $x = 0.2$ is about $6.7 times 10^(-5)$. Doubling to $x = 0.4$ makes the error roughly…],
  [$2 times$ bigger],
  [$4 times$ bigger],
  [$8 times$ bigger],
  [$16 times$ bigger],
)

== Answer: how fast does the error die? #A

#mcq-answer([D], [$16 times$ — quartic scaling],
  [Error $approx x^4 slash 24$, so doubling $x$ multiplies it by $2^4 = 16$. (Check: at $0.4$ the true error is $1.06 times 10^(-3) approx 15.9 times$ bigger.) General rule: drop terms above order $n$, and the error scales like $h^(n+1)$.])

= Where the local story breaks

== $log(1 + x)$: a series with a boundary

Same game, new function: $ln(1 + x)$ at $0$ has the series $x - x^2/2 + x^3/3 - x^4/4 + dots$ #h(4pt) Watch its partial sums at two inputs:

#pause
#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([], [$S_1$], [$S_2$], [$S_3$], [$S_4$], [$S_5$], [$S_6$], [*truth*]),
  [$x = 0.5$], [$0.5$], [$0.375$], [$0.417$], [$0.401$], [$0.407$], [$0.405$], [$ln 1.5 = 0.405$ #text(fill: GREEN)[✓]],
  [$x = 2$], [$2$], [$0$], [$2.67$], [$-1.33$], [$5.07$], [$-5.6$], [$ln 3 = 1.099$ #text(fill: RED)[*never*]],
)

#pause
At $x = 0.5$ the sums settle beautifully. At $x = 2$ they *oscillate harder with every term* — the series diverges, even though $ln 3$ is a perfectly innocent number.

== Watch it break #V

#align(center, lines(
  fn: (x => calc.ln(1 + x), logS(4), logS(9)),
  domain: (-0.9, 1.6), samples: 120, markers: false,
  colors: (INK, TEAL, ACC),
  labels: ([$ln(1 + x)$], [order 4], [order 9]),
  vlines: ((1.0, [$x = 1$], RED),),
  size: (108mm, 50mm), x-label: $x$,
))

#pause
Inside $abs(x) < 1$: higher order = better, everywhere. Past $x = 1$ the orders *disagree violently* — order 9 rockets up, order 4 dives — while the truth ambles along.

== The radius of convergence

Why $1$? The series was built entirely from data *at $x = 0$* — and $ln(1 + x)$ has a disaster at $x = -1$ (it plunges to $-oo$).

#pause
#result[A Taylor series converges inside a radius equal to the distance from $x_0$ to the *nearest disaster* — and diverges beyond it, in _every_ direction.]

#pause
#alertbox[The direction doesn't matter! The blow-up lives at $-1$, yet the series also fails at $+2$ — a point where the function itself is perfectly finite. The local story simply *cannot see* past its radius.]

#pause
#notebox[Intuition to carry: a Taylor series is *local testimony*. Near the base point it is superb evidence; past the radius it is no evidence at all.]

== And $cos$ never breaks

$cos$ has no disaster anywhere — its radius is $oo$. The mechanism: *factorials beat powers*. Terms $x^k slash k!$ at the distant point $x = 10$:

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([$k$], [$10$], [$20$], [$30$], [$40$]),
  [$10^k slash k!$], [$2756$], [$41$], [$0.0038$], [$1.2 times 10^(-8)$],
)

#pause
The terms *rise first* — then $k!$ catches up and crushes them. Enough terms deliver $cos(10)$ to any accuracy you like.

#pause
#notebox[So both endings exist: $e^x$, $sin$, $cos$ converge everywhere; $ln(1 + x)$ and $1/(1 - x)$ have a finite horizon. Always ask _where_ a local replacement is licensed — in ML we stay close to $x_0$, which is why Taylor serves us so well.]

= What this buys ML

== The hook, resolved

Zoom near a minimum $theta^*$ and Taylor-expand the loss there:

$ cal(L)(theta) thin approx thin cal(L)(theta^*) + underbrace(cal(L)'(theta^*), = thin 0 "at a minimum") (theta - theta^*) + 1/2 cal(L)''(theta^*) (theta - theta^*)^2 $

#pause
At the bottom, the *line term is dead* — the first surviving correction is the quadratic. That is why every smooth loss bottoms out as a parabola.

#pause
Anywhere else, $cal(L)' eq.not 0$: the line term dominates all others for small steps — which is why zooming elsewhere gave a *line*.

#pause
#result[Wiggly $->$ parabola at minima, line elsewhere: not a coincidence — it's Taylor, order by order.]

== See it on today's loss #V

Our hook's loss near its minimum ($theta^* = -1.497$, $cal(L)^* = 0.828$, $cal(L)'' (theta^*) = 9.39$) against the pure parabola $cal(L)^* + 4.69 thin (theta - theta^*)^2$:

#align(center, lines(
  fn: (wig, t => LSTAR + 0.5 * KSTAR * calc.pow(t - TSTAR, 2)),
  domain: (TSTAR - 0.7, TSTAR + 0.7), samples: 120, markers: false,
  colors: (INK, ACC), dashes: ("solid", "dashed"),
  labels: ([$cal(L)(theta)$], [order-2 Taylor]),
  points: ((TSTAR, LSTAR, [$theta^*$]),),
  size: (100mm, 48mm), x-label: $theta$,
))

#pause
Near $theta^*$ the parabola is the loss, for every practical purpose. Optimizers exploit exactly this — next slide's promissory note.

== A promissory note: where L7 gets cashed

Continuity ledger — *planted today, paid off later*:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*lecture*], [*what it does with today's Taylor*]),
  [L8–L9], [slope $->$ gradient, curvature $->$ Hessian: same formulas, more dimensions],
  [L16], [convexity: the smile never flips — one global basin, no bad minima],
  [L17], [*gradient descent derived*: trust the line, step downhill],
  [L18], [*Newton's method derived*: trust the parabola, jump to its bottom],
)

#pause
#result[When L17 writes $theta_(t+1) = theta_t - eta cal(L)'(theta_t)$, it is quoting today's local line; when L18 divides by $cal(L)''$, it is quoting today's parabola.]

= Summary & what's next

== Lecture 7 — summary

- *Derivative = sensitivity*: the ratio $Delta f slash h$ stabilizes — $7, 6.1, 6.01, dots -> 6$.
- *Local line*: $f(x_0 + h) approx f(x_0) + f'(x_0) h$ — the only error that beats $h$.
- *Chain rule*: pipelines multiply local ratios ($3 times 14 = 42$); ML is a deep pipeline.
- *Flat + curvature*: $f' = 0$ finds candidates; the sign of $f''$ classifies them — usually.
- *Taylor*: match $n$ derivatives, repair by $k!$; ⭐ $cos x approx 1 - x^2 slash 2$, error $approx x^4 slash 24$.
- *Radius*: local testimony expires at the nearest disaster ($ln(1 + x)$ past $abs(x) = 1$; $cos$ never).
- *Payoff*: at minima the line term dies — every smooth loss bottoms out as a parabola (L16–L18).

#pause
#notebox[*Watch:* 3b1b _Essence of Calculus_ ch. 1–4 + the Taylor video. *Read before L8:* MML §5.1. *T4* (after L8): derivative and chain-rule drills in NumPy.]

== Practice problems

Try on paper; verify against the T4 notebook.

+ Build the nudge table for $f(x) = 1 slash x$ at $x_0 = 2$ with $h = 0.1, 0.01, 0.001$. What is the ratio heading toward?
+ Estimate $e^(0.1)$ with the local line at $0$, then with the order-2 Taylor. Compare both to $1.10517dots$
+ For $y = (2x - 1)^3$ at $x = 1$: tabulate $Delta x -> Delta u -> Delta y$ for $Delta x = 0.01$ and verify the local ratios multiply to $dif y slash dif x = 6$.
+ Find and classify every critical point of $f(x) = x^3 - 6x^2 + 9x + 1$.
+ Derive the order-3 Taylor of $sin x$ at $0$, evaluate it at $x = 0.3$, and bound the error by the first dropped term. Why do only odd powers appear?
+ The $ln(1 + x)$ series diverges at $x = 2$ although $ln 3$ is finite. In one sentence, what sets the radius, and where is the offending point?

== ⭐⭐⭐ The function Taylor cannot see #OPT

$f(x) = e^(-1 slash x^2)$ (with $f(0) = 0$) is *smooth everywhere* — yet at $0$, _every_ derivative equals $0$:

#align(center, lines(
  fn: x => if calc.abs(x) < 1e-6 { 0.0 } else { calc.exp(-1 / (x * x)) },
  domain: (-1.6, 1.6), samples: 120, markers: false, colors: (INK,),
  size: (84mm, 30mm), x-label: $x$,
))

#pause
Its Taylor series at $0$ is $0 + 0 x + 0 x^2 + dots equiv 0$ — it converges beautifully, *to the wrong function*. The curve is flatter at $0$ than any polynomial; Taylor is blind to it.

#pause
#notebox[Moral: _smooth_ #sym.eq.not _analytic_ (faithfully told by its Taylor series). Our losses never hit the distinction — but honest mathematics keeps the receipt.]

#focus-slide[
  Calculus is the art of replacing a function with a line.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Gradients & the Geometry of Surfaces* — the same replacement idea, in more dimensions: the line becomes a plane, the slope becomes a vector.
]
