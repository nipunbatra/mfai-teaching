// Gradients & the Geometry of Surfaces — Lecture 8 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture8/L8-gradients.typ
//   typst compile --root . --input handout=true lecture8/L8-gradients.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Gradients & the Geometry of Surfaces],
  subtitle: [The arrow that points straight uphill],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the workhorse bowl of the lecture: f(x, y) = x² + 3y², base point (2, 1) ──
// One string, written once — autodiff supplies every gradient a figure needs.
#let fbowl = ad.expr("x^2 + 3*y^2", ("x", "y"))
#let GRADP = ad.grad(fbowl, (2, 1))              // (4, 6), exactly
#let GNORM = calc.sqrt(GRADP.map(g => g * g).sum())   // ‖∇f(2,1)‖ = √52

// a "gradient pin": a short 2-point path from p along the EXACT gradient at p
#let pin(p, s: 0.075) = {
  let g = ad.grad(fbowl, p)
  (p, (p.at(0) + s * g.at(0), p.at(1) + s * g.at(1)))
}

// ── the hook's tiny model: fit y = w·x + b to {(1,2), (2,4)} by least squares ──
#let sse = ad.expr("(2 - w - b)^2 + (4 - 2*w - b)^2", ("w", "b"))
#let sse-path = gd(ad.grad-fn(sse), (-0.5, 2.5), lr: 0.13, steps: 150)
#let sse-losses = sse-path.enumerate().map(((i, p)) => (i, ad.value(sse, p)))

// ── the two-route chain of the lecture, drawn once (fletcher, house style) ──
#let branch-diagram(ml: false) = align(center, diagram(
  spacing: (30mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    let L(body) = text(size: 15pt, fill: TEAL, body)
    if ml {
      node((0, -1), $w_1$, radius: 6mm)
      node((0, 1), $w_2$, radius: 6mm)
      node((1, 0), $hat(y)$, radius: 6mm, stroke: 0.9pt + TEAL)
      node((2, 0), $cal(L)$, radius: 6mm, stroke: 0.9pt + ACC)
      edge((0, -1), (1, 0), L[$(partial hat(y))/(partial w_1)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: left, label-sep: 3pt)
      edge((0, 1), (1, 0), L[$(partial hat(y))/(partial w_2)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: right, label-sep: 3pt)
      edge((1, 0), (2, 0), L[$(dif cal(L))/(dif hat(y))$], "-|>", stroke: 1.1pt + TEAL,
        label-side: left, label-sep: 4pt)
    } else {
      node((0, 0), $t$, radius: 6mm)
      node((1, -1), $x$, radius: 6mm, stroke: 0.9pt + TEAL)
      node((1, 1), $y$, radius: 6mm, stroke: 0.9pt + TEAL)
      node((2, 0), $f$, radius: 6mm, stroke: 0.9pt + ACC)
      edge((0, 0), (1, -1), L[$(dif x)/(dif t)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: left, label-sep: 3pt)
      edge((0, 0), (1, 1), L[$(dif y)/(dif t)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: right, label-sep: 3pt)
      edge((1, -1), (2, 0), L[$(partial f)/(partial x)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: left, label-sep: 3pt)
      edge((1, 1), (2, 0), L[$(partial f)/(partial y)$], "-|>", stroke: 1.1pt + TEAL,
        label-side: right, label-sep: 3pt)
    }
  }))

#title-slide()

= The hook: the picture behind the training curve

== Every training run hides a landscape #V

#fig("/lecture8/figures/loss_landscape.svg", w: 99%)

#pause
You will stare at thousands of the left plot. It is an *altimeter log*: height vs time of a walk. The walk itself happens on the *middle* object — and hikers navigate it with the *right* one.

== Training is a walk you cannot watch

- A real model has *millions* of weights — its loss surface lives in $RR^(10^6)$, unplottable forever.
#pause
- But every mechanism you'll ever need — slopes, contours, steepest directions — already exists, fully visible, at *two* weights.
#pause
- So today is deliberately flat-footed: $f(x, y)$, drawn every way we can.

#pause
#result[Training = walking downhill on a loss surface. The training curve is only the altimeter — today we learn to read the *terrain*.]

== Today's one idea

#result[The *gradient* collects every partial sensitivity into one vector — and that vector points *straight uphill*, perpendicular to the contour lines.]

#pause
Three questions, in order:
+ How do we even *draw* a function of two variables? → surfaces & contour maps
+ How do we *differentiate* it? → partial derivatives, bundled into the gradient
+ Which direction is *steepest*, and why? → the directional derivative ⭐

== Learning outcomes

By the end of this lecture you will be able to:

+ Draw one $f(x, y)$ two ways — *surface* and *contour map* — and translate between them.
+ Read a contour map *like a hiker*: peaks, pits, saddles, steep faces.
+ Compute *partial derivatives* by freezing variables; check them numerically.
+ Assemble the *gradient* $nabla f$ and interpret it as an arrow on the map.
+ Predict the slope in *any* direction from just $nabla f$ — via $nabla f dot bold(u)$.
+ Prove ⭐ that $nabla f$ is *perpendicular to contours* and is the *steepest* direction.
+ Run the *multivariate chain rule* along branching pipelines.

= Surfaces and contour maps

== A function of two inputs is a landscape #V

$f(x, y) = x^2 + 3y^2$ — every map point $(x, y)$ gets one number: its *height*.

#two(r: (48%, 52%),
  align(center, stack(spacing: 5pt,
    surface((x, y) => x*x + 3*y*y, xlim: (-3, 3), ylim: (-2, 2),
      samples: 22, cell: 1.9mm, height: 36mm, x-label: $x$, y-label: $y$),
    text(size: 16pt, fill: MUTED)[the *surface* view: height drawn as height])),
  align(center, stack(spacing: 5pt,
    contour((x, y) => x*x + 3*y*y, xlim: (-3, 3), ylim: (-2, 2),
      samples: 56, levels: 7, size: (72mm, 48mm), color: TEAL,
      marks: ((0, 0, [min], RED),), x-label: $x$, y-label: $y$),
    text(size: 16pt, fill: MUTED)[the *map* view: height drawn as curves])),
)

#pause
Same $f$, two pictures. The rest of the course lives in the right one — so let's earn it.

== Your first loss landscape, built by hand

Fit the line $hat(y) = w x + b$ to two points $(1, 2)$ and $(2, 4)$. The loss is a function of *two* variables — the parameters:

$ cal(L)(w, b) = (2 - (w + b))^2 + (4 - (2w + b))^2 $

#pause
#two(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 8pt,
    table.header([$w$], [$b$], [$cal(L)(w, b)$]),
    [$0$], [$0$], [$4 + 16 = 20$],
    [$1$], [$1$], [$0 + 1 = 1$],
    [$2$], [$1$], [$1 + 1 = 2$],
    [$2$], [$0$], [$0 + 0 = bold(0)$],
  ),
  [
    Each row: guess a line, square the two misses, add.

    #pause
    $(w, b) = (2, 0)$ fits perfectly — the data *is* $y = 2x$.

    #pause
    #text(fill: MUTED)[Keep this $cal(L)$ in mind; it returns at the end of the lecture, with a walker on it.]
  ],
)

== How a contour map is made #V

#fig("/lecture8/figures/surface_to_map.svg", w: 92%)

#pause
Slice the surface at constant heights $f(x, y) = K$; drop each cut edge straight down. A *contour line* (or *level set*) is everything at one height.

== Read the map like a hiker

#two(r: (55%, 45%),
  [
    - *Closed rings shrinking to a dot* — a peak or a pit lives inside.
    #pause
    - *Lines crowded together* — steep face: height changes fast per step.
    #pause
    - *Lines far apart* — gentle plain: many steps to change height.
    #pause
    - *Walking along a line* — you stay exactly level, by definition.
  ],
  align(center, contour((x, y) => x*x + 3*y*y, xlim: (-3, 3), ylim: (-2, 2),
    samples: 56, levels: 7, size: (68mm, 46mm), color: TEAL,
    marks: ((0, 0, [pit], RED),), x-label: $x$, y-label: $y$)),
)

#pause
#result[Equal height *spacing* between lines ⟹ line *crowding* = steepness. A contour map is a slope chart in disguise.]

== The map in code · `meshgrid` + `contour`

#codebox[```python
import numpy as np, matplotlib.pyplot as plt
x = np.linspace(-3, 3, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)      # X[i,j], Y[i,j]: every grid pair
Z = X**2 + 3*Y**2             # f on the whole grid at once
plt.contour(X, Y, Z, levels=10)
plt.gca().set_aspect("equal")
```]

#pause
`meshgrid` tabulates the plane; `contour` interpolates the level curves between grid values. Every loss-landscape picture you have ever seen is these five lines.

#pause
#notebox[T4 walks through `meshgrid` slowly — including the classic confusion about which axis is which.]

== A gallery of terrains #V

#align(center, stack(dir: ltr, spacing: 8mm,
  contour((x, y) => x*x + y*y, xlim: (-2, 2), ylim: (-2, 2), samples: 49, levels: 6,
    size: (46mm, 46mm), color: TEAL, title: [$x^2 + y^2$ · bowl]),
  contour((x, y) => x*x, xlim: (-2, 2), ylim: (-2, 2), samples: 49, levels: 6,
    size: (46mm, 46mm), color: TEAL, title: [$x^2$ · trench]),
  contour((x, y) => x*y, xlim: (-2, 2), ylim: (-2, 2), samples: 71, levels: 8,
    size: (46mm, 46mm), color: TEAL, title: [$x y$ · saddle]),
))

#pause
- *Bowl*: rings around one pit — the ideal loss. #h(8pt) *Trench*: $f$ *ignores* $y$ — a loss that ignores a parameter (a dead weight: contours parallel to its axis).
#pause
- *Saddle*: downhill in two quadrants, uphill in two — no ring closes.

== The saddle deserves a second look #V

#two(r: (46%, 54%),
  align(center, stack(spacing: 5pt,
    heatmap((x, y) => x*y, xlim: (-2, 2), ylim: (-2, 2), samples: 41, cell: 1.5mm,
      contour-lines: 6),
    text(size: 16pt, fill: MUTED)[heat view: #text(fill: ACC)[high] / #text(fill: TEAL)[low]])),
  [
    At the center: flat — yet *neither* a peak nor a pit. Ride $x = y$ and you climb; ride $x = -y$ and you sink.

    #pause
    #alertbox[High-dimensional loss surfaces are *full* of saddle points — flat spots that stall training without being minima. L9 gives you the tool (the Hessian) that tells bowls, saddles, and trenches apart.]
  ],
)

== Level sets you have already met

Contours are not just a plotting trick — they are how ML *draws decisions*:

#pause
- A classifier outputs a probability surface $p(x, y)$ over the plane; its *decision boundary* is precisely the level set $p = 0.5$.
#pause
- Every "decision boundary" plot in ES 335 is `plt.contour(X, Y, P, levels=[0.5])` — the same five lines of code as before.

#pause
#notebox[Same object, two costumes: in *parameter space*, level sets of the loss (today); in *input space*, level sets of the model's output (your next ML course).]

= Partial derivatives: freeze one variable

== Two knobs, but one dial at a time

L7's derivative answers: nudge *the* input, how hard does the output move? Now there are *two* inputs to nudge — and they can move together, or alone.

#pause
The multivariable move (you will use it for the rest of your life):

#result[*Freeze* every variable except one. What remains is a one-variable function — and L7 already taught you everything about those.]

#pause
Freezing has a picture, a number, and a symbol. Picture first.

== Freezing a variable saws the surface open #V

#fig("/lecture8/figures/slice_partial.svg", w: 74%)

#pause
Each freeze exposes a *cut edge* through our base point $(2, 1)$, where $f = 7$ (the dot).

== Each cut edge is an ordinary curve

#align(center, stack(dir: ltr, spacing: 9mm,
  lines(fn: (x => x*x + 3, x => 7 + 4*(x - 2)), domain: (0.4, 3.4),
    colors: (TEAL, ACC), dashes: ("solid", "dashed"), markers: false,
    points: ((2.0, 7.0, [$(2, 7)$]),), size: (62mm, 44mm),
    x-label: $x$, title: [$g(x) = f(x, 1) = x^2 + 3$]),
  lines(fn: (y => 4 + 3*y*y, y => 7 + 6*(y - 1)), domain: (-0.4, 2.2),
    colors: (ACC, TEAL), dashes: ("solid", "dashed"), markers: false,
    points: ((1.0, 7.0, [$(1, 7)$]),), size: (62mm, 44mm),
    x-label: $y$, title: [$h(y) = f(2, y) = 4 + 3y^2$]),
))

#pause
Each cut is a parabola with a *tangent line* at the base point — L7's local-line story, verbatim. Two cuts, two slopes. Let's measure them.

== The nudge table, twice

Base point $(2, 1)$, $f = 7$. Nudge *one* coordinate, hold the other — L7's ritual:

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*move*], [*new value*], [*response*], [*ratio*]),
  [$x: 2 -> 2.01$], [$f(2.01, 1) = 7.0401$], [$0.0401$], [$4.01$],
  [$x: 2 -> 2.001$], [$f(2.001, 1) = 7.004001$], [$0.004001$], [$4.001$],
  [$y: 1 -> 1.01$], [$f(2, 1.01) = 7.0603$], [$0.0603$], [$6.03$],
  [$y: 1 -> 1.001$], [$f(2, 1.001) = 7.006003$], [$0.006003$], [$6.003$],
)

#pause
Two stabilizing ratios: the $x$-ratio heads to $bold(4)$, the $y$-ratio to $bold(6)$.

#pause
#result[At $(2, 1)$ this surface is *4-steep along $x$* and *6-steep along $y$*. Two sensitivities, one point.]

== $partial$ joins the notation zoo

L7 promised "$partial$ joins the zoo in L8". Here it is — the *partial derivative*:

$ (partial f) / (partial x) (x_0, y_0) = lim_(h -> 0) (f(x_0 + h, thin y_0) - f(x_0, y_0)) / h $

#pause
- Curly $partial$ = "I nudged *one* input; every other input is frozen."
#pause
- To compute: *differentiate as usual, treating the frozen variables as constants*:

$ f = x^2 + 3y^2: quad (partial f) / (partial x) = 2x + 0 = 2x, quad (partial f) / (partial y) = 0 + 6y = 6y $

#pause
At $(2, 1)$: $2x = bold(4)$ ✓ and $6y = bold(6)$ ✓ — exactly the two stabilized ratios.

== Partials in code

#codebox[```python
import torch
f = lambda x, y: x**2 + 3*y**2
x0, y0 = torch.tensor(2.0), torch.tensor(1.0)

df_dx = torch.func.grad(f, argnums=0)(x0, y0)   # nudge arg 0 only
df_dy = torch.func.grad(f, argnums=1)(x0, y0)   # nudge arg 1 only
print(df_dx, df_dy)
# tensor(4.)  tensor(6.)
```]

#pause
`argnums` *is* the freeze: differentiate w.r.t. argument 0, hold the rest. And the values are exact — no `0.0401`-style nudge noise. How PyTorch manages that is the whole of L10–L11.

== Checkpoint: read the map like a pro #Q

#mcq([On a loss contour map, a small step in the $w_2$ direction crosses *three* contour lines; the same-length step in the $w_1$ direction stays *between* two lines. At that point:],
  [$abs(partial cal(L) slash partial w_1) > abs(partial cal(L) slash partial w_2)$],
  [$abs(partial cal(L) slash partial w_2) > abs(partial cal(L) slash partial w_1)$],
  [the two partials are equal],
  [$nabla cal(L) = bold(0)$ there],
)

== Answer: read the map like a pro #A

#mcq-answer([B], [the $w_2$ partial is bigger],
  [Contour lines sit at equal height steps — crossing three of them means the height changed three steps' worth over one small move: a big ratio, i.e. a big $abs(partial cal(L) slash partial w_2)$. Staying between lines means barely any change: small $abs(partial cal(L) slash partial w_1)$. D is the opposite of what's described: a zero gradient point has *locally no* crossings in any direction.])

= The gradient: every sensitivity in one arrow

== Bundle the partials

Two sensitivities at every point beg to be one object. Stack them:

$ nabla f (x, y) = mat((partial f) / (partial x); (partial f) / (partial y)) quad quad "here:" quad nabla f = mat(2x; 6y), quad nabla f (2, 1) = mat(4; 6) $

#pause
- $nabla f$ ("nabla f", or *the gradient*) is a *vector-valued* recipe: feed a point, get a vector.
#pause
- It packages *every* first-order fact about $f$ at that point — L7's derivative, grown up.

#pause
#notebox[Honest bookkeeping: MML writes $nabla f$ as a *row* vector (good reasons arrive with the Jacobian). We draw it as a column/arrow today; L9 settles the layout conventions.]

== The gradient is an arrow on the map

#two(r: (52%, 48%),
  align(center, contour(ad.fn2(fbowl), xlim: (-3.4, 3.4), ylim: (-2.1, 2.6),
    samples: 56, levels: (1, 3, 5, 7, 10, 14, 19), size: (74mm, 52mm), color: TEAL,
    paths: (pin((2, 1), s: 0.16),),
    marks: ((2 + 0.16 * GRADP.at(0), 1 + 0.16 * GRADP.at(1), [$nabla f$], GREEN),
            (2, 1, none, RED)), x-label: $x$, y-label: $y$)),
  [
    The pin planted at $(2, 1)$ is $nabla f = (4, 6)$, drawn *in the input plane*.

    #pause
    It leans more along $y$ than $x$ ($6 > 4$) — the surface is steeper that way.

    #pause
    #alertbox[The gradient is *not* an arrow on the mountain. It lives on the *map*, pointing along the ground — "walk that way to climb fastest."]
  ],
)

== Drop an arrow at every point: the gradient field #V

Every arrow is $nabla f$ planted at its base point — three regularities hide here:

#fig("/lecture8/figures/grad_field.svg", w: 56%)

== What the field is telling us

+ Every arrow points *away from the pit* — uphill, toward higher contours.
#pause
+ Every arrow crosses its contour line *at a right angle* — no exceptions, anywhere.
#pause
+ Arrows are *long where lines crowd* (steep!), and shrink to $nabla f = bold(0)$ at the flat bottom — the resting places of gradient descent (L16–L18's obsession).

#pause
#result[Perpendicular to contours, magnitude = steepness. Claim 2 looks like magic — it gets a proof in the next section.]

== The field in code

#codebox[```python
qx, qy = np.meshgrid(np.linspace(-2.7, 2.7, 10),
                     np.linspace(-1.7, 1.7, 8))
U, V = 2*qx, 6*qy                  # ∇f = (2x, 6y) on the grid
plt.contour(X, Y, Z, levels=10)
plt.quiver(qx, qy, U, V)           # arrows over the map
```]

#pause
`quiver` = "a case for arrows". Contours from the *values* of $f$, arrows from its *partials* — the two halves of today, overlaid.

= Which way is steepest?

== A hiker's three questions

Standing at $(2, 1)$ on our surface, gradient in hand — $nabla f = (4, 6)$:

#pause
+ If I walk *north-east* — or any direction I fancy — how fast do I climb?
#pause
+ Which direction climbs *fastest*?
#pause
+ Which directions keep me exactly *level*?

#pause
One dot product will answer all three. First, make question 1 precise — with numbers.

== Walking in a chosen direction

Pick a *unit* direction $bold(u)$, walk along $(2, 1) + t thin bold(u)$, and log altitude: $phi(t) = f((2,1) + t bold(u))$. The slope $phi'(0)$ is the *directional derivative* $D_bold(u) f$. Measure it, nudge-style:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*direction* $bold(u)$], [*measured slope*], [*recognize it?*]),
  [$(1, 0)$ — due east], [$4$], [$partial f slash partial x$ — of course: freezing $y$ *is* walking east],
  [$(0, 1)$ — due north], [$6$], [$partial f slash partial y$],
  [$(0.6, 0.8)$ — north-east-ish], [$7.2$], [$0.6 dot 4 + 0.8 dot 6 = 7.2$ …a *blend* of the two],
)

#pause
#result[The diagonal slope is the partials *weighted by the direction's components*. That pattern is the whole theorem — now we derive it.]

== ⭐ Step 1 · zoom in: the local plane #D

L7: zoom into a smooth *curve* and only a line survives. Zoom into a smooth *surface* and only a *plane* survives — tilted $4$-steep along $x$, $6$-steep along $y$:

$ f(2 + h, thin 1 + k) thin approx thin underbrace(7, f(2,1)) + underbrace(4, partial f slash partial x) thin h + underbrace(6, partial f slash partial y) thin k $

#pause
Why: move in two legs. East by $h$: the $x$-cut's tangent charges $4h$. Then north by $k$: the $y$-cut's tangent charges $6k$. (To first order the second leg's slope hasn't changed — that's what *smooth* buys.)

#pause
*Test it* at $(h, k) = (0.1, 0.05)$: plane says $7 + 0.4 + 0.3 = 7.70$; truth $f(2.1, 1.05) = 7.7175$. Off by $0.0175$ — quadratically small, L7's superlinear story again.

== ⭐ Step 2 · the slope along $bold(u)$ is a dot product #D

Walk the plane in direction $bold(u) = (u_1, u_2)$: set $(h, k) = (t u_1, t u_2)$,

$ phi(t) approx 7 + t thin (4 u_1 + 6 u_2) quad arrow.r.double quad phi'(0) = 4 u_1 + 6 u_2 $

#pause
That coefficient is exactly a dot product — and nothing about $(4, 6)$ was special:

$ D_bold(u) f = nabla f dot bold(u) $

#pause
Check against the measured table: $bold(u) = (0.6, 0.8)$: $(4)(0.6) + (6)(0.8) = 7.2$ ✓. Axes: $(1,0) -> 4$ ✓, $(0,1) -> 6$ ✓.

#pause
#result[One vector, $nabla f$, predicts the slope in *every* direction. That is why the gradient is the only thing an optimizer asks for.]

== The right angle, up close #V

Zoom onto $(2, 1)$: the exact $nabla f$ (green pin) vs the level curve $f = 7$ and its direction (teal chord):

#align(center, contour(ad.fn2(fbowl), xlim: (0.6, 3.6), ylim: (-0.1, 2.5),
  samples: 56, levels: (1, 3, 5, 7, 10, 14, 19, 25), size: (92mm, 68mm), color: TEAL,
  paths: (pin((2, 1), s: 0.16), ((1.25, 1.5), (2.75, 0.5))),
  marks: ((2 + 0.16 * GRADP.at(0), 1 + 0.16 * GRADP.at(1), [$nabla f$], GREEN),
          (2.75, 0.5, [level], TEAL), (2, 1, none, RED)),
  x-label: $x$, y-label: $y$))

#pause
The claim: *always* $90 degree$, at every point of every smooth surface. Picture argument, then algebra.

== ⭐ Step 3 · level walking ⟹ perpendicular #D

The picture argument first. Walking *along a contour*, your altitude is pinned — the level set is the set of one height. No change, no slope:

$ D_bold(u) f = 0 quad "whenever" bold(u) "hugs the contour" $

#pause
Now the algebra finishes it in one line. $D_bold(u) f = nabla f dot bold(u) = 0$, and L3 taught us exactly what a zero dot product *means*:

#result[$nabla f dot bold(u) = 0 arrow.l.r.double nabla f perp bold(u)$: #h(4pt) the gradient is *perpendicular to the level set* through its point.]

#pause
Numeric witness at $(2, 1)$: the contour-hugging direction is $prop (3, -2)$, and $(4, 6) dot (3, -2) = 12 - 12 = 0$ ✓. Claim 2 of the quiver plot — proved.

== ⭐ Step 4 · steepest ascent, from Cauchy–Schwarz #D

Among all unit $bold(u)$, which maximizes $D_bold(u) f$? L3's second reading of the dot product:

$ D_bold(u) f = nabla f dot bold(u) = norm(nabla f) thin underbrace(norm(bold(u)), = 1) cos theta = norm(nabla f) cos theta $

#pause
$cos theta <= 1$, with equality only at $theta = 0$ — this is L3's Cauchy–Schwarz, "bought for nothing":

- *Steepest ascent*: $bold(u) = nabla f slash norm(nabla f)$, slope $norm(nabla f) = sqrt(4^2 + 6^2) = sqrt(52) approx 7.21$
#pause
- *Steepest descent*: $-nabla f$, slope $-7.21$ — the direction training will take (L17)

#pause
#notebox[Our hand-picked $(0.6, 0.8)$ scored $7.2$ of a possible $7.21$ — it sits just $3degree$ off the true champion $(4, 6) slash sqrt(52) approx (0.55, 0.83)$. Close only counts because $cos theta$ is flat near $theta = 0$.]

== The slope compass #V

Slope at $(2, 1)$ vs the angle $theta$ (in degrees) between your step and $nabla f$ — a pure cosine:

#align(center, lines(fn: t => GNORM * calc.cos(t * calc.pi / 180), domain: (-180, 180),
  samples: 91, markers: false, colors: (INK,),
  points: ((0, GNORM, [ascent: +7.21]), (90, 0, [level: 0]),),
  annotations: ((-85, -4.4, text(fill: MUTED)[descent: −7.21 at ±180°]),),
  hlines: ((0, none, MUTED),),
  size: (108mm, 48mm), x-label: $theta$, y-label: [slope]))

#pause
Every hiker question is on this dial: max at $theta = 0$ (with $nabla f$), zero at $plus.minus 90 degree$ (contour directions), minimum dead against it.

== Checkpoint: the level walk #Q

#mcq([At some point on a surface, $nabla f = (3, 4)$. You step in the unit direction $bold(u) = (4, -3) slash 5$. The instantaneous rate of change of $f$ is:],
  [$5$],
  [$0$],
  [$-5$],
  [$7$],
)

== Answer: the level walk #A

#mcq-answer([B], [$0$ — you walked along the contour],
  [$D_bold(u) f = nabla f dot bold(u) = (3 dot 4 + 4 dot (-3)) slash 5 = 0$: this $bold(u)$ is perpendicular to $nabla f$, i.e. a level direction. A ($5 = norm(nabla f)$) is the rate along $nabla f slash norm(nabla f)$; C is the rate dead against it; D adds components — a units crime from L7's answer keys.])

= The chain rule grows branches

== Composition, now with two routes

L7's chain rule ran along a single pipeline. In two variables the pipeline *branches* — one clock $t$ drives both coordinates of a moving point:

#branch-diagram()

#pause
$ (dif f) / (dif t) = (partial f) / (partial x) (dif x) / (dif t) + (partial f) / (partial y) (dif y) / (dif t) quad quad #text(fill: MUTED)[multiply along each route, *add* the routes] $

== Numbers first: a drone over the bowl

A drone flies the path $x = t$, $y = t^2$ over our surface $f = x^2 + 3y^2$. How fast is the terrain rising under it at $t = 1$ (position $(1, 1)$)?

#pause
*Route by route.* $partial f slash partial x = 2$, $partial f slash partial y = 6$ at $(1,1)$; $dif x slash dif t = 1$, $dif y slash dif t = 2t = 2$:

$ (dif f) / (dif t) = 2 dot 1 + 6 dot 2 = 14 $

#pause
*Sanity check, no chain rule.* Substitute first: $f(t, t^2) = t^2 + 3t^4$, so $dif f slash dif t = 2t + 12 t^3 = 14$ at $t = 1$ ✓.

#pause
#notebox[Why *add*? Nudge $t$ by $0.01$: the $x$-route delivers $2 times 0.01 = 0.02$ of height, the $y$-route $6 times 0.02 = 0.12$. Both arrive: $Delta f approx 0.14$. (Truth: $f(1.01, 1.0201) = 4.1419$, i.e. $Delta f = 0.1419$.)]

== The chain ML actually runs

#branch-diagram(ml: true)

#pause
- Forward: weights make a prediction, prediction makes a loss — routes *into* $cal(L)$.
#pause
- Training needs $partial cal(L) slash partial w_i$ for *every* weight — the gradient $nabla_bold(w) cal(L)$, a million partials.

#pause
#result[Multiply along routes, add the routes — at industrial scale, with every route shared. Doing that bookkeeping without waste is *backpropagation*: L10–L11.]

= The hook, resolved

== Training is gradient descent on the landscape

$nabla cal(L)$ points straight *uphill* — so training repeatedly steps *against* it:

$ bold(theta)_(t+1) = bold(theta)_t - eta thin nabla cal(L)(bold(theta)_t) $

#pause
- Cross contour lines perpendicularly, downhill, at speed $prop$ steepness.
#pause
- $eta$ (the *learning rate*) scales the step — L1's diverging-vs-crawling mystery lives here, and L16–L18 derive all of it properly (from L7's Taylor, as promised).

#pause
Watch it run on the loss *you built by hand* — next slide.

== The walk and its altimeter #V

Gradient descent on $cal(L)(w, b)$ from the two-point fit, starting at $(w, b) = (-0.5, 2.5)$:

#two(r: (54%, 46%),
  align(center, contour(ad.fn2(sse), xlim: (-1, 4), ylim: (-1.2, 3.2), samples: 56,
    levels: (0.08, 0.3, 0.8, 1.8, 3.5, 7, 12, 20), size: (74mm, 52mm), color: TEAL,
    paths: (sse-path,), marks: ((2, 0, [$(2, 0)$], RED),), x-label: $w$, y-label: $b$)),
  align(center, lines((sse-losses,), colors: (INK,), markers: false,
    size: (58mm, 46mm), x-label: [step], y-label: $cal(L)$, title: [its training curve])),
)

#pause
Fast drop, then a long crawl *along the valley floor* toward $(2, 0)$ — the plateau in every real training curve is usually this: a valley, walked lengthwise.

== Interactive: walkers on landscapes #I

#interbox(link-to: IA + "optimizer-race")[
  Race gradient descent (and fancier walkers — L17's cast) across 2-D loss landscapes. You can now read *everything* on that screen: the contour map, the perpendicular downhill steps, the crowded-line cliffs, the valley crawls.
]

#pause
Try the ill-conditioned valley: watch steps zigzag *across* the valley precisely because each one is perpendicular to its contour.

== Where today's ideas get cashed

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*lecture*], [*what it does with today's gradient*]),
  [L9], [derivatives become *matrices*: Jacobian (many outputs), Hessian (curvature)],
  [L10–L11], [how $nabla f$ is actually *computed*: autodiff, backprop, `loss.backward()`],
  [L16–L18], [descent *derived and tuned*: step sizes, convergence, Newton],
  [L19–L20], [constrained optima: where the constraint's and the loss's gradients *align*],
)

#pause
#result[Every one of those lectures manipulates the object built today: the vector of all partial sensitivities.]

= Summary & what's next

== Lecture 8 — summary

- *Two views*: surface, and contour map of level sets $f = K$; crowding = steepness.
- *Partials*: freeze all but one variable, differentiate as in L7 — ratios $4$ and $6$ at $(2, 1)$.
- *Gradient*: $nabla f = (partial f slash partial x, thin partial f slash partial y)$ — one arrow per point, on the *map*.
- *Directional derivative* ⭐: $D_bold(u) f = nabla f dot bold(u)$ — one vector, every slope.
- *Geometry* ⭐: $nabla f perp$ contours (level walk $arrow.r.double$ dot $= 0$); steepest ascent $nabla f$ at rate $norm(nabla f)$, steepest descent $-nabla f$ — Cauchy–Schwarz, from L3.
- *Chain rule with branches*: multiply along routes, *add* the routes — backprop's skeleton.
- *Payoff*: training = downhill walk; the training curve is its altimeter.

#pause
#notebox[*Read before L9:* MML §5.2–§5.3 (partial differentiation & gradients). *T4* this week: `meshgrid`, contours, quivers, and your own two-parameter descent loop in NumPy.]

== Practice problems

Try on paper; verify in the T4 notebook.

+ For $f(x, y) = x^3 y + y^2$: compute both partials at $(1, 2)$ and assemble $nabla f$.
+ Sketch the contour map of $f(x, y) = 2x + y$. What shapes are the level sets? Compute $nabla f$ — why is it the *same arrow everywhere*, and how do the contours confirm it?
+ On our bowl at $(1, 1)$: find both unit directions with *zero* instantaneous change, and verify each with a dot product.
+ For $f = x y$ at $(2, 3)$: the slope along $bold(u) = (1, 1) slash sqrt(2)$, and the steepest slope available there.
+ Let $x = cos t$, $y = sin t$ on the bowl $f = x^2 + 3y^2$. Compute $dif f slash dif t$ at $t = pi slash 4$ twice: by the branching chain rule, and by substituting first.
+ A quiver plot shows every arrow pointing *toward* the origin, shrinking near it. Sketch plausible contours, and propose an $f$ that fits.

== ⭐⭐⭐ Where the map has corners #OPT

$f(x, y) = abs(x) + abs(y)$ — the L1 norm from L3, as terrain:

#two(r: (46%, 54%),
  align(center, contour((x, y) => calc.abs(x) + calc.abs(y), xlim: (-2, 2), ylim: (-2, 2),
    samples: 64, levels: 6, size: (52mm, 52mm), color: TEAL,
    marks: ((0, 0, [pit], RED),))),
  [
    Diamond rings — with *corners* wherever $x = 0$ or $y = 0$.

    #pause
    At a corner the left and right slices disagree on the slope: *no single tangent, no gradient*. Our smooth-zoom story needs a patch.

    #pause
    #notebox[The patch is the *subgradient* — any arrow between the two one-sided slopes. It's why L1-regularized models (and ReLU networks) still train fine. Boyd §6; a cameo in L16.]
  ],
)

#focus-slide[
  The gradient points where the contour lines crowd.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Jacobian, Hessian & Multivariate Taylor* — when outputs are vectors too, the derivative becomes a matrix.
]
