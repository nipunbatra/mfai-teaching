// Differentiation on a Computer II: Reverse Mode & Backprop — Lecture 11 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture11/L11-backprop.typ
//   typst compile --root . --input handout=true lecture11/L11-backprop.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Reverse Mode & Backprop],
  subtitle: [Every gradient, in one backward pass],
)
// prose em-dashes must never split a slide (touying treats a lone "—" as a
// Markdown-style horizontal-rule pagebreak by default)
#show: touying-set-config.with(config-common(horizontal-line-to-pagebreak: false))

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── the lecture's one expression, built ONCE for the chalkdust engine ──
// f(x, y) = (x + y) · x² at (3, 1) — the same 5-node graph T5 drills.
#let f5 = ad.expr("(x + y) * x^2", ("x", "y"))
#let adgraph = ad.graph.with(theme: mfai-theme)
#let fnum(v) = {
  let r = calc.round(v, digits: 4)
  if r == calc.round(r) { str(int(r)) } else { str(r) }
}

// ── the worked 5-node graph, drawn in house style ────────────────────
// One parametric drawing so every derivation step shows the SAME graph,
// progressively annotated: values (teal) and adjoints (orange).
#let g5(
  vals: none,     // dict key → content, forward values
  adjs: (:),      // dict key → content, adjoints
  hi: (),         // node keys highlighted (the node firing this step)
  back: (),       // ((from, to), …) orange backward arrows for this step
  sp: (23mm, 7.5mm),
) = {
  let P  = (x: (0, 0.1), y: (0, 1.55), b: (1.15, -0.72), a: (1.15, 0.85), f: (2.3, 0.1))
  let TT = (x: $x$, y: $y$, b: $b = x^2$, a: $a = x + y$, f: $f = a dot b$)
  let CC = (x: INK, y: INK, b: TEAL, a: TEAL, f: ACC)
  align(center, diagram(spacing: sp, {
    for (u, v) in (("x", "b"), ("x", "a"), ("y", "a"), ("b", "f"), ("a", "f")) {
      edge(P.at(u), P.at(v), "-|>", stroke: 0.8pt + MUTED)
    }
    for (u, v) in back {
      edge(P.at(u), P.at(v), "-|>", bend: 26deg, stroke: 1.25pt + ACC)
    }
    for (k, pos) in P {
      let c = CC.at(k)
      let ishi = k in hi
      let rows = (text(size: 13pt, weight: 700, fill: INK, TT.at(k)),)
      if vals != none and k in vals {
        rows.push(text(size: 11.5pt, weight: 500, fill: TEAL, vals.at(k)))
      }
      if k in adjs {
        rows.push(text(size: 11.5pt, weight: 700, fill: ACC, adjs.at(k)))
      }
      node(pos, align(center, stack(spacing: 3pt, ..rows)),
        shape: fletcher.shapes.rect,
        fill: if ishi { ACC.lighten(87%) } else if c == INK { white } else { c.lighten(91%) },
        stroke: if ishi { 1.6pt + ACC } else { 0.9pt + c },
        corner-radius: 3pt, inset: 5.5pt)
    }
  }))
}
#let V5 = (x: "= 3", y: "= 1", a: "= 4", b: "= 9", f: "= 36")   // forward values at (3, 1)

// one local node u → v → f: the chain rule as a picture (house localnode idiom)
#let localnode = align(center, diagram(spacing: (30mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0), $u$, radius: 6mm)
  node((1, 0), $v$, radius: 6mm, stroke: 0.9pt + TEAL)
  node((2, 0), $f$, radius: 6mm, stroke: 0.9pt + ACC)
  edge((0, 0), (1, 0), text(size: 13pt, fill: TEAL)[forward #h(4pt) $v = g(u)$], "-|>", stroke: 1.1pt + TEAL, label-sep: 8pt)
  edge((1, 0), (2, 0), text(size: 13pt, fill: MUTED)[rest of graph], "-|>", stroke: 1.1pt + TEAL, label-sep: 8pt)
  edge((2, 0), (0, 0), "-|>", bend: 32deg, stroke: 1.25pt + ACC,
    label: text(size: 13pt, fill: ACC)[backward #h(4pt) $overline(u) = overline(v) dot g'(u)$], label-sep: 8pt)
}))

// x feeding two nodes: the branch motif
#let branchgraph = align(center, diagram(spacing: (20mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white, {
  node((0, 0.5), $x$, radius: 5.5mm)
  node((1, -0.3), $u$, radius: 5.5mm, fill: TEAL.lighten(91%), stroke: 0.9pt + TEAL)
  node((1, 1.3), $v$, radius: 5.5mm, fill: TEAL.lighten(91%), stroke: 0.9pt + TEAL)
  node((2, 0.5), $f$, radius: 5.5mm, fill: ACC.lighten(90%), stroke: 0.9pt + ACC)
  edge((0, 0.5), (1, -0.3), "-|>", stroke: 0.8pt + MUTED)
  edge((0, 0.5), (1, 1.3), "-|>", stroke: 0.8pt + MUTED)
  edge((1, -0.3), (2, 0.5), "-|>", stroke: 0.8pt + MUTED)
  edge((1, 1.3), (2, 0.5), "-|>", stroke: 0.8pt + MUTED)
}))

// the finale neuron o = tanh(w1 x1 + w2 x2 + b), drawn as a computation graph
#let neurongraph = align(center, diagram(spacing: (15mm, 5.2mm), {
  let N(p, lbl, c: INK, bg: white) = node(p, text(size: 12.5pt, weight: 600, fill: INK, lbl),
    shape: fletcher.shapes.rect, fill: bg, stroke: 0.9pt + c, corner-radius: 3pt, inset: 5pt)
  N((0, 0), $x_1 = 2$)
  N((0, 1), $w_1 = -3$)
  N((0, 2), $x_2 = 0$)
  N((0, 3), $w_2 = 1$)
  N((0, 4), $b = 6.88$)
  N((1, 0.5), $times$, c: TEAL, bg: TEAL.lighten(91%))
  N((1, 2.5), $times$, c: TEAL, bg: TEAL.lighten(91%))
  N((2, 2.2), $+$, c: TEAL, bg: TEAL.lighten(91%))
  N((3, 2.2), $tanh$, c: TEAL, bg: TEAL.lighten(91%))
  N((4, 2.2), $o = 0.71$, c: ACC, bg: ACC.lighten(90%))
  for (u, v) in (((0, 0), (1, 0.5)), ((0, 1), (1, 0.5)), ((0, 2), (1, 2.5)), ((0, 3), (1, 2.5)),
    ((1, 0.5), (2, 2.2)), ((1, 2.5), (2, 2.2)), ((0, 4), (2, 2.2)),
    ((2, 2.2), (3, 2.2)), ((3, 2.2), (4, 2.2))) {
    edge(u, v, "-|>", stroke: 0.8pt + MUTED)
  }
}))

// funnel graph seeded two ways: forward mode (one input) vs reverse (the output)
#let sweep(rev: false) = align(center, diagram(spacing: (17mm, 6mm), {
  let ins  = ((0, 0), (0, 1), (0, 2))
  let mids = ((1, 0.5), (1, 1.5))
  let outp = (2, 1)
  let labs = ($theta_1$, $theta_2$, $theta_3$)
  for m in mids { node(m, none, radius: 3.2mm, fill: TEAL.lighten(88%), stroke: 0.9pt + TEAL) }
  for (i, p) in ins.enumerate() {
    let seeded = (not rev) and i == 0
    node(p, text(size: 13pt, labs.at(i)), radius: 4.6mm,
      fill: if seeded { TEAL.lighten(70%) } else { white },
      stroke: (if seeded { 1.6pt + TEAL } else { 0.9pt + INK }))
  }
  node(outp, $f$, radius: 4.6mm, fill: ACC.lighten(if rev { 70% } else { 90% }),
    stroke: (if rev { 1.6pt } else { 0.9pt }) + ACC)
  for p in ins { for m in mids { edge(p, m, "-|>", stroke: 0.7pt + MUTED) } }
  for m in mids { edge(m, outp, "-|>", stroke: 0.7pt + MUTED) }
  if rev {
    for m in mids { edge(outp, m, "-|>", bend: 22deg, stroke: 1.2pt + ACC) }
    for p in ins { for m in mids { edge(m, p, "-|>", bend: 22deg, stroke: 1.2pt + ACC) } }
  } else {
    for m in mids { edge(ins.at(0), m, "-|>", bend: -22deg, stroke: 1.2pt + TEAL) }
    for m in mids { edge(m, outp, "-|>", bend: -22deg, stroke: 1.2pt + TEAL) }
  }
}))

#title-slide()

= The hook: one line of code

== The most-executed line in modern AI

#codebox[```python
loss = f(parameters, data)    # forward pass: ONE number comes out
loss.backward()               # ← this line. What does it DO?
```]

#pause
Every model you have ever heard of — chatbots, image generators, speech systems — was trained by a loop whose heart is `loss.backward()`, executed *millions of times per run*.

#pause
About 20 milliseconds after the call, `∂loss/∂θ` has appeared for *every one* of the model's parameters. Billions of them. From one pass.

#pause
#notebox[By the end of today you will have run this pass *by hand*, then *written the engine yourself* (\~35 lines of Python), then matched PyTorch digit for digit.]

== Today's one idea

#result[Reverse mode computes the gradient of *one output* with respect to a *million inputs* in *one backward pass*.]

#pause
That asymmetry — not faster chips, not bigger data — is why deep learning is possible at all.

#pause
Three questions, in order:
+ What structure hides inside an expression? → *computation graphs*
+ What flows backward through it? → *adjoints* (the chain rule, organized)
+ Who does the bookkeeping? → *you* (micrograd), then *PyTorch*

== Where L10 left us: a cliffhanger

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Route*], [*Verdict from L10*], [*Cost for* $nabla f$, $f: RR^n -> RR$]),
  [symbolic], [exact, but expressions explode], [—],
  [finite differences], [truncation vs rounding — both lose], [$n$ evals, *approximate*],
  [forward-mode AD], [exact! carry $("value", "derivative")$ pairs], [#text(fill: RED)[$n$ *passes* — one per input]],
)

#pause
Forward mode ended L10 undefeated on *accuracy* — and disqualified on *cost*: a million inputs means a million passes.

#pause
#result[Today's fix: don't push derivatives *forward* from each input. Pull them *backward* from the one output.]

== Learning outcomes

By the end of this lecture you will be able to:

+ Draw the *computation graph* of any small expression — nodes, edges, DAG.
+ Run a *forward pass* and a complete *backward pass* by hand, node by node.
+ Define the *adjoint* $overline(v) = partial f slash partial v$ and apply "arriving × local, add at branches".
+ Explain the *cost asymmetry*: one pass per input (forward) vs one pass per output (reverse).
+ Implement a working scalar autograd engine — *micrograd* — and verify it.
+ Reproduce everything in PyTorch: `requires_grad`, `.backward()`, `.grad`, `zero_grad()`.

= Computation graphs

== You already compute in graphs

Evaluate $f(x, y) = (x + y) dot x^2$ at $(3, 1)$. Watch what your own hand does:

$ a = x + y = 4, quad quad b = x^2 = 9, quad quad f = a dot b = 36 $

#pause
Three small steps, each a *primitive* — a single operation whose derivative we know cold ($+$, $times$, $(dot)^2$, and later $exp$, $tanh$, …).

#pause
Each step *names* an intermediate value, and each step *uses* earlier values.

#pause
#result[An expression is a recipe: primitive steps, wired by "who uses whom". Draw the wiring and you have the *computation graph*.]

== The expression, drawn #V

#g5()

#pause
- *Leaves* (left): the inputs $x, y$. *Root* (right): the output $f$.
#pause
- Every arrow says "this value feeds that operation".
#pause
#notebox[Look at $x$: it feeds *two* nodes — it is written once but *used twice*. So the picture is a *DAG* (directed acyclic graph), not a tree. Remember this; it is where the one subtle rule of the day will live.]

== Why graphs? Because computers already build them

- When Python evaluates `(x + y) * x**2`, it *must* execute primitives in some order — the graph is just that execution, remembered.
#pause
- PyTorch, JAX, TensorFlow all do exactly this: as your code runs, every operation quietly *records* itself: result, operands, and which primitive fired.
#pause
- The recording is called the *tape* (or "autograd graph"). Recording costs almost nothing — the work was being done anyway.

#pause
#result[No new math yet: a computation graph is bookkeeping for work the machine already did.]

== Forward pass: values flow along the arrows #V

#two(r: (52%, 48%),
  g5(vals: V5),
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 7pt,
    table.header([*Node*], [*Rule*], [*Value*]),
    [$a = x + y$], [$3 + 1$], [$4$],
    [$b = x^2$], [$3^2$], [$9$],
    [$f = a dot b$], [$4 dot 9$], [$36$],
  ),
)

#pause
This is the *forward pass*: leaves are given, every other node fires once its inputs are ready.

#pause
#notebox[Keep every number in sight — the backward pass is about to *reuse all of them*.]

= Adjoints: the chain rule on a graph

== L8's chain rule, re-read as a graph rule

L8 gave the multivariate chain rule. For our $f(a(x, y), b(x))$:

$ (partial f)/(partial x) = (partial f)/(partial a) (partial a)/(partial x) + (partial f)/(partial b) (partial b)/(partial x) $

#pause
Read it *on the graph*:

#result[*Multiply* local slopes along a path. *Add* over all paths from the variable to $f$.]

#pause
#notebox[A deep graph can hold *exponentially many* paths — summing them one by one would be hopeless. Today's algorithm computes each node's "all paths downstream of me" summary *once*, and reuses it. Hold that thought: it is *memoization*.]

== The adjoint: one number per node

For every node $v$ in the graph, define its *adjoint*

$ #text(fill: ACC)[$overline(v)$] := (partial f)/(partial v) quad = quad "\"if" v "wiggles by" epsilon", " f "wiggles by" overline(v) dot epsilon"\"" $

#pause
- The adjoint already accounts for *everything between $v$ and the output* — all paths, summed.
#pause
- At the output itself: $overline(f) = partial f slash partial f = 1$. Free seed, no work.
#pause
- The two adjoints we actually want: $overline(x)$ and $overline(y)$ — together they *are* $nabla f$.

#pause
#notebox[Notation: the bar $overline(v)$ is standard in the autodiff literature; PyTorch stores exactly this number in `v.grad`.]

== Each node's job: arriving × local #V

#localnode

#pause
The node $v = g(u)$ never sees the whole graph. It receives $overline(v)$ from above, multiplies by its *own local slope*, and passes the product down:

$ #text(fill: ACC)[$overline(u)$] = #text(fill: ACC)[$overline(v)$] dot underbrace(g'(u), "local, known cold") $

#pause
#result[derivative sent back $=$ derivative arriving $times$ local slope]

== The three local rules we need today

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Node*], [*Local slopes*], [*Backward behaviour*]),
  [$v = u + w$], [$partial v slash partial u = 1, quad partial v slash partial w = 1$], [$+$ *copies* $overline(v)$ to both inputs],
  [$v = u dot w$], [$partial v slash partial u = w, quad partial v slash partial w = u$], [$times$ *sends each input the other input*: $overline(u) = overline(v) dot w$],
  [$v = u^2$], [$partial v slash partial u = 2u$], [square sends $overline(v) dot 2u$],
)

#pause
Every primitive an engine supports ships with its one-line rule like these — $exp$ sends $overline(v) dot e^u$, $tanh$ sends $overline(v)(1 - tanh^2 u)$, …

#pause
#result[An autodiff engine is: this table, plus bookkeeping.]

== Where paths meet, contributions add #V

#branchgraph

If $x$ feeds two nodes, a nudge $Delta x$ travels down *both* routes to $f$, and the two effects superpose (first-order Taylor — L7):

#pause
$ Delta f approx underbrace(overline(u) (partial u)/(partial x), "via" u) dot Delta x + underbrace(overline(v) (partial v)/(partial x), "via" v) dot Delta x quad arrow.r.double quad overline(x) = "sum of both contributions" $

#pause
#alertbox[The one subtle rule of the day: at a branch point, backward contributions *accumulate* (`+=`), never overwrite. Our graph's $x$ is exactly such a point — watch it in the derivation.]

= ⭐ The backward pass, by hand

== The plan of attack #D

#g5(vals: V5)

#pause
+ *Seed* the output: $overline(f) = 1$.
#pause
+ Visit nodes from the output back toward the inputs — each node only after *everything it feeds* is done. Here: $f$, then $b$, then $a$.
#pause
+ At each node: send *arriving × local* down every incoming edge; where edges meet, *add*.

== Step 0 · seed the output #D

#g5(vals: V5, adjs: (f: $overline(f) = 1$), hi: ("f",))

$ overline(f) = (partial f)/(partial f) = 1 quad quad "(a value can't help but move one-for-one with itself)" $

#pause
Every other adjoint will now follow from the local-rules table — no calculus, only arithmetic.

== Step 1 · the × node fires #D

#g5(vals: V5, adjs: (f: $overline(f) = 1$, a: $overline(a) = 9$, b: $overline(b) = 4$),
  hi: ("f",), back: (("f", "a"), ("f", "b")))

$f = a dot b$: *send each input the other input.*

#pause
$ overline(a) = overline(f) dot b = 1 dot 9 = 9 quad quad quad overline(b) = overline(f) dot a = 1 dot 4 = 4 $

#pause
#notebox[With $overline(f) = 1$, each factor's adjoint *is the other factor's value* — read both off the forward pass, zero new work.]

== Step 2 · the square node fires #D

#g5(vals: V5, adjs: (f: $overline(f) = 1$, a: $overline(a) = 9$, b: $overline(b) = 4$, x: [24 (via $b$)]),
  hi: ("b",), back: (("b", "x"),))

$b = x^2$: local slope $2x = 6$. Arriving × local:

$ overline(b) dot 2x = 4 dot 6 = 24 quad "flows down the" b arrow.l x "edge" $

#pause
#alertbox[Do *not* write $overline(x) = 24$ yet! A second path into $x$ (via $a$) has not reported. Park the contribution and wait.]

== Step 3 · the + node fires #D

#g5(vals: V5, adjs: (f: $overline(f) = 1$, a: $overline(a) = 9$, b: $overline(b) = 4$,
  x: [24 (via $b$) + 9 (via $a$)], y: $overline(y) = 9$),
  hi: ("a",), back: (("a", "x"), ("a", "y")))

$a = x + y$: *copy the arriving adjoint to both inputs* (local slopes are $1$).

#pause
$ "to " x: quad overline(a) dot 1 = 9 quad quad quad "to " y: quad overline(a) dot 1 = 9 $

#pause
$y$ has only this one path, so it is done: $overline(y) = 9$.

== Step 4 · accumulate at the branch #D

#g5(vals: V5, adjs: (f: $overline(f) = 1$, a: $overline(a) = 9$, b: $overline(b) = 4$,
  x: $overline(x) = 24 + 9 = 33$, y: $overline(y) = 9$), hi: ("x",))

Both of $x$'s paths have reported — the branch rule says *add*:

#pause
$ overline(x) = underbrace(24, "via " b) + underbrace(9, "via " a) = 33 $

#pause
#result[$nabla f(3, 1) = (overline(x), overline(y)) = (33, 9)$ — every input's derivative, one sweep.]

== The finished graph — photograph this #V

#g5(vals: V5, adjs: (f: $overline(f) = 1$, a: $overline(a) = 9$, b: $overline(b) = 4$,
  x: $overline(x) = 33$, y: $overline(y) = 9$), sp: (25mm, 8.5mm))

#pause
#align(center, text(size: 16pt, fill: MUTED)[teal: forward values, flowing right · orange: adjoints, flowing left — same graph, two sweeps])

#pause
The procedure you just executed has a name: *backpropagation*.

#pause
#notebox[*Tutorial 5* re-runs this exact graph with fresh numbers until it is reflex; *Quiz 2* assumes it is.]

== Does calculus agree? #D

Expand and differentiate the old way: $f = (x + y) x^2 = x^3 + x^2 y$.

#pause
$ (partial f)/(partial x) = 3x^2 + 2x y = 27 + 6 = 33 quad checkmark quad quad quad (partial f)/(partial y) = x^2 = 9 quad checkmark $

#pause
Same numbers. But look *how* each method got there:

#pause
- Calculus needed the *global formula* for $f$, expanded and re-derived per input.
#pause
- Backprop only ever used *local* facts — a table of primitive rules and the forward values. Locality is what a computer can automate.

== This slide ran a backward pass while compiling

These slides ship their own reverse-mode engine (chalkdust `autodiff`, written in Typst). This slide *runs it* on today's expression — the numbers below are computed live, not typed:

#codebox[```typ
#let f5 = ad.expr("(x + y) * x^2", ("x", "y"))
#ad.value(f5, (3, 1))   // forward pass
#ad.grad(f5, (3, 1))    // one backward pass → (x̄, ȳ)
```]

#pause
#let gv = ad.grad(f5, (3, 1))
#result[engine says: $f = #fnum(ad.value(f5, (3, 1)))$, #h(0.6em) $nabla f = (#fnum(gv.at(0)), #fnum(gv.at(1)))$ — the slide agrees with your hand.]

#pause
Hand, calculus, Typst engine: three routes, one answer. Two more coming (micrograd, PyTorch).

== The engine's own drawing of its tape #V

#adgraph(f5, (3, 1), names: ("x", "y"), spacing: (19mm, 8mm))

#pause
#align(center, text(size: 16pt, fill: MUTED)[auto-drawn from the recorded tape: each node shows op, value ("="), adjoint ("∂")])

#pause
#notebox[The tape records each *use* of $x$ as its own entry — so the two $x$ leaves carry their per-path contributions $24$ and $9$ separately. The variable $x$ collects both: $24 + 9 = 33$. The accumulation rule, made visible.]

== Checkpoint: run the graph at a new point #Q

#mcq([Same graph, new point: $x = 2, y = 5$ — so $a = 7$, $b = 4$, $f = 28$. What is $overline(y)$?],
  [$7$],
  [$4$],
  [$28$],
  [$32$],
)

== Answer: one path for $y$ #A

#mcq-answer([B], [$overline(y) = 4$],
  [$y$ reaches $f$ through one path: the $+$ node *copies* $overline(a)$, and the $times$ node made $overline(a) = overline(f) dot b = 4$. Sanity check by calculus: $partial f slash partial y = x^2 = 4$. (Option D is $overline(x) = 3x^2 + 2x y = 32$ — the *other* input's adjoint.)])

= Why reverse mode wins

== Two sweep directions, two prices #V

#two(
  [#sweep(rev: false)
   #align(center, text(size: 15.5pt, fill: TEAL)[*forward mode (L10)*: seed *one input*, push tangents up — yields $partial f slash partial theta_1$ only])],
  [#sweep(rev: true)
   #align(center, text(size: 15.5pt, fill: ACC)[*reverse mode (today)*: seed *the output*, push adjoints down — yields *all* $partial f slash partial theta_i$])],
)

#pause
#result[Forward: one pass per *input*. #h(1em) Reverse: one pass per *output*.]

== The scoreboard #V

Passes needed to assemble the full gradient of $f: RR^10 -> RR$:

#align(center, bars((10, 1), labels: ("forward", "reverse"),
  horizontal: true, annotate: true, highlight: 1, size: (125mm, 32mm)))

#pause
And the gap *scales*:

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Function*], [*Inputs $n$*], [*Forward-mode passes*], [*Reverse passes*]),
  [today's $f(x, y)$], [$2$], [$2$], [$1$],
  [small image model's loss], [$10^7$], [$10^7$], [$1$],
  [GPT-3's loss], [$1.75 times 10^11$], [$1.75 times 10^11$], [#text(fill: GREEN)[*1*]],
)

#pause
At 60 ms per pass, forward mode on GPT-3 needs $approx$ *330 years* per training step. Reverse: *60 ms*.

== ML's shape is exactly reverse mode's shape

Why does reverse mode fit machine learning so perfectly?

#pause
- Training tunes $n$ = *millions to billions* of inputs — the parameters $theta$.
#pause
- But it scores them with $m$ = *one* output — the loss $cal(L)(theta)$ (L14 builds it: a log-likelihood).
#pause
- $RR^n -> RR$, gigantic $n$, $m = 1$: the most extreme "reverse wins" shape that exists.

#pause
#result[Millions of knobs, one score. Reverse mode is not *an* option for ML — it is *the* option.]

#pause
#notebox[Flip the shape and forward mode wins: for $g: RR -> RR^(10^6)$ (one input, many outputs), forward needs 1 pass, reverse $10^6$. Rare in ML — but keep it for the checkpoint.]

== The price: memory (here is the memoization)

Reverse mode is not free. Look what the backward steps *consumed*:

- the $times$ rule needed the stored values $a = 4$, $b = 9$
- the square rule needed the stored $x = 3$ (for $2x$)

#pause
So the forward pass must *keep every intermediate value alive* until backward is done — that stored trail is the tape.

#pause
#result[Store each node's value once, reuse it for every path through that node: backprop is the chain rule with *memoization*.]

#pause
#notebox[The trade: forward mode streams in $O(1)$ extra memory; reverse pays $O("graph size")$ memory to win time. This is literally why training a network needs far more GPU memory than running one.]

== The cheap-gradient principle

How much *time* does the backward sweep add? Each node fires once, doing constant work per edge — the same order of work as the forward pass itself.

#pause
#result[Computing $nabla f$ costs a small constant multiple ($approx 2$–$3times$) of computing $f$ — *independent of $n$*. True for $n = 2$ and for $n = 1.75 times 10^11$.]

#pause
#notebox[This is the *cheap gradient principle* (Baur–Strassen; Griewank). It is why "just take the gradient" is a reasonable instruction at any scale — and why all of modern ML is gradient-based. The catch stays memory, not time.]

== Checkpoint: pick the mode #Q

#mcq([$f: RR^(10^6) -> RR$ (a loss). #h(0.4em) $g: RR -> RR^(10^6)$ (one dial, a million outputs). Which mode for which, to get all derivatives cheapest?],
  [forward for both],
  [reverse for $f$, forward for $g$],
  [forward for $f$, reverse for $g$],
  [reverse for both],
)

== Answer: match the pass to the thin side #A

#mcq-answer([B], [reverse for $f$, forward for $g$],
  [Passes = one per input (forward) or one per output (reverse). $f$ has one output → 1 reverse pass beats $10^6$ forward passes. $g$ has one input → 1 forward pass beats $10^6$ reverse passes. Always seed the *thin* side.])

= Micrograd: build the engine

== What a number must remember

To automate the hand-run, each value needs to carry exactly what *you* used at the desk:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Field*], [*What it held in the hand-run*]),
  [`data`], [the forward value (teal numbers: $3, 1, 4, 9, 36$)],
  [`grad`], [the adjoint slot, starting at $0$ (orange numbers, `+=` into it)],
  [`_prev`], [which nodes fed me (the arrows)],
  [`_backward`], [my one line from the local-rules table, frozen as a function],
)

#pause
#notebox[We now write this class in three stages — it is Karpathy's *micrograd*, the engine his assigned video builds. Same fields, same names.]

== Stage 1 · a `Value` that remembers its history

#codebox[```python
class Value:
    """A scalar that knows how it was computed."""
    def __init__(self, data, _children=()):
        self.data = data                 # forward value
        self.grad = 0.0                  # adjoint: d(output)/d(self)
        self._backward = lambda: None    # local rule (leaves: nothing)
        self._prev = set(_children)      # the nodes that made me
```]

#pause
- A plain number, plus the graph bookkeeping. `grad = 0.0` so contributions can `+=` in.
#pause
- Leaves ($x$, $y$) are just `Value(3.0)`, `Value(1.0)` — no children, nothing to send back.

== Stage 2a · addition, carrying its backward rule

#codebox[```python
    def __add__(self, other):                  # self + other
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad  += out.grad             # + copies the
            other.grad += out.grad             #   arriving adjoint
        out._backward = _backward
        return out
```]

#pause
- The forward line computes the value *and wires the graph* (`(self, other)` become `out._prev`).
#pause
- The closure is the rules-table row for $+$, waiting: "when my turn comes, copy `out.grad` down". Note the `+=` — branch accumulation, built in.

== Stage 2b · multiplication: send the other input

#codebox[```python
    def __mul__(self, other):                  # self * other
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad  += other.data * out.grad   # × sends each
            other.grad += self.data  * out.grad   #   the OTHER input
        out._backward = _backward
        return out
```]

#pause
- `other.data * out.grad` is precisely *arriving × local* — the local slope of $u dot w$ w.r.t. $u$ *is* $w$.
#pause
#result[Each operator owns one closure = one row of the rules table. The engine never needs a global formula.]

== In what order may nodes fire?

A node may push its `grad` down *only when that `grad` is complete* — after every node it feeds has already fired.

#pause
- Fire $a$ before $f$? Then $a$ pushes `a.grad = 0` — silence — and $x, y$ end up wrong forever.
#pause
- Safe order for our graph: $f$, then $b$, then $a$ (leaves receive; they never push).
#pause
- The general recipe: list nodes so every node comes *after* all nodes it feeds — a *topological order* of the DAG, walked in reverse.

#pause
#notebox[Getting this order is a classic graph traversal: depth-first, children before self ("post-order"). Ten lines, next slide.]

== Stage 3 · topological sort, then one sweep

#codebox[```python
    def backward(self):
        topo, seen = [], set()
        def build(v):                      # depth-first:
            if v not in seen:
                seen.add(v)
                for c in v._prev:          #   children first,
                    build(c)
                topo.append(v)             #   then me
        build(self)
        self.grad = 1.0                    # seed: df/df = 1
        for v in reversed(topo):           # output → inputs
            v._backward()
```]

#pause
Seed, sweep, done — steps 0–4 of the hand derivation, mechanized in twelve lines.

== Run it on the lecture's graph

#codebox[```python
x, y = Value(3.0), Value(1.0)
f = (x + y) * x * x        # the same 5-node graph
f.backward()

print(f.data)              # 36.0
print(x.grad, y.grad)      # 33.0  9.0
```]

#pause
#result[$36$, $33$, $9$ — the hand-run, reproduced by 35 lines of Python you can hold in your head.]

#pause
#notebox[We wrote `x * x` because our `Value` has no power op yet. Watch the mul closure: *both* branches are `x`, so it fires `x.grad += x.data * out.grad` twice — the $2x$ rule emerges from accumulation, unasked.]

== 35 lines, and it is real #I

#interbox(link-to: IA + "autograd")[
  Walk a computation graph in the browser: symbolic vs finite-difference vs autodiff on one screen, forward values then a step-by-step backward sweep. Rebuild today's $f = (x + y) dot x^2$ in it and watch $33$ and $9$ appear.
]

#pause
#notebox[*Assignment 1 (micrograd) goes out this week.* You take today's 35 lines and grow them Karpathy's way: `tanh`, `exp`, `__pow__`, `__neg__`, broadcasting scalars — then use *your own engine* to fit a small model. The assigned video walks the full 94-line original; watch it before you start.]

= PyTorch: the industrial engine

== The same numbers, a fifth and final time

#codebox[```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

f = (x + y) * x**2       # forward — PyTorch records the tape
f.backward()             # reverse sweep

print(f.item())          # 36.0
print(x.grad, y.grad)    # tensor(33.)  tensor(9.)
```]

#pause
#result[Hand · calculus · Typst engine · micrograd · PyTorch: *five* routes, one gradient $(33, 9)$.]

== What the industrial engine adds

Same algorithm as your 35 lines. Different engineering:

#pause
- Nodes hold *tensors*, not scalars — one graph node per matrix op, not per number.
#pause
- Each primitive (`matmul`, `exp`, `softmax`, …) ships a hand-tuned backward rule, running on GPU.
#pause
- The tape is rebuilt *on every forward run* ("define-by-run") — Python `if`s and loops just work.
#pause
- `requires_grad=True` marks which leaves deserve adjoints; everything else is constants.

#pause
#notebox[That is genuinely all. If you can read your micrograd, you can read `torch.autograd`'s documentation without fear.]

== Careful: gradients accumulate

`.grad` is an accumulator — `backward()` always `+=`s into it:

#codebox[```python
f = (x + y) * x**2
f.backward()
print(x.grad)            # tensor(33.)

f = (x + y) * x**2       # build the graph again…
f.backward()             # …sweep again
print(x.grad)            # tensor(66.)  ← 33 + 33. Not 33!
```]

#pause
Why design it this way? The engine cannot know your intent — the same `+=` that merged $x$'s two paths *inside* one graph also merges gradients *across* calls (useful for micro-batching). Clearing is your job.

== What `zero_grad()` is for

#codebox[```python
for step in range(1000):
    optimizer.zero_grad()   # 1 · flush stale adjoints
    loss = compute_loss()   # 2 · forward (tape records)
    loss.backward()         # 3 · reverse sweep → .grad
    optimizer.step()        # 4 · nudge parameters (L16!)
```]

#pause
#alertbox[Forget line 1 and nothing crashes — gradients from every past step silently pile up, and training goes sideways. The single most common autograd bug; now you know exactly which `+=` causes it.]

#pause
#notebox[Lines 3–4 are the bridge to Module 4: L16 takes the gradients you can now compute and asks *where to step*.]

== Finale: the graph AI actually trains #V

One last graph — small algebra, but shaped like the real thing: inputs weighted, summed, squashed.

$ o = tanh(w_1 x_1 + w_2 x_2 + b) $

#neurongraph

#pause
This unit is an (artificial) *neuron* — the atom of every network. To the engine it is nothing special: five leaves, two $times$, one $+$, one $tanh$. Backprop needs no new ideas.

== The neuron, through the engine

#codebox(size: 14pt)[```python
x1, x2 = torch.tensor(2.0), torch.tensor(0.0)
w1 = torch.tensor(-3.0, requires_grad=True)
w2 = torch.tensor( 1.0, requires_grad=True)
b  = torch.tensor( 6.88137, requires_grad=True)

o = torch.tanh(w1*x1 + w2*x2 + b)   # o = 0.7071
o.backward()
print(w1.grad, w2.grad, b.grad)     # 1.0   0.0   0.5
```]

#pause
#let fn = ad.expr("tanh(w1*x1 + w2*x2 + b)", ("w1", "w2", "b", "x1", "x2"))
#let gn = ad.grad(fn, (-3.0, 1.0, 6.88137, 2.0, 0.0))
Our Typst engine, live, agrees: $overline(w)_1 = #fnum(gn.at(0))$, $overline(w)_2 = #fnum(gn.at(1))$, $overline(b) = #fnum(gn.at(2))$.

#pause
#notebox[Read $overline(w)_2 = 0$ off the mul rule: $times$ *sends the other input*, and $w_2$'s partner is $x_2 = 0$. A weight attached to a silent input gets no gradient — no data, no learning signal.]

== The 20 milliseconds, itemized

Every step of `loss.backward()` is now one you have done yourself:

#pause
+ The forward pass already *recorded the tape* (your `_prev` wiring).
#pause
+ Topological sort of the tape (your twelve-line `build`).
#pause
+ Seed $overline("loss") = 1$; sweep once: *arriving × local*, `+=` at branches.
#pause
+ Adjoints land in `.grad` — millions at a time.

#pause
#align(center, scale(66%, reflow: true, mlp-diagram((3, 4, 4, 1), labels: ([inputs], [layer 1], [layer 2], [loss]))))

#result[Bigger graph, same single pass — why deep learning is possible.]

= Summary & what's next

== Lecture 11 — summary

- *Graphs*: an expression is a DAG of primitives; the forward pass fills every value.
- *Adjoints*: $overline(v) = partial f slash partial v$; seed $overline(f) = 1$; each node sends *arriving × local*; branches `+=`.
- *The graph*: $f = (x + y) x^2$ at $(3, 1)$: $f = 36$, $nabla f = (33, 9)$ — five methods, one answer.
- *Asymmetry*: one pass per input (forward) vs one pass per *output* (reverse); a loss has one output.
- *The price*: the tape stores forward values — memoization; costs memory, not time.
- *Engines*: `data / grad / _prev / _backward` + topo sort + one sweep; in PyTorch `requires_grad` → `.backward()` → `.grad`, with `zero_grad()` every step.

#pause
#notebox[*Read before L12* — MML §5.6. *Assigned:* Karpathy, "spelled-out intro to backpropagation" (builds micrograd; watch before starting A1). #sym.star.filled#sym.star.filled#sym.star.filled JAX autodiff cookbook. *This week:* T5 drills today's graph by hand · *Quiz 2* (L7–L11) · *A1 (micrograd)* goes out.]

== Practice problems

Try on paper; verify with the T5 notebook (or your own micrograd).

+ Draw the graph of $g(x, y) = (x - y) dot (x + y)$; run forward and backward at $(3, 1)$. Check both adjoints against $g = x^2 - y^2$.
+ Re-run today's graph at $x = 1, y = 2$. Before computing: which of $overline(x), overline(y)$ changes, and why?
+ Give `Value` a `__sub__`. What are its two local slopes, and its two closure lines?
+ In `backward()`, delete `reversed(...)` and trace our graph. What does `x.grad` end as, and which silent push caused it?
+ Predict PyTorch: build and `backward()` the same loss twice with no `zero_grad()` between. What is `w.grad`? Which single line fixes it?
+ For $h: RR^3 -> RR^2$, count passes to get the full Jacobian by each mode. Name a function shape where *forward* mode is the right tool.

== ⭐⭐⭐ JVPs, VJPs, and Hessians without Hessians #OPT

Both modes are matrix-free products with the Jacobian $J$ of $f: RR^n -> RR^m$ (L9):

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Sweep*], [*Computes*], [*Reading*]),
  [forward, seeded with $bold(v) in RR^n$], [$J bold(v)$ — a *JVP*], [one *column* mix: sensitivity along direction $bold(v)$],
  [reverse, seeded with $bold(u) in RR^m$], [$bold(u)^top J$ — a *VJP*], [one *row* mix: $nabla f$ is the VJP with $u = 1$],
)

#pause
Compose them and curvature comes free: for scalar $f$, the *Hessian-vector product*

$ H bold(v) = nabla (nabla f(theta)^top bold(v)) quad = quad "forward mode pushed through the reverse-mode program" $

costs a few function evaluations and *never materializes* the $n times n$ Hessian — the trick behind Newton-flavoured optimizers (L18) at scales where $n^2$ storage is fiction. JAX exposes all three as `jvp`, `vjp`, `grad`; see the autodiff cookbook.

#focus-slide[
  Backprop is just the chain rule with memoization.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Module 3 — Continuous Distributions* — we can now differentiate anything; next we need something worth differentiating: likelihoods.
]
