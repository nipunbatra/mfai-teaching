// Eigendecomposition — Lecture 5 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture5/L5-eigendecomposition.typ
//   typst compile --root . --input handout=true lecture5/L5-eigendecomposition.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Eigendecomposition],
  subtitle: [The directions a matrix can't turn],
)

#let SETOSA = "https://setosa.io/ev/eigenvectors-and-eigenvalues/"

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
// build mat() from code: a `;` straight after a #fmt(..) interpolation is
// swallowed as a code-statement terminator and flattens the matrix to one row
#let m2(M, d: 4) = math.mat(
  (fmt(M.at(0).at(0), d: d), fmt(M.at(0).at(1), d: d)),
  (fmt(M.at(1).at(0), d: d), fmt(M.at(1).at(1), d: d)),
)
#let v2(v, d: 4) = $vec(#fmt(v.at(0), d: d), #fmt(v.at(1), d: d))$

// the lecture's one matrix, its eigen-pieces, and the power-iteration run
#let A2 = ((2.0, 1.0), (1.0, 2.0))
#let vstar = la.normalize((1.0, 1.0))                    // top eigen-direction
#let iters = {
  let x = (1.0, 0.0)
  let rows = ((0, x, la.dist(x, vstar)),)
  for k in range(8) {
    x = la.normalize(la.matvec(A2, x))
    rows.push((k + 1, x, la.dist(x, vstar)))
  }
  rows                                                   // (k, x_k, error)
}
#let P2 = ((1.0, 1.0), (1.0, -1.0))                      // columns = eigenvectors
#let D2 = ((3.0, 0.0), (0.0, 1.0))
#let P2i = la.inv(P2)

// the four-page web: A→B,C · B→A,C · C→A,D · D→A  (column j = page j's votes)
#let M4 = (
  (0.0, 0.5, 0.5, 1.0),
  (0.5, 0.0, 0.0, 0.0),
  (0.5, 0.5, 0.0, 0.0),
  (0.0, 0.0, 0.5, 0.0),
)
#let prhist = {
  let r = (0.25, 0.25, 0.25, 0.25)
  let hist = (r,)
  for _ in range(12) { r = la.matvec(M4, r); hist.push(r) }
  hist
}

// the web as a fletcher graph (vector-native, house style)
#let web-diagram(weights: false, sp: (46mm, 30mm)) = {
  let lw = if weights { l => text(size: 12pt, fill: MUTED, l) } else { l => none }
  align(center, diagram(
    spacing: sp, node-stroke: 0.9pt + INK, node-fill: white,
    {
      node((0, 0), [*A*], radius: 6mm)
      node((1, 0), [*B*], radius: 6mm)
      node((1, 1), [*C*], radius: 6mm)
      node((0, 1), [*D*], radius: 6mm)
      edge((0, 0), (1, 0), "-|>", stroke: 0.7pt + MUTED, bend: 22deg, label: lw($1\/2$), label-sep: 1.5pt)
      edge((1, 0), (0, 0), "-|>", stroke: 0.7pt + MUTED, bend: 22deg, label: lw($1\/2$), label-sep: 1.5pt)
      edge((0, 0), (1, 1), "-|>", stroke: 0.7pt + MUTED, bend: 16deg, label: lw($1\/2$), label-pos: 0.2, label-sep: 1.5pt)
      edge((1, 1), (0, 0), "-|>", stroke: 0.7pt + MUTED, bend: 16deg, label: lw($1\/2$), label-pos: 0.2, label-sep: 1.5pt)
      edge((1, 0), (1, 1), "-|>", stroke: 0.7pt + MUTED, label: lw($1\/2$), label-side: left, label-sep: 1.5pt)
      edge((1, 1), (0, 1), "-|>", stroke: 0.7pt + MUTED, label: lw($1\/2$), label-side: left, label-sep: 1.5pt)
      edge((0, 1), (0, 0), "-|>", stroke: 1.1pt + ACC, label: lw(text(fill: ACC)[$1$]), label-side: left, label-sep: 1.5pt)
    },
  ))
}

#title-slide()

// ═══════════════════════════ 1 · motivating example ═══════════════════════════
= PageRank and repeated matrix multiplication

== PageRank uses a stationary direction #V

L1 demoed the teaching contract — picture, numbers, code, symbols — on *this exact picture*:

#fig("/lecture1/figures/eigenvector_demo.svg", w: 62%)

#pause
Almost every direction gets turned. Two refuse. Today we go deep: where they come from, what they cost to find, and why they quietly run modern AI.

== 1998 · the web has a ranking problem

Keyword-based ranking was easy to manipulate with repeated or hidden terms. Link structure offered a second signal: which pages point to this one?

#pause
Two Stanford grad students, Sergey Brin and Larry Page, proposed something structural: ignore what a page _says about itself_; look at *who links to it*.

#pause
#notebox[PageRank became a core component of early Google search. Its mathematical centre is an eigenvector of a link-transition matrix, which you will compute before this lecture ends.]

== A four-page internet #V

Every arrow is a hyperlink:

#web-diagram()

#pause
#align(center, text(size: 22pt)[*Which page is the most important?*])

== "Important" is circular — and that's the good part

A link is a *vote*. First guess: count votes. But surely a vote _from an important page_ should count for more…

#pause
- To score a page, you need the scores of the pages linking to it.
#pause
- To score _those_, you need the scores of the pages linking to *them*. And so on, forever.

#pause
#result[The ranking must be *self-consistent* — a vector that reproduces itself under the voting rule. That is an eigenvector equation.]

#pause
We'll compute this ranking live today. The *full* model — dead ends, teleportation, why a unique answer is guaranteed — is *L25* (Markov chains), where PageRank is fully resolved.

== Learning outcomes

By the end of this lecture you will be able to:

+ Say what an *eigenvector* and *eigenvalue* are — in pictures, numbers, code, and symbols.
+ Compute both *by hand* for any 2×2 matrix (characteristic polynomial).
+ Interpret a real $lambda$ as scaling and possible sign reversal; recognize that complex-conjugate pairs describe two-dimensional rotation–scaling behaviour.
+ Factor $A = P D P^(-1)$ and use it to compute $A^k$ instantly.
+ Run *power iteration* — and prove why it finds the top eigenvector.
+ State the *spectral theorem* for symmetric matrices (and see why ML loves them).
+ Explain the eigen-idea inside *PageRank*.

// ═══════════════════════════ 2 · directions that don't turn ═══════════════════════════
= Directions that don't turn

== Numbers first: catch a direction refusing to turn

Same matrix as L1. Feed it $bold(v) = (1, 1)$:

$ A bold(v) = mat(2, 1; 1, 2) vec(1, 1) = vec(2 dot 1 + 1 dot 1, 1 dot 1 + 2 dot 1) = vec(3, 3) = 3 bold(v) $

#pause
Output = *3 × input*. Same line through the origin — just stretched. Now feed it an ordinary vector, $bold(u) = (1, 0)$:

$ A bold(u) = vec(2, 1) quad quad #text(fill: MUTED)[— was pointing east, now points east-north-east: *turned*.] $

== The second stubborn direction

Try $bold(w) = (1, -1)$:

$ A bold(w) = mat(2, 1; 1, 2) vec(1, -1) = vec(2 - 1, 1 - 2) = vec(1, -1) = 1 dot bold(w) $

#pause
Not turned — not even stretched. Stretch factor exactly $1$.

#pause
#result[$A$ has two special directions: $(1,1)$ with factor $3$, and $(1,-1)$ with factor $1$. Everything this lecture builds runs on those two facts.]

== Code: NumPy finds both in one line

#codebox[```python
>>> A = np.array([[2., 1.], [1., 2.]])
>>> vals, vecs = np.linalg.eig(A)
>>> vals
array([3., 1.])
>>> vecs                        # eigenvectors are the COLUMNS
array([[ 0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]])
```]

#pause
Column $k$ pairs with `vals[k]`. Column 0 is $(1,1)$ normalized to unit length.

#pause
#notebox[Column 1 is $(-0.707, 0.707)$ — that is $(1,-1)$ times $-1\/sqrt(2)$. Same line, so same eigenvector: NumPy just picked a different scale and sign.]

== Symbols, at last: the definition

A *nonzero* vector $bold(v)$ is an *eigenvector* of a square matrix $A$, with *eigenvalue* $lambda$, if

$ A bold(v) = #text(fill: ACC)[$lambda$] bold(v) $

#pause
- applying $A$ to $bold(v)$ only *scales* it by $lambda$ — it never leaves its line;
#pause
- _eigen_ is German for "own / characteristic": these are $A$'s *own* directions.

#pause
#result[An eigenvector is a direction $A$ cannot turn. \ The eigenvalue is what $A$ does *instead* of turning it.]

== The rules of the club

- *$bold(v) = 0$ is banned.* $A dot 0 = lambda dot 0$ holds for every $lambda$ — no information.
#pause
- *Any nonzero multiple of $bold(v)$ is the _same_ eigenvector.* If $A bold(v) = lambda bold(v)$, then $A(c bold(v)) = lambda (c bold(v))$. Eigenvectors are *directions*, not arrows of a fixed length — we normalize purely for convenience.
#pause
- *$lambda = 0$ is allowed!* $A bold(v) = 0$ means $bold(v)$ is squashed to the origin — eigenvalue-zero directions are exactly the *null space* from L4.
#pause
- $lambda$ may be negative, repeated, or (for some matrices) not real at all — all three coming up.

== Interactive: drag a vector until it stops turning #I

#interbox(link-to: SETOSA)[
  Drag $bold(v)$ around the plane and watch $A bold(v)$ follow. When the two arrows line up, you have found an eigen-direction — feel how the plane snaps along them. Then edit the matrix and watch the special directions move.
]

#pause
Five minutes with this builds the reflex the algebra will lean on: *eigenvectors are found by watching directions, not by staring at matrix entries.*

== Why two stubborn directions run everything

Apply $A$ to $bold(v) = (1,1)$ *twice*:

$ A(A bold(v)) = A(3 bold(v)) = 3 dot A bold(v) = 9 bold(v) quad quad A^10 bold(v) = 3^10 bold(v) = 59049 thin bold(v) $

#pause
On its eigen-directions, the matrix $A^10$ — a hundred multiplications and additions — collapses to *one scalar power*.

#pause
#result[Repeated application of a matrix is governed *entirely* by its eigenvectors. That single sentence is the spine of this lecture — and of PageRank.]

// ═══════════════════════════ 3 · characteristic polynomial ═══════════════════════════
= Finding eigenvalues by hand

== Turn the definition into something solvable

$A bold(v) = lambda bold(v)$ has two unknowns tangled together: the direction $bold(v)$ and the number $lambda$. Move everything left:

$ A bold(v) - lambda bold(v) = 0 quad arrow.r.double quad (A - lambda I) bold(v) = 0 $

#pause
#notebox[The identity $I$ sneaks in because $A - lambda$ (matrix minus number) is not defined — $lambda I$ scales every direction by $lambda$, so subtracting it is legal and changes nothing else.]

#pause
So we need: a *nonzero* $bold(v)$ that the matrix $(A - lambda I)$ sends to zero.

== Nonzero-to-zero demands a collapse

L4 fact: an *invertible* matrix sends only $0$ to $0$ — every other input survives.

#pause
So if $(A - lambda I) bold(v) = 0$ has a nonzero solution, then $(A - lambda I)$ must be *singular* — it squashes some direction flat. And L4 gave us the detector for that:

#pause
#result[$ det(A - lambda I) = 0 $ \ #text(size: 17pt)[the *characteristic equation* — solve it for $lambda$, and the collapsed direction is the eigenvector]]

#pause
Most $lambda$ values leave $A - lambda I$ invertible. Eigenvalues are the handful of *exact settings* at which it collapses.

== Worked by hand: the polynomial

For our $A$, subtract $lambda$ down the diagonal and take the 2×2 determinant:

$ det mat(2 - lambda, 1; 1, 2 - lambda) = (2 - lambda)(2 - lambda) - 1 dot 1 $

#pause
$ = lambda^2 - 4 lambda + 4 - 1 = lambda^2 - 4 lambda + 3 $

#pause
A quadratic in $lambda$ — the *characteristic polynomial*. A 2×2 always gives a quadratic: *two* eigenvalues (counted with repeats).

== Worked by hand: the roots

$ lambda^2 - 4 lambda + 3 = (lambda - 3)(lambda - 1) = 0 quad arrow.r.double quad lambda_1 = 3, quad lambda_2 = 1 $

#pause
Exactly the two stretch factors we caught by hand — and exactly NumPy's `array([3., 1.])`.

#pause
#notebox[*Honesty about scale:* we solve the polynomial for 2×2 only. For a $1000 times 1000$ matrix nobody touches a degree-1000 polynomial — not you, and not NumPy either (it uses iterative methods; you'll build the simplest one today).]

== Back-substitute $lambda = 3$: recover the direction

Set $lambda = 3$ in $(A - lambda I) bold(v) = 0$:

$ (A - 3I) bold(v) = mat(-1, 1; 1, -1) vec(v_1, v_2) = vec(0, 0) quad arrow.r.double quad v_2 = v_1 $

#pause
Both rows say the same thing — the matrix has *collapsed to one honest equation*, exactly as engineered. Pick $v_1 = 1$:

$ bold(v) = vec(1, 1) quad #text(fill: MUTED)[(or any multiple — it's a direction)] $

== Back-substitute $lambda = 1$

$ (A - I) bold(w) = mat(1, 1; 1, 1) vec(w_1, w_2) = vec(0, 0) quad arrow.r.double quad w_2 = -w_1 quad arrow.r.double quad bold(w) = vec(1, -1) $

#pause
Look at those two matrices: at $lambda = 3$ and $lambda = 1$ the rows became *copies of each other* — rank collapsed from 2 to 1 (L4's rank, on duty). At any other $lambda$, the only solution would be $bold(v) = 0$.

#pause
#result[2×2 recipe: $det(A - lambda I) = 0$ → quadratic → two $lambda$'s → back-substitute each → two directions.]

== Two instant sanity checks (2×2 gold)

For any 2×2 (in fact any $n times n$) matrix:

$ "trace"(A) = a_(11) + a_(22) = lambda_1 + lambda_2 quad quad det(A) = lambda_1 lambda_2 $

#pause
Our $A$: trace $= 2 + 2 = 4 = 3 + 1$ ✓ #h(1.5em) det $= 4 - 1 = 3 = 3 dot 1$ ✓

#pause
#notebox[Ten-second exam armor: after *any* eigenvalue computation, check the sum against the trace and the product against the determinant. Catches most slips.]

== Checkpoint: read the eigenvalues #Q

#mcq([$M = mat(5, 2; 0, 3)$. Its eigenvalues are…? #text(size: 17pt, fill: MUTED)[(try to answer *without* fully expanding the polynomial)]],
  [$5$ and $3$],
  [$5$ and $2$],
  [$2$ and $3$],
  [$4$ and $4$],
)

== Answer: read the eigenvalues #A

#mcq-answer([A], [$lambda = 5$ and $lambda = 3$],
  [$det(M - lambda I) = (5 - lambda)(3 - lambda) - 2 dot 0$: the off-diagonal term dies, so for a *triangular* matrix the eigenvalues sit on the diagonal. Checks: trace $8 = 5 + 3$ ✓, det $15 = 5 dot 3$ ✓. (Careful: this shortcut is for triangular matrices only — our $mat(delim: "(", 2, 1; 1, 2)$ has eigenvalues $3, 1$, not $2, 2$.)],
)

// ═══════════════════════════ 4 · geometric meaning ═══════════════════════════
= Interpreting eigenvalues

== One number per direction — what can it say? #V

#fig("/lecture5/figures/eigen_gallery.svg", w: 96%)

#pause
Three matrices, three behaviors: stretch along the eigen-line ($lambda > 0$), *reverse* along it ($lambda < 0$), or refuse to have a real eigen-line at all (rotation).

== $lambda < 0$: the direction that flips

The mirror matrix $F = mat(0, 1; 1, 0)$ swaps coordinates — reflection across $y = x$. Check $bold(u) = (1, -1)$:

$ F bold(u) = vec(-1, 1) = -1 dot bold(u) $

#pause
$F bold(u)$ lies on the *same line* as $bold(u)$ — no turning! — but points the opposite way. That is $lambda = -1$: kept in line, reversed in sense.

#pause
#notebox[And $(1,1)$? Reflection leaves it alone: $lambda = +1$. Trace $0 = 1 + (-1)$ ✓, det $-1 = 1 dot (-1)$ ✓. A negative *determinant* always betrays a flip hiding somewhere (L4).]

== Rotation: the matrix with no real eigenvectors

Rotate every vector by $90°$: $R = mat(0, -1; 1, 0)$. *Every* direction turns — so no eigenvector should exist. The algebra agrees:

$ det(R - lambda I) = lambda^2 + 1 = 0 quad arrow.r.double quad lambda = plus.minus i $

#pause
No real roots. The eigenvalues have become a *complex pair* — and that is precisely the algebra's way of saying "this map rotates".

#pause
#notebox[We will not use complex arithmetic beyond this slide: a complex eigenvalue pair represents a rotation, possibly combined with scaling. We will see the same effect once more today as a wobble in a live computation. A formal treatment of complex vector spaces is beyond this course.]

== Eigenvalue cases in one table

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Eigenvalue*], [*What happens along that direction*]),
  [$lambda > 1$], [stretched by $lambda$],
  [$lambda = 1$], [untouched],
  [$0 < lambda < 1$], [shrunk toward the origin],
  [$lambda = 0$], [squashed flat — the null-space direction (L4)],
  [$-1 <= lambda < 0$], [flipped, and kept-or-shrunk],
  [$lambda < -1$], [flipped *and* stretched],
  [complex pair], [no invariant direction — a rotation (+ scale) in that plane],
)

#pause
#result[Eigen-directions turn a matrix into a *dashboard of dials* — one independent stretch factor per direction.]

// ═══════════════════════════ 5 · diagonalization ═══════════════════════════
= Diagonalization: the eigen-basis

== Package the eigen-data as matrices

Two facts, $A bold(v) = 3 bold(v)$ and $A bold(w) = 1 bold(w)$, stack into one equation. Put the eigenvectors in the *columns* of $P$, the eigenvalues in a diagonal $D$:

$ P = mat(1, 1; 1, -1) quad quad D = mat(#text(fill: ACC)[$3$], 0; 0, #text(fill: ACC)[$1$]) quad quad "then" quad A P = P D $

#pause
($A P$ hits each column: $(A bold(v), A bold(w)) = (3 bold(v), 1 bold(w))$ — which is exactly $P D$.)

#pause
$P$'s columns are independent directions, so $P$ is invertible (L4). Multiply on the right by $P^(-1)$:

$ #text(fill: ACC)[$A = P D P^(-1)$] quad quad "— the eigendecomposition (a.k.a. diagonalization)." $

== Check it live — this slide multiplies the matrices

The slide computes $P^(-1)$, then the triple product, at compile time (chalkdust linalg):

$ P^(-1) = #m2(P2i, d: 1) $

#pause

$ P D P^(-1) = mat(1, 1; 1, -1) mat(3, 0; 0, 1) #m2(P2i, d: 1) = #m2(la.matmul(la.matmul(P2, D2), P2i), d: 0) quad #text(fill: GREEN)[= $A$ ✓] $

#pause
#align(center, text(size: 16pt, fill: MUTED)[computed, not typed — change $P$ or $D$ in the source and this slide would catch the lie])

== Read it right to left: three moves

$A bold(x) = P D P^(-1) bold(x)$ is a *recipe*:

#align(center, diagram(
  spacing: (13mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    node((0, 0), $bold(x)$, radius: 6mm)
    node((1, 0), text(size: 13pt)[*1 · $P^(-1)$* \ rewrite $bold(x)$ in \ eigen-coordinates], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: rgb("#EFEEEB"))
    node((2, 0), $bold(c)$, radius: 6mm, stroke: 0.9pt + TEAL)
    node((3, 0), text(size: 13pt)[*2 · $D$* \ stretch coordinate $i$ \ by $lambda_i$ — just dials], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: rgb("#FDECD6"), stroke: 0.9pt + ACC)
    node((4, 0), $D bold(c)$, radius: 6mm, stroke: 0.9pt + ACC)
    node((5, 0), text(size: 13pt)[*3 · $P$* \ translate back to \ standard coordinates], shape: fletcher.shapes.rect, corner-radius: 3pt, inset: 6pt, fill: rgb("#EFEEEB"))
    node((6, 0), $A bold(x)$, radius: 6mm)
    edge((0, 0), (1, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((1, 0), (2, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((2, 0), (3, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((3, 0), (4, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((4, 0), (5, 0), "-|>", stroke: 0.7pt + MUTED)
    edge((5, 0), (6, 0), "-|>", stroke: 0.7pt + MUTED)
  },
))

#pause
In the right coordinate system, $A$ *is* just its dials.

== This is a change of basis (L3, paying off)

The eigenvectors ${bold(v), bold(w)}$ are two independent directions — a perfectly good *basis* for the plane (L3). Any $bold(x)$ has coordinates in it:

$ bold(x) = c_1 bold(v) + c_2 bold(w), quad quad bold(c) = P^(-1) bold(x) $

#pause
In eigen-coordinates the tangled map $mat(2, 1; 1, 2)$ becomes the *diagonal* map $mat(3, 0; 0, 1)$: coordinate 1 tripled, coordinate 2 kept. No crosstalk between coordinates, ever.

#pause
#result[Diagonalization = switching to the basis $A$ itself prefers. \ Same map, honest coordinates.]

== Matrix powers in an eigenbasis

Multiply the factored form by itself and watch the middle cancel:

$ A^2 = P D underbrace(P^(-1) thin P, = thin I) D P^(-1) = P D^2 P^(-1) $

#pause
Every extra factor of $A$ cancels the same way, $k - 1$ times:

$ A^k = P D^k P^(-1), quad quad D^k = mat(3^k, 0; 0, 1^k) quad #text(fill: MUTED)[— powers of a diagonal matrix are free] $

#pause
#result[$k$ matrix multiplications become *two scalar powers* and one change of basis.]

== $A^(10)$, two ways — both computed live

*Way 1 (brute force):* the slide multiplies $A$ by itself ten times:

$ A^(10) = #m2({ let R = la.identity(2); for _ in range(10) { R = la.matmul(R, A2) }; R }, d: 0) $

#pause
*Way 2 (eigen):* the slide computes $P D^(10) P^(-1)$ with $3^(10) = #fmt(calc.pow(3.0, 10), d: 0)$:

$ P mat(3^10, 0; 0, 1) P^(-1) = #m2(la.matmul(la.matmul(P2, ((calc.pow(3.0, 10), 0.0), (0.0, 1.0))), P2i), d: 0) quad #text(fill: GREEN)[same matrix ✓] $

#pause
Every entry is $approx 3^(10)\/2$: after ten applications, the $lambda = 3$ direction has utterly buried the $lambda = 1$ direction.

== Watch a vector under repeated application #V

#fig("/lecture5/figures/repeated_application.svg", w: 73%)

#pause
#align(center, text(size: 16pt, fill: MUTED)[for a generic start, the component along the dominant eigenvector eventually determines the direction of $A^k bold(x)$])

#pause
#align(center, text(size: 18pt, fill: ACC)[That observation gives a practical algorithm.])

// ═══════════════════════════ 6 · power iteration ═══════════════════════════
= Power iteration, live

== The algorithm: multiply, normalize, repeat

To find the top eigen-direction of $A$ — *without* polynomials:

$ bold(x)_(k+1) = (A bold(x)_k) / (norm(A bold(x)_k)) quad quad #text(fill: MUTED)[start from (almost) any $bold(x)_0$] $

#pause
- *Multiply:* let $A$ drag the direction wherever it wants.
#pause
- *Normalize:* rescale to unit length. The lengths would grow like $3^k$ — in float32 that overflows around $k approx 81$, and you know exactly why (L2). Normalizing keeps only what matters: the *direction*.

#pause
#result[This is *power iteration* — the simplest eigenvalue algorithm in existence, and the one that ranked the web.]

== Eight iterations, computed inside this slide

Start at $bold(x)_0 = (1, 0)$ — pointing due east, $45°$ away from the answer:

#text(size: 15.5pt)[
  #table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 5.6pt,
    align: (center, right, right, right),
    table.header([*$k$*], [*first coord of $bold(x)_k$*], [*second coord*], [*error $norm(bold(x)_k - bold(v)_1)$*]),
    ..iters.map(((k, x, e)) => ([#k], [#fmt(x.at(0))], [#fmt(x.at(1))], [#fmt(e, d: 5)])).flatten(),
  )
]

#pause
#align(center, text(size: 16pt, fill: MUTED)[every number above is computed by the slide at compile time (chalkdust `la.matvec` + `la.normalize`) — converging to $(0.7071, 0.7071) = (1,1)\/sqrt(2)$])

== The error, on a log axis #V

#align(center, lines(
  (
    iters.map(r => (r.at(0), r.at(2))),
    range(1, 9).map(k => (k, iters.at(1).at(2) * calc.pow(1.0 / 3.0, k - 1))),
  ),
  labels: ([measured error], [$(1\/3)^k$]),
  colors: (ACC, TEAL),
  dashes: ("solid", "dashed"),
  markers: (true, false),
  log-y: true, legend: "tr",
  x-label: [iteration $k$], y-label: [error],
  size: (105mm, 48mm),
))

#pause
A straight line on a log axis = *geometric* decay: the error divides by almost exactly $3$ every step. Where does the $3$ come from? Time for this lecture's one real proof.

== ⭐ Why it converges · step 1: change glasses #D

Write the start vector in the *eigen-basis* (we can — eigenvectors span the plane):

$ bold(x)_0 = c_1 bold(v)_1 + c_2 bold(v)_2 quad quad #text(fill: MUTED)[$bold(v)_1$: $lambda_1 = 3$; $quad bold(v)_2$: $lambda_2 = 1$] $

#pause
For our run: $bold(x)_0 = (1, 0)$ has $c_1 = c_2 = 1\/sqrt(2) approx 0.7071$ — a half-and-half mix of the two eigen-directions.

#pause
#notebox[This is THE move of the lecture: to understand what a matrix does to *any* vector, first express that vector in the matrix's own directions.]

== ⭐ Step 2: apply $A$, $k$ times #D

On eigenvectors, applying $A$ is just multiplying by $lambda$ — so on the mix, term by term:

$ A bold(x)_0 = c_1 lambda_1 bold(v)_1 + c_2 lambda_2 bold(v)_2 $

#pause
$ A^k bold(x)_0 = c_1 lambda_1^k bold(v)_1 + c_2 lambda_2^k bold(v)_2 quad = quad c_1 thin 3^k bold(v)_1 + c_2 thin 1^k bold(v)_2 $

#pause
No matrix multiplication survived. The entire history of $k$ applications is *two scalar powers* — the $bold(v)_1$ ingredient grows like $3^k$, the $bold(v)_2$ ingredient like $1^k$.

== ⭐ Step 3: factor out the winner #D

Pull the dominant growth $lambda_1^k$ out front:

$ A^k bold(x)_0 = lambda_1^k [ thin c_1 bold(v)_1 + c_2 (lambda_2 / lambda_1)^k bold(v)_2 thin ] $

#pause
Normalization kills the prefactor $lambda_1^k$. Inside the bracket, since $abs(lambda_2) < abs(lambda_1)$:

$ (lambda_2 / lambda_1)^k = (1/3)^k arrow.r 0 quad quad #text(fill: MUTED)[the $bold(v)_2$ contamination decays geometrically] $

#pause
#result[Power iteration converges to the top eigenvector, and the error shrinks by the factor $abs(lambda_2 \/ lambda_1)$ per step — here $1\/3$, exactly the slope the live plot measured.]

== The fine print (three honest conditions)

+ *An eigen-basis and a gap:* the derivation assumes the start can be expanded in eigenvectors (as for our symmetric matrix) and $abs(lambda_1) > abs(lambda_2)$ strictly. The closer the ratio to 1, the slower the convergence.
#pause
+ *A foothold:* need $c_1 != 0$ — the start must contain _some_ $bold(v)_1$. Random starts do with probability one under a continuous distribution. For this matrix, starting exactly at $(1, -1)$ stays on $bold(v)_2$ even in float32 because the relevant operations are exactly representable.
#pause
+ *Normalize:* not for correctness — for survival. Un-normalized, $norm(A^k bold(x)) tilde 3^k$ overflows float32 near $k approx 81$.

== The code: six honest lines

#codebox[```python
A = np.array([[2., 1.], [1., 2.]])
x = np.array([1., 0.])
for k in range(8):
    x = A @ x                    # multiply: let A pull
    x = x / np.linalg.norm(x)    # normalize: keep the direction
print(x)                         # [0.70718 0.70703]  -> v1
```]

#pause
#codebox[```python
>>> np.linalg.eig(A).eigenvectors[:, 0]     # the library agrees
array([0.70710678, 0.70710678])
```]

#pause
Six lines, no characteristic polynomial. The method can scale to huge *sparse or implicit* matrices when a matrix-vector product is cheap; a dense billion-row matrix would still be infeasible to store.

== Checkpoint: power-iteration rate #Q

#mcq([$B$ has eigenvalues $lambda_1 = 4$ and $lambda_2 = 2$. Per step of power iteration, the error shrinks by roughly a factor of…],
  [$1\/4$],
  [$1\/2$],
  [$1\/16$],
  [it doesn't shrink — $lambda_1 > 1$ makes it grow],
)

== Answer: the eigenvalue ratio sets the rate #A

#mcq-answer([B], [$times 1\/2$ per step],
  [The contamination decays like $(lambda_2\/lambda_1)^k = (2\/4)^k$. The absolute sizes of the $lambda$'s are irrelevant after normalization — only the *ratio* matters. (D confuses growth of *length*, which normalization discards, with drift of *direction*.)],
)

// ═══════════════════════════ 7 · PageRank ═══════════════════════════
= PageRank, almost answered

== Back to the four pages — votes, made precise

#two(r: (46%, 54%))[
  #web-diagram(weights: true, sp: (34mm, 26mm))
][
  Each page splits *one unit of importance* equally over its outgoing links:

  - A and B and C each cast two half-votes;
  - D links only to A — *one wholehearted vote*.

  #pause
  So importance flows: $r_A^"new" = 1/2 r_B + 1/2 r_C + r_D$, and likewise for the others.
]

== The voting rule is a matrix

Stack the four flow equations; column $j$ holds page $j$'s outgoing vote shares:

$ bold(r)^"new" = M bold(r), quad quad M = mat(0, 1/2, 1/2, 1; 1/2, 0, 0, 0; 1/2, 1/2, 0, 0; 0, 0, 1/2, 0) $

#pause
"Self-consistent ranking" now has an exact meaning — a vector the voting rule *reproduces*:

$ bold(r) = M bold(r) quad quad #text(fill: MUTED)[i.e. an eigenvector of $M$ with $lambda = 1$] $

#pause
#result[PageRank is an eigenvector. Ranking the web = power iteration on the link matrix.]

== Power-iterate the web — computed in this slide

Start fair — a quarter each — and let the votes flow:

#text(size: 15.5pt)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 5.6pt,
    align: (center, right, right, right, right),
    table.header([*$k$*], [*A*], [*B*], [*C*], [*D*]),
    ..(0, 1, 2, 3, 4, 5, 8, 12).map(k => {
      let r = prhist.at(k)
      ([#k], [#fmt(r.at(0), d: 3)], [#fmt(r.at(1), d: 3)], [#fmt(r.at(2), d: 3)], [#fmt(r.at(3), d: 3)])
    }).flatten(),
  )
]

#pause
#align(center, text(size: 16pt, fill: MUTED)[no normalization needed — each column of $M$ sums to 1, so the total importance stays exactly #fmt(prhist.at(12).sum(), d: 2). Suspicious? Hold that thought.])

== Power iteration on the four-page graph #V

#align(center, bars(
  prhist.at(12),
  labels: ([A], [B], [C], [D]),
  annotate: true, digits: 3,
  highlight: 0,
  size: (95mm, 44mm),
))

#pause
- *A wins*: it collects votes from everyone — including D's undivided one.
#pause
- The exact limit is rational: $bold(r) = (8, 4, 6, 3)\/21$. Check the first row by hand: $1/2 dot 4/21 + 1/2 dot 6/21 + 3/21 = 8/21$ ✓ — the ranking votes for itself.

== Did you see the wobble?

Look back at column A: #range(5).map(k => fmt(prhist.at(k).at(0), d: 3)).join(" → ") → … — it *overshoots and rings* like a struck bell, rather than sliding in monotonically.

#pause
Our convergence theory says the error decays by $abs(lambda_2 \/ lambda_1)$. For this $M$ the subdominant eigenvalues are $-1/2$ and a *complex pair* $-1/4 plus.minus 0.43 i$…

#pause
#notebox[A complex-conjugate pair represents a rotation combined with scaling, so the error spirals toward zero. Its magnitude decreases by $abs(lambda_2\/lambda_1) = 1\/2$ per step.]

== One slide of honesty: this matrix is a Markov matrix

$M$'s columns are nonnegative and sum to $1$ — a *stochastic (Markov) matrix*. Equivalent view: a "random surfer" clicks a uniformly random outgoing link forever; $r_j$ becomes the fraction of time spent on page $j$.

#pause
- Why did $lambda = 1$ exist at all? Rows of $M^top$ sum to 1, so $M^top bold(1) = bold(1)$ — and $M$, $M^top$ share eigenvalues (same characteristic polynomial). *Every* Markov matrix has $lambda = 1$.
#pause
- Our little web was safe: no dead-end pages, no isolated clique, so power iteration converged to a *unique* answer.

#pause
#alertbox[Real link graphs contain pages with no outlinks and closed communicating classes. PageRank adds a $0.15$ teleport probability; L25 studies the Markov-chain conditions that make the stationary ranking unique.]

// ═══════════════════════════ 8 · symmetric matrices ═══════════════════════════
= Symmetric matrices: the nicest case

== Our matrix was secretly the nicest kind

$A = mat(2, 1; 1, 2)$ equals its own transpose: $A = A^top$ — *symmetric*. Now look at its eigenvectors:

$ bold(v) = vec(1, 1), quad bold(w) = vec(1, -1), quad quad bold(v) dot bold(w) = 1 - 1 = 0 quad #text(fill: ACC)[— orthogonal!] $

#pause
Perpendicular eigen-directions, real eigenvalues, no drama. Coincidence?

#pause
#result[Not a coincidence. For symmetric matrices, *all of it is guaranteed*.]

== The spectral theorem (stated, savored, not proved)

*Theorem.* Every real symmetric matrix $A = A^top$ has:

+ all eigenvalues *real* — no rotation pairs can hide in it,
#pause
+ an *orthonormal eigenbasis* can be chosen — a perpendicular grid, including within repeated-eigenvalue subspaces,
#pause
+ hence the clean factorization $A = Q Lambda Q^top$, with $Q$'s columns orthonormal ($Q^(-1) = Q^top$ — inverting it is free).

#pause
#notebox[Stated with a straight face, used forever, proved in MML §4.2 for the curious — the proof is not our derivation today. What you must own is the *picture*, next slide.]

== The picture: symmetric matrices act along a perpendicular grid #V

#fig("/lecture5/figures/spectral_ellipse.svg", w: 58%)

#pause
#align(center, text(size: 16pt, fill: MUTED)[symmetric = pure stretch along a perpendicular grid: eigenvectors *are* the ellipse axes — non-symmetric eigenvectors miss them])

#pause
#notebox[*Planted for L13:* a Gaussian data cloud's covariance matrix is symmetric — its eigenvectors will be the *axes of the data ellipse*, its eigenvalues the spreads.]

== The slide double-checks the theorem

Chalkdust's symmetric eigensolver (`la.eig-sym`, Jacobi rotations), run on $A$ at compile time:

#let es = la.eig-sym(A2)
#let canon(v) = if v.at(0) < 0 { la.scale(v, -1.0) } else { v }

$ lambda = (#fmt(es.at(0).at(0), d: 0), #fmt(es.at(0).at(1), d: 0)) quad quad bold(q)_1 = #v2(canon(es.at(1).at(0))) quad bold(q)_2 = #v2(canon(es.at(1).at(1))) $

#pause
$ bold(q)_1 dot bold(q)_2 = #fmt(la.dot(es.at(1).at(0), es.at(1).at(1)), d: 4) quad #text(fill: GREEN)[— orthogonal ✓] $

#pause
The reported value is zero to machine precision, the appropriate numerical interpretation after L2.

#pause
Real $lambda$'s, perpendicular $bold(q)$'s — the theorem, holding up in silicon.

== Why ML is wall-to-wall symmetric matrices

Symmetric matrices are not a corner case — ML manufactures them:

- *Covariance matrices* (L13) — the geometry of every data cloud;
#pause
- *$X^top X$* — at the heart of least squares (L4's preview) and of PCA (L6);
#pause
- *Hessians* (L17–L18) — their eigenvalues determine curvature and stable learning-rate ranges.

#pause
#result[Whenever ML hands you a matrix that is symmetric, the spectral theorem hands you back a perpendicular grid of dials. That guarantee powers L6, L13, and L17.]

// ═══════════════════════════ 9 · the take-home ═══════════════════════════
= The take-home

== Lecture 5 — summary

- *Eigen-pair:* $A bold(v) = lambda bold(v)$ — for a real eigenpair, $bold(v)$ keeps its line and $lambda$ sets scaling and possible sign reversal. A complex-conjugate pair describes a real two-dimensional invariant plane.
- *2×2 by hand:* solve $det(A - lambda I) = 0$; audit with trace $= sum lambda_i$, det $= product lambda_i$.
- *Diagonalize:* $A = P D P^(-1)$ — in the eigen-basis $A$ is pure dials, so $A^k = P D^k P^(-1)$.
- *Power iteration:* multiply-normalize → dominant eigenvector; error $times abs(lambda_2 \/ lambda_1)$ per step. PageRank uses the eigenvector with $lambda = 1$ (L25).
- *Spectral theorem:* symmetric $arrow.r.double$ real $lambda$'s, orthogonal eigenvectors, $A = Q Lambda Q^top$ (→ L6, L13).

#pause
#notebox[*Before L6* — MML §4.1–4.2 · replay the #link(SETOSA)[Setosa explorable] · 3Blue1Brown, _Essence of Linear Algebra_, ch. 14 — 17 minutes, gold.]

== Practice problems

Try on paper; then check yourself with `np.linalg.eig` and your six-line power iteration.

+ Eigenvalues and eigenvectors of $mat(3, 1; 1, 3)$, by hand. Verify trace and det.
+ For $mat(0, 2; 2, 0)$: predict the eigenvalues' *signs* from the det, then compute.
+ Two steps of power iteration on our $A$ from $bold(x)_0 = (0, 1)$. Why the same limit as from $(1, 0)$?
+ For the Markov matrix $mat(0.9, 0.5; 0.1, 0.5)$: find the $lambda = 1$ eigenvector — you have just computed a stationary distribution (L25).
+ Why does rotation (by anything but $0°$ or $180°$) have no real eigenvectors? Picture first, then the discriminant.
+ Power-iterate $A$ from exactly $bold(x)_0 = (1, -1)$: what does theory predict? Why does float32 also remain on that direction for this particular integer matrix?

== ⭐⭐⭐ When diagonalization fails #OPT

The shear $J = mat(1, 1; 0, 1)$ has characteristic polynomial $(1 - lambda)^2$: eigenvalue $lambda = 1$, *twice*. But hunt for eigenvectors:

$ (J - I) bold(v) = mat(0, 1; 0, 0) bold(v) = 0 quad arrow.r.double quad "only" bold(v) = vec(1, 0) $

#pause
One direction for a 2×2 matrix — not enough to build $P$. $J$ is *defective*: it cannot be diagonalized, so the eigen-basis formula $A = P D P^(-1)$ and the power proof built from it do not apply. The eigenvalue equation and characteristic polynomial still do.

#pause
#notebox[Two consolations. Symmetric matrices are *never* defective (spectral theorem — one more reason ML loves them). And next lecture's SVD factors *every* matrix, defective or not, square or not.]

#focus-slide[
  Eigenvectors are the directions a matrix can't turn.
  #v(12pt)
  #set text(size: 22pt)
  Next: *SVD & PCA* — what about non-square matrices? Every matrix gets a version of this.
]
