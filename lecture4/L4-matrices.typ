// Matrices as Linear Maps — Lecture 4 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture4/L4-matrices.typ
//   typst compile --root . --input handout=true lecture4/L4-matrices.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Matrices as Linear Maps],
  subtitle: [A matrix is a function that moves space],
)

#title-slide()

= Matrices as transformations in neural networks

== The trick a photo app runs a billion times a day #V

#fig("/lecture4/figures/image_transform.svg", w: 82%)

#pause
Rotating a photo touches a million pixels — yet the whole operation is *one $2 times 2$ matrix*, applied to every pixel's coordinates.

#pause
Change four numbers → rotate becomes shear, zoom, mirror. The *matrix is the move*.

== Inside every layer of every neural network

Every layer of every neural network computes the same thing: $bold(h) = phi(W bold(x) + bold(b))$.

#mlp-diagram((3, 4, 4, 2), labels: ([$bold(x)$], [$W_1 bold(x) + bold(b)_1$], [$W_2 bold(h)_1 + bold(b)_2$], [output]))

#pause
The $+ bold(b)$ merely slides space; $phi$ bends it (a later course's story). The geometry — the part *training actually learns* — is $W bold(x)$: *a matrix moving space*.

#pause
#notebox[A GPT-class model is thousands of matrices applied in sequence. To understand the machine, first understand what one matrix does to space. That is today.]

== Learning outcomes

By the end of this lecture you will be able to:

+ Read a small matrix as a *move of space* — and sketch what it does to a grid.
+ Compute $A bold(x)$ two ways: the *row view* (dot products) and the *column view* (mix of columns).
+ Explain why matrix × matrix is *composition* of moves — and why $A B eq.not B A$.
+ Compute *rank* on small matrices and say what it means geometrically: how much of space survives.
+ Decide when a matrix is *invertible* — and recognize when information is destroyed for good.
+ Solve $A bold(x) = bold(b)$ the professional way (`solve`, never `inv`), and set up *least squares* for tall systems.

= A matrix is a function

== A second identity for an old object

So far a matrix has been a *table*: in L3, rows were data points ($n$ students × $d$ features).

#pause
Today the same object gets a second life. Feed a $2 times 2$ matrix a vector — it hands back a vector:

$ T(bold(x)) = A bold(x), quad quad T : RR^2 arrow.r RR^2 $

#pause
That makes $A$ a *function*. Its input is every point of the plane; its output is where each point moves.

#pause
#notebox[Same numbers, two readings — the table stores measurements; the function *does* something. Keep both in your head: ML constantly switches between them mid-equation.]

== Exhibit 1 · rotation #V

#fig("/lecture4/figures/tf_rotate.svg", w: 71%)

#pause
$ R_(30 degree) = mat(cos 30 degree, -sin 30 degree; sin 30 degree, cos 30 degree) approx mat(0.87, -0.50; 0.50, 0.87) quad — "the whole plane turns; the house rides along" $

== Exhibit 2 · stretch and squeeze #V

#fig("/lecture4/figures/tf_scale.svg", w: 71%)

#pause
$ D = mat(2, 0; 0, 1\/2) quad — "×2 along " x ", ×" 1\/2 " along " y "; the grid squares become bricks" $

== Exhibit 3 · shear #V

#fig("/lecture4/figures/tf_shear.svg", w: 71%)

#pause
$ S = mat(1, 1; 0, 1) quad — "the ground floor holds still; every higher layer slides right" $

== Exhibit 4 · reflection #V

#fig("/lecture4/figures/tf_reflect.svg", w: 66%)

#pause
$ F = mat(0, 1; 1, 0) quad — "mirror across the line " y = x $

#pause
The chimney switched sides — no rotation can ever do that. Four matrices, four personalities.

== What all four exhibits preserved

Look back at the after-grids. Every move kept three promises:

- straight lines stayed *straight* (nothing bent)
#pause
- parallel lines stayed *parallel*, evenly spaced — grid squares became identical tilted bricks
#pause
- the *origin stayed put*

#pause
Moves that keep these promises are called *linear*, and they obey two laws:

$ A(bold(u) + bold(v)) = A bold(u) + A bold(v), quad quad A(c bold(u)) = c(A bold(u)) $

#pause
#result[Linear = the move respects addition and scaling. Every claim in today's lecture flows from these two laws.]

== The mechanics · one entry at a time

How is $A bold(x)$ actually computed? Each output entry is a *row of $A$ dotted with $bold(x)$*:

$ mat(a, b; c, d) mat(x_1; x_2) = mat(a x_1 + b x_2; c x_1 + d x_2) $

#pause
Check it on the shear, moving the point $(2, 1)$:

$ mat(1, 1; 0, 1) mat(2; 1) = mat(2 + 1; 0 + 1) = mat(3; 1) quad #text(fill: GREEN)[✓ slid right, height kept] $

#pause
#notebox[L3 callback: each output entry is a *dot product* — row $i$ asks the input "how much of you lies along me?" A neural layer with 4096 rows asks 4096 such questions at once.]

== The shape contract, in color

Matrices multiply vectors of *matching* shape — dimensions in #text(fill: ACC)[orange] must agree, and they are consumed:

$ underbrace(mat(1, 2, 0; 0, 1, 3), #[#text(fill: BLUE)[$2$] $times$ #text(fill: ACC)[$3$]]) med underbrace(mat(1; 2; 1), #[#text(fill: ACC)[$3$] $times$ #text(fill: GREEN)[$1$]]) = underbrace(mat(1 dot 1 + 2 dot 2 + 0 dot 1; 0 dot 1 + 1 dot 2 + 3 dot 1), #[#text(fill: BLUE)[$2$] $times$ #text(fill: GREEN)[$1$]]) = mat(5; 5) $

#pause
The general contract — inner dimensions kiss and cancel, outer dimensions survive:

$ (#text(fill: BLUE)[$m$] times #text(fill: ACC)[$n$]) dot (#text(fill: ACC)[$n$] times #text(fill: GREEN)[$1$]) arrow.r (#text(fill: BLUE)[$m$] times #text(fill: GREEN)[$1$]) $

#pause
#result[An $m times n$ matrix eats vectors from $RR^#text(fill: ACC)[$n$]$ and returns vectors in $RR^#text(fill: BLUE)[$m$]$ — it can move you *between* spaces.]

== Code · one `@` moves a million points

#codebox[```python
import numpy as np
th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)],
              [np.sin(th),  np.cos(th)]])

R @ np.array([2.0, 1.0])        # array([1.232, 1.866]) — one point moved
```]

#pause
#codebox[```python
pts = np.random.rand(1_000_000, 2)   # a million pixel coordinates
new = pts @ R.T                      # ...all rotated in one line
```]

#pause
That second line *is* the photo app from the opening slide. No loops — the matrix moves all of space at once, and hardware loves it.

== Symbols · what "linear map" means

A map $T : RR^n arrow.r RR^m$ is *linear* if for all vectors $bold(u), bold(v)$ and scalars $c$:

$ T(bold(u) + bold(v)) = T(bold(u)) + T(bold(v)), quad quad T(c bold(u)) = c thin T(bold(u)) $

#pause
- Every matrix gives a linear map $bold(x) arrow.r.bar A bold(x)$ — the two laws are exactly how the row-dot-product formula behaves.
#pause
- The converse holds too (stated, not proved): *every* linear map $RR^n arrow.r RR^m$ is multiplication by some $m times n$ matrix. There is nothing else.

#pause
#result[“Matrix” and “linear move of space” are one concept in two costumes.]

= Read the columns

== The trick · watch just two vectors #V

#fig("/lecture4/figures/basis_tracking.svg", w: 74%)

#pause
You never need to track the whole plane — *two arrows carry the entire story*.

== The columns are the landed basis #D

Where does $A = mat(2, 1; 1, 2)$ send the basis? Just multiply:

$ A bold(e)_1 = mat(2, 1; 1, 2) mat(1; 0) = mat(2; 1) = "column 1", quad quad A bold(e)_2 = mat(2, 1; 1, 2) mat(0; 1) = mat(1; 2) = "column 2" $

#pause
This is completely general — feeding in $bold(e)_j$ picks out column $j$:

$ A bold(e)_j = "the " j"-th column of" A $

#pause
#notebox[Sanity check with the do-nothing move: the identity $I = mat(1, 0; 0, 1)$ has columns $bold(e)_1, bold(e)_2$ — "the basis lands exactly where it started." $I bold(x) = bold(x)$ for every $bold(x)$.]

== The column picture of $A bold(x)$ #D

Why do two arrows determine everything? Any input is a mix of basis vectors:

$ bold(x) = mat(x_1; x_2) = x_1 bold(e)_1 + x_2 bold(e)_2 $

#pause
Now apply $A$ and let linearity do all the work:

$ A bold(x) = A(x_1 bold(e)_1 + x_2 bold(e)_2) = x_1 (A bold(e)_1) + x_2 (A bold(e)_2) $

#pause
But $A bold(e)_1, A bold(e)_2$ are the *columns* $bold(a)_1, bold(a)_2$:

$ A bold(x) = x_1 bold(a)_1 + x_2 bold(a)_2 $

#pause
#result[$A bold(x)$ is a *recipe*: mix the columns of $A$, using the entries of $bold(x)$ as amounts. (Strang's column picture.)]

== Numbers · two readings, one answer

Compute $A bold(x)$ for $A = mat(2, 1; 1, 2)$, $bold(x) = mat(2; 1)$ — both ways:

#pause
*Row view* — two dot products:

$ mat(2, 1; 1, 2) mat(2; 1) = mat(2 dot 2 + 1 dot 1; 1 dot 2 + 2 dot 1) = mat(5; 4) $

#pause
*Column view* — a mix of columns, 2 parts and 1 part:

$ 2 mat(2; 1) + 1 mat(1; 2) = mat(4; 2) + mat(1; 2) = mat(5; 4) quad #text(fill: GREEN)[✓ same answer] $

#pause
The row view is how machines compute. The column view is how *you should think* — it will explain rank, $A bold(x) = bold(b)$, and least squares within the hour.

== Construct a matrix for a desired transformation

Want a matrix that rotates by $90 degree$? Don't memorize — *place the columns*:

#pause
$bold(e)_1$ must land at $(0, 1)$; $bold(e)_2$ at $(-1, 0)$. The landings *are* the columns:

$ R_(90 degree) = mat(0, -1; 1, 0) $

#pause
A "shadow" matrix that flattens everything onto the $x$-axis? $bold(e)_1$ stays, $bold(e)_2$ dies:

$ P = mat(1, 0; 0, 0) $

#pause
#notebox[That second design *crushed a whole dimension on purpose*. Hold the thought — it returns in ten minutes with a name: rank 1.]

== Checkpoint: identify the transformation #Q

#fig("/lecture4/figures/mcq_transform.svg", w: 52%)

#mcq([Which matrix $M$ made this move?],
  [$M = mat(1, 1; 0, 1)$],
  [$M = mat(1, 0; 1, 1)$],
  [$M = mat(0, 1; 1, 0)$],
  [$M = mat(2, 0; 0, 2)$],
)

== Answer · read the columns #A

#mcq-answer([B], [$M = mat(1, 0; 1, 1)$],
  [Read the landings off the picture: $bold(e)_1$ landed at $(1, 1)$ — that must be column 1. $bold(e)_2$ stayed at $(0, 1)$ — that must be column 2. Only $mat(1, 0; 1, 1)$ has those columns. (Option A is the *horizontal* shear from Exhibit 3 — it would leave $bold(e)_1$ untouched instead.)])

#pause
Every "which matrix did this?" puzzle is solved the same way: *watch where the basis lands*.

= Multiplication is composition

== Two moves, one after the other #V

#fig("/lecture4/figures/composition_steps.svg", w: 94%)

#pause
Rotate with $R$, *then* shear with $S$: the input $bold(x)$ becomes $S(R bold(x))$.

#pause
Two linear moves in a row are still one linear move — so *some single matrix* must do the whole trip. We name it $S R$, written right-to-left: the move nearest $bold(x)$ acts first.

== The combined matrix, without any formula

Where does the *combined* move send the basis? Just follow each arrow through both moves:

$ (S R) bold(e)_1 = S(R bold(e)_1) = S mat(0; 1) = mat(1; 1), quad quad (S R) bold(e)_2 = S(R bold(e)_2) = S mat(-1; 0) = mat(-1; 0) $

#pause
The landings are the columns — the product is already done:

$ S R = mat(1, -1; 1, 0) $

#pause
#result[To multiply matrices, push the columns of the right one through the left one: column $j$ of $A B$ is $A bold(b)_j$.]

== The general recipe (and its shape)

$ (A B)_(i j) = sum_(k) A_(i k) B_(k j) quad — "row " i " of " A " dotted with column " j " of " B $

#pause
The shape contract again — inner dimensions consumed, outer survive:

$ (#text(fill: BLUE)[$m$] times #text(fill: ACC)[$n$]) dot (#text(fill: ACC)[$n$] times #text(fill: GREEN)[$p$]) arrow.r (#text(fill: BLUE)[$m$] times #text(fill: GREEN)[$p$]) $

#pause
- The entry formula is the *bookkeeping* view — what hardware executes.
- Column-through view is the *thinking* view: $A B$ = "$A$ applied to each column of $B$" — a matmul is just $p$ matvecs standing side by side.

== Order matters #V

#fig("/lecture4/figures/composition_order.svg", w: 61%)

#pause
Rotate-then-shear and shear-then-rotate end in *different places* — so $S R eq.not R S$ before we compute a thing.

== $S R$ vs $R S$ · the numbers agree with the picture

Track the basis both ways (or grind the entry formula — same result):

$ S R = mat(1, -1; 1, 0), quad quad R S = mat(0, -1; 1, 1) $

#pause
Different matrices, as the houses foretold.

#alertbox[$A B eq.not B A$ in general. Matrix multiplication remembers the *order of moves* — "put on socks, then shoes" is not "put on shoes, then socks." Every identity you write this semester must respect it.]

#pause
What *does* survive from ordinary algebra: associativity, $(A B) C = A(B C)$ — grouping is free, order is not.

== Interactive · watch the machine run #I

#interbox(link-to: "http://matrixmultiplication.xyz/")[
  Enter $S = mat(1, 1; 0, 1)$ and $R = mat(0, -1; 1, 0)$ and watch the row-through-column machine grind out each entry — the row physically tips over and slides down the column. In class: predict every entry of $R S$ *before* the animation lands.
]

#pause
Then feed it a $(2 times 3)(3 times 1)$ pair and watch the inner $3$s consume each other — the shape contract, animated.

== Code · `@` composes

#codebox[```python
S = np.array([[1., 1.], [0., 1.]])
R = np.array([[0., -1.], [1., 0.]])

S @ R                 # array([[ 1., -1.], [ 1.,  0.]])
R @ S                 # array([[ 0., -1.], [ 1.,  1.]])  — different!

x = np.array([2., 1.])
np.allclose((S @ R) @ x, S @ (R @ x))    # True — associativity
```]

#pause
That last line is quietly profound: *precompute the combined move once*, or apply moves one at a time — identical result. Graphics engines and deep-learning compilers live on this choice.

== Why stacked linear layers need a nonlinearity

Stack two purely linear layers, $bold(h) = W_1 bold(x)$ then $bold(y) = W_2 bold(h)$:

$ bold(y) = W_2 (W_1 bold(x)) = (W_2 W_1) bold(x) = W_("eq") bold(x) $

#pause
Composition collapses the stack: a 100-layer *linear* network is secretly *one matrix* — depth bought nothing.

#pause
#result[That is exactly why every real layer wraps $W bold(x) + bold(b)$ in a nonlinear $phi$: it blocks the collapse, forcing each matrix to contribute a genuinely new move.]

#pause
Move of space → bend → move of space → bend… — that alternation *is* deep learning. The moves are today's math; the bends belong to ES 667.

= Rank: how much of space survives

== Not every move preserves the plane #V

#fig("/lecture4/figures/rank_collapse.svg", w: 79%)

#pause
$C = mat(1, 2; 2, 4)$ takes the *entire 2-D plane* — houses, point clouds, everything — and lands it all on a single line. Something fundamental was lost.

== The collapse, in numbers

Watch the columns of $C = mat(1, 2; 2, 4)$: column 2 is exactly $2 times$ column 1.

#pause
So every output — every mix $x_1 bold(a)_1 + x_2 bold(a)_2$ — is a multiple of $mat(1; 2)$:

$ C bold(x) = x_1 mat(1; 2) + x_2 mat(2; 4) = (x_1 + 2 x_2) mat(1; 2) quad — "one direction, scaled. The line " y = 2x. $

#pause
And one whole direction of inputs is *doomed*:

$ C mat(2; -1) = 2 mat(1; 2) - 1 mat(2; 4) = mat(0; 0) $

#pause
The columns fought over the same direction — and an entire dimension of information died in the crossfire.

== What the columns can reach · column space #V

#fig("/lecture4/figures/column_space.svg", w: 70%)

#pause
$"col"(A) = "span"{bold(a)_1, ..., bold(a)_n}$ — L3's *span* of the columns: everything $A bold(x)$ can ever reach.

#pause
Dependent columns shrink the span to a line, and most targets $bold(b)$ become *unreachable* — exactly when $A bold(x) = bold(b)$ has no solution. (File that away; it returns in twenty minutes.)

== Rank · the dimension of what survives

$ "rank"(A) = "number of linearly independent columns" = dim("col"(A)) $

#pause
#table(
  columns: (1fr, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Move*], [*Matrix*], [*Rank*], [*What survives*]),
  [rotation $R_(30 degree)$], [$mat(0.87, -0.5; 0.5, 0.87)$], [2], [the whole plane],
  [shear $S$], [$mat(1, 1; 0, 1)$], [2], [the whole plane],
  [shadow $P$], [$mat(1, 0; 0, 0)$], [1], [the $x$-axis],
  [collapse $C$], [$mat(1, 2; 2, 4)$], [1], [the line $y = 2x$],
  [zero matrix $O$], [$mat(0, 0; 0, 0)$], [0], [just the origin],
)

#pause
For any $m times n$ matrix: $"rank"(A) <= min(m, n)$ — you can't have more independent columns than columns, nor than the dimension they live in. Hitting the ceiling is called *full rank*.

== The null space · what dies

The flip side of "what survives" is *which inputs get crushed to zero*:

$ "null"(A) = {bold(x) : A bold(x) = bold(0)} quad quad "for " C ": the whole line through" mat(2; -1) $

#pause
Dimensions are conserved — every input dimension either survives or dies:

$ underbrace("rank"(A), "survives") + underbrace(dim "null"(A), "dies") = underbrace(n, "input dimensions") quad quad "for " C ": " 1 + 1 = 2 #text(fill: GREEN)[✓] $

(The *rank–nullity theorem* — stated here, proved in MML 2.7.)

#pause
#notebox[L5 plant: a direction that $A$ merely *scales* — stretches without turning — is called an *eigenvector*. A doomed direction is the extreme case: an eigenvector with stretch factor $0$. Next lecture is entirely about these special directions.]

== Rank by hand, one size up

$ A = mat(0, 1, 2; 1, 2, 1; 2, 7, 8) quad — 3 times 3", so rank" <= 3". Hunt for a dependency." $

#pause
Row 3 is a mix of the others: $3 dot "row 1" + 2 dot "row 2" = (2, 7, 8) = "row 3"$. Only 2 independent rows → $"rank"(A) = 2$.

#pause
#notebox[Row rank always equals column rank (MML 2.6), so test dependencies on whichever side is easier. Here the row dependency is immediate.]

#pause
One more: $X = mat(1, 2, 4, 4; 3, 4, 8, 0)$ is $2 times 4$, so $"rank" <= 2$; its two rows aren't proportional → $"rank"(X) = 2$ — full rank, even though *some* columns repeat each other.

== The thinnest matrices · outer products #D

Take $bold(u) = mat(1; 2; 3)$ and $bold(v) = mat(4; 5; 6)$. Their *outer product* $bold(u) bold(v)^top$ is a full $3 times 3$ matrix:

#two(r: (58%, 42%),
  [
    $ bold(u) bold(v)^top = mat(1; 2; 3) mat(4, 5, 6) = mat(4, 5, 6; 8, 10, 12; 12, 15, 18) $
    #v(6pt)
    Cell $(i, j)$ is simply $u_i v_j$ — a multiplication table.
  ],
  align(center, grid-map(
    ((4, 5, 6), (8, 10, 12), (12, 15, 18)),
    cell: 14mm,
    row-labels: ($u_1 = 1$, $u_2 = 2$, $u_3 = 3$),
    col-labels: ($v_1 = 4$, $v_2 = 5$, $v_3 = 6$),
  )),
)

#pause
Nine entries, but look at the shading: every column repeats the *same pattern*, just rescaled. This matrix is enormous bureaucracy around very little information.

== Why $"rank"(bold(u) bold(v)^top) = 1$ #D

*Claim:* for any nonzero $bold(u), bold(v)$, the outer product $bold(u) bold(v)^top$ has rank exactly 1.

#pause
*Look at the columns.* Column $j$ of $bold(u) bold(v)^top$ is $v_j bold(u)$ — every column is a multiple of the *same* vector:

$ "col 1" = 4 bold(u), quad "col 2" = 5 bold(u), quad "col 3" = 6 bold(u) $

#pause
So the column space is just the line through $bold(u)$: dimension 1 → rank 1. $qed$

#pause
*Second look, via the action:* $(bold(u) bold(v)^top) bold(x) = bold(u) (bold(v)^top bold(x))$ — a *number* times $bold(u)$. Every input in $RR^3$ lands on $bold(u)$'s line: the map measures you against $bold(v)$, then points along $bold(u)$.

#pause
#result[An outer product $bold(u) bold(v)^top$ is the thinnest nonzero matrix there is: rank 1.]

== Every matrix is a sum of rank-1 atoms

Read the claim backwards: rank-1 outer products are *atoms*, and every matrix is a molecule —

$ A = sigma_1 bold(u)_1 bold(v)_1^top + sigma_2 bold(u)_2 bold(v)_2^top + dots.c + sigma_r bold(u)_r bold(v)_r^top, quad r = "rank"(A) $

#pause
- L6 preview: the *SVD* finds the best atoms, importance-sorted $sigma_1 >= sigma_2 >= dots$ — keep the top few → compression, PCA.
#pause
- Modern fine-tuning (*LoRA*) freezes a giant $W$ and learns only a low-rank update — a few outer products instead of billions of entries.

#pause
#notebox[The statement "every column is a multiple of $bold(u)$" is the rank-one structure used by low-rank parameter updates in L6.]

= Invertibility: undoing a move

== Can you undo the move? #V

#fig("/lecture4/figures/invert_or_not.svg", w: 79%)

#pause
The rotation is *reversible* — every landing point has exactly one origin story. The collapse is not: $bold(p)$ and $bold(q)$ land on the *same* output, and "which input was it?" has no answer.

== The inverse · undo, in numbers

$A^(-1)$ is the move that cancels $A$ exactly — do, then undo, and nothing happened:

$ A^(-1) A = A A^(-1) = I $

#pause
For our exhibits the undo is guessable from the *geometry*, no formula needed:

$ R_(30 degree)^(-1) = R_(-30 degree) quad "(rotate back)", quad quad S^(-1) = mat(1, -1; 0, 1) quad "(slide the top back left)" $

#pause
Verify the shear by composing:

$ mat(1, -1; 0, 1) mat(1, 1; 0, 1) = mat(1, 0; 0, 1) = I quad #text(fill: GREEN)[✓] $

#pause
When an inverse exists, it is *unique* — there is only one way to perfectly undo a move.

== When can you undo? One test in three costumes

For a *square* matrix $A$, these are the same single question:

#pause
- *full rank* — columns independent: the outputs fill all of $RR^n$, nothing was flattened
#pause
- *null space is only ${bold(0)}$* — no nonzero direction is removed, so distinct inputs do not collide
#pause
- $det(A) eq.not 0$ — the determinant measures the *area scale factor* of the move; $det = 0$ means areas got crushed flat (we'll meet $det$ properly in L5)

#pause
A matrix passing the test is *invertible* (nonsingular); failing, *singular*.

#pause
#result[Invertible $arrow.l.r.double$ full rank $arrow.l.r.double$ nothing is crushed. You can undo a move exactly when it destroyed no information.]

== Checkpoint · spot the un-undoable #Q

#mcq([Without computing a single determinant: which matrix has *no inverse*? (Hint: one of them flattens the plane.)],
  [$mat(0, -1; 1, 0)$],
  [$mat(1, 2; 2, 4)$],
  [$mat(1, 1; 0, 1)$],
  [$mat(2, 0; 0, 1)$],
)

== Answer · spot the un-undoable #A

#mcq-answer([B], [$mat(1, 2; 2, 4)$ is singular],
  [Its second column is $2 times$ the first — dependent columns → rank 1 → the plane is flattened onto a line → two different inputs share each output, so no undo can exist. The rotation (A), shear (C), and scale (D) all keep rank 2: nothing dies, so each can be undone.])

#pause
Notice you *never* needed arithmetic — reading the columns answered a determinant question.

= Solving Ax = b

== The most-solved equation in computing

Flip the question. So far: given $bold(x)$, where does it land? Now *the landing $bold(b)$ is known; the input is not*:

$ A bold(x) = bold(b) quad — "which input lands on " bold(b)"?" $

#pause
The column picture makes it concrete — *find the recipe* mixing the columns into $bold(b)$:

$ mat(2, 1; 1, 2) bold(x) = mat(4; 5) quad arrow.r.double quad bold(x) = mat(1; 2)": " quad 1 mat(2; 1) + 2 mat(1; 2) = mat(4; 5) quad #text(fill: GREEN)[✓] $

#pause
Solvable for every $bold(b)$, uniquely, exactly when $A$ is invertible — nothing crushed.
#pause
By hand for two unknowns; but one weather step or network layer has $n approx 10^6$. We need the machine — *carefully*.

== On paper, $bold(x) = A^(-1) bold(b)$ · on silicon, never

$bold(x) = A^(-1) bold(b)$ is a theorem, *not an algorithm*. Watch both routes on a notoriously nasty (but tiny!) matrix — the $12 times 12$ Hilbert matrix of simple fractions $H_(i j) = 1\/(i + j + 1)$, with true solution all-ones:

#codebox[```python
n = 12; i = np.arange(n)
H = 1.0 / (i[:, None] + i[None, :] + 1)      # Hilbert matrix
x_true = np.ones(n);  b = H @ x_true

np.abs(np.linalg.inv(H) @ b - x_true).max()   # 8.66    ← inv route
np.abs(np.linalg.solve(H, b) - x_true).max()  # 0.143   ← solve route
```]

#pause
The true entries are all $1$. The `inv` route is off by *8.66* — a 60× larger error than `solve`, on a matrix that fits on this slide.

== Why `solve` wins

- *Fewer operations* — `inv` secretly solves $n$ systems (one per column of $I$), then still multiplies: measured $approx 2.3 times$ slower at $n = 2000$, and every extra operation is an extra rounding.
#pause
- *One rounding cascade, not two* — L2 callback: each float op tells a tiny lie. `solve` factors $A$ once (LU factorization — L17 opens the box) and back-substitutes; `inv` commits its lies into an explicit $A^(-1)$, then compounds them in the multiply.

#pause
#alertbox[Write `np.linalg.solve(A, b)`. An `inv(A) @ b` in numerical code is a code-review flag across the entire industry — same theorem, worse arithmetic.]

#pause
#notebox[Honest footnote: $"cond"(H) approx 10^16$, so even `solve` sweated here (error 0.14). Why some matrices *amplify* rounding lies — the condition number — is Lecture 17's opening act.]

== Too many equations · tall systems

Fit a line $hat(y) = a + b t$ through *three* points $(0, 1), (1, 3), (2, 4)$ — three equations, two unknowns:

$ underbrace(mat(1, 0; 1, 1; 1, 2), #[#text(fill: BLUE)[$3$] $times$ #text(fill: ACC)[$2$] — tall]) mat(a; b) = underbrace(mat(1; 3; 4), bold(b) in RR^3) $

#pause
Now the column picture delivers bad news: $"col"(A)$ — everything the two columns can mix — is only a *plane* inside $RR^3$, and our $bold(b)$ is off it.

#pause
#result[A tall system generically has *no exact solution*: three demands, two dials.]

== No line fits · the miss, drawn #V

#fig("/lecture4/figures/least_squares.svg", w: 74%)

#pause
Left: no line threads all three points — every candidate leaves *residuals*. Right: the same fact in column-space language — $bold(b)$ hovers off the reachable plane.

== Least squares · aim for the shadow

If you can't reach $bold(b)$, reach the *closest point you can* — minimize the miss:

$ hat(bold(x)) = arg min_(bold(x)) norm(A bold(x) - bold(b))^2 $

#pause
Geometrically (right panel, a moment ago): $A hat(bold(x))$ is the *projection* — L3's shadow! — of $bold(b)$ onto $"col"(A)$; the leftover error sticks out perpendicular.

#pause
#codebox[```python
A = np.array([[1., 0.], [1., 1.], [1., 2.]])
b = np.array([1., 3., 4.])
np.linalg.lstsq(A, b, rcond=None)[0]   # (7/6, 3/2): best line 7/6 + 1.5·t
```]

#pause
#notebox[Two IOUs: *L6* computes least squares at scale (via SVD), and *L14* reveals why *squared* error is the statistically right miss to minimize. Fitting = projection is the bridge.]

== Lecture 4 — summary

- *A matrix is a function*: $bold(x) arrow.r.bar A bold(x)$ moves all of space — lines stay straight, parallels parallel, origin fixed.
- *Columns = landed basis*: $A bold(e)_j = bold(a)_j$, so $A bold(x) = x_1 bold(a)_1 + dots + x_n bold(a)_n$ — a recipe mixing columns.
- *Multiplication is composition*: column $j$ of $A B$ is $A bold(b)_j$; order matters, grouping doesn't.
- *Rank* = dimension of what survives; *null space* = what dies; $"rank" + dim "null" = n$; $bold(u) bold(v)^top$ = the rank-1 atom.
- *Invertible $arrow.l.r.double$ full rank*: undo exists iff no information was destroyed.
- *`solve`, never `inv`*, for $A bold(x) = bold(b)$ — fewer roundings, faster (conditioning: L17).
- *Tall systems*: no exact solution — least squares projects $bold(b)$ onto $"col"(A)$ (L6, L14).

#pause
#notebox[*Read before L5* — MML Ch 2.5–2.8. *Tutorial 2* this week: the transformation gallery, rank experiments, and the Hilbert `solve`-vs-`inv` shoot-out, in NumPy.]

== Practice problems · I

Try on paper; verify in the Tutorial 2 notebook.

+ Design by columns: the matrix reflecting across the $x$-axis, and the one rotating by $180 degree$. Check by tracking $bold(e)_1, bold(e)_2$.
+ Compute $mat(1, 2; 3, 4) mat(0, 1; 1, 0)$ and $mat(0, 1; 1, 0) mat(1, 2; 3, 4)$ — one swaps columns, one swaps rows. Which is which, and why?
+ Ranks of $mat(1, 2, 4, 4; 3, 4, 8, 0)$ and $mat(5, 4, 3; 4, 3, 2; 3, 2, 1)$? Hunt for dependencies before trusting your gut.

== Practice problems · II

#set enum(start: 4)
+ For $bold(u) = mat(1; 2)$, $bold(v) = mat(3; 1)$: write $bold(u) bold(v)^top$ and find the direction it kills. Hint: what is perpendicular to $bold(v)$?
+ Socks and shoes: why is $(A B)^(-1) = B^(-1) A^(-1)$? Think "undo two moves in the right order" — then verify on $S R$.
+ On the $n = 8$ Hilbert matrix, predict the more accurate of `solve` and `inv @ b`; run both. At what $n$ does `solve` itself lose digits?

#focus-slide[
  A matrix is a verb, not a table.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Eigendecomposition* — the directions a matrix cannot turn, and how Google used one matrix, applied over and over, to rank the entire web.
]
