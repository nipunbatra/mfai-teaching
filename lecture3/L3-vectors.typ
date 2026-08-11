// Vectors: the Geometry of Data — Lecture 3 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture3/L3-vectors.typ
//   typst compile --root . --input handout=true lecture3/L3-vectors.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Vectors: the Geometry of Data],
  subtitle: [Data becomes arrows; similarity becomes an angle],
)

#title-slide()

= Word embeddings as vectors

== The word-embedding example from Lecture 1 #V

Lecture 2 explained the `nan` loss. The learning-rate example returns in L17. We now study the vector relation:

$ "king" - "man" + "woman" approx "queen" $

#fig("/lecture1/figures/embedding_arithmetic.svg", w: 38%)

#pause
The relation is a statement about vector displacement and direction; we will verify both numerically.

== A second example · "people like you"

Every streaming app claims to know "people like you". But taste isn't a number — what could "like you" even *mean*, mathematically?

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Film*], [*You*], [*Priya*], [*Rohan*]),
  [Dangal], [5], [4], [0],
  [3 Idiots], [4], [5], [1],
  [Interstellar], [1], [0], [5],
  [Chak De! India], [#text(fill: ACC)[*?*]], [5], [2],
)

#pause
Who is more "like you" — Priya or Rohan? *By how much?* And should the app pitch you _Chak De!_? Within the hour you'll compute all three answers — with the same formula that explains king − man + woman.

== Today's one idea

#result[A data point is a *vector*. Similarity between data points is *geometry* — ultimately, an angle.]

#pause
The route, tool by tool:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Step*], [*Tool*], [*Use*]),
  [length], [norms $norm(bold(x))$], [how big, how far apart],
  [angle], [dot product, cosine], [both questions above],
  [shadow], [projection], [the germ of least squares & PCA],
  [grid], [span, basis], [what coordinates even mean],
)

#pause
Module 1 starts here: L3 vectors → L4 matrices (they *move* vectors) → L5 eigenvectors → L6 SVD & PCA.

== Learning outcomes

By the end of this lecture you will be able to:

+ Read a vector both ways: an *arrow* in space and a *row* of a dataset.
+ Measure size with *L1 / L2 norms* — and draw each norm's "unit ball".
+ Compute the *dot product* three ways: multiply–add, lengths × cos θ, shadow.
+ Derive $cos theta = bold(x)^top bold(y) \/ (norm(bold(x)) norm(bold(y)))$ from the school law of cosines (⭐).
+ Rank *real word embeddings* by cosine similarity and evaluate the opening analogy.
+ *Project* one vector onto another and verify the leftover is orthogonal.
+ Say when vectors *span* a line vs the plane, and what an (orthonormal) *basis* buys.

= Vectors: arrows and data rows

== One object, two readings #V

#fig("/lecture3/figures/vector_two_views.svg", w: 70%)

#pause
Left: $(3, 2)$ is an *arrow* — a direction plus a length. Right: the same kind of pair is a *row of data* — Asha scored $(8, 6)$ in (maths, physics).

#pause
*One object, two readings* — an arrow you can draw, a row you can store. All of today is about switching between them at will.

== Many AI objects can be represented this way

#table(
  columns: (1fr, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Object*], [*The vector*], [*Lives in*]),
  [Asha's marks], [$(8, 6)$], [$RR^2$],
  [your film ratings], [$(5, 4, 1, ...)$], [$RR^(\#"films")$],
  [the word "king" (GloVe)], [50 learned numbers], [$RR^50$],
  [one MNIST digit], [$28 times 28 = 784$ pixel values], [$RR^784$],
  [3 s of speech at 16 kHz], [48,000 samples], [$RR^48000$],
)

#pause
L2 callback: these entries are often stored as float32 or a lower-precision format — each coordinate then lives on an unevenly spaced number line.

#pause
#result[From here on, "data" means: $n$ points in $RR^d$ — $n$ = how many, $d$ = how rich.]

== Two moves only: add, and scale #V

#fig("/lecture3/figures/vector_ops.svg", w: 66%)

#pause
Addition: walk $bold(u)$, then walk $bold(v)$ — tip to tail. Scaling: stretch, shrink, or flip along the same line.

#pause
These two moves are the whole meaning of "linear". Everything in Module 1 — combinations, span, matrices — is built from exactly these.

== The moves, in numbers and code

With $bold(u) = (2, 1)$, $bold(v) = (1, 2)$: everything happens slot by slot —

$ bold(u) + bold(v) = (3, 3) quad quad 2bold(u) = (4, 2) quad quad -bold(u) = (-2, -1) $

#pause
#codebox[```python
u, v = np.array([2., 1.]), np.array([1., 2.])
u + v        # array([3., 3.])
2 * u        # array([4., 2.])
u * v        # array([2., 2.])  ← slot-by-slot, NOT the dot product
```]

#pause
#notebox[`u * v` multiplies slot-by-slot and keeps a *vector*. The product that carries geometry — `u @ v` — collapses two vectors into one number. It gets its own section, two sections from now.]

== The notation contract

$ bold(x) = vec(x_1, dots.v, x_d) in RR^d quad quad bold(x)^top = (x_1, ..., x_d) $

#pause
- *bold lowercase* $bold(x)$: vectors (columns by default) · _italic_ $x_i$: scalars & components
- CAPS $A, X$: matrices — from L4 on; the dataset is $X in RR^(n times d)$, one point per row
- $n$ data points, $d$ dimensions; an unadorned $norm(bold(x))$ always means the L2 norm

#pause
#notebox[This is the same contract as MML and the CS229 linear-algebra handout — decks, tutorials, and assignments all speak it. Learn it once, read everything.]

= Norms: how big is a vector?

== Two honest ways to measure one trip #V

#fig("/lecture3/figures/taxicab.svg", w: 44%)

#pause
Walk the streets: $|3| + |2| = 5$ blocks. Fly straight: $sqrt(3^2 + 2^2) = sqrt(13) approx 3.61$.

#pause
Both are legitimate "sizes" of the same vector — they answer different questions. Let's formalize both.

== The L2 norm · as the crow flies

$ norm(bold(x))_2 = sqrt(x_1^2 + x_2^2 + dots.c + x_d^2) = sqrt(bold(x)^top bold(x)) $

For the trip: $norm((3, 2))_2 = sqrt(13) approx 3.61$. Works in any dimension — for $(1, -2, 2) in RR^3$: $sqrt(1 + 4 + 4) = 3$.

#pause
- This is *the* default: $norm(bold(x))$ with no subscript means L2.
- It is Pythagoras, applied $d-1$ times — flight distance in $RR^d$.

#pause
Keep the inner form $sqrt(bold(x)^top bold(x))$ in view: *the length is secretly a product of $bold(x)$ with itself* — that observation runs the next section.

== The L1 norm · street by street

$ norm(bold(x))_1 = |x_1| + |x_2| + dots.c + |x_d| $

#pause
Worked example — a model's errors on four students, $bold(e) = (2, -1, 3, -4)$:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Norm*], [*Value*], [*What it judges*]),
  [$norm(bold(e))_1$], [$2 + 1 + 3 + 4 = 10$], [total miss — every error counts equally],
  [$norm(bold(e))_2$], [$sqrt(30) approx 5.48$], [squares amplify — big misses dominate],
  [$norm(bold(e))_infinity$], [$max = 4$], [only the single worst case],
)

#pause
One model, three verdicts. *Choosing a norm is choosing what you care about.*

== Every norm draws a different "circle" #V

#fig("/lecture3/figures/unit_balls.svg", w: 46%)

#pause
All three earn the name "norm" by the same three axioms (MML 3.1): positive unless $bold(x) = bold(0)$, $norm(c bold(x)) = |c| thin norm(bold(x))$, triangle inequality.

#pause
The L1 diamond has *corners* on the axes — the geometric clue for why L1 regularization can set weights exactly to zero. (ES 335 develops that result.)

== Unit vectors · direction, distilled

$ hat(bold(x)) = bold(x) / norm(bold(x)) quad quad "same direction, length exactly 1" $

Numbers: $bold(x) = (3, 2)$: $hat(bold(x)) = (3, 2)\/sqrt(13) approx (0.832, 0.555)$ — check: $0.832^2 + 0.555^2 approx 1$ ✓

#pause
Every nonzero vector splits cleanly: $bold(x) = norm(bold(x)) dot hat(bold(x))$ — *how much* × *which way*.

#pause
Keep this distinction: cosine similarity will use direction and discard magnitude.

== Distance · a norm in disguise

$ d(bold(x), bold(y)) = norm(bold(x) - bold(y)) $

Asha $(8, 6)$ vs Vikram $(3, 4)$: difference $(5, 2)$ → flight distance $sqrt(29) approx 5.39$, street distance $7$.

#pause
#codebox[```python
asha, vikram = np.array([8., 6.]), np.array([3., 4.])
np.linalg.norm(asha - vikram)       # 5.385...  (L2)
np.linalg.norm(asha - vikram, 1)    # 7.0       (L1)
```]

#pause
#notebox[Every "nearest" in ML — nearest neighbour, nearest cluster centre, nearest embedding — is this one line of code plus a choice of norm.]

= One dot product, three readings

== Reading 1 · multiply matching slots, then add

$ bold(x)^top bold(y) = x_1 y_1 + x_2 y_2 + dots.c + x_d y_d = sum_(i=1)^d x_i y_i $

The running pair for the rest of this lecture: $bold(x) = (3, 2)$, $bold(y) = (4, 2)$:

$ bold(x)^top bold(y) = 3 dot 4 + 2 dot 2 = 16 $

#pause
Two vectors in, *one number* out. But what does "16" mean? Hold that thought.

#pause
One meaning is already ours — pair $bold(x)$ with itself: $bold(x)^top bold(x) = 9 + 4 = 13 = norm(bold(x))^2$.

#result[$norm(bold(x)) = sqrt(bold(x)^top bold(x))$ — the length was a dot product all along.]

== Reading 2 · lengths × alignment

The claim (to be *earned* next section — it is today's ⭐):

$ bold(x)^top bold(y) = norm(bold(x)) thin norm(bold(y)) cos #text(fill: ACC)[$theta$] $

#pause
Check it against the running pair: $norm(bold(x)) norm(bold(y)) = sqrt(13) sqrt(20) approx 16.12$, so

$ cos theta = 16 / 16.12 approx 0.992 quad => quad theta approx 7.1 degree quad "— two nearly parallel arrows." $

#pause
Pause on how strange this is: pure multiply–add on coordinates apparently *knows the angle*. No protractor was consulted. We will demand proof shortly.

== Reading 3 · the shadow #V

#fig("/lecture3/figures/projection_shadow.svg", w: 44%)

#pause
Dot $bold(x)$ against a *unit* direction: $bold(x)^top hat(bold(y)) = norm(bold(x)) cos theta$ = the signed *length of $bold(x)$'s shadow* on $bold(y)$'s line. Running pair: $16\/sqrt(20) approx 3.58$.

#pause
Read it as: "how much of $bold(x)$ points the $bold(y)$ way."

== The alignment detector

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Angle*], [$cos theta$], [$bold(x)^top bold(y)$], [*Interpretation*]),
  [acute ($theta < 90 degree$)], [$+$], [$+$], [the vectors agree],
  [right ($theta = 90 degree$)], [$0$], [$0$], [*orthogonal* — $bold(x) perp bold(y)$, zero signed projection],
  [obtuse ($theta > 90 degree$)], [$-$], [$-$], [they oppose],
)

#pause
Check: $(3, 2)^top (-2, 3) = -6 + 6 = 0$ → perpendicular. The axes too: $bold(e)_1^top bold(e)_2 = 0$.

#pause
Orthogonal directions have zero signed projection onto each other. This is a geometric statement; it does *not* by itself imply that two measured features are statistically independent. Orthogonality returns twice today: residuals, and the nicest bases.

== Three readings, one number

#codebox[```python
x, y = np.array([3., 2.]), np.array([4., 2.])
x @ y                                  # 16.0    reading 1: multiply-add
nx, ny = np.linalg.norm(x), np.linalg.norm(y)
theta = np.arccos(x @ y / (nx * ny))
np.degrees(theta)                      # 7.125   the angle between them
nx * np.cos(theta)                     # 3.578   reading 3: shadow length
(x @ y) / ny                           # 3.578   same shadow, no trig needed
```]

#pause
Plant for L4: a matrix multiply is nothing but these, en masse — every entry of $W bold(x)$ is one dot product $bold(w)_i^top bold(x)$, one "how much of $bold(x)$ lies along my direction?" question.

== Checkpoint: the perpendicular detector #Q

#mcq([$bold(x) = (3, 2)$. Which $bold(y)$ is *orthogonal* to $bold(x)$?],
  [$bold(y) = (2, 3)$],
  [$bold(y) = (-2, 3)$],
  [$bold(y) = (3, -2)$],
  [$bold(y) = (-3, -2)$],
)

== Answer: the perpendicular detector #A

#mcq-answer([B], [$(-2, 3)$, since $3 dot (-2) + 2 dot 3 = 0$],
  [in 2-D, "flip the slots, negate one": $(-x_2, x_1)$ is always $perp (x_1, x_2)$. A gives $12$ and C gives $5$ (acute); D gives $-13$ — *opposing*, but only exactly zero means perpendicular.])

= Why multiply–add measures angle

== Two currencies, one triangle #D

*Claim (⭐):* for any two nonzero vectors, $bold(x)^top bold(y) = norm(bold(x)) norm(bold(y)) cos theta$.

#fig("/lecture3/figures/law_cosines.svg", w: 40%)

#pause
The two sides of the claim speak different languages — coordinates vs geometry. The bridge: compute the third side $norm(bold(x) - bold(y))^2$ *twice*, once in each language, and compare.

== Step 1 · algebra expands the third side #D

$ norm(bold(x) - bold(y))^2 = (bold(x) - bold(y))^top (bold(x) - bold(y)) quad quad "(a norm"^2 "is a self-dot)" $

#pause
$ = bold(x)^top bold(x) - bold(x)^top bold(y) - bold(y)^top bold(x) + bold(y)^top bold(y) quad quad "(expand the brackets)" $

#pause
$ = norm(bold(x))^2 + norm(bold(y))^2 - 2 thin bold(x)^top bold(y) quad quad "(since" bold(y)^top bold(x) = bold(x)^top bold(y) ")" $

#pause
#notebox[Only two facts were used — brackets distribute, and order doesn't matter — and both read straight off the multiply–add formula: check $bold(x)^top (bold(y) + bold(z)) = sum_i x_i (y_i + z_i) = bold(x)^top bold(y) + bold(x)^top bold(z)$.]

== Step 2 · school geometry measures the same side #D

The *law of cosines* (Class 10): any triangle with sides $a, b, c$ and angle $gamma$ opposite $c$ obeys

$ c^2 = a^2 + b^2 - 2 a b cos gamma quad quad "(Pythagoras, plus a tilt correction)" $

#pause
Our triangle has sides $a = norm(bold(x))$, $b = norm(bold(y))$, $c = norm(bold(x) - bold(y))$, and $gamma = theta$:

$ norm(bold(x) - bold(y))^2 = norm(bold(x))^2 + norm(bold(y))^2 - 2 norm(bold(x)) norm(bold(y)) cos theta $

#pause
Sanity check the correction: at $theta = 90 degree$ the cosine dies and pure Pythagoras remains. ✓

== Step 3 · equate, cancel, done #D

Both computations measured the *same number* $norm(bold(x) - bold(y))^2$:

$ norm(bold(x))^2 + norm(bold(y))^2 - 2 thin bold(x)^top bold(y) = norm(bold(x))^2 + norm(bold(y))^2 - 2 norm(bold(x)) norm(bold(y)) cos theta $

#pause
Cancel the squared lengths, divide by $-2$:

#result[$bold(x)^top bold(y) = norm(bold(x)) norm(bold(y)) cos theta quad <==> quad cos theta = (bold(x)^top bold(y)) / (norm(bold(x)) thin norm(bold(y)))$]

#pause
Check on the running pair: $16 = 16.12 times 0.992$ ✓ ($theta approx 7.1 degree$, as promised).

#pause
Free corollary: $|cos theta| <= 1$ forces $|bold(x)^top bold(y)| <= norm(bold(x)) norm(bold(y))$ — that's *Cauchy–Schwarz*, bought for nothing.

= Cosine similarity: geometry becomes meaning

== Angles in $RR^300$, by decree

In $RR^2$ we *saw* $theta$ and proved the formula. In $RR^300$ nobody sees anything — so flip the logic and let the formula *define* the angle:

$ "cos_sim"(bold(x), bold(y)) := (bold(x)^top bold(y)) / (norm(bold(x)) norm(bold(y))) = hat(bold(x))^top hat(bold(y)) quad quad "— a dot product of unit vectors." $

#pause
This is legal and safe: Cauchy–Schwarz keeps it inside $[-1, 1]$, and any two non-collinear vectors span an ordinary 2-D plane where it *is* the visible angle. Collinear vectors give the endpoint angles $0 degree$ or $180 degree$.

#pause
#result[Cosine is scale-invariant: $"cos_sim"(10bold(x), bold(y)) = "cos_sim"(bold(x), bold(y))$. Lengths cancel — only *directions* are compared.]

== Cosine similarity at three angles #V

#fig("/lecture3/figures/angle_gallery.svg", w: 96%)

#pause
$+1$: same direction · $0$: perpendicular · $-1$: opposite direction. For data, cosine can be a useful similarity score, but zero cosine does not automatically mean statistical or semantic independence.

== "People like you", by hand

Ratings on the three films everyone has seen: you $= (5, 4, 1)$, Priya $= (4, 5, 0)$, Rohan $= (0, 1, 5)$.

#pause
$ "you" dot "Priya": quad (5 dot 4 + 4 dot 5 + 1 dot 0) / (sqrt(42) sqrt(41)) = 40 / 41.5 approx 0.96 quad => quad theta approx 15 degree $

#pause
$ "you" dot "Rohan": quad (0 + 4 + 5) / (sqrt(42) sqrt(26)) = 9 / 33.0 approx 0.27 quad => quad theta approx 74 degree $

#pause
#result[In this toy representation, a smaller angle means a more similar rating direction. Priya is therefore the nearer neighbour, and her 5★ rating suggests _Chak De!_.]

== What the toy recommender omits

The cosine calculation isolated one geometric step. A real system must also:

- distinguish "not rated" from an actual zero or low rating,
- account for users who rate everything high or low and films that are broadly popular,
- learn from many neighbours or fit latent user/item vectors rather than use three raw ratings.

#pause
#result[The example is useful for learning cosine geometry; it is not a complete recommendation algorithm.]

== A word is 50 floats

GloVe (2014): read 6 billion words of text; assign every word a vector so that words appearing in *similar contexts* get similar vectors. No dictionary, no grammar — just co-occurrence.

#codebox[```python
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")   # 400k words × 50 floats
glove["king"][:6]
# array([ 0.5045,  0.6861, -0.5952, -0.0228,  0.6005, -0.135 ])

def cos_sim(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
cos_sim(glove["king"], glove["queen"])       # 0.784  →  θ ≈ 38.4°
```]

#pause
Coordinate 17 of "king" is usually not interpretable on its own. The useful signal lies mainly in relations among the 50 coordinates — *the geometry between word vectors*.

== Cosine similarity for real word embeddings #V

#align(center, lines(
  fn: x => calc.cos(x * calc.pi / 180),
  domain: (0, 180), samples: 120, markers: false, colors: (INK,),
  points: (
    (38.4, 0.784, [king · queen]),
    (79.4, 0.184, [king · python]),
    (135.4, -0.711, [king · "rules-based"]),
  ),
  hlines: ((0.0,),),
  x-label: [angle between GloVe vectors (degrees)], y-label: [$cos theta$],
  size: (150mm, 54mm),
))

#pause
More readings: uncle · aunt lands at $40.3 degree$, king · cricket at $65.1 degree$. Semantic neighbours sit at small angles; strangers drift toward $90 degree$ and beyond.

== All 28 pairs at once #V

#fig("/lecture3/figures/cosine_heatmap.svg", w: 46%)

#pause
Blocks are the point: a royal cluster (king–prince $0.82$, queen–princess $0.85$), a tight man–woman pair ($0.89$) — and python cold to all, even *opposed* to prince ($-0.08$).

== Evaluate the word analogy

The bet behind the arithmetic:

$ bold(v)_"king" - bold(v)_"man" + bold(v)_"woman" = bold(v)_"king" + underbrace((bold(v)_"woman" - bold(v)_"man"), "a learned gender shift") $

#pause
#codebox[```python
glove.most_similar(positive=["king", "woman"], negative=["man"])[0]
# ('queen', 0.852)  ← nearest of all 400k words, by cosine
```]

#pause
The subtraction never "understood" gender. *Directions in this space consistently encode relationships* — man→woman and king→queen are nearly parallel arrows.

#pause
#notebox[Honest small print: `most_similar` excludes the query words; keep them in and *king itself* outranks queen. The parallelogram is real but approximate — hence "$approx$", never "$=$".]

== Random high-dimensional directions are nearly orthogonal #V

#fig("/lecture3/figures/high_d_angles.svg", w: 58%)

#pause
For independent isotropic random vectors, pairs in $RR^3$ cover a wide range of angles. In $RR^300$, their angles concentrate near $90 degree$.

#pause
So king · queen at $38 degree$ is far from this random baseline. Learned embedding clouds need not be isotropic, so the honest comparison is with angles from the same embedding model, not with a universal $90 degree$ rule.

= Projections: shadows and residuals

== From shadow length to shadow vector

Reading 3 gave a *number* — the shadow's length $bold(x)^top hat(bold(y))$. The shadow itself is that length, aimed along $hat(bold(y))$:

$ "proj"_bold(y)(bold(x)) = (bold(x)^top hat(bold(y))) thin hat(bold(y)) = (bold(x)^top bold(y)) / (bold(y)^top bold(y)) thin bold(y) $

#pause
(The right form just substitutes $hat(bold(y)) = bold(y) \/ norm(bold(y))$ twice and uses $norm(bold(y))^2 = bold(y)^top bold(y)$.)

#pause
Read it: *the multiple of $bold(y)$ that best imitates $bold(x)$*. Whatever's left — the residual $bold(x) - "proj"$ — is the part of $bold(x)$ that $bold(y)$ cannot explain.

== The shadow, in numbers you can check #V

#fig("/lecture3/figures/projection_numbers.svg", w: 48%)

$ (bold(x)^top bold(y)) / (bold(y)^top bold(y)) = 16 / 20 = 0.8 quad => quad "proj" = 0.8 thin (4, 2) = (3.2, 1.6) $

#pause
Residual: $bold(x) - "proj" = (-0.2, 0.4)$ — and $(-0.2, 0.4)^top (4, 2) = -0.8 + 0.8 = 0$. Perpendicular, *exactly*.

== Why perpendicular means "closest"

Slide a candidate point along $bold(y)$'s line and connect it to $bold(x)$: the connector is shortest where it meets the line at a right angle — any other landing point makes it a hypotenuse over that leg.

#pause
#result[$"proj"_bold(y)(bold(x))$ is the *best approximation of $bold(x)$ on $bold(y)$'s line* — and "best" is certified by an orthogonal residual.]

#pause
#notebox[Widen "$bold(y)$'s line" to a whole plane of allowed directions and this same right-angle picture becomes *least squares* (L16) and *PCA* (L6). A surprising amount of "learning" is projecting.]

== Projection in three lines

#codebox[```python
x, y = np.array([3., 2.]), np.array([4., 2.])
proj = (x @ y) / (y @ y) * y     # array([3.2, 1.6])
r = x - proj                     # array([-0.2,  0.4])
r @ y                            # 0.0 — the residual is ⊥, always
```]

#pause
The zero is not luck; it is the definition doing its job. You will reuse exactly these lines inside least squares (L16) and PCA (L6).

= Basis and span: the grid is a choice

== Span · where can your moves take you? #V

#fig("/lecture3/figures/span_fig.svg", w: 70%)

#pause
$"span"{bold(u)} = {alpha bold(u)}$: scaling alone keeps you on a *line*. $"span"{bold(u), bold(v)} = {alpha bold(u) + beta bold(v)}$: with two genuinely different directions, walk $bold(u)$'s then $bold(v)$'s and *every* point of the plane is reachable.

== When a second vector adds nothing

Take $bold(u) = (2, 1)$ and $bold(v) = (4, 2) = 2bold(u)$:

$ alpha bold(u) + beta bold(v) = (alpha + 2 beta) thin bold(u) quad quad "— every 'combination' is still a multiple of" bold(u). $

#pause
*Linearly independent* = no vector in the set is a combination of the others. Dependent = a redundant direction; the span doesn't grow.

#pause
Data version: a feature `total = maths + physics` adds a *column* to your table but *no new direction* to the data. That redundancy is why matrices have rank (L4) and why data compresses (L6).

== Drag a span yourself #I

#interbox(link-to: "https://textbooks.math.gatech.edu/ila/spans.html")[*Interactive Linear Algebra* (GaTech) — in the span demos, drag $bold(u)$ and $bold(v)$ and watch $"span"{bold(u), bold(v)}$ collapse from the whole plane to a line the instant the arrows align.]

#pause
Two minutes of dragging beats twenty static slides — do it tonight. This free textbook is the reference explorable for all of Module 1 (L3–L6).

== A basis is a chosen grid #V

#fig("/lecture3/figures/basis_coords.svg", w: 64%)

#pause
A *basis* = a set of vectors that is (1) independent and (2) spans the space. Consequence: every vector gets *exactly one address* — its coordinates.

#pause
Same arrow, two grids, two addresses: $(3, 2)$ vs $(3.54, -0.71)$. *The arrow is real; the numbers are an artifact of the grid you chose.*

== Orthonormal grids read themselves

If $bold(b)_1, bold(b)_2$ are *unit length* and *mutually perpendicular* (orthonormal), addresses come free — each coordinate is a projection:

$ bold(x) = (bold(x)^top bold(b)_1) thin bold(b)_1 + (bold(x)^top bold(b)_2) thin bold(b)_2 $

#pause
Check against the figure's rotated grid — $bold(b)_1 = (1,1)\/sqrt(2)$, $bold(b)_2 = (-1,1)\/sqrt(2)$, $bold(x) = (3,2)$: $bold(x)^top bold(b)_1 = 5\/sqrt(2) approx 3.54$ and $bold(x)^top bold(b)_2 = -1\/sqrt(2) approx -0.71$ ✓ — the figure's numbers.

#pause
#result[In an orthonormal basis, *coordinates are projections*: reading an address costs $d$ dot products.]

#pause
L6 teaser: PCA = let the data cloud elect its own favourite orthonormal grid.

== Checkpoint: four quick calls #Q

#mcq([Commit to all four before the reveal.],
  [$cos theta$ between $bold(x)$ and $bold(y)$ is $0.8$. Your teammate rescales $bold(x) arrow.r 10bold(x)$. New cosine?],
  [$"span"{(2, 1), (4, 2)}$: a point, a line, or the whole plane?],
  [$"proj"_bold(y)(bold(y)) = $ ?],
  [Can *three* vectors in $RR^2$ be linearly independent?],
)

== Answers: four quick calls #A

*A.* $0.8$ — unchanged. The $10$ scales $bold(x)^top bold(y)$ and $norm(bold(x))$ alike and cancels. Cosine sees direction only.

#pause
*B.* A line — $(4, 2) = 2 dot (2, 1)$: the second move is the first in disguise.

#pause
*C.* $bold(y)$ itself — the coefficient is $bold(y)^top bold(y) \/ bold(y)^top bold(y) = 1$: a vector is its own shadow on its own line.

#pause
*D.* No — two independent arrows already span $RR^2$; any third is a combination of them. "How many independent directions fit" is exactly what *dimension* means.

== Practice problems

Try on paper; verify in NumPy.

+ For $bold(x) = (1, -2, 2)$: compute $norm(bold(x))_1, norm(bold(x))_2, norm(bold(x))_infinity$. Which is biggest — and is that always so?
+ A new user rates $(1, 1, 5)$. Cosine with you $= (5, 4, 1)$? More or less "like you" than Rohan's $0.27$?
+ Find $t$ so that $(3, 2) perp (4, t)$.
+ Project $bold(x) = (1, 4)$ onto $bold(y) = (2, 0)$; write $bold(x)$ = shadow + residual and verify the residual is $perp bold(y)$.
+ Show $(1, 1) perp (1, -1)$, normalize both, and write $(3, 2)$ in this basis via two dot products.
+ In the heatmap, king · cricket $approx 0.42$ but prince · python $approx -0.08$. Give one geometric interpretation of each number.

== The reading map

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Source*], [*What it gives you*]),
  [MML Ch 2.1–2.4], [vectors and vector spaces — today's arrows, done carefully],
  [MML Ch 3.1–3.3], [norms, inner products, distances — today's rulers],
  [CS229 linear-algebra notes], [the course's reference handout; keep it at hand through L6],
  [3b1b _Essence of Linear Algebra_ ch 1–3], [today in moving pictures; chapter 3 is the perfect L4 pre-watch],
)

#pause
#notebox[All free: `mml-book.github.io` · `cs229.stanford.edu/section/cs229-linalg.pdf` · the 3b1b playlist on YouTube. Quiz questions come from lecture beats, and every beat above has a book home.]

== Lecture 3 — summary

- *A vector is an arrow AND a data row*; a dataset is $n$ arrows in $RR^d$ (each entry a float32 — L2).
- *Norms measure size*: L2 flight, L1 streets, L∞ worst case — unit balls: circle, diamond, square.
- *One dot product, three readings*: $sum_i x_i y_i = norm(bold(x)) norm(bold(y)) cos theta$ = shadow × length.
- ⭐ The *law of cosines* turns multiply–add into an angle meter — and *defines* angles in $RR^300$.
- *Cosine similarity* ranks words, films, and people; `king − man + woman ≈ queen` compares approximately parallel displacement vectors.
- *Projection* = closest point on a line; the residual is orthogonal — the germ of least squares and PCA.
- *Span* is where your moves reach; a *basis* is the grid you narrate coordinates in; orthonormal grids read coordinates by dot products.

#focus-slide[
  Similarity is an angle.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Matrices as Linear Maps* — a dense neural-network layer uses a matrix to transform its input vector.
]
