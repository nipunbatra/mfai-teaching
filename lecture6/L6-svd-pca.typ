// SVD & PCA — Lecture 6 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture6/L6-svd-pca.typ
//   typst compile --root . --input handout=true lecture6/L6-svd-pca.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#let IA = "https://nipunbatra.github.io/interactive-articles/"

#show: metropolis-deck.with(
  title: [SVD & PCA],
  subtitle: [The best low-rank story of your data],
)

#title-slide()

= Low-rank updates for large models

== Fine-tuning moves every weight

A pretrained 7-billion-parameter language model already knows language. You want it to write _your_ hospital's radiology reports.

#pause
- *Fine-tuning* = nudge the weights: every matrix $W arrow.r W + Delta W$
#pause
- Naively, $Delta W$ is as big as the model: *7 billion* new numbers to learn, store, and ship
#pause
- One fine-tune per task, per clinic, per customer… each a full model copy

#pause
#result[Question of the day: does the _update_ really need all 7,000,000,000 numbers?]

== LoRA's bet: the update is low-rank #I

One weight matrix inside such a model: $W in RR^(4096 times 4096)$ — *16,777,216* numbers. *LoRA* (Hu et al., 2021) learns the update as a product of two skinny matrices:

$ Delta W = underbrace(B, 4096 times 8) space underbrace(A, 8 times 4096) quad arrow.r quad 2 dot 4096 dot 8 = 65,536 "numbers — " bold(0.4%) ", a " 256 times " saving" $

#pause
- $Delta W = B A$ has *rank at most 8* — a sum of 8 outer products (L4)
- GPT-3 (175B): *\~10,000×* fewer trainable parameters, matching accuracy

#pause
#interbox(link-to: IA + "lora-adapter")[Slide the adapter rank $r$; watch the parameter meter collapse.]

== Why would low-rank be enough?

That is a *mathematical* question, and this lecture is the answer:

#pause
+ Every matrix secretly is a *rotation, a stretch, and a rotation* (the SVD)
+ The stretches come *sorted by importance* — most matrices are dominated by a few
+ Keeping the top $k$ is *provably the best* rank-$k$ approximation (Eckart–Young)

#pause
#notebox[Training LoRA for real is the deep-learning course (ES 667). Today we build the theorem it stands on — and cash a cheque you can _see_…]

== The same trick, visible: image compression #V

#fig("/lecture6/figures/teaser_pair.svg", w: 58%)

#align(center, text(size: 16pt, fill: MUTED)[A 256 × 256 image _is_ a matrix. Right: the same matrix, rebuilt from its top 20 "stretches".])

#pause
By the end of class you will compute exactly _which_ 16% to keep — and prove nothing better exists at that budget.

== Learning outcomes

By the end of this lecture you will be able to:

+ Describe what *any* matrix does to space: rotate · stretch · rotate — and read singular values as the ellipse's semi-axes.
+ Compute a full *2×2 SVD by hand* via $A^top A$, and verify it multiplies back.
+ Expand a matrix into *rank-1 layers*, ranked by energy $sigma_i^2$.
+ State *Eckart–Young* and use the $sigma$-tail to predict compression error.
+ Derive *PCA* as best-fit axes / max-variance directions, and run it on real data.
+ Show *PCA = SVD of the centered data matrix*, and pick the right route in code.

= Every matrix: rotate · stretch · rotate

== L5 gave us eigenvectors — and left a gap

L5: eigenvectors are the directions a matrix _can't turn_ — and symmetric matrices got the dream: $S = Q Lambda Q^top$, real stretches along perpendicular axes.

#pause
But the dream has fine print:

- A plain square matrix may have *complex* eigenvalues (a rotation turns _every_ direction)
#pause
- And most matrices in AI are not even square: a dataset is $1797 times 64$, an MLP weight is $4096 times 11008$ — "$A bold(v) = lambda bold(v)$" doesn't typecheck when input and output live in different spaces

#pause
#result[Today's theorem: *every* matrix — square or not — factors perfectly. The price: two orthonormal bases instead of one.]

== What a matrix does to the unit circle #V

Feed every unit vector through $A = mat(5, 3; 0, 4)$ and watch:

#fig("/lecture6/figures/svd_action.svg", w: 100%)

#pause
A circle in, an *ellipse* out — always. The three arrows between the panels are the three machines hiding inside $A$.

== Step 1 · $V^top$ rotates the input into position #V

#fig("/lecture6/figures/svd_step_v.svg", w: 62%)

#pause
- A rotation: the circle doesn't change — but two special directions $bold(v)_1, bold(v)_2$ (the *right singular vectors*) land exactly on the coordinate axes
#pause
- Perpendicular before, perpendicular after — rotations preserve angles

== Step 2 · $Sigma$ stretches along the axes #V

#fig("/lecture6/figures/svd_step_s.svg", w: 62%)

#pause
- A pure axis-aligned stretch: ×#text(fill: ACC)[6.32] horizontally, ×#text(fill: TEAL)[3.16] vertically
#pause
- The *only* place sizes change — all of $A$'s "action", concentrated in two positive numbers $sigma_1 >= sigma_2$: the *singular values*

== Step 3 · $U$ rotates to the output pose #V

#fig("/lecture6/figures/svd_step_u.svg", w: 62%)

#pause
- A final rotation sets the ellipse's tilt: the axes land on $bold(u)_1, bold(u)_2$ — the *left singular vectors*
#pause
- Net effect: exactly what $A$ does — rebuilt from a rotation, a stretch, and a rotation

== The claim, in symbols

$ A = underbrace(U, "rotate (output)") space underbrace(Sigma, "stretch") space underbrace(V^top, "rotate (input)") $

read right-to-left, like function composition (L4).

#pause
- $U, V$: *orthonormal* columns (perpendicular unit vectors — pure rotations/reflections)
- $Sigma$: *diagonal*, entries $sigma_1 >= sigma_2 >= dots.c >= 0$

#pause
#result[*Singular value decomposition:* every matrix is a rotation, a stretch, and a rotation. No exceptions — not even rectangular ones.]

== Rectangular is fine — the shapes

For $A in RR^(m times n)$ with rank $r$, the (thin) SVD reads

$ underbrace(A, m times n) = underbrace(U, m times r) space underbrace(Sigma, r times r) space underbrace(V^top, r times n) $

#pause
- $V^top$ rotates in the *input* space $RR^n$; $U$ poses the result in the *output* space $RR^m$ — a map between different worlds needs two casts
#pause
- $Sigma$ stretches $r$ surviving directions and drops the rest

#pause
#notebox[The "full" SVD pads $U$ to $m times m$ and $V$ to $n times n$ with basis vectors for the leftover space. ML code wants the thin one: `np.linalg.svd(A, full_matrices=False)`.]

== Reading $Sigma$: semi-axes, importance, rank

The singular values are the *semi-axes of the output ellipse* — and they come sorted:

#pause
- $sigma_1$: the largest stretch $A$ applies to any unit vector (its "amplification record")
#pause
- $sigma_i = 0$: that direction is flattened away entirely — the ellipse loses a dimension
#pause
- *rank$(A)$ = number of nonzero $sigma_i$* — L4's "how much of space survives", now a countable quantity

#pause
#alertbox[Eigenvalues can be negative or complex. Singular values are *always real and $>= 0$* — signs and turns are absorbed by $U$ and $V$.]

= A 2×2 SVD, by hand

== Where do $V$ and $Sigma$ come from? Ask $A^top A$

How strongly does $A$ stretch a unit vector $bold(v)$? Square it:

$ norm(A bold(v))^2 = (A bold(v))^top (A bold(v)) = bold(v)^top (A^top A) bold(v) $

#pause
- $A^top A$ is *symmetric* — L5's spectral theorem applies: real eigenvalues $lambda_1 >= lambda_2 >= dots.c >= 0$, *orthonormal* eigenvectors
#pause
- Along an eigenvector: $norm(A bold(v)_i)^2 = lambda_i$ — so the stretch there is $sqrt(lambda_i)$

#pause
#result[$bold(v)_i$ = eigenvectors of $A^top A$, and $sigma_i = sqrt(lambda_i)$. L5's spectral theorem is the engine inside the SVD.]

== Worked 2×2 · the eigenvalues of $A^top A$

$ A = mat(5, 3; 0, 4) quad arrow.r quad A^top A = mat(5, 0; 3, 4) mat(5, 3; 0, 4) = mat(25, 15; 15, 25) $

#pause
Characteristic polynomial (L5, on a symmetric 2×2):

$ (25 - lambda)^2 - 15^2 = 0 quad arrow.r quad 25 - lambda = plus.minus 15 quad arrow.r quad lambda_1 = 40, quad lambda_2 = 10 $

#pause
- Singular values: $sigma_1 = sqrt(40) = 2 sqrt(10) approx 6.32$, #h(6pt) $sigma_2 = sqrt(10) approx 3.16$
#pause
- Eigenvectors: $lambda_1$: $-15 v_x + 15 v_y = 0 arrow.r bold(v)_1 = 1/sqrt(2) vec(1, 1)$; #h(4pt) perpendicular: $bold(v)_2 = 1/sqrt(2) vec(-1, 1)$ — the ±45° directions in the step-1 picture

== The left side: where the axes land

$bold(u)_i$ = the unit vector that $bold(v)_i$ is sent to. Since $norm(A bold(v)_i) = sigma_i$, normalize:

$ bold(u)_1 = (A bold(v)_1) / sigma_1 = 1/(2 sqrt(10)) dot 1/sqrt(2) vec(8, 4) = 1/sqrt(5) vec(2, 1) $

#pause
$ bold(u)_2 = (A bold(v)_2) / sigma_2 = 1/sqrt(10) dot 1/sqrt(2) vec(-2, 4) = 1/sqrt(5) vec(-1, 2) $

#pause
- Check: $bold(u)_1 dot bold(u)_2 = (-2 + 2)\/5 = 0$ — automatically perpendicular
- $U = 1/sqrt(5) mat(2, -1; 1, 2)$ is a rotation by $arctan(1\/2) approx 26.6 degree$ — the tilt in the step-3 picture

== Multiply back — it really is $A$

Assemble the three machines. First the stretch applied to the rotated frame:

$ Sigma V^top = mat(2 sqrt(10), 0; 0, sqrt(10)) dot 1/sqrt(2) mat(1, 1; -1, 1) = sqrt(5) mat(2, 2; -1, 1) $

#pause
then the output rotation — the $sqrt(5)$'s cancel and the arithmetic goes integer:

$ U (Sigma V^top) = 1/sqrt(5) mat(2, -1; 1, 2) dot sqrt(5) mat(2, 2; -1, 1) = mat(4+1, 4-1; 2-2, 2+2) = mat(5, 3; 0, 4) space checkmark $

#pause
#result[$U Sigma V^top = A$, exactly. Rotate · stretch · rotate is not a metaphor — it's a factorization you can verify by hand.]

== Two sanity checks that generalize

*Areas.* A rotation preserves area; the stretch scales it by $sigma_1 sigma_2$:

$ sigma_1 sigma_2 = 2 sqrt(10) dot sqrt(10) = 20 = 5 dot 4 - 3 dot 0 = det A space checkmark $

#pause
*Total content.* Sum the squares of all entries (the _Frobenius norm_ $norm(A)_F^2$):

$ 5^2 + 3^2 + 0^2 + 4^2 = 50 = 40 + 10 = sigma_1^2 + sigma_2^2 space checkmark $

#pause
Both are theorems, not luck: $|det A| = product_i sigma_i$, and $norm(A)_F^2 = sum_i sigma_i^2$ — rotations change neither areas nor lengths, so only $Sigma$ can.

== The machine agrees

#codebox[```python
>>> A = np.array([[5., 3.], [0., 4.]])
>>> U, S, Vt = np.linalg.svd(A)
>>> S
array([6.3246, 3.1623])            # 2*sqrt(10), sqrt(10)
>>> U
array([[ 0.8944, -0.4472],         # (2,1)/sqrt(5) in column 1
       [ 0.4472,  0.8944]])
>>> np.allclose(U @ np.diag(S) @ Vt, A)
True
```]

#pause
#notebox[The SVD is unique up to *paired sign flips*: $(bold(u)_i, bold(v)_i) arrow.r (-bold(u)_i, -bold(v)_i)$ is equally valid, so your hand solution may differ from NumPy's by minus signs. $Sigma$ (sorted descending) is always the same.]

== Checkpoint: reading the singular values #Q

#mcq([$A$ is $2 times 2$ with $sigma_1 = 10$ and $sigma_2 = 0.01$. Which conclusion is _forced_?],
  [$A$ maps the unit circle to a circle of radius ≈ 10],
  [$A$ maps the unit circle to a near-1-D sliver — $A$ is numerically almost rank 1],
  [$A approx 10 I$: roughly "multiply by 10"],
  [Nothing can be said without knowing $U$ and $V$],
)

== Answer: reading the singular values #A

#mcq-answer([B], [a 10 × 0.01 sliver — numerically almost rank 1],
  [$U$ and $V$ only _pose_ the ellipse; its shape is fixed by $Sigma$ alone. With $sigma_2 approx 0$ the second axis is flattened: outputs hug a single line, one rank-1 layer carries $10^2 \/ (10^2 + 0.01^2) approx 99.9999%$ of the energy. (A and C would need $sigma_1 approx sigma_2$.)])

= Keep only the big stretches

== A matrix is a stack of rank-1 layers

Multiply $U Sigma V^top$ out column-by-row and the SVD becomes a *sum*:

$ A = sum_(i=1)^r sigma_i bold(u)_i bold(v)_i^top quad — "each" bold(u)_i bold(v)_i^top "a rank-1 layer (L4's" star "fact), weighted by" sigma_i $

#pause
Our worked matrix, split into its two layers:

$ mat(5, 3; 0, 4) = underbrace(mat(4, 4; 2, 2), sigma_1 bold(u)_1 bold(v)_1^top) + underbrace(mat(1, -1; -2, 2), sigma_2 bold(u)_2 bold(v)_2^top) $

#pause
#result[The SVD ships every matrix *pre-sliced into layers, sorted by importance*. Now: what if we keep only the top of the stack?]

== "Importance" made precise: energy

From the sanity check: $norm(A)_F^2 = sum_i sigma_i^2$. So layer $i$ *owns* $sigma_i^2$ of the matrix's total squared content — its *energy*:

#pause
#align(center, bars((40.0, 10.0), labels: ($sigma_1^2$, $sigma_2^2$), annotate: true, digits: 0, size: (62mm, 40mm)))

#pause
For our 2×2: an *80% / 20%* split — the first layer already _is_ most of the matrix. Real data matrices are far more lopsided, as you're about to see.

== Truncate the stack: the storage arithmetic

Keep the top $k$ layers and discard the rest:

$ A_k = sum_(i=1)^k sigma_i bold(u)_i bold(v)_i^top quad arrow.r quad "store" k "of the" bold(u)"s," k "of the" bold(v)"s," k " " sigma"s" = k(m + n + 1) "numbers" $

#pause
For a $256 times 256$ image ($65,536$ numbers): rank $k$ costs $513 k$.

- $k = 20$: $10,260$ numbers — *15.7%* of the original, a 6.4× saving
#pause
- LoRA's $4096 times 4096$, $k = 8$: $65,536$ numbers — *0.4%*. Same formula.

#pause
But is blunt truncation a _good_ approximation — or just a cheap one?

== Eckart–Young: truncation is optimal

#result[Among *all* matrices $B$ of rank $<= k$, the truncated SVD $A_k$ is the one closest to $A$: #h(6pt) $min_("rank"(B) <= k) norm(A - B)_F = norm(A - A_k)_F = sqrt(sigma_(k+1)^2 + dots.c + sigma_r^2)$]

#pause
- The error is exactly the *energy left in the tail* — read your loss straight off the $sigma$'s, before committing to anything
#pause
- No cleverer rank-$k$ matrix exists. Searching is pointless; *truncating is already optimal*
#pause
- This is the course's first "*learning = approximation*" theorem: replace a big object by the best small one and _know_ what you lost. (Also true in the spectral norm — not shown here.)

== Which direction does $A$ amplify most? #D

Let's earn the $k = 1$ case. The best single story about $A$ should be built from the direction it cares most about:

$ max_(norm(bold(v)) = 1) norm(A bold(v)) quad = quad ? $

#pause
Claim: the maximizer is $bold(v)_1$ — the top right singular vector — with value $sigma_1$.

#pause
Strategy: don't fight in standard coordinates. *Change to the basis the SVD hands you.*

== Expand in the singular basis #D

$bold(v)_1, dots, bold(v)_n$ are orthonormal (spectral theorem, L5), so any unit $bold(v)$ is

$ bold(v) = c_1 bold(v)_1 + dots.c + c_n bold(v)_n, quad "with" sum_i c_i^2 = 1 $

#pause
Apply $A$, using $A bold(v)_i = sigma_i bold(u)_i$:

$ A bold(v) = sum_i c_i sigma_i bold(u)_i quad arrow.r quad norm(A bold(v))^2 = sum_i sigma_i^2 c_i^2 $

#pause
(the cross terms die because the $bold(u)_i$ are orthonormal too — both bases pull their weight)

== The bound — and the best rank-1 layer #D

$norm(A bold(v))^2 = sum_i sigma_i^2 c_i^2$ is a *weighted average* of the $sigma_i^2$ with weights $c_i^2$ summing to 1:

$ norm(A bold(v))^2 <= sigma_1^2 dot sum_i c_i^2 = sigma_1^2, quad "equality" arrow.l.r "all weight on" c_1 arrow.l.r bold(v) = plus.minus bold(v)_1 $

#pause
- So the strongest direction is $bold(v)_1$, amplified $sigma_1$-fold onto $bold(u)_1$ — and the best rank-1 layer is $A_1 = sigma_1 bold(u)_1 bold(v)_1^top$, leaving error$""^2 = sigma_2^2 + dots.c + sigma_r^2$
#pause
- Rank $k$: subtract the layer, repeat on the remainder — greedy, and here *greedy is optimal*

#pause
#notebox[The honest gap: we showed $bold(v)_1$ wins the amplification contest; that _no other_ rank-1 matrix removes more energy is the full Eckart–Young proof — MML §4.5.]

= The demo: image compression

== One image, four budgets #V

$A$ = a 256 × 256 grayscale image. Compute `np.linalg.svd(A)`, keep $k$ layers:

#fig("/lecture6/figures/compress_grid.svg", w: 100%)

#pause
$k = 1$ is literally one column times one row — yet the sky gradient is already there. By $k = 20$ you'd hand this to a human.

== The demo's ledger

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*rank $k$*], [*numbers stored*], [*of 65,536*], [*saving*], [*energy kept*], [*rel. error*]),
  [1], [513], [0.8%], [128×], [93.4%], [25.7%],
  [5], [2,565], [3.9%], [26×], [98.4%], [12.8%],
  [20], [10,260], [15.7%], [6.4×], [99.8%], [4.2%],
  [50], [25,650], [39.1%], [2.6×], [\~100%], [2.2%],
)

#pause
- "Energy kept" = $sum_(i <= k) sigma_i^2 \/ sum_i sigma_i^2$; "rel. error" = $norm(A - A_k)_F \/ norm(A)_F$
#pause
- $k = 1$ keeps 93% of the energy — one layer of 256 already tells most of the story

== Why it works: the spectrum falls off a cliff #V

#fig("/lecture6/figures/sv_decay.svg", w: 80%)

#pause
- Left: $sigma_i$ on a log scale — *structured* images spend their energy in few layers
#pause
- Right: measured error (dots) lands _exactly_ on the Eckart–Young tail $sqrt(sum_(i > k) sigma_i^2)$ (line) — the theorem, photographed

== LoRA uses a low-rank parameter update

Replace "image" by "weight update" and rerun the argument:

#pause
- LoRA's bet: the _useful_ change to a weight matrix concentrates its energy in a few directions — like the image, a cliff-shaped spectrum
#pause
- Then Eckart–Young says a rank-$r$ update loses almost nothing — while the storage line reads $k(m + n)$ instead of $m n$: *0.4%* of the parameters for $r = 8$
#pause
- Fine-tune = ship the tiny $B, A$ pair; the 7B base model stays frozen, shared by every task

#pause
#result[Same theorem, three costumes: compress an image, adapt a model — and next, _summarize a dataset_.]

== Compression in six lines

#codebox[```python
img = plt.imread("campus.png").mean(axis=2)      # (256, 256) grayscale
U, S, Vt = np.linalg.svd(img, full_matrices=False)

k = 20
img_k = U[:, :k] * S[:k] @ Vt[:k]                # broadcast: U_k diag(S_k) V_k^T
np.linalg.norm(img - img_k) / np.linalg.norm(img)    # 0.042  — as the table said
```]

#pause
- `U[:, :k] * S[:k]` scales column $i$ by $sigma_i$ — the standard idiom for $U_k Sigma_k$
#pause
- Tutorial 3: you'll run this on your own photo and plot the whole error-vs-$k$ curve

== Checkpoint: the price of truncation #Q

#mcq([$A$ is $4 times 4$ with singular values $sigma = (6, 3, 2, 1)$. You keep $A_2$, the best rank-2 approximation. What is $norm(A - A_2)_F$?],
  [$2$],
  [$3$],
  [$sqrt(5)$],
  [$sqrt(14)$],
)

== Answer: the price of truncation #A

#mcq-answer([C], [$sqrt(5) approx 2.24$],
  [Eckart–Young: the error is the tail in _quadrature_, $sqrt(sigma_3^2 + sigma_4^2) = sqrt(4 + 1)$ — discarded layers are orthogonal, so their energies add like Pythagoras. A forgets $sigma_4$; B adds instead of squaring; D starts the tail at $sigma_2$, which $A_2$ still keeps.])

= PCA: the best flat story of a dataset

== Datasets are wide; structure is thin

Stack $n$ data points as the rows of a matrix $X in RR^(n times d)$:

- 1797 handwritten digits × 64 pixel values
- a smartwatch: millions of timestamps × dozens of sensor channels
- L3's word embeddings: vocabulary × 300 coordinates

#pause
- Coordinates are *redundant*: neighboring pixels agree, sensors co-move, features correlate. The cloud is $d$-dimensional on paper but hugs a *much flatter* subspace
#pause

#result[The matrix most worth compressing is the dataset itself. Finding its best flat summary has a name: *PCA*.]

== Which axis keeps the most story? #V

Project a centered 2-D cloud onto one line — which line loses least?

#fig("/lecture6/figures/pca_bestfit.svg", w: 76%)

#pause
Same 90 points: the left axis keeps 30% of the spread, the *principal axis* 91%. PCA is the search for that winning direction (then the runner-up, and so on).

== Two definitions, one direction

For centered data, split each point against a candidate unit axis $bold(w)$ — Pythagoras:

$ underbrace(norm(bold(x)_i)^2, "fixed") = underbrace((bold(w)^top bold(x)_i)^2, "spread along" bold(w)) + underbrace(norm(bold(x)_i - (bold(w)^top bold(x)_i) bold(w))^2, "residual"^2) $

#pause
Sum over the cloud: the left side is fixed, so *max variance $arrow.l.r$ min reconstruction error*. Best-fit line and max-spread direction are the _same_ axis.

#pause
#result[Total spread along $bold(w)$ is $norm(X bold(w))^2$ and is maximized by the top right singular vector $bold(v)_1$ of $X$. This is the first principal direction.]

== The PCA recipe

Given raw data $X in RR^(n times d)$, want $k$ axes:

+ *Center*: $X arrow.l X - macron(bold(x))^top$ (subtract the mean of each column)
#pause
+ *Covariance*: $C = 1/n X^top X in RR^(d times d)$ — entry $(a, b)$: how features $a, b$ co-vary
#pause
+ *Eigendecompose* (symmetric! L5): $C = V Lambda V^top$, $lambda_1 >= dots.c >= lambda_d >= 0$
#pause
+ *Project*: $bold(z)_i = V_k^top bold(x)_i in RR^k$ — the point in "principal coordinates"

#pause
$lambda_j$ = variance captured by axis $j$; reconstruct with $hat(bold(x))_i = V_k bold(z)_i + macron(bold(x))$.

#notebox[Convention: we divide by $n$; statisticians and sklearn use $n - 1$. Same axes either way — only the $lambda$ scale shifts.]

== The recipe in NumPy

#codebox[```python
Xc = X - X.mean(axis=0)          # 1. center            (n, d)
C = Xc.T @ Xc / len(Xc)          # 2. covariance        (d, d)
lam, V = np.linalg.eigh(C)       # 3. eigh -> ASCENDING order...
lam, V = lam[::-1], V[:, ::-1]   #    ...flip: biggest first
Z = Xc @ V[:, :k]                # 4. project           (n, k)
```]

#pause
#alertbox[`eigh` returns eigenvalues *smallest-first* — forgetting the flip silently hands you the $k$ _least_ informative axes. A classic, painful bug.]

== 1797 digits: 64 dimensions → 2 #V

#fig("/lecture6/figures/digits_pca.svg", w: 52%)

#pause
- Two axes keep 28.5% of the variance — yet 0s, 6s, 4s already form islands
#pause
- PCA never saw the labels; the colors were painted on _afterwards_ — the structure was in the pixels all along

== A real dataset's spectrum

Variance share of the top principal axes (digits, $d = 64$):

#align(center, bars((14.9, 13.6, 11.8, 8.4, 5.8, 4.9, 4.3, 3.7),
  labels: ($1$, $2$, $3$, $4$, $5$, $6$, $7$, $8$), annotate: true, digits: 1,
  size: (118mm, 44mm)))

#align(center, text(size: 16pt, fill: MUTED)[percent of total variance, by principal-axis index])

#pause
The image's cliff again, in dataset form: *21 of 64* axes already carry 90% of the variance.

== Rebuilding a digit from $k$ components #V

Reconstruct one digit from its first $k$ principal coordinates ($hat(bold(x)) = V_k bold(z) + macron(bold(x))$):

#fig("/lecture6/figures/digits_recon.svg", w: 100%)

#pause
- $k = 1$: a generic average-ish digit; $k = 8$: recognizably a 3; $k = 32$: essentially perfect from *half* the numbers
#pause
- This is Eckart–Young acting on a _data_ matrix — compression and PCA are one operation

= The bridge: PCA is the SVD of centered data

== Two routes, one set of axes

Let $X$ be *centered*, with SVD $X = U Sigma V^top$. Feed it into the covariance:

$ C = 1/n X^top X = 1/n (U Sigma V^top)^top (U Sigma V^top) = 1/n V Sigma U^top U Sigma V^top = V (Sigma^2/n) V^top $

#pause
That _is_ an eigendecomposition of $C$ — orthonormal $V$, diagonal middle. Read it off:

- principal axes = *right singular vectors* of $X$; #h(8pt) variances $lambda_i = sigma_i^2 \/ n$
#pause
- and the mirror fact: $X X^top = U Sigma^2 U^top$ — the $n times n$ point-similarity matrix shares the same spectrum

#pause
#result[*PCA = SVD of the centered data matrix.* Eigen-of-covariance and SVD-of-data are the same theorem wearing different clothes.]

== The bridge, checked in code

#codebox[```python
Xc = X - X.mean(axis=0)                    # digits: (1797, 64)
n = len(Xc)

lam, V = np.linalg.eigh(Xc.T @ Xc / n)     # route 1: covariance
lam, V = lam[::-1], V[:, ::-1]
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # route 2: SVD

np.allclose(S**2 / n, lam)                          # True
np.allclose(np.abs(Vt[:20]), np.abs(V.T[:20]))      # True (signs may flip)
```]

#pause
#notebox[Why compare only the top 20? Some digit pixels are constant → variance 0 → those tail axes are arbitrary in _both_ routes. Zero-eigenvalue directions have no preferred basis.]

== Centering is the fine print

Skip step 1 and "PCA" quietly answers a different question:

#pause
- Three points hugging $(100, 100)$: $(99, 101), (100, 100), (101, 99)$
#pause
- *Uncentered* top singular direction: $(0.707, 0.707)$ with $sigma_1 approx 245$ — it points *at the mean*, because reaching the cloud from the origin dwarfs the spread
- *Centered*: $(0.707, -0.707)$ — the direction the data actually varies

#pause
#alertbox[SVD of raw data ≈ "where is the cloud?" SVD of centered data = "which way does it vary?" Only the second is PCA. Forgetting to center is the field's favorite silent bug.]

== In practice, take the SVD route

Both routes are correct mathematics. Numerically they are not equals:

#pause
- *Memory*: for $d = 10^5$ features, $C$ is $10^5 times 10^5$ — $10^10$ entries you never needed; SVD works on $X$ directly
#pause
- *Precision*: forming $A^top A$ squares the singular values — $sigma_i = 10^(-8)$ becomes $lambda_i = 10^(-16)$, close to float64 $epsilon$. The numerical example reports a tail eigenvalue of $-0.0$, a rounding artifact rather than negative variance
#pause
- sklearn's `PCA` therefore never forms $C$: it runs an SVD of $X_c$ under the hood

#pause
#result[Think in covariance eigenvectors; _compute_ with the SVD.]

= Wrap-up

== Practice problems

Try on paper; verify in the Tutorial 3 notebook.

+ Hand-SVD $mat(0, 2; 1, 0)$: find $sigma_1, sigma_2$ and both bases. What are the "rotations" here?
+ The SVD of $mat(3, 0; 0, -2)$ has $sigma = (3, 2)$ — a diagonal matrix that is _not_ its own $Sigma$. Where did the minus sign and the ordering go?
+ $sigma = (10, 6, 3, 1)$: compute the _relative_ Frobenius error of $A_2$. (≈ 26%.)
+ A $512 times 512$ image: what is the largest $k$ that still compresses at least 4×?
+ Compute the top singular direction of the three-point cloud $(99, 101), (100, 100), (101, 99)$ — uncentered, then centered. Explain the flip.
+ Prove $norm(A)_F^2 = sum_i sigma_i^2$. (Hint: rotations preserve lengths — or take $"tr"(A^top A)$.)

== ⭐⭐⭐ Too big to SVD? Randomize #OPT

Exact SVD costs $O(m n min(m, n))$ — hopeless for a $10^6 times 10^5$ matrix. The modern fix is *randomized SVD*:

#pause
+ Sketch: $Y = A Omega$ with a random $n times (k + 10)$ Gaussian $Omega$ — a few random probes almost surely capture the span of the top layers
#pause
+ Orthonormalize $Y$ (QR), project $A$ into that thin subspace, SVD the small matrix
#pause
+ Guarantee: error close to the optimal $sigma_(k+1)$, with high probability — at a fraction of the cost

#pause
#notebox[Halko–Martinsson–Tropp (2011). fast.ai's free _Computational Linear Algebra_ course builds it in NumPy — and sklearn's `PCA(svd_solver="randomized")` is what actually runs on large data.]

== Lecture 6 — summary

- *SVD*: every matrix is $U Sigma V^top$ — rotate, stretch, rotate; $sigma_i$ = ellipse semi-axes; rank = surviving axes; computed by hand via the eigenvectors of $A^top A$ (L5's spectral theorem).
- *Layers*: $A = sum_i sigma_i bold(u)_i bold(v)_i^top$, importance-sorted; energy $norm(A)_F^2 = sum_i sigma_i^2$.
- *Eckart–Young*: truncating at $k$ is the _best_ rank-$k$ approximation; error = the $sigma$-tail. One theorem behind image compression *and* LoRA's 0.4% fine-tuning.
- *PCA*: best-fit axes = max-variance directions = eigenvectors of the covariance.
- *The bridge*: PCA is the SVD of _centered_ data ($lambda_i = sigma_i^2\/n$) — compute via SVD.

#pause
#notebox[*Read before L7* — MML Ch 4.4–4.5 (SVD) and Ch 10 (PCA); Solomon Ch 7. *Tutorial 3* this week: SVD compression on your own photo + PCA on digits. *Quiz 1* covers Module 1 (L1–L6) — the worked 2×2 here is exam-shaped.]

#focus-slide[
  Every matrix is a rotation, a stretch, and a rotation.
  #v(12pt)
  #set text(size: 22pt)
  Module 2 begins. Next: *Univariate Calculus & Taylor Series* — we stop rearranging space and start measuring change.
]
