// Multivariate Gaussians — Lecture 13 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture13/L13-multivariate-gaussians.typ
//   typst compile --root . --input handout=true lecture13/L13-multivariate-gaussians.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Multivariate Gaussians],
  subtitle: [A mean point and a covariance ellipse],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let CS229 = "https://cs229.stanford.edu/section/cs229-prob.pdf"

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
// build mat() from code: a `;` straight after a #fmt(..) interpolation would be
// swallowed as a code-statement terminator and flatten the matrix to one row
#let m2(M, d: 4) = math.mat(
  (fmt(M.at(0).at(0), d: d), fmt(M.at(0).at(1), d: d)),
  (fmt(M.at(1).at(0), d: d), fmt(M.at(1).at(1), d: d)),
)
#let v2(v, d: 4) = $vec(#fmt(v.at(0), d: d), #fmt(v.at(1), d: d))$

// the lecture's running covariance — L5's and L9's matrix, returning as Σ
#let SIG  = ((2.0, 1.0), (1.0, 2.0))
#let SIGI = la.inv(SIG)                          // Σ⁻¹, computed live
#let (lam, evs) = la.eig-sym(SIG)                // λ = (3, 1), eigenvectors (1,1)/√2 & (1,-1)/√2
#let canon(v) = if v.at(0) < 0 { la.scale(v, -1.0) } else { v }
#let q1 = canon(evs.at(0))
#let q2 = canon(evs.at(1))
#let mah2(v) = la.dot(v, la.matvec(SIGI, v))     // squared Mahalanobis distance at μ = 0

// the worked four-point dataset → its covariance, computed live (population, ÷ n)
#let DPTS = ((1.0, 5.0), (3.0, 3.0), (3.0, 5.0), (5.0, 7.0))
#let DMU = la.mean(DPTS)
#let DCOV = la.mscale(
  DPTS.map(p => la.vsub(p, DMU))
    .fold(la.zeros(2, 2), (S, d) => la.madd(S,
      ((d.at(0) * d.at(0), d.at(0) * d.at(1)), (d.at(1) * d.at(0), d.at(1) * d.at(1))))),
  1.0 / DPTS.len())

// the running densities (dist.gaussian-2d inverts Σ internally)
#let g2c = dist.gaussian-2d(sigma: SIG)                       // centered
#let g2m = dist.gaussian-2d(mu: (3.0, 5.0), sigma: SIG)      // the worked-data Gaussian
#let qm(x, y) = SIGI.at(0).at(0)*x*x + 2.0*SIGI.at(0).at(1)*x*y + SIGI.at(1).at(1)*y*y

#title-slide()

// ═══════════════════════════ 1 · motivation ═══════════════════════════
= Why multivariate Gaussians appear in ML

== Common uses of Gaussian distributions in ML

#table(
  columns: (1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*You will read…*], [*Where*]),
  [_"weights initialized from $cal(N)(0, 0.02^2)$"_], [GPT-2's actual init],
  [_"add noise $epsilon tilde cal(N)(0, I)$ at every step"_], [diffusion image generators],
  [_"sample a latent $bold(z) tilde cal(N)(0, I)$"_], [variational autoencoders],
  [_"we assume Gaussian observation noise"_], [regression, Kalman filters, GPs],
)

#pause
These applications use Gaussians for different reasons: initialization, injected noise, latent variables, and observation models.

#pause
#align(center, text(size: 22pt)[*Which mathematical properties make the Gaussian useful?*])

== Three useful properties

+ *Limit behavior.* Standardized sums of many i.i.d. finite-variance effects approach a Gaussian. (The *Central Limit Theorem*; demonstrated today.)
#pause
+ *Maximum entropy under moment constraints.* Among densities with a specified mean and covariance, the Gaussian has maximum entropy. (Stated today, proved in Module 5.)
#pause
+ *Closure.* Marginals, conditionals, affine transforms, and sums of independent Gaussians remain Gaussian, with explicit parameter formulas.

#pause
#notebox[These three properties explain many Gaussian models, but using a Gaussian remains a modeling assumption whose adequacy should be checked against the application.]

== L12 gave us bumps. But data is a vector.

Last lecture: densities for *one* continuous quantity — $p(x)$, area = probability, $cal(N)(mu, sigma^2)$.

#pause
But since L3, our data has been *vectors*: (height, weight) · a 784-pixel image · an embedding · a parameter vector $bold(theta)$.

#pause
- Height and weight are not two separate stories: tall people tend to be heavier. A model of each alone *misses the relationship*.
#pause
- We need one density over $RR^d$ — a *joint* — that knows how coordinates co-vary.

#pause
#result[Today's object: $bold(x) tilde cal(N)(bold(mu), Sigma)$ — a bump over $RR^d$. \ One point $bold(mu)$ says _where_; one matrix $Sigma$ says _what shape_.]

== Learning outcomes

By the end of this lecture you will be able to:

+ Write and *dissect the MVN density* — and say what every symbol is doing there.
+ Read a *covariance matrix* like a picture: variances, correlation, tilt.
+ Compute a 2×2 covariance *by hand* from raw data.
+ *Marginalize* (project) and *condition* (slice) a Gaussian — with formulas.
+ Score outliers with *Mahalanobis distance* — "how many σ's, direction-aware".
+ Prove the ⭐ fact: *contours are ellipses along the eigenvectors of $Sigma$* (L5, cashed in).
+ Explain why Gaussians are everywhere: *CLT, max entropy, closure*.

// ═══════════════════════════ 2 · from one bump to two ═══════════════════════════
= From one bump to a 2-D bump

== The 1-D bell, recalled (L12) #V

$ p(x) = 1/(sigma sqrt(2 pi)) exp(-(x - mu)^2 / (2 sigma^2)) quad quad #text(fill: MUTED)[two numbers: center $mu$, spread $sigma$] $

#align(center, lines(
  fn: (
    x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), x),
    x => dist.pdf(dist.normal(mu: 1.0, sigma: 0.5), x),
  ),
  domain: (-3.6, 3.6), samples: 160,
  labels: ([$cal(N)(0, 1)$], [$cal(N)(1, 0.25)$]),
  colors: (TEAL, ACC), markers: false, legend: "tl",
  x-label: [$x$], y-label: [$p(x)$],
  size: (108mm, 44mm),
))

#pause
The parameters of $cal(N)(mu, sigma^2)$ are $(mu, sigma)$. What replaces them when $x$ becomes a *vector*?

== First 2-D density: two independent bells, multiplied

Let $x$ and $y$ be *independent* Gaussians (L12: independence means the joint density factorizes):

$ p(x, y) = p(x) dot p(y) = 1/(sigma_1 sqrt(2 pi)) e^(-(x - mu_1)^2 / (2 sigma_1^2)) times 1/(sigma_2 sqrt(2 pi)) e^(-(y - mu_2)^2 / (2 sigma_2^2)) $

#pause
Collect the pieces — constants together, exponents added:

$ p(x, y) = 1/(2 pi thin sigma_1 sigma_2) exp(-1/2 [ (x - mu_1)^2 / sigma_1^2 + (y - mu_2)^2 / sigma_2^2 ]) $

#pause
#result[A perfectly legal density over $RR^2$ — our first 2-D bump. Every multivariate Gaussian is this, plus one twist we'll add shortly.]

== The 2-D bump, two views #V

#two(r: (48%, 52%),
  align(center, surface(dist.gaussian-2d(), xlim: (-3, 3), ylim: (-3, 3),
    samples: 24, cell: 3.0mm, height: 30mm,
    x-label: [$x$], y-label: [$y$], title: [the hill, from the side])),
  align(center, contour(dist.gaussian-2d(), xlim: (-3, 3), ylim: (-3, 3),
    samples: 56, levels: 7, size: (50mm, 50mm),
    x-label: [$x$], y-label: [$y$], title: [the map, from above])),
)

#pause
#align(center, text(size: 16pt, fill: MUTED)[same object ($mu_1 = mu_2 = 0$, $sigma_1 = sigma_2 = 1$), sampled live from its formula — the right view's rings are *contours*: curves of equal density])

#pause
Cartographers had this figured out long ago: a hill *is* its contour map. We will live in the map view.

== Probability is volume now

L12's rule was "probability = *area* under the curve". One dimension up:

$ P((x, y) in A) = integral integral_A p(x, y) thin d x thin d y $
#align(center, text(size: 16pt, fill: MUTED)[probability = *volume* under the hill, above the region $A$])

#pause
- Total volume under the whole hill: exactly $1$.
#pause
- $p(x, y)$ is still a *density*, not a probability: $P$ of any single point is $0$, and $p$ can exceed $1$ (both L12 facts, still in force).

#pause
#notebox[Contours are the map's way of drawing the hill. Everything today — shape, slicing, projecting — will be read off contour pictures.]

== Repackage the exponent: a quadratic form appears

Write $bold(x) = vec(x, y)$, $bold(mu) = vec(mu_1, mu_2)$, $bold(d) = bold(x) - bold(mu)$. The exponent was a sum of squares — watch it become a *quadratic form* (L9!):

$ (x - mu_1)^2 / sigma_1^2 + (y - mu_2)^2 / sigma_2^2 = bold(d)^top mat(1\/sigma_1^2, 0; 0, 1\/sigma_2^2) bold(d) = bold(d)^top Sigma^(-1) bold(d) $

#pause
where $Sigma = mat(delim: "(", sigma_1^2, 0; 0, sigma_2^2)$ collects the two variances — our first covariance matrix, and $Sigma^(-1)$ simply divides each squared deviation by its own variance.

#pause
#notebox[L9 planted exactly this: _"the Gaussian's exponent IS a quadratic form, with $A = Sigma^(-1)$."_ The IOU starts being paid right here.]

== Repackage the constant: a determinant appears

The normalizing constant can be written with a determinant — for a diagonal matrix, $det$ is the product of the diagonal entries (L4):

$ 1/(2 pi thin sigma_1 sigma_2) = 1/(2 pi thin sqrt(sigma_1^2 thin sigma_2^2)) = 1/(2 pi thin det(Sigma)^(1\/2)) $

#pause
#result[$p(bold(x)) = 1/(2 pi thin det(Sigma)^(1\/2)) exp(-1/2 bold(d)^top Sigma^(-1) bold(d))$ \ nothing was added — the independent case *re-packaged itself*.]

== The general formula: keep the packaging, free the matrix

The repackaged formula makes sense for *any* symmetric positive-definite $Sigma$ — not just diagonal ones. That generalization *is* the multivariate Gaussian, in any dimension $d$:

$ p(bold(x)) = 1/((2 pi)^(d\/2) thin det(Sigma)^(1\/2)) exp(-1/2 (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu))), quad quad bold(x) tilde cal(N)(bold(mu), Sigma) $

#pause
- $bold(mu) in RR^d$: the *mean vector* — where the peak sits.
#pause
- $Sigma in RR^(d times d)$: the *covariance matrix* — symmetric, positive definite. Off-diagonal entries are the "twist": they *tilt* the bump, encoding co-variation that independence forbade.

#pause
This formula is the independent case rewritten in matrix notation, then extended from diagonal covariance to any symmetric positive-definite covariance. The next sections interpret each term geometrically.

== The formula, dissected

$ p(bold(x)) = underbrace(1/((2 pi)^(d\/2) thin det(Sigma)^(1\/2)), "volume bookkeeping") thin exp(-1/2 thin underbrace((bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu)), "a quadratic form in " bold(x) - bold(mu))) $

#pause
- The *exponent* is L9's quadratic form $bold(d)^top A bold(d)$ with $A = Sigma^(-1)$ — L9 planted exactly this: _"the bowl's ellipse contours are a Gaussian's, with $A = Sigma^(-1)$"_. Today that IOU is paid.
#pause
- The *normalizer* only rescales height so total volume is 1 — $det(Sigma)^(1\/2)$ is the bump's footprint volume (L4: determinants measure volume). It never affects the shape.
#pause
- *Sanity check* $d = 1$, $Sigma = (sigma^2)$: $det(Sigma)^(1\/2) = sigma$, $Sigma^(-1) = 1\/sigma^2$ — L12's bell, exactly.

// ═══════════════════════════ 3 · reading Σ ═══════════════════════════
= Reading the covariance matrix $Sigma$

== The covariance matrix, defined

$ Sigma_(i j) = EE[(X_i - mu_i)(X_j - mu_j)] $

#pause
- *Diagonal* ($i = j$): $Sigma_(i i) = EE[(X_i - mu_i)^2] = "Var"(X_i)$ — each coordinate's own spread.
#pause
- *Off-diagonal* ($i != j$): do $X_i$ and $X_j$ move *together*? Positive: above-average together. Negative: one up, the other down. Zero: no linear relationship.
#pause
- Swap $i$ and $j$: same product, same expectation. So $Sigma_(i j) = Sigma_(j i)$ — *every covariance matrix is symmetric*, automatically. (You know what that buys us: L5's spectral theorem. Hold that thought.)

== Numbers: four data points, by hand

Four measurements of $(x, y)$: $(1, 5), (3, 3), (3, 5), (5, 7)$. Mean: $bold(mu) = #v2(DMU, d: 0)$. Deviations $bold(d)_i = bold(x)_i - bold(mu)$, then average the products:

#pause
#text(size: 17pt)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + MUTED.lighten(40%),
    inset: 6.5pt,
    align: (center, center, right, right, right),
    table.header([*point*], [*deviation $(d_x, d_y)$*], [*$d_x^2$*], [*$d_y^2$*], [*$d_x d_y$*]),
    [$(1, 5)$], [$(-2, 0)$], [$4$], [$0$], [$0$],
    [$(3, 3)$], [$(0, -2)$], [$0$], [$4$], [$0$],
    [$(3, 5)$], [$(0, 0)$], [$0$], [$0$], [$0$],
    [$(5, 7)$], [$(2, 2)$], [$4$], [$4$], [$4$],
    [*mean*], [], [*$2$*], [*$2$*], [*$1$*],
  )
]

#pause
$ Sigma = mat("mean" d_x^2, "mean" d_x d_y; "mean" d_x d_y, "mean" d_y^2) = #m2(DCOV, d: 0) $
#align(center, text(size: 16pt, fill: GREEN)[the slide recomputes this $Sigma$ from the raw points at compile time (chalkdust) ✓])

== Wait — you have met this matrix before

$ Sigma = mat(2, 1; 1, 2) $

#pause
*L5's running matrix.* Its eigenvectors are $(1, 1)$ and $(1, -1)$, with eigenvalues $3$ and $1$. The contour derivation later in the lecture will turn these into ellipse axes and squared axis scales.

#pause
For any direction $bold(a)$, the variance of the projection $bold(a)^top bold(x)$ is

$ "Var"(bold(a)^top bold(x)) = bold(a)^top Sigma thin bold(a) quad #text(fill: MUTED)[— a quadratic form again, and variances can't be negative…] $

#pause
#result[$bold(a)^top Sigma thin bold(a) >= 0$ for every $bold(a)$: covariance matrices are exactly L9's *positive semi-definite* club. $Sigma$ is a machine that turns directions into variances.]

== Correlation: the unit-free reading

Covariance carries units (cm·kg — unreadable). Normalize by both spreads:

$ rho = Sigma_(12) / (sigma_1 sigma_2) = Sigma_(12) / sqrt(Sigma_(11) Sigma_(22)), quad quad rho in [-1, 1] $

#pause
- $rho = plus.minus 1$: a perfect line · $rho = 0$: no *linear* co-movement · sign = tilt direction.
#pause
- Our running $Sigma$: $rho = 1 / sqrt(2 dot 2) = 0.5$ — moderately, positively coupled.

#pause
#notebox[For *Gaussians* (and only for them, in general): $rho = 0 arrow.l.r.double$ independent. For arbitrary distributions, $rho = 0$ does NOT imply independence — a caution flag worth remembering.]

== The gallery: what $Sigma$ does to the bump #V

#let gpan(S, ttl) = contour(dist.gaussian-2d(sigma: S), xlim: (-3.4, 3.4), ylim: (-3.4, 3.4),
  samples: 44, levels: 6, size: (35mm, 35mm), color: TEAL, title: ttl)
#align(center, grid(columns: 4, column-gutter: 8mm, row-gutter: 2.5mm, align: center,
  gpan(((1.0, 0.0), (0.0, 1.0)), [isotropic]),
  gpan(((2.25, 0.0), (0.0, 0.36)), [diagonal]),
  gpan(((2.0, 1.0), (1.0, 2.0)), [tilted $+$]),
  gpan(((2.0, -1.0), (-1.0, 2.0)), [tilted $-$]),
  text(size: 15pt)[$sigma^2 I = mat(1, 0; 0, 1)$],
  text(size: 15pt)[$mat(2.25, 0; 0, 0.36)$],
  text(size: 15pt)[$mat(2, 1; 1, 2)$],
  text(size: 15pt)[$mat(2, -1; -1, 2)$],
))

#pause
#align(center, text(size: 16pt, fill: MUTED)[four live densities, one formula — only $Sigma$ changed: circles $arrow.r$ axis-aligned stretch $arrow.r$ tilt (sign of $Sigma_(12)$ picks the tilt's direction)])

#pause
#result[$bold(mu)$ moves the bump. The diagonal of $Sigma$ stretches it. The off-diagonal *tilts* it.]

== Sampled clouds obey the same $Sigma$ #V

#fig("/lecture13/figures/gallery_clouds.svg", w: 72%)

#pause
900 samples from each density; the $1 sigma$ and $2 sigma$ ellipses are drawn from $Sigma$ *alone*. The cloud fills its ellipse — simulation and formula agree.

#pause
#notebox[The picture to keep: *a Gaussian is a data cloud's idealization* — center $bold(mu)$, shape $Sigma$. Where the ellipse comes from: the ⭐ derivation, soon.]

== Interactive: bend the ellipse yourself #I

#interbox(link-to: IA + "multivariate-normal")[
  Drag the entries of $Sigma$ (variances and correlation) and watch three things respond at once: the sampled *cloud*, the *contour ellipse*, and the *marginals* on the walls. Then slide the conditioning bar and watch a slice move — the two operations of the next section, live under your mouse.
]

#pause
Five minutes with the sliders builds the reflex: you should *see* $mat(4, -3; -3, 9)$ as a shape before you compute anything about it.

== Code: $Sigma$ in one line

#codebox[```python
X = np.array([[1., 5.], [3., 3.], [3., 5.], [5., 7.]])
X.mean(axis=0)              # array([3., 5.])          = μ
np.cov(X.T, bias=True)      # [[2., 1.], [1., 2.]]     = Σ  (divide by n)
np.cov(X.T)                 # [[2.67, 1.33], ...]      (divide by n-1)
```]

#pause
#codebox[```python
d = torch.distributions.MultivariateNormal(
        loc=torch.tensor([3., 5.]),
        covariance_matrix=torch.tensor([[2., 1.], [1., 2.]]))
xs = d.sample((100_000,))   # 100k points from our Gaussian
torch.cov(xs.T)             # ≈ [[2.00, 1.00], [1.00, 2.00]] ✓
```]

#pause
#notebox[`bias=True` divides by $n$ (what we did by hand); the default divides by $n - 1$. L14 explains the estimation distinction between these conventions.]

== Checkpoint: read the matrix #Q

#mcq([$Sigma = mat(4, -3; -3, 9)$ describes a data cloud. Which reading is correct?],
  [$sigma_x = 4$, $sigma_y = 9$, cloud tilted up-right],
  [$sigma_x = 2$, $sigma_y = 3$, cloud tilted down-right],
  [Invalid — a covariance matrix can never contain negative entries],
  [$x$ and $y$ are independent, since $Sigma$ is symmetric],
)

== Answer: read the matrix #A

#mcq-answer([B], [$sigma_x = 2$, $sigma_y = 3$, tilted down-right],
  [The *diagonal* holds variances, so $sigma_x = sqrt(4)$, $sigma_y = sqrt(9)$ — A read them as standard deviations. $rho = -3\/(2 dot 3) = -0.5 < 0$: as $x$ rises, $y$ tends to fall — a down-right tilt. C: only the _diagonal_ must be non-negative; off-diagonals may be negative freely (validity is positive semi-definiteness: here $det = 36 - 9 = 27 > 0$, trace $> 0$ — L9's hand test says PD ✓). D: independence needs $Sigma_(12) = 0$; symmetry is free.],
)

// ═══════════════════════════ 4 · slice and project ═══════════════════════════
= Slice and project: marginals and conditionals

== Two questions you ask every joint

Your model is $p(x, y)$ over (height, weight). Two everyday questions:

#pause
+ _"What's the distribution of weight — ignoring height entirely?"_ \
  #h(1.2em) → the *marginal* $p(y) = integral p(x, y) thin d x$: sum away what you don't care about.
#pause
+ _"What's the distribution of weight *among 180 cm people*?"_ \
  #h(1.2em) → the *conditional* $p(y | x = 180) = p(180, y) \/ p(180)$: freeze what you know, renormalize.

#pause
#result[Geometrically, marginalization projects the joint density and conditioning takes and renormalizes a slice. The next two figures show both operations.]

== Project = marginalize #V

#two(r: (58%, 42%),
  fig("/lecture13/figures/joint_marginal.svg", w: 96%),
  [
    Squash the hill onto a wall — every $y$ collapses into its $x$ column.

    #v(6pt)
    For our worked Gaussian:
    $ p(x) = cal(N)(3, 2) $
    $ p(y) = cal(N)(5, 2) $

    #pause
    #v(6pt)
    #result[Marginals: *read them off* $bold(mu)$ and $Sigma$ — keep your entries, delete the rest. No integral needed.]
  ],
)

== Slice = condition #V

#fig("/lecture13/figures/conditional_slices.svg", w: 88%)

#pause
Cut along $y = y_0$: the profile revealed is $p(x, y_0)$ — a bump-shaped sliver. Renormalize (divide by $p(y_0)$) and it becomes a legal 1-D density: $p(x | y = y_0)$.

== The slice, quantified

Both panels of that figure obey two formulas (for 2-D; general $d$: same shapes, block form):

$ mu_(x|y) = mu_x + Sigma_(12)/Sigma_(22) (y_0 - mu_y) quad quad quad sigma_(x|y)^2 = Sigma_(11) - Sigma_(12)^2/Sigma_(22) $

#pause
Our numbers ($bold(mu) = (3, 5)$, $Sigma_(12) = 1$, $Sigma_(22) = 2$): observe $y_0 = 7$ →

$ mu_(x|7) = 3 + 1/2 (7 - 5) = 4 quad quad sigma_(x|y)^2 = 2 - 1/2 = 1.5 quad #text(fill: MUTED)[(matches the figure's orange slice)] $

#pause
- The mean *shifts linearly* with $y_0$ — the dashed line through the slice means is a preview of *regression* (L14's linear regression lives inside this picture).
#pause
- The variance *shrank* ($1.5 < 2$) and doesn't depend on $y_0$ at all — knowing $y$ helps by the same amount everywhere. A distinctly Gaussian privilege.

== Closure: every answer stayed in the family

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*operation on $cal(N)(bold(mu), Sigma)$*], [*result*]),
  [marginalize (project)], [Gaussian — keep the relevant entries of $bold(mu)$, $Sigma$],
  [condition (slice)], [Gaussian — mean shifts linearly, variance shrinks],
  [linear map $A bold(x) + bold(b)$], [Gaussian — $cal(N)(A bold(mu) + bold(b), thin A Sigma A^top)$],
  [sum of independent Gaussians], [Gaussian — means add, covariances add],
)

#pause
The Gaussian family is closed under common linear operations used in ML, each with an explicit formula.

#pause
#notebox[This closure is why *Kalman filters* track rockets in real time and *Gaussian processes* fit in a page of code: every update is "Gaussian in, Gaussian out". General block formulas: MML §6.5.]

== Code: check the slice by filtering samples

Don't trust the formula? Condition *empirically* — keep only samples where $y approx 7$:

#codebox[```python
xs = d.sample((400_000,)).numpy()          # our N([3,5], [[2,1],[1,2]])
band = xs[np.abs(xs[:, 1] - 7.0) < 0.05]   # the thin slice y ≈ 7
band[:, 0].mean(), band[:, 0].var()
# (4.003, 1.497)     ← formula predicted N(4, 1.5) ✓
```]

#pause
The filtered histogram is the renormalized slice; simulation agrees with the conditional-density formula.

#pause
#result[Marginal = project. Conditional = slice. Both stay Gaussian — that's the spine of this lecture.]

// ═══════════════════════════ 5 · Mahalanobis ═══════════════════════════
= Mahalanobis distance: direction-aware $sigma$'s

== Two points, same distance, different stories #V

#fig("/lecture13/figures/mahalanobis_scene.svg", w: 68%)

#pause
$A = (1, 1)$ and $B = (1, -1)$ sit at *identical* Euclidean distance $sqrt(2)$ from the mean — yet the cloud disagrees violently: $A$ is in the thick of it, $B$ is out in the cold.

#pause
Euclidean distance is *shape-blind*. We need a ruler that has seen $Sigma$.

== In 1-D, you already own the fix

L12's *z-score*: don't ask "how far?", ask *"how many standard deviations?"*

$ z = (x - mu)/sigma quad quad #text(fill: MUTED)[3 cm from the mean means nothing until you know $sigma$] $

#pause
In $d$ dimensions the question sharpens: standard deviations *in which direction?* Along the cloud's long axis, spread is large — a big step is cheap. Across it, spread is small — the same step is expensive.

#pause
#result[We need a distance that *rescales every direction by that direction's own spread*. $Sigma$ knows every direction's spread — so the ruler must be built from $Sigma$.]

== Mahalanobis distance, defined

$ d_M^2 (bold(x)) = (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu)) $

#pause
- Why $Sigma^(-1)$: dividing by variance, matrix-style — big-spread directions get *discounted*, tight directions get *amplified*.
#pause
- Sanity, $d = 1$: $d_M^2 = (x - mu)^2 \/ sigma^2$ — the z-score squared, exactly.
#pause
- Now look again at the Gaussian: $p(bold(x)) prop exp(-1/2 d_M^2 (bold(x)))$. *The exponent we dissected IS Mahalanobis distance.*

#pause
#result[Equal density $arrow.l.r.double$ equal Mahalanobis distance. \ A Gaussian's contours are Mahalanobis "circles".]

== One ruler, two geometries #V

#two(r: (50%, 50%),
  align(center, contour((x, y) => x*x + y*y, xlim: (-3.2, 3.2), ylim: (-3.2, 3.2),
    samples: 48, levels: (0.5, 2.0, 4.5, 8.0), size: (46mm, 43mm), color: BLUE,
    x-label: [$x$], y-label: [$y$], title: [Euclidean $bold(d)^top bold(d)$: circles])),
  align(center, contour((x, y) => qm(x, y), xlim: (-3.2, 3.2), ylim: (-3.2, 3.2),
    samples: 48, levels: (0.5, 2.0, 4.5, 8.0), size: (46mm, 43mm), color: ACC,
    x-label: [$x$], y-label: [$y$], title: [Mahalanobis $bold(d)^top Sigma^(-1) bold(d)$: ellipses])),
)

#pause
#align(center, text(size: 16pt, fill: MUTED)[same four level values on both plots, $Sigma = mat(delim: "(", 2, 1; 1, 2)$ — the Mahalanobis unit ball is the covariance ellipse itself])

== Compute the conditional distribution

The slide inverts $Sigma$ live (chalkdust `la.inv`):

$ Sigma^(-1) = #m2(SIGI, d: 4) = 1/3 mat(2, -1; -1, 2) $

#pause
Score both suspects — hand-checkable in two lines each:

$ d_M^2 (A) = (1, 1) thin Sigma^(-1) vec(1, 1) = #fmt(mah2((1.0, 1.0)), d: 4) quad arrow.r.double quad d_M (A) = #fmt(calc.sqrt(mah2((1.0, 1.0))), d: 2) " ‘standard deviations’" $

$ d_M^2 (B) = (1, -1) thin Sigma^(-1) vec(1, -1) = #fmt(mah2((1.0, -1.0)), d: 4) quad arrow.r.double quad d_M (B) = #fmt(calc.sqrt(mah2((1.0, -1.0))), d: 2) $

#pause
Why exactly those numbers? $A$ lies along $(1,1)$ — the direction with variance $3$. $B$ lies along $(1,-1)$ — variance $1$. Eigen-directions again… which brings us to the ⭐.

// ═══════════════════════════ 6 · the ⭐ derivation ═══════════════════════════
= ⭐ The ellipse is the eigendecomposition of $Sigma$

== The claim — and where you first saw it #V

*Claim.* The contours of $cal(N)(bold(mu), Sigma)$ are ellipses centered at $bold(mu)$, with axes along the *eigenvectors of $Sigma$* and semi-axes proportional to $sqrt(lambda_i)$.

#fig("/lecture5/figures/spectral_ellipse.svg", w: 48%)

#pause
L5 anticipated the claim: a Gaussian data cloud's covariance eigenvectors give the axes of its density ellipses. The next three slides derive it.

== ⭐ Step 1: a contour is a level set of the exponent #D

Fix a density value $c$ and take $log$ of both sides of $p(bold(x)) = c$ — the normalizer is a constant, so it folds into the right-hand side:

$ p(bold(x)) = c quad arrow.l.r.double quad (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu)) = r^2 quad #text(fill: MUTED)[(some constant $r^2 >= 0$)] $

#pause
($exp$ is strictly monotone, so no information is lost taking the $log$ — L2's log-space move, working geometry duty.)

#pause
#result[Contours of the Gaussian = level sets of the quadratic form $bold(d)^top A thin bold(d)$ with $A = Sigma^(-1)$ — precisely L9's bowl. What remains: *which* ellipses, pointed *where*?]

== ⭐ Step 2: $Sigma$ and $Sigma^(-1)$ share eigenvectors #D

Let $Sigma bold(q) = lambda bold(q)$ (spectral theorem, L5: $Sigma$ symmetric → orthonormal $bold(q)_i$, real $lambda_i$; positive definite → all $lambda_i > 0$, so $Sigma^(-1)$ exists). Multiply both sides by $Sigma^(-1)$ and divide by $lambda$:

$ Sigma bold(q) = lambda bold(q) quad arrow.r.double quad bold(q) = lambda thin Sigma^(-1) bold(q) quad arrow.r.double quad Sigma^(-1) bold(q) = 1/lambda thin bold(q) $

#pause
#result[Same eigenvectors, reciprocal eigenvalues. The bowl $bold(d)^top Sigma^(-1) bold(d)$ and the covariance $Sigma$ agree about the special directions — they only disagree about which one is "long".]

#pause
Where $Sigma$ has a *large* $lambda$ (much spread), $Sigma^(-1)$ has a *small* $1\/lambda$ (cheap distance) — the discount from the Mahalanobis slide, now with a mechanism.

== ⭐ Step 3: switch to the eigen-basis #D

L5/L9's signature move. Write $Sigma = Q Lambda Q^top$, change coordinates $bold(t) = Q^top (bold(x) - bold(mu))$ — a rotation onto the eigen-axes. The cross-terms die:

$ (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu)) = bold(t)^top Lambda^(-1) bold(t) = t_1^2/lambda_1 + t_2^2/lambda_2 $

#pause
So the contour $= r^2$ reads, in eigen-coordinates:

$ t_1^2/(r^2 lambda_1) + t_2^2/(r^2 lambda_2) = 1 quad #text(fill: MUTED)[— the standard ellipse equation, semi-axes $r sqrt(lambda_1)$ and $r sqrt(lambda_2)$] $

#pause
#result[Contours of $cal(N)(bold(mu), Sigma)$: ellipses centered at $bold(mu)$, axes along the eigenvectors $bold(q)_i$ of $Sigma$, semi-axes $prop sqrt(lambda_i)$. $qed$ \ #text(size: 17pt)[L5 planted it, L9 armed it, L13 proved it.]]

== The theorem, checked live #V

#two(r: (55%, 45%),
  align(center, contour(g2c, xlim: (-3.4, 3.4), ylim: (-3.4, 3.4),
    samples: 56, levels: 6, size: (56mm, 50mm), color: MUTED.lighten(10%),
    paths: (
      (la.scale(q1, -1.65 * calc.sqrt(lam.at(0))), la.scale(q1, 1.65 * calc.sqrt(lam.at(0)))),
      (la.scale(q2, -1.65 * calc.sqrt(lam.at(1))), la.scale(q2, 1.65 * calc.sqrt(lam.at(1)))),
    ),
    marks: (
      (la.scale(q1, 1.9 * calc.sqrt(lam.at(0))).at(0), la.scale(q1, 1.9 * calc.sqrt(lam.at(0))).at(1), [$bold(q)_1$], ACC),
      (la.scale(q2, 1.9 * calc.sqrt(lam.at(1))).at(0), la.scale(q2, 1.9 * calc.sqrt(lam.at(1))).at(1), [$bold(q)_2$], RED),
    ),
    x-label: [$x$], y-label: [$y$])),
  [
    `la.eig-sym(Σ)`, run by this slide:

    $ lambda_1 = #fmt(lam.at(0), d: 0), quad bold(q)_1 = #v2(q1, d: 3) $
    $ lambda_2 = #fmt(lam.at(1), d: 0), quad bold(q)_2 = #v2(q2, d: 3) $

    #pause
    Axis half-length ratio:
    $ sqrt(lambda_1) : sqrt(lambda_2) = sqrt(3) : 1 approx #fmt(calc.sqrt(lam.at(0) / lam.at(1)), d: 2) : 1 $
  ],
)

#pause
#align(center, text(size: 16pt, fill: MUTED)[contours sampled from the density; axis segments drawn from `eig-sym`'s output, lengths $prop sqrt(lambda_i)$ — computed, not sketched])

== Where this one fact keeps cashing out

- *PCA (L6), demystified*: PCA's principal directions are the eigenvectors of the data covariance — i.e., *the axes of this ellipse*. L6 found them by SVD; today you know what they mean probabilistically.
#pause
- *Whitening*: rotate by $Q^top$, divide by $sqrt(lambda_i)$ — any Gaussian becomes $cal(N)(0, I)$. Data preprocessing = ellipse → circle.
#pause
- *Optimization preview (L17)*: quadratic loss surfaces have these same elliptical contours; the ratio $lambda_max \/ lambda_min$ (how *stretched* the ellipse is) will decide how fast gradient descent can go.

#pause
#result[One picture — eigenvectors of a symmetric matrix as ellipse axes — now runs statistics (today), compression (L6), and optimization (L17).]

== Checkpoint: the long axis #Q

#mcq([$cal(N)(bold(mu), Sigma)$ with $Sigma = mat(3, 2; 2, 3)$ has elliptical contours. Their *long* axis points along…],
  [$(1, 0)$ — the first diagonal entry is largest],
  [$(1, 1)$],
  [$(1, -1)$],
  [Can't say without knowing $bold(mu)$],
)

== Answer: the long axis #A

#mcq-answer([B], [along $(1, 1)$],
  [Same drill as our running matrix: for $mat(delim: "(", 3, 2; 2, 3)$, trace $= 6$ and $det = 5$ give $lambda = 5, 1$ (L5's audit), with $(1,1)$ the $lambda = 5$ direction: most variance, longest axis ($sqrt(5) : 1 approx 2.2 : 1$). A: diagonal entries are per-*coordinate* variances — equal here (3 and 3), yet the ellipse is far from round; the tilt lives in the off-diagonal. D: $bold(mu)$ only *slides* the ellipse; orientation is $Sigma$'s alone.],
)

// ═══════════════════════════ 7 · why Gaussians are everywhere ═══════════════════════════
= Why the Gaussian is everywhere

== Sums of independent variables #V

Take the *least* Gaussian thing available — a flat uniform — and add independent copies:

#fig("/lecture13/figures/clt_uniforms.svg", w: 96%)

#pause
By $n = 4$ the bell shape is visible; by $n = 16$ the histogram closely follows $cal(N)(0, 1)$ at the plotted resolution. Other i.i.d. finite-variance summands show the same standardized limit behavior.

== The Central Limit Theorem, stated

*Theorem (CLT, informal).* If $X_1, dots, X_n$ are i.i.d. with mean $mu$ and finite variance $sigma^2$, then the standardized sum

$ (X_1 + dots.c + X_n - n mu) / (sigma sqrt(n)) arrow.r quad cal(N)(0, 1) quad "as" n arrow.r infinity $

#pause
- After centering and scaling, the limiting distribution depends on the summands through their mean and variance; the finite-$n$ approximation quality still depends on the original distribution.
#pause
- Sensor error, measurement jitter, and aggregate effects often combine many small independent contributions, so the central limit theorem motivates Gaussian noise models.

#pause
#notebox[The CLT is stated and illustrated here but not proved; a proof uses tools such as characteristic functions. See MML §6.4 for pointers.]

== The Gaussian maximizes entropy at fixed variance

*Fact (maximum entropy).* Among *all* densities on $RR$ with a given mean $mu$ and variance $sigma^2$, the Gaussian has the largest *entropy* — it is the least committed, most spread-out choice consistent with those two facts.

#pause
- If a modeling principle retains only $bold(mu)$ and $Sigma$ and otherwise maximizes entropy, it selects $cal(N)(bold(mu), Sigma)$. This does not make every dataset Gaussian; it states the optimization criterion used to choose the model.
#pause
- "Entropy" — a number measuring non-commitment? Defining it properly is the whole business of *Module 5* (L22). This fact is planted today and *proved there*.

#pause
#result[The CLT motivates Gaussian approximations for standardized sums; the maximum-entropy theorem selects a Gaussian when only mean and covariance are constrained.]

== Summary of the Gaussian modelling assumptions

Why Gaussian models recur in ML:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*property*], [*modelling consequence*], [*status*]),
  [CLT], [some aggregated effects are approximately Gaussian after standardization], [demoed today],
  [max entropy], [maximum-entropy density given $bold(mu)$, $Sigma$], [stated → L22],
  [closure], [marginals, conditionals, sums: *closed forms*], [seen today],
)

#pause
GPT-2's initialization, diffusion noise, VAE latents, and Kalman filters use different subsets of these properties.

#pause
#result[Gaussian models combine tractable linear algebra with useful limit and maximum-entropy characterizations.]

// ═══════════════════════════ 8 · the take-home ═══════════════════════════
= Summary

== Lecture 13 — summary

- *The MVN*: a bump on $RR^d$ with density $prop exp(-1/2 (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu)))$ — the exponent is L9's quadratic form, $A = Sigma^(-1)$.
- *Reading $Sigma$*: diagonal = variances, off-diagonal = co-movement ($rho = Sigma_(12)\/sigma_1 sigma_2$); PSD because $bold(a)^top Sigma bold(a) = "Var"(bold(a)^top bold(x)) >= 0$.
- *Marginal = project, conditional = slice* — both stay Gaussian; conditional mean moves linearly, variance shrinks by a constant.
- *Mahalanobis* $d_M^2 = (bold(x) - bold(mu))^top Sigma^(-1) (bold(x) - bold(mu))$: "how many σ's, direction-aware"; equal density ⇔ equal $d_M$.
- *⭐ The ellipse*: contour axes = eigenvectors of $Sigma$, semi-axes $prop sqrt(lambda_i)$ (L5, paid).
- *Why everywhere*: CLT + max entropy (→ L22) + closure.

#pause
#notebox[*Read before L14* — MML §6.5–6.6 · CS229 #link(CS229)[probability notes] handout. *Tutorial 6* this week: covariances by hand, slicing/projecting, the ellipse in NumPy.]

== Practice problems

Try on paper; check with NumPy (`np.cov`, `np.linalg.eigh`, `np.linalg.inv`).

+ Read $Sigma = mat(9, -5; -5, 4)$: $sigma_x$, $sigma_y$, $rho$, tilt direction. Then verify it *is* a valid covariance (L9 hand test: $det$, trace).
+ Why can no dataset ever have covariance $mat(1, 2; 2, 1)$? (Hint: L9 classified this exact matrix. Find a direction $bold(a)$ with "negative variance".)
+ Compute the covariance of $(0, 0), (2, 1), (4, 2)$ by hand. What is $det(Sigma)$? What has the "ellipse" collapsed into, and which L4 concept is showing up?
+ For our running $Sigma = mat(2, 1; 1, 2)$: compute $d_M^2$ of $(2, 2)$ and of $(-1, 1)$. Which is the outlier? Reconcile with the eigenvalues.
+ Sketch the 1-contour of $cal(N)(bold(0), mat(5, 3; 3, 5))$: axis directions, axis-length ratio. (Trace/det shortcut welcome.)
+ For $bold(mu) = (3, 5)$, $Sigma = mat(2, 1; 1, 2)$: write $p(y | x = 1)$, then verify by sampling + filtering in code (as we did for $y = 7$).

== ⭐⭐⭐ Sampling: every Gaussian is a warped $cal(N)(0, I)$ #OPT

L12 taught the machine to make 1-D standard normals. Stack $d$ of them: $bold(z) tilde cal(N)(0, I)$ — a round cloud. How do we get $cal(N)(bold(mu), Sigma)$ from it?

#pause
Apply a linear map and use *closure*: $bold(x) = bold(mu) + A bold(z)$ is Gaussian with mean $bold(mu)$ and covariance

$ "Cov"(A bold(z)) = A thin "Cov"(bold(z)) thin A^top = A A^top $
#align(center, text(size: 16pt, fill: MUTED)[so we need a matrix with $A A^top = Sigma$ — a "square root" of $Sigma$])

#pause
The workhorse choice: the *Cholesky factor* $L$ — lower-triangular, $L L^top = Sigma$, exists for every positive-definite $Sigma$, cheap to compute.

#pause
#result[$ bold(z) tilde cal(N)(0, I), quad bold(x) = bold(mu) + L bold(z) quad arrow.r.double quad bold(x) tilde cal(N)(bold(mu), Sigma) $]

== ⭐⭐⭐ Cholesky in code — and in pictures #OPT

#codebox[```python
Sigma = np.array([[2., 1.], [1., 2.]])
L = np.linalg.cholesky(Sigma)          # L @ L.T == Sigma
z = rng.standard_normal((100_000, 2))  # round cloud  N(0, I)
x = np.array([3., 5.]) + z @ L.T       # our Gaussian N(μ, Σ)
np.cov(x.T)                            # ≈ [[2.00, 1.00], [1.00, 2.00]] ✓
```]

#pause
- Geometry: $L$ maps the unit circle of $bold(z)$ onto *the covariance ellipse*; the same linear map is also a Gaussian sampling construction.
#pause
- Another valid square root, straight from the ⭐: $Sigma^(1\/2) = Q Lambda^(1\/2) Q^top$ — stretch by $sqrt(lambda_i)$ along the eigen-axes, rotate. Different $A$, same $A A^top$, same distribution.
#pause
- This is precisely how `torch.distributions.MultivariateNormal` samples (it stores `scale_tril` — the Cholesky factor).

#focus-slide[
  A Gaussian is a point and an ellipse.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Maximum Likelihood Estimation* — we can now describe data with distributions; next we _fit_ them.
]
