// MAP & Conjugate Priors — Lecture 15 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture15/L15-map-conjugate-priors.typ
//   typst compile --root . --input handout=true lecture15/L15-map-conjugate-priors.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [MAP & Conjugate Priors],
  subtitle: [MLE, with a prior's vote],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let amax = math.op("arg max", limits: true)
#let amin = math.op("arg min", limits: true)

// ── Beta pdf, computed in-deck (chalkdust-dist has no Beta) ────────────────
// Normalizing constant B(α,β) = Γ(α)Γ(β)/Γ(α+β) via the Lanczos log-Γ
// approximation (g = 7) — every α, β this deck plots is ≥ 0.5, and so are
// their sums, so the reflection branch is never needed.
#let lgamma(x) = {
  assert(x >= 0.5, message: "lgamma: this deck's inline Lanczos needs x >= 0.5")
  let c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
           771.32342877765313, -176.61502916214059, 12.507343278686905,
           -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
  let y = x - 1.0
  let a = c.at(0)
  let tt = y + 7.5
  for i in range(1, 9) { a += c.at(i) / (y + i) }
  0.5 * calc.ln(2 * calc.pi) + (y + 0.5) * calc.ln(tt) - tt + calc.ln(a)
}
#let beta-pdf(a, b) = {
  let lnB = lgamma(a) + lgamma(b) - lgamma(a + b)
  x => {
    let xx = calc.clamp(x, 0.000001, 0.999999)
    calc.exp((a - 1.0) * calc.ln(xx) + (b - 1.0) * calc.ln(1.0 - xx) - lnB)
  }
}

#title-slide()

// ═══════════════════════════ 1 · motivating example ═══════════════════════════
= Three heads in three flips

== Where Lecture 14 left us

L14 fixed the data, varied the parameter, and maximized the likelihood.

$ hat(theta)_"MLE" = amax_theta P(D | theta) $

#pause
- The coin: $n_H$ heads in $n$ flips $arrow.r.double hat(theta)_"MLE" = n_H \/ n$ — count and divide.
#pause
- With fixed-variance Gaussian observation noise, least squares and maximum likelihood have the same minimizer.
#pause
- Several common supervised losses are negative log-likelihoods for explicit observation models.

#pause
#notebox[With only three observations, the Bernoulli MLE can lie on the boundary and report a zero probability for an unobserved outcome.]

== Three flips, three heads #V

You flip a coin from your pocket three times. Heads, heads, heads. MLE does what it always does:

$ hat(theta)_"MLE" = n_H / n = 3 / 3 = 1.0 $

#align(center, lines(fn: t => t * t * t, domain: (0, 1), markers: false,
  colors: (GREEN,), size: (95mm, 34mm), x-label: $theta$,
  y-label: text(size: 12pt)[likelihood $theta^3$],
  vlines: ((1.0, text(size: 12pt)[$hat(theta)_"MLE" = 1$], RED),)))

#pause
The plug-in model at $hat(theta) = 1$ assigns probability zero to every future tail.

== Consequence of the boundary estimate

$hat(theta) = 1$ is a checkable claim: it prices "at least one tail in the next 100 flips" at *exactly zero*.

#pause
- Background knowledge about ordinary coins makes this plug-in prediction implausible.
#pause
- The three flips are real evidence — but they are *three flips*.
#pause
- A Bayesian model represents such pre-data information with a prior distribution.

#pause
#result[MLE has no slot for what you knew before the data. Today we build that slot: the *prior*.]

== Zero-count estimates in simple models

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*System*], [*The three-heads bug, at work*]),
  [spam filter], ["invoice" never appeared in the spam folder → $P("invoice" | "spam") = 0$ → one word acquits any email, forever],
  [drug trial], [3 successes in 3 patients → "this drug _cannot_ fail"],
  [language model], [a letter pair never seen in training → probability $0$ → $log 0 = -infinity$ loss on any text containing it],
)

#pause
#notebox[L26 applies the corresponding Dirichlet update to character counts, so unseen transitions receive nonzero posterior-predictive probability.]

== What would a fix even look like?

Three demands, straight from your intuition:

+ Tiny datasets must *not* produce absolute certainty.
#pause
+ For a fixed prior that assigns positive density near the data-generating parameter, its influence should diminish as evidence accumulates.
#pause
+ The hunch itself must be *adjustable* — a fairness zealot and a con-artist-spotter should be able to encode different starting beliefs.

#pause
Today's route: Bayes' rule defines the posterior; MAP extracts its mode; conjugacy gives closed-form posterior updates.

== Learning outcomes

By the end of this lecture you will be able to:

+ Read Bayes' rule as a *belief update*, naming all four factors and their jobs.
+ Compute *MAP* estimates and say exactly when MAP $eq$ MLE.
+ Use the *Beta distribution* as a prior over probabilities, reading $alpha, beta$ as pseudo-counts.
+ Derive ⭐ the *Beta–Bernoulli conjugate update*, Beta$(alpha + n_H, beta + n_T)$.
+ Update *sequentially* as data streams in — and explain why order can't matter.
+ State the *Gaussian–Gaussian* posterior as a precision-weighted average.
+ Interpret *L2 regularization* as Gaussian-prior MAP (and L1 as Laplace-prior MAP).

// ═══════════════════════════ 2 · Bayes rule ═══════════════════════════
= Bayes' rule: the belief-update machine

== The four factors, named

One rule merges what you believed with what you saw:

$ #text(fill: ACC)[$p(theta | D)$] = (#text(fill: GREEN)[$P(D | theta)$] dot #text(fill: TEAL)[$p(theta)$]) / #text(fill: MUTED)[$P(D)$] $

#pause
#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Factor*], [*Name*], [*Its job*]),
  [#text(fill: ACC)[$p(theta | D)$]], [#text(fill: ACC, weight: 700)[posterior]], [density of $theta$ _after_ the data],
  [#text(fill: GREEN)[$P(D | theta)$]], [#text(fill: GREEN, weight: 700)[likelihood]], [how loudly the data argues for each $theta$ — L14's object],
  [#text(fill: TEAL)[$p(theta)$]], [#text(fill: TEAL, weight: 700)[prior]], [density of $theta$ _before_ the data],
  [#text(fill: MUTED)[$P(D)$]], [#text(fill: MUTED, weight: 700)[evidence]], [one number that makes the left side a distribution],
)

== Numbers first: a bag of coins

A bag holds 10 coins: *9 fair* ($theta = 0.5$), *1 rigged* ($theta = 0.9$). You draw one at random, flip it 3 times: *3 heads*. Did you draw the rigged one?

#pause
Two hypotheses → Bayes' rule is a two-row table:

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Coin*], [#text(fill: TEAL)[*prior*]], [#text(fill: GREEN)[*likelihood* $theta^3$]], [*prior $times$ lik*], [#text(fill: ACC)[*posterior*]]),
  [fair], [$0.9$], [$0.5^3 = 0.125$], [$0.1125$], [$0.1125 \/ 0.1854 = bold(0.607)$],
  [rigged], [$0.1$], [$0.9^3 = 0.729$], [$0.0729$], [$0.0729 \/ 0.1854 = bold(0.393)$],
)

#pause
Three heads moved the posterior probability of "rigged" from *10% to 39%*. The update multiplies prior by likelihood and rescales the two numbers to sum to 1.

== The update, drawn — and coded #V

#two(
  bars((0.9, 0.1), labels: ([fair], [rigged]), annotate: true, digits: 2,
    size: (46mm, 26mm), title: [before: #text(fill: TEAL)[prior]]),
  bars((0.607, 0.393), labels: ([fair], [rigged]), annotate: true, digits: 3,
    size: (46mm, 26mm), title: [after 3 heads: #text(fill: ACC)[posterior]]),
)

#pause
#codebox[```python
prior = np.array([0.9, 0.1])          # P(fair), P(rigged)
like  = np.array([0.5**3, 0.9**3])    # P(HHH | each coin)
post  = prior * like                  # multiply...
post /= post.sum()                    # ...normalize -> [0.607, 0.393]
```]

#pause
This finite-hypothesis example implements Bayes' rule by multiplication and normalization.

== The evidence is bookkeeping

For discrete hypotheses, $P(D) = sum_h P(D | h)P(h)$. For a continuous parameter, $P(D) = integral P(D | theta)p(theta) dif theta$. In either case it is constant with respect to $theta$ and cannot move the posterior mode.

#pause
$ #text(fill: ACC)[$p(theta | D)$] prop #text(fill: GREEN)[$P(D | theta)$] dot #text(fill: TEAL)[$p(theta)$] $

#pause
#result[*Posterior $prop$ likelihood $times$ prior.* The evidence is the normalization constant that makes the posterior integrate to one.]

#pause
The proportional form is sufficient whenever only the posterior shape or mode is needed.

== MAP: maximize the posterior density

Same recipe as MLE — but climb the *posterior* instead of the likelihood:

$ theta_"MLE" = amax_theta #text(fill: GREEN)[$P(D | theta)$] quad quad quad theta_"MAP" = amax_theta #text(fill: GREEN)[$P(D | theta)$] dot #text(fill: TEAL)[$p(theta)$] $

#pause
In log-space (products → sums — L2's rule, as always):

$ theta_"MAP" = amax_theta [ underbrace(log P(D | theta), "log-likelihood") + underbrace(log p(theta), "log-prior density") ] $

#pause
*MAP = maximum a posteriori.* On the log scale, MAP maximizes the sum of log-likelihood and log-prior density.

== A flat prior changes nothing

Take $p(theta) = 1$ on $[0, 1]$. Then $log p(theta) = 0$:

$ theta_"MAP" = amax_theta [ log P(D | theta) + 0 ] = theta_"MLE" $

#pause
#result[In this bounded parameterization, a uniform prior makes the MAP and MLE optimizers coincide. This does not mean that MLE assumes a prior: MLE and MAP are distinct procedures, and a density uniform in one parameterization need not be uniform after reparameterization.]

#pause
A prior changes the posterior and MAP estimate most visibly in small samples. For fixed regular priors with support near the data-generating parameter, its influence on the MAP estimate typically diminishes as $n$ grows.

// ═══════════════════════════ 3 · the coin, worked ═══════════════════════════
= The coin gets a prior

== A prior must live on $[0, 1]$

$theta$ is a probability, so a prior is a *density on $[0,1]$* (L12: continuous belief is area). Two candidates:

#align(center, lines(
  fn: (x => 1.0, x => 6.0 * x * (1.0 - x)),
  domain: (0, 1), markers: false, samples: 100,
  colors: (INK, TEAL), dashes: ("dashed", "solid"),
  labels: ([flat: any bias is fine], [hill: probably near-fair]),
  size: (86mm, 36mm), x-label: $theta$, y-label: $p(theta)$))

#pause
The hill is $p(theta) prop theta (1 - theta)$ — zero at the impossible extremes, gentle peak at $1\/2$. (It has a family name; we'll meet the family in a few slides.)

== Worked: the flat prior

Multiply the likelihood of three heads by the flat prior:

$ p(theta | D) prop underbrace(theta^3, "likelihood") dot underbrace(1, "prior") = theta^3 $

#pause
Climbing $theta^3$ on $[0,1]$ ends at the right wall:

$ theta_"MAP" = 1.0 = theta_"MLE" $

#pause
The boundary estimate survives a flat prior. Moving it requires an informative prior.

== Worked: the hill prior, by hand

Same three heads, hill prior $p(theta) prop theta(1 - theta)$:

$ p(theta | D) prop theta^3 dot theta (1 - theta) = theta^4 (1 - theta) $

#pause
Maximize with one derivative (drop constants; they can't move a peak):

$ dif / (dif theta) [ theta^4 (1 - theta) ] = 4 theta^3 - 5 theta^4 = theta^3 (4 - 5 theta) = 0 $

#pause
$ arrow.r.double quad theta_"MAP" = 4 / 5 = 0.8 $

#pause
#result[Same three heads: the flat prior says $1.00$; a gentle near-fair belief says $0.80$.]

== Prior $times$ likelihood $=$ posterior, drawn #V

#align(center, lines(
  fn: (beta-pdf(2, 2), beta-pdf(4, 1), beta-pdf(5, 2)),
  domain: (0, 1), markers: false, samples: 120,
  colors: (TEAL, GREEN, ACC), dashes: ("dashed", "solid", "solid"),
  labels: ([prior], [likelihood], [posterior]),
  legend: "tl", size: (108mm, 44mm), x-label: $theta$,
  vlines: ((0.8, [$theta_"MAP"$], ACC), (1.0, [$theta_"MLE"$], RED),)))

#align(center, text(size: 16pt, fill: MUTED)[Likelihood rescaled to area 1 so the three shapes share one axis.])

#pause
The posterior mode lies between the prior mode ($0.5$) and likelihood maximum ($1.0$), with the relative concentration of the two factors determining its position.

== Check it by computer

No calculus, just a grid — the same argmax, numerically:

#codebox[```python
th   = np.linspace(0, 1, 1001)
flat = th**3                        # likelihood x flat prior
hill = th**3 * th * (1 - th)        # likelihood x hill prior

th[flat.argmax()], th[hill.argmax()]     # -> (1.0, 0.8)
```]

#pause
- Shapes only — no normalizing constants anywhere, because `argmax` ignores scale.
#pause
- This grid trick works for *any* prior you can evaluate — calculus optional. (T7 uses it to cross-check every derivation.)

== Checkpoint: MLE vs MAP #Q

#mcq([A coin shows 9 heads in 10 flips. With a *flat* prior on $theta$, the MAP estimate is…],
  [$0.50$ — the prior always pulls to fair],
  [$0.90$ — exactly the MLE],
  [$0.83$ — pulled part-way toward $1\/2$],
  [$1.00$ — MAP always sharpens the MLE],
)

== Answer: MLE vs MAP #A

#mcq-answer([B], [$theta_"MAP" = 0.90 = theta_"MLE"$],
  [A flat prior adds $log 1 = 0$ to the log-posterior — the argmax cannot move, so MAP $=$ MLE $= 9\/10$. Answer C ($approx 0.83$) is what the _hill_ prior Beta$(2,2)$ would give — right instinct, wrong prior; A forgets the data entirely; D isn't a thing MAP does.])

// ═══════════════════════════ 4 · the Beta family ═══════════════════════════
= The Beta family as a prior over probabilities

== The hill has a family name

Tilt the exponents and the hill becomes a whole family of beliefs on $[0,1]$:

$ "Beta"(theta; alpha, beta) = (theta^(alpha - 1) (1 - theta)^(beta - 1)) / (B(alpha, beta)), quad quad B(alpha, beta) = (Gamma(alpha) Gamma(beta)) / (Gamma(alpha + beta)) $

#pause
- $Gamma(k) = (k - 1)!$ for positive integers, and the gamma function extends this definition so $alpha$ and $beta$ need not be integers.
#pause
- Our hill $prop theta(1 - theta)$ is $"Beta"(2, 2)$; the flat prior is $"Beta"(1, 1)$.

#pause
#notebox[$B(alpha, beta)$ exists so the area is 1. We will *never* integrate it in this lecture — inspection will hand it to us for free.]

== The shape gallery #V

All dials from ${0.5, 1, 2, 5}$ — one family, five personalities:

#grid(columns: 3, gutter: 14pt, row-gutter: 6pt,
  lines(fn: beta-pdf(0.5, 0.5), domain: (0.004, 0.996), markers: false, samples: 120,
    colors: (RED,), y-ticks: false, size: (38mm, 22mm), title: [Beta(0.5, 0.5) — extremist]),
  lines(fn: beta-pdf(1, 1), domain: (0, 1), markers: false,
    colors: (MUTED,), y-ticks: false, size: (38mm, 22mm), title: [Beta(1, 1) — agnostic]),
  lines(fn: beta-pdf(2, 2), domain: (0, 1), markers: false,
    colors: (TEAL,), y-ticks: false, size: (38mm, 22mm), title: [Beta(2, 2) — gentle hunch]),
  lines(fn: beta-pdf(5, 5), domain: (0, 1), markers: false,
    colors: (BLUE,), y-ticks: false, size: (38mm, 22mm), title: [Beta(5, 5) — firmer hunch]),
  lines(fn: beta-pdf(5, 2), domain: (0, 1), markers: false,
    colors: (ACC,), y-ticks: false, size: (38mm, 22mm), title: [Beta(5, 2) — leans heads]),
  lines(fn: beta-pdf(2, 5), domain: (0, 1), markers: false,
    colors: (GREEN,), y-ticks: false, size: (38mm, 22mm), title: [Beta(2, 5) — leans tails]),
)

#pause
One knob pair covers superstition, indifference, hunches, conviction, and lopsided suspicion.

== A mode-based pseudo-count mnemonic

For $alpha, beta > 1$ the peak sits at $"mode" = (alpha - 1) \/ (alpha + beta - 2)$ — exactly the MLE of a record with $alpha - 1$ heads and $beta - 1$ tails.

#pause
#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Prior*], [*Imaginary record*], [*Peak*]),
  [Beta$(1, 1)$], [0 H, 0 T — no memory at all], [flat],
  [Beta$(2, 2)$], [1 H, 1 T — the gentlest fairness hint], [$0.5$],
  [Beta$(5, 2)$], [4 H, 1 T — suspects a heads bias], [$0.8$],
  [Beta$(50, 50)$], [49 H, 49 T — fairness conviction], [$0.5$],
)

#pause
#result[For MAP, $alpha - 1$ and $beta - 1$ act like added head and tail counts. This is a mode mnemonic; the posterior update itself adds observations directly to $alpha$ and $beta$.]

== Concentration is controlled by $alpha + beta$

Same center, different bankroll: mean $= alpha \/ (alpha + beta)$ fixes _where_; the total $alpha + beta$ fixes _how firmly_.

#align(center, lines(
  fn: (beta-pdf(2, 2), beta-pdf(50, 50)),
  domain: (0, 1), markers: false, samples: 140,
  colors: (TEAL, BLUE),
  title: [#text(fill: TEAL)[Beta(2, 2) — low concentration] #h(14pt) #text(fill: BLUE)[Beta(50, 50) — high concentration]],
  size: (100mm, 38mm), x-label: $theta$))

#pause
A Beta$(50,50)$ prior is much more concentrated than Beta$(2,2)$, so the same dataset moves it less.

== Play the update before we derive it #I

#interbox(link-to: IA + "bayesian-posterior")[Drag the prior's $alpha, beta$, stream in evidence across 8 experiments, and watch prior $times$ likelihood $=$ posterior re-render live.]

#pause
#interbox(link-to: "https://seeing-theory.brown.edu/bayesian-inference/index.html")[_Seeing Theory_ · Bayesian Inference — flip a coin one toss at a time and watch a Beta posterior fill in.]

#pause
Both toys are running one identity — the one we now derive by hand.

// ═══════════════════════════ 5 · the ⭐ derivation ═══════════════════════════
= Conjugacy: the one-line update ⭐

== Set up the multiplication #D

Data: $n_H$ heads, $n_T$ tails ($n = n_H + n_T$). L14's likelihood, and a Beta$(alpha, beta)$ prior — constants dropped, shapes kept ($prop$ absorbs $1\/B$):

$ #text(fill: ACC)[$p(theta | D)$] prop underbrace(#text(fill: GREEN)[$theta^(n_H) (1 - theta)^(n_T)$], "likelihood") dot underbrace(#text(fill: TEAL)[$theta^(alpha - 1) (1 - theta)^(beta - 1)$], "prior density") $

#pause
Both factors speak the *same language*: powers of $theta$ and powers of $(1 - theta)$. That coincidence is about to do all the work.

== Exponents simply add #D

Same base, so multiplication is addition upstairs:

$ p(theta | D) prop theta^(n_H + alpha - 1) thin (1 - theta)^(n_T + beta - 1) $

#pause
Stare at it: this is *the Beta shape again*, with new dials

$ alpha' = alpha + n_H, quad quad beta' = beta + n_T $

#pause
The head and tail counts add to the two exponents. The data do not change the family, only its parameters.

== Normalize by inspection #D

We hold a function $prop theta^(alpha' - 1)(1 - theta)^(beta' - 1)$ that must integrate to 1 (it's a posterior). Among all rescalings $c dot theta^(alpha' - 1)(1 - theta)^(beta' - 1)$, *exactly one* has area 1 — and Beta$(alpha', beta')$ is that one, by definition.

#pause
So the constant is forced, no integral computed:

#result[$ "Beta"(alpha, beta) "prior" + n_H "heads", n_T "tails" quad arrow.r.double quad "posterior" = "Beta"(alpha + n_H, thin beta + n_T) $]

#pause
#notebox["By inspection" means recognizing the Beta kernel and using its known normalizing constant.]

== Conjugacy, named

*Definition.* A prior family is *conjugate* to a likelihood if the posterior always lands back in the same family.

#pause
- Beta in → Bernoulli data → Beta out: the Beta family is conjugate to coin flips.
#pause
- The entire update is *two integer additions*: $alpha arrow.l alpha + n_H$, $beta arrow.l beta + n_T$. No integral, no grid, no optimizer.
#pause
- Our worked example: Beta$(2,2)$ + 3 heads → Beta$(5, 2)$, whose mode is $4\/5$.

== The MAP formula drops out

The posterior is Beta$(alpha + n_H, beta + n_T)$, and we know where a Beta peaks:

$ theta_"MAP" = (n_H + alpha - 1) / (n + alpha + beta - 2) $

#pause
- Check: 3 heads, Beta$(2,2)$: $(3 + 1) \/ (3 + 2) = 4\/5 = 0.8$ — matches the hand derivative. ✓
#pause
- Check: flat prior $alpha = beta = 1$: $theta_"MAP" = n_H \/ n = theta_"MLE"$. ✓

#pause
#result[The MAP estimate is the empirical count formula with the mode offsets $alpha - 1$ and $beta - 1$.]

== Stress test: 9 heads, 1 tail

Ten flips, nine heads ($theta_"MLE" = 0.9$). Three believers meet the same data:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Prior*], [*$theta_"MAP"$*], [*Reading*]),
  [Beta$(1, 1)$ — agnostic], [$9\/10 = 0.900$], [no opinion → data speaks alone],
  [Beta$(2, 2)$ — gentle hunch], [$10\/12 approx 0.833$], [two pseudo-flips nudge back toward fair],
  [Beta$(50, 50)$ — conviction], [$58\/108 approx 0.537$], [98 pseudo-flips: ten real ones barely dent it],
)

#pause
Same data, three defensible answers — the prior is a *published, criticizable assumption*, not a hidden thumb on the scale. (Which believer is being unreasonable? That argument is now about $alpha, beta$ — in the open.)

== The conjugate catalog

The same algebra appears beyond coins:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Likelihood*], [*Conjugate prior*], [*Posterior update*]),
  [Bernoulli (coin)], [Beta], [$alpha arrow.l alpha + n_H$, $beta arrow.l beta + n_T$],
  [Categorical ($d$ faces)], [Dirichlet], [add each face's count to its pseudo-count],
  [Gaussian mean ($sigma$ known)], [Gaussian], [precision-weighted average — next section],
)

#pause
#notebox[*Continuity ledger — planted today, paid off in L26:* the Dirichlet row is Beta's $d$-sided sibling. When L26's character language model meets an unseen letter pair, _Dirichlet smoothing_ (add-$alpha$ pseudo-counts) is exactly this table's second row.]

== Checkpoint: the one-line update #Q

#mcq([Prior Beta$(2, 3)$; you then observe 4 heads and 1 tail. The posterior is…],
  [Beta$(6, 4)$],
  [Beta$(3, 7)$],
  [Beta$(5, 2)$],
  [Beta$(7, 8)$],
)

== Answer: the one-line update #A

#mcq-answer([A], [Beta$(2 + 4, thin 3 + 1) = $ Beta$(6, 4)$],
  [Heads land on $alpha$, tails on $beta$ — nothing else moves. B swapped the two counts; C threw the prior away and updated a flat Beta$(1,1)$; D added the total $n = 5$ to both dials, double-counting every flip.])

// ═══════════════════════════ 6 · sequential updating ═══════════════════════════
= Watching beliefs sharpen

== Today's posterior is tomorrow's prior

Nothing forces the flips to arrive in one batch. Feed them one at a time, and *each posterior becomes the next prior*:

$ "Beta"(alpha, beta) arrow.r^("H") "Beta"(alpha + 1, beta) arrow.r^("T") "Beta"(alpha + 1, beta + 1) arrow.r^("H") dots $

#pause
#codebox[```python
a, b = 2, 2                     # Beta(2,2): 1 imaginary head, 1 tail
for y in flips:                 # y = 1 for heads, 0 for tails
    a, b = a + y, b + 1 - y     # the ENTIRE Bayesian update
# at any moment: posterior = Beta(a, b), MAP = (a-1)/(a+b-2)
```]

#pause
Two integers of memory, one addition per observation. This is *online learning* in its purest form — the model never re-reads old data.

== One hundred flips, four snapshots #V

A coin with true $theta = 0.7$ streams flips into a Beta$(2,2)$ prior:

#fig("/lecture15/figures/seq_updating.svg", w: 100%)

#pause
After 1 flip the prior still dominates; by 100 flips (75 H, 25 T → Beta$(77, 27)$) the posterior is a needle — parked at the *sample's* rate $0.75$, a whisker above the truth. That residual wobble is the data's own sampling noise, not the prior's doing.

== The posterior sharpens like $1 \/ sqrt(n)$

How fast is certainty earned? The posterior's spread obeys

$ "sd" approx sqrt( (hat(theta) (1 - hat(theta))) / n ) $

#pause
- Quadruple the flips → *halve* the uncertainty — the same $sqrt(n)$ law you met in L13's CLT demo, now running _inside_ the posterior.
#pause
- By $n = 100$, the prior's 4 pseudo-flips are 4 votes in an electorate of 104: present, out-voted, and fading.

== Order cannot matter

Flip H then T, starting from Beta$(alpha, beta)$: $(alpha + 1, beta)$, then $(alpha + 1, beta + 1)$. Flip T then H: $(alpha, beta + 1)$, then $(alpha + 1, beta + 1)$. *Same place.*

#pause
No accident: the posterior is a *product* of per-flip factors, and multiplication commutes —

$ p(theta | D) prop p(theta) product_(i = 1)^n P(y_i | theta) $

#pause
#result[Stream the data or dump it in one batch, in any order — *exactly* the same posterior.]

== Three scientists, three priors #V

An optimist, a skeptic, and an agnostic watch the same coin ($theta = 0.7$):

#fig("/lecture15/figures/prior_washout.svg", w: 100%)

#pause
Ten flips in (5 H, 5 T), the posteriors still differ. After a thousand flips they are nearly indistinguishable.

== When the prior washes out

The formula says exactly when belief stops mattering:

$ theta_"MAP" = (n_H + alpha - 1) / (n + alpha + beta - 2) arrow.r n_H / n = theta_"MLE" quad "as" n arrow.r infinity $

#pause
- The prior is a *fixed* number of pseudo-votes in a growing electorate: decisive at $n = 3$, a rounding error at $n = 10^6$.
#pause
- With a fixed regular prior and a growing i.i.d. sample, MLE and MAP typically converge to the same value.

#pause
#result[For this fixed Beta prior, the difference between MAP and MLE is of order $1\/n$. Priors with zero support near the truth or priors that change with $n$ need not wash out this way.]

== Posterior inference for the opening coin example

Three heads in three flips, Beta$(2,2)$ prior → posterior Beta$(5,2)$:

#align(center, lines(
  fn: beta-pdf(5, 2), domain: (0, 1), markers: false, samples: 120,
  colors: (ACC,), size: (96mm, 32mm), x-label: $theta$,
  fill-under: (from: 0.0, to: 0.5, color: RED.transparentize(65%), series: 0),
  vlines: ((0.8, text(size: 12pt)[$theta_"MAP" = 0.8$], ACC),)))

#pause
- The MAP estimate is $theta_"MAP" = 0.8$, while the posterior still assigns about $11%$ probability to $theta < 0.5$.
#pause
- After three flips, the posterior remains uncertain and assigns nonzero probability to future tails.

== Slide the prior yourself #I

#interbox(link-to: IA + "mle-map-coin")[A coin you control: set the true bias, choose $alpha, beta$, and flip. Watch $theta_"MAP"$ glide from the prior's peak toward $theta_"MLE"$ as the pseudo-counts get out-voted.]

#pause
Try a strong prior far from the data, then compare it with a flat prior at $n = 3$.

// ═══════════════════════════ 7 · Gaussian meets Gaussian ═══════════════════════════
= Gaussian meets Gaussian

== A continuous analogue

New estimation problem: a sensor reads a true value $mu$ through Gaussian noise (L13):

$ x_i tilde cal(N)(mu, sigma^2), quad sigma "known", quad i = 1, dots, n $

#pause
You also have a hunch — the calibration sheet says readings sit near $mu_0$:

$ mu tilde cal(N)(mu_0, sigma_0^2) $

#pause
Beta was the prior for a *probability*; the Gaussian is the prior for a *location*. And the Gaussian is conjugate to itself: Gaussian in → Gaussian out.

== Stated: Gaussian in, Gaussian out

Multiply the two bells and complete the square (algebra in T7 and MML 8.3.2 — today, the shape of the answer). Call $1\/"variance"$ *precision* — confidence, as a number: prior precision $tau_0 = 1\/sigma_0^2$, data precision $tau_D = n\/sigma^2$.

#pause
$ p(mu | D) = cal(N)(mu_n, sigma_n^2), quad quad 1 / sigma_n^2 = tau_0 + tau_D $

#pause
$ mu_n = (tau_0 thin mu_0 + tau_D thin macron(x)) / (tau_0 + tau_D) $

#pause
*Precisions add*, and the posterior mean is a *precision-weighted average* of hunch $mu_0$ and data average $macron(x)$ — the surer voice counts more.

== Picture and numbers #V

Hunch $mu_0 = 0$ (precision $1\/sigma_0^2 = 1$); data $n = 4$, $macron(x) = 2$, $sigma^2 = 4$ (precision $n\/sigma^2 = 1$). Equal confidence → meet exactly halfway:

$ mu_n = (1 dot 0 + 1 dot 2) / (1 + 1) = 1, quad quad sigma_n^2 = 1 / (1 + 1) = 0.5 $

#align(center, lines(
  fn: (x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), x),
       x => dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), x),
       x => dist.pdf(dist.normal(mu: 1.0, sigma: 0.7071), x)),
  domain: (-3, 4.5), markers: false, samples: 130,
  colors: (TEAL, GREEN, ACC), dashes: ("dashed", "solid", "solid"),
  labels: ([prior], [data alone], [posterior]),
  legend: "tl", size: (102mm, 36mm), x-label: $mu$,
  vlines: ((1.0, text(size: 12pt)[$mu_n = 1$], ACC),)))

#pause
The posterior is *narrower than both parents* — two independent confidences stacked.

== Precision is the currency

- Posterior mean $=$ average of the two centers, *weighted by precision* — the surer voice counts more.
#pause
- The prior behaves like $sigma^2 \/ sigma_0^2$ *virtual observations* pinned at $mu_0$ — pseudo-counts again, in continuous clothing.
#pause
- As $n$ grows, data precision $n\/sigma^2 arrow.r infinity$, so a fixed finite prior precision has diminishing influence.

#pause
#result[Conjugate updating is an *average where precision is the currency*: pseudo-flips for coins, virtual readings for means.]

// ═══════════════════════════ 8 · regularization = prior ═══════════════════════════
= MAP interpretations of common penalties

== Recall the Gaussian-noise result from L14

Linear regression, Gaussian noise: $y_i = bold(w)^top bold(x)_i + epsilon_i$, $epsilon_i tilde cal(N)(0, sigma^2)$. L14 showed:

$ "NLL"(bold(w)) = 1 / (2 sigma^2) norm(bold(y) - X bold(w))^2 + "const" $

#pause
Under independent Gaussian observation noise with fixed variance, minimizing NLL is equivalent to least squares.

#pause
To form a MAP estimator, place an explicit prior on the weights.

== Give the weights a prior

Belief: weights shouldn't be huge — a model that needs $w = 10^6$ is probably memorizing noise. Encode it:

$ bold(w) tilde cal(N)(bold(0), sigma_0^2 I) quad arrow.r.double quad -log p(bold(w)) = 1 / (2 sigma_0^2) norm(bold(w))^2 + "const" $

#pause
MAP swaps argmax-posterior for argmin of the negative log-posterior:

$ bold(w)_"MAP" = amin_(bold(w)) thin underbrace(1 / (2 sigma^2) norm(bold(y) - X bold(w))^2, "data misfit (L14)") + underbrace(1 / (2 sigma_0^2) norm(bold(w))^2, "prior's fine on big weights") $

#pause
#codebox[```python
nll  = ((y - X @ w)**2).sum() / (2 * sig2)     # L14: Gaussian NLL
nlp  = (w**2).sum() / (2 * sig0_2)             # -log N(0, sig0^2) prior
loss = nll + nlp                               # MAP objective
```]

== MAP with a Gaussian prior is ridge regression

Multiply through by $2 sigma^2$ (argmins don't care):

$ bold(w)_"MAP" = amin_(bold(w)) thin norm(bold(y) - X bold(w))^2 + lambda norm(bold(w))^2, quad quad lambda = sigma^2 / sigma_0^2 $

#pause
#result[*Ridge regression is MAP with a Gaussian prior on the weights.* The coefficient $lambda$ is the noise-to-prior-variance ratio.]

#pause
#notebox[A later ML course may introduce ridge regression or weight decay as regularization, with $lambda$ tuned by validation. In this probabilistic model, increasing $lambda$ corresponds to decreasing the prior variance $sigma_0^2$ and concentrating more prior mass near zero.]

== Negative log-prior to penalty #V

A prior density contributes its negative logarithm to the MAP objective:

#two(
  lines(
    fn: (x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), x),
         x => dist.pdf(dist.laplace(mu: 0.0, b: 1.0), x)),
    domain: (-3, 3), markers: false, samples: 130,
    colors: (TEAL, BLUE), size: (58mm, 34mm), x-label: $w$,
    title: [the prior — #text(fill: TEAL)[Gaussian] vs #text(fill: BLUE)[Laplace]]),
  lines(
    fn: (x => x * x / 2.0, x => calc.abs(x)),
    domain: (-3, 3), markers: false, samples: 130,
    colors: (TEAL, BLUE), labels: ([$w^2 \/ 2$], [$abs(w)$]),
    size: (58mm, 34mm), x-label: $w$, title: [the penalty $-log p(w)$]),
)

#pause
Gaussian prior → quadratic penalty → *L2 / ridge*. Laplace prior → $abs(w)$ penalty → *L1 / lasso*; the corner at zero allows exact zero-valued coefficients.

== The regularization dictionary

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*You add to the loss…*], [*You just assumed…*], [*Its street name*]),
  [$lambda norm(bold(w))^2$], [Gaussian prior $cal(N)(0, sigma_0^2)$], [ridge / weight decay],
  [$lambda norm(bold(w))_1$], [Laplace prior], [lasso],
  [add-$alpha$ pseudo-counts], [Beta / Dirichlet prior], [Laplace smoothing (→ L26)],
)

#pause
If $exp(-R(bold(w)))$ is integrable, a penalty $R$ can be interpreted as a negative log-prior in MAP estimation. Some training regularizers and constraints do not admit this simple interpretation.

// ═══════════════════════════ 9 · wrap ═══════════════════════════
= Summary

== Lecture 15 — summary

- *Bayes*: posterior $prop$ likelihood $times$ prior — the evidence only normalizes.
- *MAP* climbs log-likelihood $+$ log-prior; flat prior $arrow.r.double$ MAP $=$ MLE.
- *Beta$(alpha, beta)$*: the MAP mode has offsets $alpha - 1$, $beta - 1$; $alpha + beta$ controls concentration.
- *Conjugacy* ⭐: Beta prior $+$ coin data $arrow.r$ Beta$(alpha + n_H, beta + n_T)$ — two additions.
- *Sequential $=$ batch*, any order; width $tilde 1\/sqrt(n)$; data swamps the prior.
- *Gaussians*: precisions add; posterior mean $=$ precision-weighted average.
- *MAP penalty interpretations*: Gaussian → ridge, Laplace → lasso, Beta/Dirichlet → smoothing.

#pause
#notebox[*Read before L16* — MML 6.6.1 & 8.3.2 · MacKay Ch. 3. *T7* pairs with this lecture (grid-check every derivation); *Quiz 3 / midsem* covers Module 3.]

== Practice problems

Try on paper; verify with the T7 notebook's grid trick.

+ Re-derive the ⭐ update: Beta$(alpha, beta)$ prior, $n_H$ heads, $n_T$ tails → posterior, from a blank page.
+ Prior Beta$(5, 5)$, data 2 H, 8 T: posterior, $theta_"MAP"$, posterior mean — and compare with $theta_"MLE"$.
+ Show that Beta$(1, 1)$ makes $theta_"MAP" = theta_"MLE"$ for *every* possible dataset.
+ How many *consecutive heads* push $theta_"MAP"$ above $0.99$ under a Beta$(2, 2)$ prior? (Solve $(n + 1)\/(n + 2) > 0.99$.)
+ Sensor: prior $cal(N)(0, 1)$; data $n = 9$, $macron(x) = 3$, $sigma^2 = 9$. Posterior mean and variance?
+ A colleague doubles ridge's $lambda$. What happens to $sigma_0^2$ in the MAP interpretation, and which prior would produce an L1 penalty instead?

== ⭐⭐⭐ MAP keeps the peak, throws the width #OPT

MAP reports *one number* — but the posterior was a whole curve, and the curve knows more.

#pause
- Predicting the next flip by *averaging over the posterior* gives the posterior _mean_, not the mode: $P("next H" | D) = (alpha + n_H) / (alpha + beta + n)$.
#pause
- Three heads, Beta$(2,2)$: predict $5\/7 approx 0.71$ — milder than $theta_"MAP" = 0.8$, and a world away from MLE's $1.0$.
#pause
- Keeping the *entire* posterior (and integrating over it) is Bayesian machine learning proper — the Probabilistic ML course lives there.

== ⭐⭐⭐ Laplace's rule of succession #OPT

Flat prior $+$ posterior mean $arrow.r$ $P("next H") = (n_H + 1)/(n + 2)$ — "add one imaginary head and one imaginary tail."

#pause
- Laplace, 1774: what is the probability the sun rises tomorrow, given $n$ recorded sunrises? Answer: $(n + 1)\/(n + 2)$ — huge, but *never 1*.
#pause
- NLP's *add-one smoothing* is this rule, verbatim, applied to word counts — and its add-$alpha$ generalization is the Dirichlet machinery L26 uses to keep the language model's $log 0$ at bay.

#pause
#notebox[250 years old, still shipping in production systems. Priors age well.]

#focus-slide[
  L2 and L1 penalties are negative log-priors in MAP estimation.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Module 4 — The Optimization Landscape*. We now know _what_ to minimize; L16 begins the story of _how_.
]
