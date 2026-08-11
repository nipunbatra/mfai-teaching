// Maximum Likelihood Estimation — Lecture 14 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture14/L14-mle.typ
//   typst compile --root . --input handout=true lecture14/L14-mle.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Maximum Likelihood Estimation],
  subtitle: [Where every loss function comes from],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"

// ── one consistent legend swatch treatment for multi-series figures ──
#let sw(color, label) = box(width: 12pt, height: 3pt, fill: color, radius: 1pt) + h(4pt) + label

// ── computed scientific notation (so tiny likelihood values are never hand-typed) ──
#let sci(v, digits: 1) = {
  let e = calc.floor(calc.log(v))          // calc.log is base 10
  let m = calc.round(v / calc.pow(10.0, e), digits: digits)
  $#m times 10^(#e)$
}

// ── sample a function into a series (to mix curves with data-dot series in one plot) ──
#let curve(f, a, b, n: 96) = range(n + 1).map(i => { let x = a + i * (b - a) / n; (x, f(x)) })

// ── the three-courses pop quiz (scores ~ Gaussians; L12's zoo, L13's Gaussians) ──
#let C1 = dist.normal(mu: 80.0, sigma: 10.0)
#let C2 = dist.normal(mu: 70.0, sigma: 10.0)
#let C3 = dist.normal(mu: 90.0, sigma: 5.0)

// ── the running Gaussian dataset: mean exactly 2.0, MLE variance exactly 1.0 ──
#let D5 = (0.5, 1.5, 2.0, 2.5, 3.5)
#let g-lik(m) = D5.map(x => dist.pdf(dist.normal(mu: m, sigma: 1.0), x)).product()
#let g-ll(m) = D5.map(x => dist.logpdf(dist.normal(mu: m, sigma: 1.0), x)).sum()
// dataset NLL over BOTH parameters (μ, σ): Σ(xᵢ−2)² = 5, so the sum collapses in closed form
#let g-nll2(m, s) = 5.0 * calc.ln(s) + 2.5 * calc.ln(2 * calc.pi) + (5.0 * (m - 2.0) * (m - 2.0) + 5.0) / (2.0 * s * s)

// ── the coin: 7 heads in 10 flips ──
#let coin-lik(t) = calc.pow(t, 7) * calc.pow(1 - t, 3)
#let coin-ll(t) = 7 * calc.ln(t) + 3 * calc.ln(1 - t)

#title-slide()

= Deriving losses from probability models

== Three lines you will type for the rest of your career

Every ML training loop — every paper, every framework — ends the same way:

#codebox[```python
loss = F.mse_loss(y_hat, y)            # regression
loss = F.binary_cross_entropy(p, y)    # yes/no classification
loss = F.cross_entropy(logits, y)      # multi-class classification
```]

#pause
Squaring errors. Taking logs of probabilities. *Who decided these formulas?* Why squared — not absolute, not fourth power? Why is there a logarithm inside a classification loss?

#pause
#notebox[Nobody tuned these by trial and error. All three fall out of *one idea*, mechanically. Today: the idea, and the full derivation for MSE. Cross-entropy's turn comes in L24 — but you'll leave today knowing exactly where it lives.]

== From a probability model to an objective

#align(center, diagram(
  spacing: (14mm, 8mm), node-stroke: 0.9pt,
  {
    node((0,0), align(center)[model \ $p(x | theta)$], shape: fletcher.shapes.rect, fill: white, stroke: INK, inset: 7pt)
    node((1,0), align(center)[likelihood \ $L(theta)$], shape: fletcher.shapes.rect, fill: TEAL.lighten(90%), stroke: TEAL, inset: 7pt)
    node((2,0), align(center)[log-likelihood \ $ell(theta)$], shape: fletcher.shapes.rect, fill: TEAL.lighten(90%), stroke: TEAL, inset: 7pt)
    node((3,0), align(center)[*NLL* \ $-ell(theta)$], shape: fletcher.shapes.rect, fill: ACC.lighten(88%), stroke: ACC, inset: 7pt)
    node((4,0), align(center, text(fill: white, weight: 600)[MSE · BCE \ CE · MAE]), shape: fletcher.shapes.rect, fill: TEAL, stroke: none, inset: 8pt)
    edge((0,0), (1,0), text(size: 11.5pt)[fix data, vary $theta$], "-|>", stroke: 0.8pt + MUTED, label-side: left, label-sep: 21pt)
    edge((1,0), (2,0), text(size: 11.5pt)[$log$], "-|>", stroke: 0.8pt + MUTED, label-side: left, label-sep: 21pt)
    edge((2,0), (3,0), text(size: 11.5pt)[negate], "-|>", stroke: 0.8pt + MUTED, label-side: left, label-sep: 21pt)
    edge((3,0), (4,0), text(size: 11.5pt)[pick a noise model], "-|>", stroke: 0.8pt + MUTED, label-side: left, label-sep: 21pt)
  }
))

#pause
Fix the observed data and vary the parameters. The following slides compute every step in this map.

#pause
#notebox[You have never formally "fit a model" in this course. That is the point: today is where *fitting* becomes a mathematical operation instead of a vibe.]

== Learning outcomes

By the end of this lecture you will be able to:

+ Read $p(x | theta)$ *two ways*: as a distribution over $x$, and as a likelihood over $theta$.
+ Write the likelihood of an entire dataset using the *IID assumption*.
+ Explain *three independent reasons* the log goes in — and recap L2's $0.03^1000$ disaster.
+ Derive $hat(theta)_"MLE"$ for the *Bernoulli* coin (⭐) and the *Gaussian* mean.
+ Derive the course's keystone: *minimizing MSE = maximizing Gaussian likelihood* (⭐).
+ Map noise models to the losses they generate: Gaussian→MSE, Bernoulli→BCE, Categorical→CE, Laplace→MAE.

= The flip: from probability to likelihood

== Pop quiz · one student, three courses

Three courses grade on curves. Their score distributions:

- C1: $cal(N)(mu = 80, sigma = 10)$
- C2: $cal(N)(mu = 70, sigma = 10)$
- C3: $cal(N)(mu = 90, sigma = 5)$

#pause
I stop a random student in the corridor. They took exactly one of the three, and they scored *82*.

#pause
*Which course would you guess?* Commit to an answer — and notice what your brain just did to produce it.

== The guess, made honest #V

#align(center, text(size: 14pt)[
  #sw(TEAL, [C1: $cal(N)(80, 10^2)$]) #h(14pt)
  #sw(BLUE, [C2: $cal(N)(70, 10^2)$]) #h(14pt)
  #sw(RED, [C3: $cal(N)(90, 5^2)$])
])
#v(2pt)
#align(center, lines(
  fn: (x => dist.pdf(C1, x), x => dist.pdf(C2, x), x => dist.pdf(C3, x)),
  domain: (42, 112), samples: 160, markers: false,
  colors: (TEAL, BLUE, RED),
  vlines: ((82.0, [$x = 82$], ACC),),
  y-ticks: false, x-label: [score], y-label: [density],
  size: (108mm, 40mm),
))

#pause
Read each curve *at the observed score*: $p(82 | "C1") = #calc.round(dist.pdf(C1, 82.0), digits: 4)$, $p(82 | "C2") = #calc.round(dist.pdf(C2, 82.0), digits: 4)$, $p(82 | "C3") = #calc.round(dist.pdf(C3, 82.0), digits: 4)$.

#pause
#result[C1 wins: you picked the model under which the data is *least surprising*. That's maximum likelihood — you already do it by eye.]

== The flip, on the simplest model there is #V

A coin with $P("heads") = theta$. One formula, $p(x | theta)$ — *two completely different readings*:

#two(r: (1fr, 1.05fr),
  [
    #text(fill: TEAL, weight: 700)[Fix $theta = 0.7$, vary $x$]
    #v(2pt)
    #align(center, bars((0.3, 0.7), labels: ([$x = 0$], [$x = 1$]), highlight: 1, size: (44mm, 30mm)))
    #v(3pt)
    a *distribution* over outcomes \ #text(size: 16pt, fill: MUTED)[bars sum to 1 — guaranteed]
  ],
  [
    #pause
    #text(fill: ACC, weight: 700)[Fix $x = 1$ (one head), vary $theta$]
    #v(2pt)
    #align(center, lines(fn: t => t, domain: (0, 1), markers: false, colors: (ACC,),
      x-label: [$theta$], y-label: [$L(theta)$], size: (52mm, 30mm)))
    #v(3pt)
    the *likelihood* of that outcome \ #text(size: 16pt, fill: MUTED)[area under this line: $1\/2$ — every $theta$ gives the same value]
  ],
)

#pause
Two bars became a line: *different axis, different object* — same formula.

== The flip, on the corridor student #V

Now suppose we know the course has $sigma = 10$ but *nobody told us $mu$*. The student says 82.

#two(r: (1fr, 1fr),
  [
    #align(center, text(size: 15pt, fill: TEAL, weight: 700)[DISTRIBUTION: fix $mu = 80$, vary $x$])
    #align(center, lines(
      fn: x => dist.pdf(dist.normal(mu: 80.0, sigma: 10.0), x),
      domain: (45, 115), samples: 120, markers: false, colors: (TEAL,),
      fill-under: (0, TEAL.transparentize(85%)),
      vlines: ((80.0, [$mu = 80$], MUTED),),
      y-ticks: false, x-label: [score $x$], size: (58mm, 30mm),
    ))
  ],
  [
    #pause
    #align(center, text(size: 15pt, fill: ACC, weight: 700)[LIKELIHOOD: fix $x = 82$, vary $mu$])
    #align(center, lines(
      fn: m => dist.pdf(dist.normal(mu: m, sigma: 10.0), 82.0),
      domain: (45, 115), samples: 120, markers: false, colors: (ACC,),
      fill-under: (0, ACC.transparentize(85%)),
      vlines: ((82.0, [$hat(mu) = 82$], ACC),),
      y-ticks: false, x-label: [candidate $mu$], size: (58mm, 30mm),
    ))
  ],
)

#pause
The right curve peaks at $mu = 82$: the single best explanation centers the bell *on the data*. You just estimated a parameter.

== Naming the move

#result[$L(theta) := p(cal(D) | theta)$, read as a function of $theta$ with the data $cal(D)$ *frozen*. \ $hat(theta)_"MLE" = arg max_theta L(theta)$.]

#pause
- $p(x | theta)$ as a function of $x$: a *distribution* — normalized, sums/integrates to 1
#pause
- the same formula as a function of $theta$: the *likelihood* — a plain score, one number per candidate model

#pause
#alertbox[$L(theta)$ is *not* a probability distribution over $theta$: nothing makes its area 1 (the one-head line had area $1\/2$). "Which $theta$ is most probable" is a different, stronger question — that's Bayes and L15. Today we only ask which $theta$ *explains the data best*.]

= The likelihood of a dataset

== From one observation to a dataset

One data point gives one score: $L(theta) = p(x_1 | theta)$. A dataset $cal(D) = {x_1, dots, x_n}$ needs the *joint* probability $p(x_1, dots, x_n | theta)$ — and joints are hard. We buy simplicity with an assumption. *IID* is two separate claims:

#pause
- *Independent* — given $theta$, no observation tells you about another → the joint *factors into a product*
#pause
- *Identically distributed* — every observation uses the *same* $p(dot | theta)$, one shared $theta$

#pause
$ L(theta) = product_(i=1)^n p(x_i | theta) $

#pause
#notebox[Each claim can fail alone: tomorrow's weather depends on today's (→ L25's Markov chains); two different-noise sensors share no rule. IID is a modeling *choice* — say it out loud.]

== IID at work · two coins, two tiny datasets

Candidate coins: fair ($theta = 0.5$) vs head-heavy ($theta = 0.8$). Score both on data:

#align(center, table(
  columns: (auto, auto, auto, 1fr), stroke: 0.5pt + MUTED.lighten(40%),
  inset: (x: 10pt, y: 7pt), align: (center, center, center, left),
  table.header([*Observed*], [*$L(0.5)$*], [*$L(0.8)$*], [*Better explanation*]),
  [$H, T$], [$0.5 times 0.5 = 0.25$], [$0.8 times 0.2 = 0.16$], [the fair coin],
  [$H, H$], [$0.5^2 = 0.25$], [$0.8^2 = 0.64$], [the head-heavy coin],
))

#pause
Same two candidates, different frozen data → different winner. The likelihood *ranks models by the data they actually saw*.

#pause
#result[Independence licenses the multiplication; identical distribution licenses reusing one $theta$ in every factor.]

== Pop quiz, round 2 · now with a dataset

Back in the corridor. *Two* students this time — scores $82$ and $72$ — and both took the *same* (unknown) course. Multiply each course's two density readings:

#let pq(c, x) = calc.round(dist.pdf(c, x), digits: 4)
#align(center, table(
  columns: (auto, auto, auto, auto), stroke: 0.5pt + MUTED.lighten(40%),
  inset: (x: 10pt, y: 6pt), align: (center, center, center, center),
  table.header([*Course*], [$p(82 | dot)$], [$p(72 | dot)$], [*$L = $ product*]),
  [C1: $cal(N)(80, 10^2)$], [#pq(C1, 82.0)], [#pq(C1, 72.0)], [#sci(dist.pdf(C1, 82.0) * dist.pdf(C1, 72.0))],
  [C2: $cal(N)(70, 10^2)$], [#pq(C2, 82.0)], [#pq(C2, 72.0)], [#sci(dist.pdf(C2, 82.0) * dist.pdf(C2, 72.0))],
  [C3: $cal(N)(90, 5^2)$], [#pq(C3, 82.0)], [#pq(C3, 72.0)], [#sci(dist.pdf(C3, 82.0) * dist.pdf(C3, 72.0))],
))

#pause
C1 wins again — but look at C3: *one* far-off point ($72$, at $z = -3.6$) collapsed its entire product. A product is a chain of vetoes: every point must be plausible.

#pause
#result[A dataset's likelihood multiplies one reading per point — computed with the IID recipe you just learned.]

== Five readings, one score #V

Our running dataset: $cal(D) = {0.5, #h(2pt) 1.5, #h(2pt) 2.0, #h(2pt) 2.5, #h(2pt) 3.5}$, modeled as $cal(N)(mu, 1)$ with $sigma$ known. Try $mu = 2$: read the density at each point, multiply.

#align(center, lines(
  fn: x => dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), x),
  domain: (-1.2, 5.2), samples: 130, markers: false, colors: (INK,),
  points: D5.map(x => (x, dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), x))),
  y-ticks: false, x-label: [$x$], y-label: [density at $mu = 2$],
  size: (100mm, 37mm),
))

#pause
$ L(2) = #calc.round(dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), 0.5), digits: 3) times #calc.round(dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), 1.5), digits: 3) times #calc.round(dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), 2.0), digits: 3) times #calc.round(dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), 2.5), digits: 3) times #calc.round(dist.pdf(dist.normal(mu: 2.0, sigma: 1.0), 3.5), digits: 3) approx #sci(g-lik(2.0)) $

#pause
One number for the candidate $mu = 2$. Now the key mental animation: *slide the bell and watch the score move*.

== Slide the Gaussian over the data #V

#let panel(m, tag) = align(center, stack(spacing: 5pt,
  text(size: 15pt, weight: 700, fill: if m == 2.0 { GREEN } else { RED }, tag),
  lines(
    (curve(x => dist.pdf(dist.normal(mu: m, sigma: 1.0), x), -1.6, 5.6),)
      + D5.map(x => ((x, 0.0), (x, 0.06))),          // the data: orange stems, fixed in place
    markers: false, colors: (INK,) + D5.map(_ => ACC),
    vlines: ((m, [$mu = #m$], MUTED),),
    y-ticks: false, x-label: none, size: (52mm, 26mm),
  ),
  text(size: 16pt)[$L(#m) approx #sci(g-lik(m))$],
))

#grid(columns: (1fr, 1fr, 1fr), gutter: 10pt,
  panel(1.0, [too far left]),
  panel(2.0, [just right]),
  panel(3.2, [too far right]),
)

#pause
#v(4pt)
Same five orange data points every time; only the model slides. Centered wins by a factor of *#calc.round(g-lik(2.0) / g-lik(1.0), digits: 0)×* over $mu = 1$.

#pause
#result[Fitting a model = sliding its parameters until the data's likelihood is maximal.]

== The whole slide, as one curve #V

Evaluate $L(mu)$ at *every* candidate $mu$ — the likelihood function of this dataset:

#align(center, lines(
  fn: m => g-lik(m), domain: (0.0, 4.0), samples: 140, markers: false, colors: (ACC,),
  fill-under: (0, ACC.transparentize(86%)),
  vlines: ((2.0, [$hat(mu) = 2.0$], ACC),),
  y-ticks: false, x-label: [candidate $mu$], y-label: [$L(mu)$],
  size: (100mm, 40mm),
))

#pause
- Each point of this curve is one "slide position" from the previous slide — a *product of five densities*
#pause
- The peak sits at $hat(mu) = 2.0$ — which happens to be the sample mean of $cal(D)$. Coincidence? We'll *prove* it shortly.

== Checkpoint: what kind of object is $L(mu)$? #Q

#mcq([The curve $L(mu)$ we just drew for $cal(D) = {0.5, 1.5, 2.0, 2.5, 3.5}$ — which statement is *true*?],
  [It is a probability density over $mu$, so its area must be 1],
  [$L(2.0)$ is the probability that $mu = 2.0$ is the true parameter],
  [$L(mu)$ is the joint density of the five *observed* points, re-read as a function of $mu$],
  [Its peak location depends on the order in which the five points were multiplied],
)

== Answer: what kind of object is $L(mu)$? #A

#mcq-answer([C], [the joint density of the observed data, as a function of $mu$],
  [The data is frozen; $mu$ varies. Nothing normalizes the curve over $mu$ (A), and no probability is ever assigned *to* $mu$ — that requires a prior and Bayes' rule, which is L15's business (B). Multiplication is commutative, so order is irrelevant (D).])

= Log-likelihood: products become sums

== Products of probabilities are numerically doomed

You proved this in L2 — the recap: a language model scores a 1000-character text, each character with probability $approx 0.03$:

$ P("text") = product_(i=1)^1000 p_i approx 0.03^1000 approx 10^(-1523) arrow.r.double_("even float64") bold(0.0) quad #text(fill: RED, size: 17pt)[argmax of all-zeros = garbage] $

#pause
Our five-point likelihood already sat at $10^(-4)$. In log-space, tiny becomes ordinary:

$ log P("text") = sum_(i=1)^1000 log p_i approx 1000 times (-3.51) = -3507 quad #text(fill: GREEN)[✓ a boring, safe float] $

#pause
#notebox[L2's exact words: "Every loss you will ever meet is a _log_-likelihood — floating point demands it (*L14 shows the deeper reason*)." Today is that day — and underflow is only reason 1 of 3.]

== Reason 2 · the log moves the peak nowhere #V

$log$ is *strictly increasing*: $a > b > 0 arrow.r.double log a > log b$. So it preserves every comparison of likelihoods — including where the maximum sits:

#two(r: (1fr, 1fr),
  [
    #align(center, text(size: 15pt, fill: ACC, weight: 700)[LIKELIHOOD $L(mu)$])
    #align(center, lines(
      fn: m => g-lik(m), domain: (0.0, 4.0), samples: 120, markers: false, colors: (ACC,),
      vlines: ((2.0, [$2.0$], ACC),), y-ticks: false, x-label: [$mu$], size: (56mm, 30mm),
    ))
  ],
  [
    #align(center, text(size: 15pt, fill: TEAL, weight: 700)[LOG-LIKELIHOOD $ell(mu)$])
    #align(center, lines(
      fn: m => g-ll(m), domain: (0.0, 4.0), samples: 120, markers: false, colors: (TEAL,),
      vlines: ((2.0, [$2.0$], TEAL),), y-ticks: false, x-label: [$mu$], size: (56mm, 30mm),
    ))
  ],
)

#pause
$ arg max_theta L(theta) = arg max_theta log L(theta) quad quad #text(fill: MUTED, size: 16pt)[— same peak, tamer numbers] $

== Reason 3 · sums are what calculus (and autodiff) want

$ ell(theta) := log L(theta) = log product_(i=1)^n p(x_i | theta) = sum_(i=1)^n log p(x_i | theta) $

#pause
To maximize, we differentiate (L7–L8) and set to zero. Compare the two forms:

- $dif/(dif theta) product_i p_i$: the product rule over $n$ factors — $n$ terms, each a product of $n - 1$ densities
#pause
- $dif/(dif theta) sum_i log p_i = sum_i (dif/(dif theta)) log p_i$: differentiate each term *alone*, add

#pause
#result[Three independent wins — no underflow (L2), same argmax, per-point derivatives. This is why *every* training objective you will ever see is built from $log p$.]

== Flip the sign, and you have invented "the loss"

Optimizers by convention *minimize* (Module 4 is built that way). So negate:

$ "NLL"(theta) = -ell(theta) = -sum_(i=1)^n log p(x_i | theta) quad quad "and often the mean" quad 1/n sum_i -log p(x_i | theta) $

#pause
- maximize likelihood #h(6pt) $arrow.l.r.double$ #h(6pt) maximize log-likelihood #h(6pt) $arrow.l.r.double$ #h(6pt) *minimize NLL*
#pause
- each data point contributes $-log p(x_i | theta)$: *its personal penalty* — large when the model found it implausible

#pause
#result[A "loss" is not a design flourish. Training loss = per-example negative log-likelihood, averaged. This sentence is the hinge of the whole course.]

= The coin: your first MLE, derived ⭐

== The model, and a clever way to write it

Flip a coin 10 times: $H, H, T, H, H, H, T, H, H, T$ — *7 heads, 3 tails*. Model: each flip is Bernoulli, $P(H) = theta$, IID. Code heads as $x = 1$, tails as $x = 0$:

$ p(x | theta) = theta^x (1 - theta)^(1 - x) quad quad #text(fill: MUTED, size: 16pt)[one formula, both cases: $x = 1 arrow.r theta$; $x = 0 arrow.r 1 - theta$] $

#pause
The dataset's likelihood — products of $theta$'s and $(1-theta)$'s, one per flip:

$ L(theta) = product_(i=1)^10 theta^(x_i) (1 - theta)^(1 - x_i) = theta^7 (1 - theta)^3 $

#pause
#notebox[Only the *counts* survive: 7 heads, 3 tails. The order of flips never mattered — IID told us so before we computed anything.]

== The two curves, computed #V

#two(r: (1fr, 1fr),
  [
    #align(center, text(size: 15pt, fill: ACC, weight: 700)[$L(theta) = theta^7 (1 - theta)^3$])
    #align(center, lines(
      fn: t => coin-lik(t), domain: (0.001, 0.999), samples: 140, markers: false, colors: (ACC,),
      fill-under: (0, ACC.transparentize(86%)),
      vlines: ((0.7, [$0.7$], ACC),),
      y-ticks: false, x-label: [$theta$], size: (56mm, 31mm),
    ))
  ],
  [
    #pause
    #align(center, text(size: 15pt, fill: TEAL, weight: 700)[$ell(theta) = 7 log theta + 3 log(1 - theta)$])
    #align(center, lines(
      fn: t => coin-ll(t), domain: (0.15, 0.99), samples: 140, markers: false, colors: (TEAL,),
      vlines: ((0.7, [$0.7$], TEAL),),
      y-ticks: false, x-label: [$theta$], size: (56mm, 31mm),
    ))
  ],
)

#pause
Both peak at $theta = 0.7$. Spot values: $L(0.5) = 0.5^10 approx #sci(coin-lik(0.5))$, $L(0.7) approx #sci(coin-lik(0.7))$ — the data is #calc.round(coin-lik(0.7) / coin-lik(0.5), digits: 1)× more plausible under $theta = 0.7$ than under a fair coin.

== ⭐ Derivation · set the derivative to zero #D

Maximize $ell(theta) = 7 log theta + 3 log (1 - theta)$ the L7 way — stationary point:

#pause
$ (dif ell)/(dif theta) = 7/theta - 3/(1 - theta) = 0 $

#pause
$ 7 (1 - theta) = 3 theta quad arrow.r.double quad 7 = 10 theta quad arrow.r.double quad hat(theta) = 7/10 = 0.7 $

#pause
Is it a maximum? Check curvature (L7's second-derivative test):

$ (dif^2 ell)/(dif theta^2) = -7/theta^2 - 3/(1 - theta)^2 < 0 quad "everywhere on" (0, 1) $

#pause
Strictly concave → the single stationary point is the *global* maximum. No luck involved.

== ⭐ Derivation · now for any dataset #D

General case: $n$ flips, $n_H$ heads, $n_T = n - n_H$ tails. Same three moves:

$ ell(theta) = sum_(i=1)^n [x_i log theta + (1 - x_i) log(1 - theta)] = n_H log theta + n_T log(1 - theta) $

#pause
$ (dif ell)/(dif theta) = n_H/theta - n_T/(1 - theta) = 0 quad arrow.r.double quad n_H (1 - theta) = n_T theta $

#pause
$ n_H = (n_H + n_T) theta $

#pause
#result[$hat(theta)_"MLE" = n_H / n$ — the fraction of heads.]

#pause
The answer your intuition would have guessed — but now it is a *theorem*, from a principle that will still work when intuition has nothing to say (a billion-parameter model has no "fraction of heads").

== Sanity checks, one honest and one alarming

The machine agrees — brute-force the curve:

#codebox[```python
heads, n = 7, 10
theta  = np.linspace(0.01, 0.99, 981)
loglik = heads * np.log(theta) + (n - heads) * np.log(1 - theta)
theta[np.argmax(loglik)]                    # 0.7  — the head fraction
```]

#pause
#alertbox[Now flip a coin *3 times*, get *3 heads*: $hat(theta)_"MLE" = 3\/3 = 1.0$ — this coin *cannot ever* land tails. Three flips convinced MLE of a certainty. Your intuition says "probably just a normal coin". Your intuition is using a *prior* — and giving MLE one is exactly L15.]

== Interactive · climb the coin's likelihood yourself #I

#interbox(link-to: IA + "mle-map-coin")[
  The *mle-map-coin* article flips coins live: the likelihood curve rebuilds after every flip, and you drag $theta$ to feel the score change under your cursor. Recreate today's 7-of-10, then try 3-of-3 and watch the peak slam into the boundary at $theta = 1$. (Ignore the prior slider for now — after L15 it will be the whole point.)
]

#pause
#notebox[Two-minute experiment before next lecture: keep the head *fraction* at 70% but grow $n$: 7/10 → 70/100 → 700/1000. Watch what happens to the curve's *width* — we name that phenomenon at the end of today.]

= MLE for the Gaussian

== The Gaussian log-likelihood, simplified once and for all

Model $cal(D) = {x_1, dots, x_n}$ as IID $cal(N)(mu, sigma^2)$ (L12's density, L13's star):

$ ell(mu, sigma) = sum_(i=1)^n log [ 1/sqrt(2 pi sigma^2) exp(-(x_i - mu)^2/(2 sigma^2)) ] $

#pause
$log$ of (product of exp) — everything unfolds:

$ ell(mu, sigma) = -n/2 log(2 pi) - n log sigma - 1/(2 sigma^2) sum_(i=1)^n (x_i - mu)^2 $

#pause
#result[All the $mu$-dependence sits in one term: $-sum_i (x_i - mu)^2$. Maximizing $ell$ over $mu$ *is* minimizing a sum of squares. Remember this shape — it returns in ten minutes as MSE.]

== The mean, earned

Differentiate w.r.t. $mu$ (the constants and $-n log sigma$ vanish):

$ (partial ell)/(partial mu) = 1/sigma^2 sum_(i=1)^n (x_i - mu) = 0 quad arrow.r.double quad sum_i x_i = n mu $

#pause
$ hat(mu)_"MLE" = 1/n sum_(i=1)^n x_i = macron(x) $

#pause
On our running dataset: $hat(mu) = (0.5 + 1.5 + 2.0 + 2.5 + 3.5)\/5 = 2.0$ — *exactly where the likelihood curve peaked* in Section 3. The picture and the calculus agree.

#pause
#notebox[Note what $hat(mu)$ does *not* depend on: $sigma$. However noisy the ruler, the best center is the average. (With $sigma$ unknown, the $mu$-equation is unchanged.)]

== The variance — and an honest footnote

Same recipe for $sigma$: differentiate $ell = dots - n log sigma - 1/(2 sigma^2) sum_i (x_i - mu)^2$, set to zero, solve:

$ (partial ell)/(partial sigma) = -n/sigma + 1/sigma^3 sum_i (x_i - hat(mu))^2 = 0 quad arrow.r.double quad hat(sigma)^2_"MLE" = 1/n sum_(i=1)^n (x_i - macron(x))^2 $

#pause
On $cal(D)$: deviations $(-1.5, -0.5, 0, 0.5, 1.5)$, squares sum to $5.0$, so $hat(sigma)^2 = 5.0\/5 = 1.0$.

#pause
#notebox[*Honest footnote.* This estimator is *biased*: on average across datasets, $EE[hat(sigma)^2] = (n-1)/n dot sigma^2$ — systematically a bit small, because $macron(x)$ chased the sample before we measured spread around it. NumPy's `np.var(x, ddof=1)` applies the $n/(n-1)$ repair (Bessel's correction). We state this; a statistics course proves it. MLE is *excellent*, not sacred.]

== Estimation is optimization · the NLL landscape #V

Free both parameters: the *dataset NLL* over the $(mu, sigma)$ plane, computed live from $cal(D)$ —

#align(center, contour(
  (m, s) => g-nll2(m, s),
  xlim: (0.5, 3.5), ylim: (0.55, 2.1), samples: 52,
  levels: (7.3, 7.8, 8.5, 9.5, 11, 13, 16, 20, 26),
  size: (96mm, 36mm), color: TEAL,
  paths: (gd(grad2d((m, s) => g-nll2(m, s)), (0.7, 1.85), lr: 0.05, steps: 60),),
  marks: ((2.0, 1.0, [$(hat(mu), hat(sigma)) = (2, 1)$], RED),),
  x-label: [$mu$], y-label: [$sigma$],
))

#pause
- The bowl bottoms out at $(hat(mu), hat(sigma)) = (2.0, 1.0)$ — both closed forms at once
#pause
- The green path: *gradient descent* walking downhill on this exact surface

#pause
#result[Fitting = minimizing NLL. Module 4 (L16–L21) is the art of *walking downhill* on such surfaces.]

= The keystone: linear regression → MSE

== A model that finally has an input #V

Everything so far fit a *constant* distribution. Real problems predict $y$ *from* $x$ — rent from area, temperature from hour. The simplest such model is a line:

#fig("/lecture14/figures/linreg_residuals.svg", w: 54%)

#pause
$hat(y) = w x + b$; the *residual* $r_i = y_i - (w x_i + b)$ is what the line cannot explain. Which line is *best*? We need a likelihood — so we need a *distribution over $y$*.

== The noise model: a bell standing on the line #V

Assume the world draws $y$ as *the line plus Gaussian noise*:

$ y_i = w x_i + b + epsilon_i, quad epsilon_i tilde cal(N)(0, sigma^2) "IID" quad arrow.r.double quad y_i | x_i tilde cal(N)(w x_i + b, #h(2pt) sigma^2) $

#fig("/lecture14/figures/mse_keystone_bells.svg", w: 52%)

#pause
A vertical Gaussian stands *on the line* at every $x$: near the line, high density; stragglers, expensive. That is the whole assumption — now we turn the crank.

== The likelihood of a *labeled* dataset

New situation, same machinery. The data comes in pairs $(x_i, y_i)$, and our model only says how $y$ arises *given* $x$ — it never models the inputs:

$ L(theta) = product_(i=1)^n p(y_i | x_i, theta) quad quad #text(fill: MUTED, size: 16pt)[$theta = (w, b)$; the $x_i$ enter as fixed givens] $

#pause
- Same flip: data $(x_i, y_i)$ frozen, parameters $(w, b)$ varying
#pause
- Same IID product — independence *of the noise* $epsilon_i$, one shared line
#pause
- No $p(x)$ anywhere: whoever chose the inputs is not our problem

#pause
#notebox[This conditional form is the shape of *every* supervised objective — including the trillion-token ones: L26's language model maximizes exactly such a product, one conditional per next character.]

== ⭐ Derivation · one point's log-likelihood #D

The density of observing label $y_i$ at input $x_i$, given parameters $theta = (w, b)$:

$ p(y_i | x_i, theta) = 1/sqrt(2 pi sigma^2) exp(-(y_i - (w x_i + b))^2/(2 sigma^2)) $

#pause
Take the log — the same unfold as the Gaussian section:

$ log p(y_i | x_i, theta) = -(y_i - (w x_i + b))^2/(2 sigma^2) - log sigma - 1/2 log(2 pi) $

#pause
$ -log p(y_i | x_i, theta) = underbrace((y_i - hat(y)_i)^2, "squared residual!")/(2 sigma^2) + underbrace(log sigma + 1/2 log(2 pi), "no" #h(3pt) theta #h(3pt) "inside — constant") $

#pause
One data point's NLL *is* its squared residual, rescaled and shifted by constants.

== ⭐ Derivation · sum, strip constants, done #D

Sum over the IID dataset and maximize over $theta = (w, b)$, with $sigma$ fixed:

$ hat(theta)_"MLE" = arg max_theta ell(theta) = arg min_theta sum_(i=1)^n [ (y_i - hat(y)_i)^2/(2 sigma^2) + log sigma + 1/2 log(2 pi) ] $

#pause
Adding a constant never moves an argmin — drop $n log sigma + n/2 log(2 pi)$. Scaling by a positive constant never moves an argmin — drop $1\/(2 sigma^2)$:

#pause
#result[$hat(theta)_"MLE" = arg min_(w, b) display(sum_(i=1)^n) (y_i - w x_i - b)^2$ — Gaussian MLE *is* least squares. \ Nobody "invented" MSE: it was hiding inside $cal(N)$ all along.]

== What MSE silently assumes

Run the derivation backwards: using MSE *is* declaring a noise model. Three claims, made every time you call `mse_loss`:

+ noise is *Gaussian* — bell-shaped errors around the prediction
#pause
+ noise has *the same $sigma$ everywhere* — no input is harder than another
#pause
+ observations are *independent* given the inputs

#pause
#alertbox[Assumption 1 is how outliers wreck least squares: a point 5σ off the line pays $(5 sigma)^2\/2 sigma^2 = 12.5$ nats — the Gaussian is *certain* such points are freakish, so the fitted line contorts to appease them. Heavier-tailed noise models forgive them — that's the MAE row coming up.]

== The machine agrees · one grid, two objectives

#codebox[```python
x  = np.array([0., 1., 2., 3., 4.]); y = np.array([1.1, 1.9, 2.8, 4.3, 4.9])
ws = np.linspace(0.2, 1.8, 801)                 # candidate slopes (b = 1 fixed)
mse = [np.mean((y - w*x - 1.0)**2) for w in ws]
ll  = [np.sum(-0.5*np.log(2*np.pi) - 0.5*(y - w*x - 1.0)**2) for w in ws]
print(ws[np.argmin(mse)], ws[np.argmax(ll)])    # 1.0 1.0  — the same slope
```]

#pause
Two objectives, one winner — on every dataset, always, by the derivation you just did.

#pause
#notebox[This is why earlier stats courses could teach least squares without ever saying "likelihood": with $sigma$ fixed, the two problems are *literally the same problem*. You now know which one is fundamental.]

= Every loss is an NLL

== The table this course orbits around

Choose a noise/observation model; the NLL *is* your loss:

#align(center, table(
  columns: (auto, auto, 1fr, auto), stroke: 0.5pt + MUTED.lighten(40%),
  inset: (x: 10pt, y: 7pt), align: (left, left, left, center),
  table.header([*You observe*], [*Model*], [*NLL per example*], [*Its famous name*]),
  [a real $y$], [Gaussian], [$(y - hat(y))^2 \/ 2 sigma^2 + c$], [*MSE* #text(fill: GREEN, size: 16pt)[✓ today]],
  [yes / no], [Bernoulli], [$-[y log p + (1 - y) log(1 - p)]$], [*BCE* #text(fill: GREEN, size: 16pt)[✓ today]],
  [1 of $K$ classes], [Categorical], [$-log p_"true class"$], [*cross-entropy* #text(fill: ACC, size: 16pt)[→ L24]],
  [$y$ with outliers], [Laplace], [$abs(y - hat(y)) \/ b + c$], [*MAE* #text(fill: MUTED, size: 16pt)[bonus]],
))

#pause
#result[Loss functions are not a menu of arbitrary formulas. Each row is *the same construction* — only the noise model changes.]

== The BCE row costs one line

For a yes/no label $y in {0, 1}$ with predicted head-probability $p$, the Bernoulli pmf in exponent form (from the coin section!) gives the per-example NLL:

$ -log [ p^y (1 - p)^(1 - y) ] = -[ y log p + (1 - y) log(1 - p) ] $

#pause
This is `binary_cross_entropy`, derived directly from the Bernoulli log-likelihood.

#pause
- $y = 1$, model said $p = 0.9$: penalty $-log 0.9 approx 0.105$ — cheap
#pause
- $y = 1$, model said $p = 0.01$: penalty $-log 0.01 approx 4.6$ — confident wrongness is *expensive*, by the shape of $-log$

#pause
#notebox[The categorical / cross-entropy row is the same move over $K$ classes — L24 does it properly and connects it to information theory (why "entropy" is in the name), and L2's stable log-softmax is exactly how it's computed.]

== Two noise models, two loss shapes #V

The *true* per-residual loss curves, computed from the densities themselves ($-log p$, shifted to 0 at $r = 0$):

#align(center, text(size: 14pt)[
  #sw(ACC, [Gaussian → $r^2\/2$ (MSE-shaped)]) #h(16pt)
  #sw(TEAL, [Laplace → $abs(r)$ (MAE-shaped)])
])
#v(2pt)
#align(center, lines(
  fn: (r => dist.nll0(dist.normal(mu: 0.0, sigma: 1.0), r), r => dist.nll0(dist.laplace(mu: 0.0, b: 1.0), r)),
  domain: (-3, 3), samples: 130, markers: false, colors: (ACC, TEAL),
  x-label: [residual $r$], y-label: [$-log p(r)$],
  size: (96mm, 38mm),
))

#pause
The parabola *accelerates*: doubling a large residual quadruples its cost — outliers steer the fit. The Laplace's thin tails in log-space charge a flat rate — outliers get a shrug. Choosing a loss *is* choosing how much you believe in outliers.

== Match each common loss to an observation model

#align(center, table(
  columns: (auto, 1fr, auto), stroke: 0.5pt + MUTED.lighten(40%),
  inset: (x: 12pt, y: 8pt), align: (left, left, center),
  table.header([*Loss*], [*Where it comes from*], [*Status*]),
  [`mse_loss`], [Gaussian noise on the prediction — derived, boxed, yours], [#text(fill: GREEN)[*closed*]],
  [`binary_cross_entropy`], [Bernoulli NLL — the coin, renamed], [#text(fill: GREEN)[*closed*]],
  [`cross_entropy`], [Categorical NLL + information theory: *why it's called entropy*, what it has to do with compression], [#text(fill: ACC)[*open → L24*]],
))

#pause
#notebox[L2 told you `CrossEntropyLoss` fuses log-softmax for stability and takes raw logits. Today you learned why it's a *loss* at all. L24 explains why it's called *entropy*. Three lectures, one function — that's how central it is.]

== Properties, in brief · more data sharpens the peak #V

Hold the head fraction at 70%, grow $n$ — the peak-aligned log-likelihood sharpens:

#align(center, text(size: 14pt)[
  #sw(MUTED, [$n = 10$]) #h(14pt) #sw(TEAL, [$n = 40$]) #h(14pt) #sw(ACC, [$n = 160$])
])
#v(2pt)
#align(center, lines(
  fn: (
    t => 10 * (0.7 * calc.ln(t) + 0.3 * calc.ln(1 - t) - (0.7 * calc.ln(0.7) + 0.3 * calc.ln(0.3))),
    t => 40 * (0.7 * calc.ln(t) + 0.3 * calc.ln(1 - t) - (0.7 * calc.ln(0.7) + 0.3 * calc.ln(0.3))),
    t => 160 * (0.7 * calc.ln(t) + 0.3 * calc.ln(1 - t) - (0.7 * calc.ln(0.7) + 0.3 * calc.ln(0.3))),
  ),
  domain: (0.45, 0.92), samples: 130, markers: false, colors: (MUTED, TEAL, ACC),
  vlines: ((0.7, [$theta = 0.7$], INK),),
  y-ticks: false, x-label: [$theta$], y-label: [$ell(theta) - ell(0.7)$],
  size: (94mm, 30mm),
))

#pause
- *Consistency* (stated): as $n arrow.r infinity$, $hat(theta)_"MLE" arrow.r$ the true $theta$ — alternatives get buried
#pause
- Small $n$ misbehaves: biased $hat(sigma)^2$, certainty from 3 flips. Data cures MLE; priors (L15) cure small data.

== Checkpoint: choose the loss like a modeler #Q

#mcq([Your sensor's errors are usually small but show occasional huge spikes — well modeled by *Laplace* noise. Maximum likelihood hands you which training loss?],
  [MSE — squared error is always optimal],
  [MAE — the mean *absolute* error],
  [BCE — it has a log, logs handle spikes],
  [Cross-entropy over discretized bins],
)

== Answer: choose the loss like a modeler #A

#mcq-answer([B], [MAE — mean absolute error],
  [Laplace density $prop e^(-abs(r)\/b)$, so per-point NLL $= abs(r)\/b + c$: the absolute residual. The quadratic MSE would let each spike bully the fit ($r^2$ explodes); the linear penalty stays calm. Same construction as MSE — different noise belief. That is the table doing real work.])

== Practice problems

Try on paper; the T7 notebook (after L15) checks all of them in NumPy.

+ 12 flips, 9 heads: write $ell(theta)$, maximize it, and compute $ell(0.75) - ell(0.5)$.
+ For $cal(D) = {1.0, 3.0}$ under $cal(N)(mu, 1)$: sketch $L(mu)$; find $hat(mu)$, then $hat(sigma)^2$. (Answers: 2.0 and 1.0.)
+ Exponential waiting times $p(x | lambda) = lambda e^(-lambda x)$ (L12's zoo): derive $hat(lambda)_"MLE" = 1 \/ macron(x)$.
+ Redo the keystone with *Laplace* noise and show the objective becomes $sum_i abs(y_i - w x_i - b)$.
+ For $n = 10^6$, $p_i approx 0.1$: estimate $product_i p_i$ and compare with float64's smallest positive number ($approx 10^(-324)$, L2). Why must training code never form this product?
+ True or false, one sentence: "the MLE of $theta$ is the most probable value of $theta$ given the data." Explain the distinction introduced in L15.

== Lecture 14 — summary

- *The flip*: freeze the data, vary $theta$ — the likelihood is a score, not a distribution over $theta$.
- *Datasets*: IID → a product of per-point densities; slide the model, climb the score.
- *Logs*: same argmax, no underflow (L2's $0.03^1000$, cashed), per-point derivatives. NLL = *the loss*.
- *Coin ⭐*: $hat(theta) = n_H\/n$ from $ell'(theta) = 0$; 3-for-3 heads exposes MLE's blind spot (→ L15).
- *Gaussian*: $hat(mu) = macron(x)$; $hat(sigma)^2 = 1/n sum (x_i - macron(x))^2$ — biased, honestly footnoted.
- *Keystone ⭐*: Gaussian noise on a line → max likelihood $equiv$ min squared error.
- *The table*: Gaussian→MSE, Bernoulli→BCE, Categorical→CE (L24), Laplace→MAE.

#pause
#notebox[*Read before L15* — MML §8.1–8.3 (skim 8.1–8.2, focus 8.3) · MacKay Ch. 2. *T7* (after L15) drills MLE + MAP together.]

== ⭐⭐⭐ Where MLE points as $n arrow.r infinity$ #OPT

Divide the log-likelihood by $n$: an *average* that (by large numbers) approaches an expectation over the true data source $p^star$:

$ 1/n ell(theta) = 1/n sum_(i=1)^n log p(x_i | theta) quad arrow.r_(n arrow.r infinity) quad EE_(x tilde p^star)[log p(x | theta)] $

#pause
Add and subtract the entropy of $p^star$ (L22 defines it properly):

$ EE_(p^star)[log p(x | theta)] = underbrace(EE_(p^star)[log p^star (x)], "fixed: no" #h(3pt) theta) - underbrace("KL"(p^star #h(3pt) || #h(3pt) p_theta), ">= 0," #h(3pt) "gap to reality") $

#pause
#result[Maximizing likelihood is equivalent to minimizing $"KL"(p^star || p_theta)$ up to a constant independent of $theta$. L24 derives this identity from cross-entropy.]

#focus-slide[
  Every loss function is a negative log-likelihood in disguise.
  #v(12pt)
  #set text(size: 22pt)
  Next: *MAP & Conjugate Priors* — what if you have beliefs _before_ seeing data?
]
