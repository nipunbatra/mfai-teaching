// Continuous Distributions — Lecture 12 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture12/L12-continuous-distributions.typ
//   typst compile --root . --input handout=true lecture12/L12-continuous-distributions.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Continuous Distributions],
  subtitle: [Probability is area under a curve],
)

#let DE = "https://distribution-explorer.github.io/"
#let ST = "https://seeing-theory.brown.edu/"

// ── Beta pdf for integer α, β ≥ 1 (normalizer via factorials — no Γ needed) ──
#let beta-pdf(a, b) = {
  let c = calc.fact(a + b - 1) / (calc.fact(a - 1) * calc.fact(b - 1))
  let pw(v, e) = if e == 0 { 1.0 } else { calc.pow(v, e) }   // 0⁰ = 1 for our purposes
  x => if x < 0 or x > 1 { 0.0 } else { c * pw(x, a - 1) * pw(1 - x, b - 1) }
}
// exponential CDF (chalkdust dist carries pdfs; the cdf is one line)
#let exp-cdf(rate) = x => if x < 0 { 0.0 } else { 1 - calc.exp(-rate * x) }

// ── motivating pipeline: noise → learned transform → sample ──
#let genpipe = align(center, diagram(spacing: (16mm, 8mm), {
  let N(p, top, bottom, c, bg) = node(p,
    align(center)[#text(size: 15pt, weight: 700, fill: INK)[#top] \ #text(size: 12.5pt, fill: MUTED)[#bottom]],
    shape: fletcher.shapes.rect, fill: bg, stroke: 1.0pt + c, corner-radius: 3pt, inset: 8pt)
  N((0, 0), [simple noise], [$z tilde$ something dumb], TEAL, TEAL.lighten(91%))
  N((1.6, 0), [a transformation $f$], [learned, expensive], INK, white)
  N((3.2, 0), [sample $x = f(z)$], [a face · a voice · a molecule], ACC, ACC.lighten(90%))
  edge((0, 0), (1.6, 0), "-|>", stroke: 0.9pt + INK)
  edge((1.6, 0), (3.2, 0), "-|>", stroke: 0.9pt + INK)
}))

#title-slide()

= Generating samples by transforming noise

== A common generative-model construction

Many generative models begin with a simple random variable and transform it into a sample:

#genpipe

#pause
In a simplified PyTorch example:

#codebox[```python
z = torch.randn(...)     # simple Gaussian noise
x = f(z)                 # a learned transformation does the rest
```]

#pause
Diffusion models, VAEs, normalizing flows — different learned $f$, *same recipe*: sample cheap noise, transform it.

== Two mathematical questions

+ The noise $z$ and the sample $x$ are *continuous* quantities. In ES 114, probability meant counting outcomes: $P(X = x)$. What does probability even *mean* when $x$ can be any real number?

#pause
+ The transformation $f$ moves samples around. What does it do to their *probabilities*?

#pause
#notebox[In applications, $f$ may be a high-dimensional neural network. Today we develop the one-dimensional density and transformation rules, then implement inverse-CDF sampling for a distribution with a closed-form inverse.]

== Today, in one line

#result[For continuous quantities, interval probabilities are areas under a density. An invertible differentiable transformation changes density by an inverse-Jacobian factor.]

#pause
The route:

+ *Density* — the object that replaces point masses
+ *CDF* — the running total that connects density to real probabilities
+ *Expectation & variance* — the old formulas, with $sum -> integral$
+ *The zoo* — uniform, exponential, Gaussian, Beta
+ *Change of variables* — the Jacobian factor, then inverse-CDF sampling from $cal(U)(0,1)$

== What ES 114 already gave you

Two dice, $Z = $ sum of the faces — the PMF you have drawn many times:

#align(center, bars(
  (1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36),
  labels: ("2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"),
  annotate: false, size: (152mm, 30mm),
))

#pause
Everything lived on this scaffolding: $p_Z (z) = P(Z = z)$ is a genuine probability, probabilities of events are *sums* of bar heights, $EE[Z] = sum z thin p_Z (z)$.

#pause
Today each operation gets its continuous counterpart.

== Learning outcomes

By the end of this lecture you will be able to:

+ Explain why $P(X = x) = 0$ for continuous $X$ — and why a *density* fixes it.
+ Read a pdf correctly: probability = *area*; densities may exceed 1.
+ Convert between *pdf and CDF* in both directions (integrate / differentiate).
+ Compute *expectations and variances* as integrals (uniform, exponential).
+ Recognize the zoo on sight: *uniform, exponential, Gaussian, Beta*.
+ Derive the 1-D *change-of-variables* rule $p_Y (y) = p_X (x) abs(dif x \/ dif y)$.
+ Prove *inverse-CDF sampling* correct and implement it in 5 lines of NumPy.

= Density: probability per unit length

== A question ES 114 cannot answer

ES 114's dataset of *25,000 measured heights*. Pick a person at random; let $H$ be their height.

#pause
Easy question: $P(66 <= H <= 70)$? Count the rows, divide by 25,000. Fine.

#pause
Different question: what is

$ P(H = italic("exactly") 68 "inches") = P(H = 68.000000000 dots) = thin ? $

#pause
Under a continuous model, probability is assigned to intervals. A singleton has zero width and hence zero probability, even though a realized measurement takes some value.

#pause
#alertbox[For a continuous quantity, $P(X = x) = 0$ at *every single* $x$. The PMF — our whole ES 114 toolkit — reads 0 everywhere. We need a new object.]

== Picture: squeeze the histogram until it becomes a curve #V

Same 25,000 heights, plotted as *density histograms* (bar area = fraction of data), with ever finer bins:

#fig("/lecture12/figures/l12_hist_density.svg", w: 96%)

#pause
The staircase stops changing shape and settles onto a smooth curve. *That limiting curve* — not any single bar — is the new object.

== The limit object: a probability density

#result[At points where the density is continuous, $ p(x) = lim_(delta -> 0^+) P(x <= X <= x + delta) / delta $ — probability *per unit length*, not probability.]

#pause
Read it like the speedometer of probability: for a tiny interval,

$ P(x <= X <= x + delta) approx p(x) dot delta quad quad "(height × width = a sliver of area)" $

#pause
- The units are real: our height curve peaks at $p approx 0.21$ *per inch*.
#pause
- Probability of a *point*: width $delta -> 0$, so $P(X = x) = 0$ — the paradox is now a feature, not a bug.

== Zero probability is not "impossible"

Uncomfortable but true: the dart *does* land somewhere, yet every exact landing point had probability 0.

#pause
- A probability-zero event need not be logically impossible. For a continuous variable, every singleton has probability zero even though the realized value belongs to one of them.
#pause
- Nothing is broken: probability was never promised to individual points, only to *sets with width*.

#pause
#notebox[This is the door to measure theory — which we will keep respectfully closed. MML §6.1 does the same; the pictures-and-areas view is all this course (and most of ML) needs.]

== A probability density can exceed one #V

$X tilde cal(U)(0, 1\/2)$ — all mass spread evenly on an interval of width $1\/2$:

#align(center, lines(
  fn: x => dist.pdf(dist.uniform(a: 0.0, b: 0.5), x),
  domain: (-0.25, 0.75), samples: 240, markers: false, colors: (INK,),
  fill-under: (from: 0.0, to: 0.5, color: TEAL.transparentize(65%)),
  hlines: ((1.0, [not a ceiling], RED),),
  x-label: $x$, y-label: $p(x)$, size: (100mm, 40mm),
))

#pause
The height is $p(x) = 2$ — *bigger than 1* — yet nothing is wrong: area $= 2 times 1\/2 = 1$. ✓

#pause
#result[Densities are not probabilities. $p(x) > 1$ is legal, common, and means only: *mass is concentrated*.]

== Density values and interval probabilities

A second offender: $cal(N)(0, sigma = 0.1)$, a Gaussian squeezed tight —

#align(center, lines(
  fn: x => dist.pdf(dist.normal(mu: 0.0, sigma: 0.1), x),
  domain: (-0.6, 0.6), samples: 160, markers: false, colors: (TEAL,),
  hlines: ((1.0, [1], RED),),
  x-label: $x$, y-label: $p(x)$, size: (92mm, 36mm),
))

#pause
Peak value $1\/(0.1 sqrt(2 pi)) approx 3.99$. Shrink $sigma$ and the peak grows *without bound* — the area stays exactly 1.

#pause
#notebox[Spot-the-bug skill for later courses: a "probability" of $2.7$ in your log output is *fine* if it is a density (`log_prob` of a narrow Gaussian, say) — and a catastrophe if it is supposed to be a probability. Know which one you are holding.]

== Numbers: three questions on $cal(U)(0, 1\/2)$

Height 2 on $[0, 1\/2]$, zero elsewhere. Probability = area = height × width:

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Question*], [*Area computation*], [*Answer*]),
  [$P(0.2 <= X <= 0.3)$], [$2 times 0.1$], [$0.2$],
  [$P(X = 0.25)$], [$2 times 0$], [$0$],
  [$P(0 <= X <= 1)$], [$2 times 0.5 + 0 times 0.5$], [$1$],
)

#pause
Row 2 in words: the tallest density in the world still assigns zero to a *point* — width is everything.

#pause
Row 3 in words: integrating past the support costs nothing — the density is 0 out there.

== Probability is area, officially

The ES 114 axioms, translated for a density $p$:

$ p(x) >= 0 quad "everywhere", quad quad integral_(-oo)^(oo) p(x) dif x = 1, quad quad P(a <= X <= b) = integral_a^b p(x) dif x $

#pause
#align(center, lines(
  fn: x => dist.pdf(dist.normal(), x),
  domain: (-4, 4), samples: 160, markers: false, colors: (INK,),
  fill-under: (from: -1.0, to: 1.0, color: ACC.transparentize(55%)),
  vlines: ((-1.0, [$a$], MUTED), (1.0, [$b$], MUTED)),
  x-label: $x$, y-label: $p(x)$, size: (96mm, 38mm),
))

#pause
Nonnegative curve, total area 1, questions answered by shading. That's the whole contract.

== Code: a density is just a function you can call

#codebox[```python
from scipy.stats import uniform, norm

u = uniform(loc=0, scale=0.5)        # U(0, 1/2)
u.pdf(0.25)                          # 2.0  ← density above 1; no alarm rings
u.cdf(0.3) - u.cdf(0.2)              # 0.2  ← an actual probability

norm(0, 0.1).pdf(0.0)                # 3.989 — the squeezed Gaussian's peak
```]

#pause
- `.pdf` returns *heights* — compare, plot, take logs, but never call them probabilities.
#pause
- Real probabilities come from `.cdf` differences — areas, exactly as the axioms demand. (The CDF is next.)

== Checkpoint: density vs probability #Q

#mcq([$X tilde cal(U)(0, 1\/2)$, so $p(x) = 2$ on $[0, 1\/2]$. Which statement is *true*?],
  [$P(X = 1\/4) = 2$ — the density at that point],
  [$p(1\/4) = 2$ *and* $P(X = 1\/4) = 0$ — no contradiction],
  [A valid pdf can never exceed 1, so $cal(U)(0, 1\/2)$ is not a real distribution],
  [$P(0 <= X <= 1\/2) = 2$, because height 2 covers the whole support],
)

== Answer: density vs probability #A

#mcq-answer("B", [$p(1\/4) = 2$ and $P(X = 1\/4) = 0$ coexist happily],
  [The density is probability *per unit length*: multiply by a width to get probability. A point has width 0, so $P(X = 1\/4) = 2 times 0 = 0$, while the height happily sits at 2. (A is heights-as-probabilities, C is the ceiling myth, and D would make total probability 2 — areas, never heights.)])

= The CDF: probability's running total

== One function sidesteps the whole paradox

$ F(x) = P(X <= x) quad quad "— the *cumulative distribution function*" $

#pause
- $F(x)$ is a *genuine probability* for every $x$ — the event $X <= x$ has width, no paradox.
#pause
- It climbs from $0$ (far left) to $1$ (far right), never decreasing — a running total of mass.
#pause
- And it is *one definition for every random variable* — discrete, continuous, or mixed. The PMF/pdf split disappears at the CDF level.

#pause
#notebox[ES 114 flashback: you met CDFs as staircases — flat, then a jump of height $p_Z (z)$ at each outcome. Today's CDFs are smooth ramps; the jumps have melted.]

== pdf and CDF, side by side #V

$"Exp"(1)$: density $p(x) = e^(-x)$, running total $F(x) = 1 - e^(-x)$.

#two(
  align(center, lines(
    fn: x => dist.pdf(dist.exponential(rate: 1.0), x),
    domain: (0, 5), samples: 140, markers: false, colors: (INK,),
    fill-under: (from: 0.0, to: 1.2, color: TEAL.transparentize(60%)),
    vlines: ((1.2, [$x = 1.2$], MUTED),),
    x-label: $x$, y-label: $p(x)$, size: (70mm, 38mm),
  )),
  align(center, lines(
    fn: exp-cdf(1.0),
    domain: (0, 5), samples: 140, markers: false, colors: (ACC,),
    points: ((1.2, 0.699, [$F(1.2) = 0.70$]),),
    x-label: $x$, y-label: $F(x)$, size: (70mm, 38mm),
  )),
)

#pause
The shaded area on the left *is* the marked height on the right. Same information, two displays: $p$ shows *where* mass sits; $F$ shows *how much* has accumulated.

== Direction 1: integrate the density → CDF

The running total is the area collected so far:

$ F(x) = integral_(-oo)^x p(t) dif t $

#pause
Worked on $"Exp"(1)$, every step visible:

$ F(x) = integral_0^x e^(-t) dif t = [-e^(-t)]_0^x = (-e^(-x)) - (-e^0) = 1 - e^(-x) $

#pause
Numbers: $F(1.2) = 1 - e^(-1.2) = 1 - 0.301 = 0.699$ — the 0.70 read off the picture. ✓

== Direction 2: differentiate the CDF → density

By the fundamental theorem of calculus, the derivative of accumulated probability is the density:

$ F'(x) = p(x) $

#pause
Round trip on $"Exp"(1)$:

$ dif / (dif x) (1 - e^(-x)) = e^(-x) = p(x) quad ✓ $

#pause
#result[Integrating a pdf gives its CDF; at points where the CDF is differentiable, $F'(x) = p(x)$.]

#pause
Equivalently, a density is a function whose integral reproduces the CDF.

== Every interval question routes through $F$

$ P(a < X <= b) = F(b) - F(a) quad quad "(total so far at " b ", minus total so far at " a ")" $

#pause
Worked on $"Exp"(1)$ — the chance the wait lands between 1 and 2:

$ F(2) - F(1) = (1 - e^(-2)) - (1 - e^(-1)) = e^(-1) - e^(-2) = 0.368 - 0.135 = 0.233 $

#pause
No new integral was computed — the CDF pre-paid it. This is why libraries ship `.cdf` and let you subtract.

#pause
#notebox[Whether you write $<$ or $<=$ changes nothing here: the endpoints carry probability 0. (For discrete variables it *does* matter — the staircase jumps.)]

== Code: `.cdf` answers real questions

Marks modeled as $cal(N)(70, 8^2)$ — the toy model from ES 114's notebook:

#codebox[```python
from scipy.stats import norm
marks = norm(70, 8)

marks.cdf(60)                    # 0.1056 → ~11% score below 60
marks.cdf(80) - marks.cdf(60)    # 0.7887 → ~79% land in [60, 80]
1 - marks.cdf(90)                # 0.0062 → the A+ tail
```]

#pause
Three managerial questions, three CDF calls, zero integrals by hand — the Gaussian's CDF has no closed formula, so *everyone* routes through the library.

= Expectation & variance, as integrals

== The dictionary move: $sum$ becomes $integral$

ES 114: $EE[X] = sum_x x thin p_X (x)$. Continuous twin: chances become slivers $p(x) dif x$ —

$ EE[X] = integral_(-oo)^(oo) x thin p(x) dif x quad quad quad EE[g(X)] = integral_(-oo)^(oo) g(x) thin p(x) dif x $

#pause
Same picture as ever: $EE[X]$ is the *balance point* of the density cutout:

#align(center, lines(
  fn: x => dist.pdf(dist.exponential(rate: 1.0), x),
  domain: (0, 5), samples: 140, markers: false, colors: (INK,),
  fill-under: (0, TEAL.transparentize(80%)),
  vlines: ((1.0, [$EE[X] = 1$], ACC),),
  x-label: $x$, y-label: $p(x)$, size: (80mm, 29mm),
))

#pause
The mean is not the mode: the mode is at $0$, while the long right tail moves the mean to $1$.

== Worked: the uniform's balance point

$X tilde cal(U)(a, b)$, so $p(x) = 1\/(b-a)$ on $[a, b]$:

$ EE[X] = integral_a^b x dot 1/(b-a) dif x = 1/(b-a) dot [x^2/2]_a^b = (b^2 - a^2)/(2(b-a)) $

#pause
Factor the difference of squares:

$ EE[X] = ((b-a)(b+a))/(2(b-a)) = (a + b)/2 $

#pause
The midpoint — exactly what symmetry promised. The integral just confirms the picture.

#pause
For $cal(U)(0, 1)$: $EE[X] = 1\/2$.

== Worked: the exponential waits $1\/lambda$

$X tilde "Exp"(lambda)$, $p(x) = lambda e^(-lambda x)$ on $[0, oo)$. Integrate by parts ($u = x$, $dif v = lambda e^(-lambda x) dif x$):

$ EE[X] = integral_0^oo x thin lambda e^(-lambda x) dif x = underbrace([-x e^(-lambda x)]_0^oo, = thin 0) + integral_0^oo e^(-lambda x) dif x $

#pause
$ EE[X] = [-(1/lambda) e^(-lambda x)]_0^oo = 1/lambda $

#pause
Sanity: $lambda$ is a *rate* (events per unit time), so the mean wait is $1\/lambda$ time units — buses arriving at rate 2 per hour are a half-hour wait on average. Units cooperate; always check they do.

== Variance is an integral too

Spread = expected squared distance from the balance point:

$ "Var"(X) = EE[(X - mu)^2] = integral (x - mu)^2 thin p(x) dif x = EE[X^2] - mu^2 $

#pause
Worked on $cal(U)(0, 1)$ via the shortcut:

$ EE[X^2] = integral_0^1 x^2 dif x = 1/3, quad quad "Var"(X) = 1/3 - (1/2)^2 = 1/12 approx 0.083 $

#pause
And the exponential: $"Var"(X) = 1\/lambda^2$ (same integration-by-parts trick twice — practice problem 3).

#pause
#notebox[Old friends survive unchanged: $EE[a X + b] = a EE[X] + b$ and $"Var"(a X + b) = a^2 "Var"(X)$ — linearity never cared whether the sum was $sum$ or $integral$.]

== Code: Monte Carlo audits our integrals

Can't integrate? *Sample and average* — the mean of draws approximates $EE[X]$:

#codebox[```python
rng = np.random.default_rng(0)

x = rng.uniform(0, 1, 1_000_000)
x.mean(), x.var()          # (0.50002, 0.08333)   = 1/2, 1/12  ✓

t = rng.exponential(1.0, 1_000_000)
t.mean(), t.var()          # (0.99950, 0.99852)   = 1/λ, 1/λ²  ✓
```]

#pause
Both hand-derived integrals confirmed to three digits by brute randomness.

#pause
#result[Sampling is an integral evaluator. Half of modern ML lives on this one sentence — and it needs a *sampler*, which the last section builds.]

= The zoo: four cards to know on sight

== How to read a zoo card

Choosing a distribution = choosing *where mass is allowed to live* (the support), then *how it is shaped* (the knobs):

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Name*], [*Support*], [*Knobs*], [*Typical job in ML*]),
  [Uniform $cal(U)(a,b)$], [$[a, b]$], [endpoints], [raw randomness; initializations],
  [Exponential $"Exp"(lambda)$], [$[0, oo)$], [rate $lambda$], [waiting times, positive scales],
  [Gaussian $cal(N)(mu, sigma^2)$], [$(-oo, oo)$], [center, spread], [noise, priors, embeddings — everything],
  [Beta$(alpha, beta)$], [$[0, 1]$], [two shapes], [beliefs about a *probability*],
)

#pause
Support first, always: a Gaussian for waiting times leaks mass into negative time; a Beta can't model heights. Match the room before decorating it.

== Card 1 · Uniform: the raw material

$ X tilde cal(U)(a, b) quad quad p(x) = cases(1\/(b-a) & quad a <= x <= b, 0 & quad "otherwise") quad quad EE[X] = (a+b)/2, quad "Var" = (b-a)^2/12 $

#pause
- The CDF is a straight ramp: $F(x) = (x - a)\/(b - a)$ on the support.
#pause
- Fair dice made continuous: no value in the support is more special than another.
#pause
#result[`rng.random()` supplies $cal(U)(0, 1)$ draws; inverse-CDF and other algorithms transform uniform draws into many useful distributions.]

== Card 2 · Exponential: waiting times

$ X tilde "Exp"(lambda) quad quad p(x) = lambda e^(-lambda x) "  on " [0, oo) quad quad EE[X] = 1/lambda, quad "Var" = 1/lambda^2 $

#align(center, lines(
  fn: (x => dist.pdf(dist.exponential(rate: 0.5), x),
       x => dist.pdf(dist.exponential(rate: 1.0), x),
       x => dist.pdf(dist.exponential(rate: 2.0), x)),
  domain: (0, 4), samples: 140, markers: false,
  colors: (TEAL, INK, ACC),
  labels: ([$lambda = 0.5$], [$lambda = 1$], [$lambda = 2$]), legend: "tr",
  x-label: $x$, y-label: $p(x)$, size: (92mm, 36mm),
))

#pause
Time between requests at a server, between clicks, between mutations. One famous quirk: it is *memoryless* — having already waited 5 minutes changes nothing about the wait ahead.

== Card 3 · Gaussian

$ X tilde cal(N)(mu, sigma^2) quad quad p(x) = 1/(sigma sqrt(2 pi)) thin e^(-(x - mu)^2 \/ (2 sigma^2)) $

#align(center, lines(
  fn: (x => dist.pdf(dist.normal(mu: 0.0, sigma: 1.0), x),
       x => dist.pdf(dist.normal(mu: 0.0, sigma: 2.0), x),
       x => dist.pdf(dist.normal(mu: 2.0, sigma: 0.7), x)),
  domain: (-6, 6), samples: 160, markers: false,
  colors: (INK, TEAL, ACC),
  labels: ([$mu=0, sigma=1$], [$mu=0, sigma=2$], [$mu=2, sigma=0.7$]), legend: "tl",
  x-label: $x$, y-label: $p(x)$, size: (92mm, 32mm),
))

#pause
$mu$ slides, $sigma$ stretches — that is the *entire* family. Tape measure: $approx 68%$ of mass within $1sigma$, $95%$ within $2sigma$, $99.7%$ within $3sigma$.

#pause
Why everywhere? Standardized sums of many independent effects often approach a Gaussian (CLT — demoed in L13). Among densities with a fixed mean and variance, the Gaussian has maximum entropy (proved in the information module).

== Card 4 · Beta: a distribution over probabilities #V

Beta draws live in $[0, 1]$ — so a draw can *be* a probability (a coin's heads-rate $theta$):

#grid(columns: 2, gutter: 4pt,
  align(center, lines(fn: beta-pdf(1, 1), domain: (0, 1), samples: 120, markers: false,
    colors: (MUTED,), fill-under: (0, MUTED.transparentize(85%)),
    title: [Beta(1, 1) — know nothing], size: (56mm, 16mm))),
  align(center, lines(fn: beta-pdf(2, 2), domain: (0, 1), samples: 120, markers: false,
    colors: (INK,), fill-under: (0, INK.transparentize(88%)),
    title: [Beta(2, 2) — probably fair], size: (56mm, 16mm))),
  align(center, lines(fn: beta-pdf(2, 5), domain: (0, 1), samples: 120, markers: false,
    colors: (TEAL,), fill-under: (0, TEAL.transparentize(85%)),
    title: [Beta(2, 5) — leans tails], size: (56mm, 16mm))),
  align(center, lines(fn: beta-pdf(5, 2), domain: (0, 1), samples: 120, markers: false,
    colors: (ACC,), fill-under: (0, ACC.transparentize(85%)),
    title: [Beta(5, 2) — leans heads], size: (56mm, 16mm))),
)

#pause
#notebox[$p(theta) prop theta^(alpha - 1) (1 - theta)^(beta - 1)$, with $EE[theta] = alpha\/(alpha + beta)$. In L15, $alpha - 1$ and $beta - 1$ become mode-based pseudo-count offsets, while observed counts update $alpha$ and $beta$ directly.]

== The four, on one wall #V

#grid(columns: 2, gutter: 6pt,
  align(center, lines(fn: x => dist.pdf(dist.uniform(a: 0.0, b: 1.0), x), domain: (-0.3, 1.3),
    samples: 200, markers: false, colors: (INK,), title: [Uniform — flat], size: (64mm, 24mm))),
  align(center, lines(fn: x => dist.pdf(dist.exponential(rate: 1.0), x), domain: (0, 4),
    samples: 120, markers: false, colors: (TEAL,), title: [Exponential — decaying], size: (64mm, 24mm))),
  align(center, lines(fn: x => dist.pdf(dist.normal(), x), domain: (-3.5, 3.5),
    samples: 120, markers: false, colors: (ACC,), title: [Gaussian — the bell], size: (64mm, 24mm))),
  align(center, lines(fn: beta-pdf(2, 5), domain: (0, 1),
    samples: 120, markers: false, colors: (GREEN,), title: [Beta — shapes on [0, 1]], size: (64mm, 24mm))),
)

#pause
Flat box, falling slide, bell, and the shape-shifter on $[0,1]$. From these four (plus L13's multivariate bell) you can read 90% of the distributions in ML papers.

== Explore the zoo hands-on #I

#interbox(link-to: DE)[*Distribution Explorer* — every distribution in tonight's reading with sliders for each knob. Watch pdf and CDF move together; find the parameter settings that push a density above 1.]

#pause
#interbox(link-to: ST)[*Seeing Theory* (Brown) — the "Probability Distributions" and "Random Variables" chapters animate exactly today's arc: PMF → pdf, area = probability, CDF as running total. 15 beautiful minutes.]

= The Jacobian toll: change of variables

== Transform a standard noise variable

The generative recipe was: $z tilde$ simple noise, $x = f(z)$. For the recipe to be *mathematics* rather than vibes, we must answer:

#result[If $X$ has density $p_X$ and $Y = g(X)$ — what is $p_Y$?]

#pause
Start with the humblest transformation and be *completely* sure of it:

$ Y = a X + b quad quad "(stretch by " a ", slide by " b ")" $

#pause
Concrete pilot case: $X tilde cal(U)(0, 1)$ and $Y = 2X$. Before any formula — what *should* $p_Y$ look like?

== Picture: stretch the axis, squash the curve #V

#align(center, lines(
  fn: (x => dist.pdf(dist.uniform(a: 0.0, b: 1.0), x),
       x => dist.pdf(dist.uniform(a: 0.0, b: 2.0), x)),
  domain: (-0.3, 2.3), samples: 250, markers: false,
  colors: (TEAL, ACC),
  labels: ([$X tilde cal(U)(0,1) thick "(height 1)"$], [$Y = 2X thick "(height 1/2)"$]), legend: "tr",
  x-label: $x$, size: (108mm, 40mm),
))

#pause
$Y$ lives on $[0, 2]$ — twice the room. Total probability is still 1 — same area. Twice the width forces *half the height*.

#pause
#result[Stretching the axis by $a$ *dilutes* the density by $a$. Area is conserved; height pays.]

== Numbers: follow one interval through the stretch

Track the interval $[0.2, 0.3]$ under $y = 2x$:

#pause
- Its probability under $X$: width $0.1$, height 1 → $P = 0.1$.
#pause
- Where does it land? $[0.4, 0.6]$ — the *same* outcomes, relabeled. The $0.1$ of probability rides along untouched.
#pause
- New width $0.2$. For the area to still be $0.1$: height $= 0.1 \/ 0.2 = 1\/2$. ✓ matches the picture.

#pause
#notebox[Probability belongs to *events* (sets of outcomes), and a transformation only renames outcomes — so event probabilities are frozen. Densities are the ones that must adjust, because *widths* changed underneath them.]

== The tempting wrong answer

"Just substitute": $p_Y (y) = p_X (y \/ 2)$?

#pause
Test it. $p_X (y\/2) = 1$ for all $y in [0, 2]$ — a box of height 1 over width 2:

$ integral_0^2 1 dif y = 2 quad #text(fill: RED)[— total probability 2. Not a distribution.] $

#pause
#alertbox[Densities do *not* transform by substitution alone. Substitution relabels the $x$-axis but forgets that the axis itself was stretched — the area bookkeeping is missing. Forgetting this factor is the classic hand-rolled-likelihood bug.]

#pause
The fix must multiply by exactly the factor that repairs the area. Let's derive it, not guess it.

== ⭐ The derivation, step 1: route through the CDF #D

For transformations, *CDFs are genuine probabilities* and are easier to work with. So compute $F_Y$ first. Take $Y = a X + b$ with $a > 0$:

#pause
$ F_Y (y) = P(Y <= y) = P(a X + b <= y) $

#pause
Solve the inequality for $X$ (allowed: $a > 0$ keeps the direction):

$ F_Y (y) = P(X <= (y - b)/a) = F_X ((y - b)/a) $

#pause
No approximation, no density yet — just the event $\{Y <= y\}$ renamed as an event about $X$. This "CDF first, then differentiate" move is *the* standard tool for transformed variables.

== ⭐ Step 2: differentiate with the chain rule #D

Density = derivative of the CDF (Direction 2, minutes old):

$ p_Y (y) = dif / (dif y) F_X ((y - b)/a) = p_X ((y - b)/a) dot underbrace(1/a, "chain rule's" \ "inner derivative") $

#pause
If $a < 0$ the inequality flips ($P(X >= dots) = 1 - F_X$), and differentiating brings $-1\/a$ — positive again. Both signs in one formula:

#pause
#result[$ p_Y (y) = p_X ((y - b)/a) dot 1/abs(a) quad quad "— the stretch factor, paid in density" $]

== ⭐ Step 3: check the formula #D

*Check against the pilot:* $X tilde cal(U)(0,1)$, $Y = 2X$: $p_Y (y) = p_X (y\/2) dot 1\/2 = 1\/2$ on $[0, 2]$ — the half-height box we measured by hand. ✓

#pause
*Check the area is repaired:* substitute $x = (y-b)\/a$, $dif y = abs(a) dif x$:

$ integral p_X ((y-b)/a) 1/abs(a) dif y = integral p_X (x) 1/abs(a) abs(a) dif x = integral p_X (x) dif x = 1 quad ✓ $

#pause
The $1\/abs(a)$ is precisely what the substitution rule of integration (Module 2) demands — change of variables for densities *is* change of variables for integrals, wearing a probability costume.

== The general 1-D rule

For a strictly monotone differentiable $g$ with nonzero derivative, the same two moves (CDF, then chain rule) give:

$ p_Y (y) = p_X (g^(-1)(y)) dot abs((dif x)/(dif y)) quad quad "where " x = g^(-1)(y) $

#pause
Read it aloud: *old density at the source point, times how much the axis was locally stretched.* For linear $g$, the stretch $abs(dif x\/dif y) = 1\/abs(a)$ everywhere; for curved $g$ it varies point by point.

#pause
#notebox[If $X=e^Z$ and $Z$ is Gaussian, then $X$ is log-normal. Its density contains $1\/x$, which is exactly the inverse derivative $abs(dif z \/ dif x)$.]

== Multivariate change of variables uses $abs(det J)$

L9 showed that $abs(det J)$ is the local volume-scaling factor of a square map.

#pause
#table(
  columns: (1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*L9 (geometry)*], [*L12 (probability)*]),
  [map $Phi: RR^2 -> RR^2$], [transform $g: RR -> RR$],
  [little square → parallelogram], [little interval → stretched interval],
  [$abs(det J)$ = local area scaling], [$abs(dif y\/dif x)$ = local length scaling],
  [areas multiply by $abs(det J)$], [densities divide by it: $p_Y = p_X dot abs(dif x\/dif y)$],
)

#pause
For an invertible differentiable map in $d$ dimensions, $p_Y (bold(y)) = p_X (bold(x)) abs(det J)^(-1)$ when $J = dif bold(y)\/dif bold(x)$. This determinant factor is central to normalizing flows; diffusion models generally use different likelihood constructions.

== Build a Gaussian from a standard normal vector

Let $Z tilde cal(N)(0, 1)$ — density $phi(z) = e^(-z^2\/2) \/ sqrt(2 pi)$ — and stretch-and-slide: $X = mu + sigma Z$. The toll rule with $a = sigma$, $b = mu$:

$ p_X (x) = phi((x - mu)/sigma) dot 1/sigma = 1/(sigma sqrt(2 pi)) thin e^(-(x - mu)^2/(2 sigma^2)) $

#pause
That is the $cal(N)(mu, sigma^2)$ density, *derived* rather than memorized — the $1\/sigma$ out front is a Jacobian toll and always was.

#pause
#codebox[```python
z = torch.randn(10_000)      # N(0, 1) — the only Gaussian anyone samples
x = mu + sigma * z           # N(mu, sigma²) — transformation does the rest
```]

#pause
This two-line transformation is the reparameterization used in variational autoencoders.

== Checkpoint: pay the toll #Q

#mcq([$X tilde cal(U)(0, 1)$ and $Y = 3X$. What is $p_Y (1.5)$?],
  [$1$ — uniform densities stay height 1],
  [$3$ — the stretch multiplies the density by 3],
  [$1\/3$ — the stretch dilutes the density by 3],
  [$0$ — $y = 1.5$ is outside the support of $Y$],
)

== Answer: pay the toll #A

#mcq-answer("C", [$p_Y (1.5) = 1 dot 1\/abs(3) = 1\/3$],
  [$Y$ lives on $[0, 3]$ (so D is out), and stretching by 3 spreads the same area over triple the width — density *divides* by 3 (so B has the toll backwards, and A forgot the toll entirely). Sanity: height $1\/3$ × width $3$ = area 1. ✓])

= Inverse-CDF sampling: uniform → anything

== Uniform draws as a starting point

Most numerical random-number APIs expose uniform pseudorandom draws (ultimately generated from random bits):

#codebox[```python
u = rng.random()        # U(0, 1)
```]

#pause
Gaussian, exponential, and many other samplers can be constructed from uniform draws, although production libraries use several specialized algorithms.

#pause
#result[Claim: $cal(U)(0, 1)$ + a CDF $F$ = a sampler for $F$'s distribution. The transform is $x = F^(-1)(u)$.]

#pause
This is the simplest instance of sampling by transformation.

== The picture: darts at the $u$-axis #V

Throw uniform darts at the *vertical* axis of the CDF; pull each through $F^(-1)$ down to the $x$-axis:

#fig("/lecture12/figures/l12_inverse_cdf.svg", w: 96%)

#pause
Left: equal-density darts on $[0, 1]$. Right: their landing spots wear the exponential's density exactly.

== Why it works, in words first

Where the CDF is *steep*, a wide band of $u$-values funnels into a narrow band of $x$ — many darts, small region: *high density*.

#pause
Where the CDF is *flat*, a narrow band of $u$ spreads over a huge range of $x$ — few darts per unit length: *low density* (the exponential's tail).

#pause
Steepness of $F$ is $F' = p$ — the density itself. The funnel automatically deals darts in proportion to $p$. That is the entire mechanism; now we make it a theorem.

== ⭐ Correctness, in two lines #D

Let $U tilde cal(U)(0, 1)$ and define $X = F^(-1)(U)$, with $F$ strictly increasing. Compute $X$'s CDF:

#pause
$ P(X <= x) = P(F^(-1)(U) <= x) = P(U <= F(x)) quad quad "(apply the increasing map " F " to both sides)" $

#pause
And for a uniform, $P(U <= t) = t$ for any $t in [0, 1]$:

$ P(X <= x) = F(x) quad qed $

#pause
#result[When $F$ is strictly increasing, $F^(-1)(U)$ has CDF $F$. More generally, the quantile function gives the same construction for CDFs with flat regions or jumps.]

#pause
(Both ⭐ derivations today were the same trick: don't touch densities — push events through monotone maps and read off the CDF.)

== Worked: the exponential sampler, by hand

Target $"Exp"(lambda)$: $F(x) = 1 - e^(-lambda x)$. Invert it — set $u = F(x)$, solve for $x$:

#pause
$ u = 1 - e^(-lambda x) quad => quad e^(-lambda x) = 1 - u quad => quad x = -ln(1 - u)/lambda $

#pause
Spot-check $lambda = 1$: $u = 0.5 -> x = ln 2 approx 0.693$ — half the darts land below $0.693$, the exponential's median. The mean (1.0) is further right, dragged by the tail — consistent with the balance-point slide. ✓

#pause
#notebox[Since $1 - U$ is *also* $cal(U)(0,1)$ (flip the interval), $x = -ln(U)\/lambda$ works too — that symmetry is practice problem 6, and what NumPy actually implements.]

== Code: five lines, ten thousand samples

#codebox[```python
lam = 1.0
rng = np.random.default_rng(0)

u = rng.random(10_000)           # step 1: the only randomness we own
x = -np.log(1 - u) / lam         # step 2: F⁻¹ — one deterministic line

x.mean()                         # 1.003  ≈ 1/λ  ✓ (the L12 audit, again)
```]

#pause
The right panel of the darts figure *is* this code's histogram against $lambda e^(-lambda x)$ — sampling theory you can falsify in one plot.

#pause
Works verbatim for any distribution whose $F^(-1)$ you can write: uniform, exponential, Laplace, Cauchy, logistic…

== The transformation view of a generative model

#genpipe

#pause
The pipeline is now explicit: draw $u tilde cal(U)(0,1)$ and transform it by $F^(-1)$ to obtain exponential samples.

#pause
The industrial versions upgrade each box, not the idea: the noise goes multivariate Gaussian (L13), the transformation becomes *learned* (ES 667), and the toll $abs(det J)$ prices every step.

#pause
#notebox[The Gaussian's own $F^(-1)$ has no closed formula — libraries use `erfinv` or the Box–Muller trick. `torch.randn` is still "uniform bits, transformed" under the hood.]

= Summary & what's next

== The discrete → continuous dictionary

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([], [*Discrete (ES 114)*], [*Continuous (today)*]),
  [chance object], [PMF: $p(x) = P(X = x)$], [pdf: probability *per unit length*],
  [$P(X = x)$], [can be positive], [always $0$],
  [can exceed 1?], [never], [happily ($cal(U)(0, 1\/2)$ has $p = 2$)],
  [combine], [$sum$], [$integral$ — probability is *area*],
  [CDF], [staircase], [smooth ramp; $F' = p$],
  [$EE[X]$, $"Var"$], [$sum x thin p(x)$, …], [$integral x thin p(x) dif x$, …],
  [transform $Y = g(X)$], [relabel outcomes], [relabel *and pay* $abs(dif x\/dif y)$],
)

#pause
One table, the whole lecture. The left column you owned already; the right column is now yours too.

== Lecture 12 — summary

- *Density* = probability per unit length: $P(X = x) = 0$; $p(x) > 1$ legal; *areas* are probabilities.
- *CDF*: $F(x) = integral_(-oo)^x p$, and $F' = p$; interval questions are $F(b) - F(a)$.
- *Moments are integrals*: uniform balances at $(a+b)\/2$; exponential waits $1\/lambda$.
- *Common families*: uniform, exponential, Gaussian, and Beta.
- *The toll*: $p_Y (y) = p_X (x) abs(dif x\/dif y)$ — the Gaussian's $1\/sigma$ is a receipt.
- *The sampler*: $F^(-1)(U) tilde F$, proved in two lines — uniform darts become anything.

#pause
#notebox[*Read before L13* — MML §6.1–6.4 (densities, CDFs, moments) and *§6.7* (today's change of variables, in print). Play: Distribution Explorer · Seeing Theory. *Tutorial this week*: densities and samplers, by hand and in NumPy.]

== Practice problems

Try on paper; verify against the tutorial notebook.

+ $X tilde cal(U)(2, 6)$: the pdf's height, $P(3 <= X <= 4)$, $EE[X]$, $"Var"(X)$.
+ The density $p(x) = 3x^2$ on $[0, 1]$: check the area, find $F$, compute $EE[X]$, and build the sampler $F^(-1)(u)$.
+ Show $"Var"("Exp"(lambda)) = 1\/lambda^2$ by computing $EE[X^2]$ (integrate by parts twice).
+ $Y = -2X + 3$ with $X tilde cal(U)(0, 1)$: find $Y$'s support and density — mind the $abs(a)$.
+ Heights in inches → centimeters ($times 2.54$): what happens to the density's height, and to its units? Reconcile with "area = 1".
+ Why does $-ln(U)\/lambda$ sample $"Exp"(lambda)$ just as well as $-ln(1-U)\/lambda$? Which property of $cal(U)(0,1)$ did you use?

== ⭐⭐⭐ When $F^(-1)$ is out of reach #OPT

Many densities (most Bayesian posteriors!) have no invertible CDF. Two escape hatches, in cartoon:

#pause
- *Rejection sampling* — rain uniform points under an easy envelope curve that covers $p$; keep the points landing under $p$, discard the rest. Kept points are exact samples from $p$. Cost: the gap between envelope and target is pure waste.
#pause
- *Importance sampling* — sample from a wrong-but-easy $q$, then repair the bias by weighting each sample by $p(x)\/q(x)$. Nothing discarded, but weights can degenerate.

#pause
#notebox[Pointers only, by design: MacKay Ch. 29 is the beautiful reference; the probabilistic-ML course (and MCMC) picks up this thread properly. For this course, inverse-CDF + the toll rule are the samplers you own.]

#focus-slide[
  Continuous probability is area, and transformations charge a Jacobian toll.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Multivariate Gaussians* — one bump, many dimensions: the ellipse that runs half of ML.
]
