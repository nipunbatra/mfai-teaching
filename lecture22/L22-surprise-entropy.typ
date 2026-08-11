// Surprise and Entropy — Lecture 22 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture22/L22-surprise-entropy.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Surprise and Entropy],
  subtitle: [Measuring uncertainty in bits],
)

#title-slide()

// ═══════════════════ 1 · questions ═══════════════════
= How many yes/no questions?

== Guess a number from 1 to 8

I choose one number uniformly from $1,dots,8$. You may ask yes/no questions.

#pause
A good first question splits the possibilities in half:

“Is the number greater than 4?”

#pause
Then split the remaining four into two, and the remaining two into one.

#pause
Three balanced questions always suffice.

== A balanced question tree #V

#fig("/lecture22/figures/question_tree.svg", w: 70%)

#pause
One yes/no answer carries at most one bit. Distinguishing eight equally likely outcomes takes $log_2 8=3$ bits.

== Why balanced questions work

After one balanced answer, $8$ possibilities become $4$.

#pause
After two answers, $4$ become $2$.

#pause
After three answers, $2$ become $1$.

#pause
$ 8 arrow.r 4 arrow.r 2 arrow.r 1. $

#pause
Each answer halves the remaining uncertainty.

== Sixteen outcomes

If the outcomes are equally likely, the same reasoning gives

$ 16 arrow.r 8 arrow.r 4 arrow.r 2 arrow.r 1. $

#pause
Four binary questions are needed:

$ log_2 16=4. $

#pause
For $N$ equally likely outcomes, the ideal number is $log_2 N$ bits.

== Outcomes need not be equally likely

Suppose tomorrow's weather probabilities are

$ P("sun")=0.75, quad P("cloud")=0.20, quad P("rain")=0.05. $

#pause
“Sun” should be easy to identify because it happens often. “Rain” can receive a longer description because it is rare.

#pause
Information theory assigns a numerical surprise to each outcome before averaging over them.

== Learning outcomes

By the end of this lecture you will be able to:

+ derive the form $-log p$ from basic requirements,
+ compute self-information in bits and nats,
+ compute entropy for a finite distribution,
+ interpret entropy as average surprise and ideal average question count,
+ analyze the entropy of a biased coin,
+ derive ⭐ the maximum-entropy distribution on a finite set,
+ and estimate entropy from observed frequencies.

// ═══════════════════ 2 · self-information ═══════════════════
= Surprise of one outcome

== Three requirements

Let $I(p)$ denote the information gained when an event of probability $p$ occurs.

We want:

#pause
1. *Rarer events are more informative:* if $p<q$, then $I(p)>I(q)$.
#pause
2. *A certain event gives no information:* $I(1)=0$.
#pause
3. *Independent surprises add:* $I(p q)=I(p)+I(q)$.

== Why independent surprises should add

For two independent fair coin flips,

$ P(HH)=1/2 times 1/2=1/4. $

#pause
Learning the first result takes one bit; learning the second takes one more bit.

#pause
So learning $HH$ should take two bits:

$ I(1/4)=I(1/2)+I(1/2). $

== The logarithm is forced by addition

The logarithm turns products into sums:

$ log(p q)=log p+log q. $

#pause
Since $log p<=0$ for $0<p<=1$, add a minus sign so rarer events receive larger positive values:

#result[$I(p)=-log p$.]

#pause
Under mild regularity assumptions, the three requirements determine this form up to the choice of log base.

== Bits and nats

Base 2 measures information in *bits*:

$ I_2(p)=-log_2 p. $

#pause
Base $e$ measures information in *nats*:

$ I_e(p)=-ln p. $

#pause
Conversion:

$ 1 " nat"=1/(ln 2) " bits" approx 1.443 " bits". $

== Familiar probabilities

#table(
  columns: (1fr, 1fr, 1.25fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([$p$], [$-log_2 p$], [interpretation]),
  [$1$], [$0$ bits], [certain],
  [$1/2$], [$1$ bit], [one fair binary choice],
  [$1/4$], [$2$ bits], [two fair binary choices],
  [$1/8$], [$3$ bits], [three fair binary choices],
  [$1/1024$], [$10$ bits], [one of $2^10$ outcomes],
)

== Surprise as a curve #V

#fig("/lecture22/figures/surprise_curve.svg", w: 63%)

#pause
Surprise grows without bound as probability approaches zero.

== Weather outcomes

For

$ P("sun")=0.75, quad P("cloud")=0.20, quad P("rain")=0.05, $

the surprises are

$ I("sun")=-log_2 0.75 approx 0.415 " bits", $
$ I("cloud")=-log_2 0.20 approx 2.322 " bits", $
$ I("rain")=-log_2 0.05 approx 4.322 " bits". $

#pause
Rain is about ten times more surprising than sun on this scale.

== Log-likelihood already used this quantity

For an observed outcome $x$ under model $q$,

$ -log q(x) $

is both:

+ the surprise assigned by $q$ to $x$, and
+ the negative log-likelihood contribution of $x$.

#pause
L14 minimized sums of these terms. L24 will interpret that sum as a coding cost.

== Question: two independent events #Q

#mcq(
  [Independent events have probabilities $1/4$ and $1/8$. What is the surprise of both occurring?],
  [$3$ bits],
  [$5$ bits],
  [$12$ bits],
  [$32$ bits],
)

== Answer #A

#mcq-answer("B", [$5$ bits], [$I((1/4)(1/8))=I(1/4)+I(1/8)=2+3=5$ bits.])

// ═══════════════════ 3 · entropy ═══════════════════
= Average surprise of a source

== From one outcome to a distribution

Let a discrete random variable $X$ have outcomes $x_1,dots,x_K$ with probabilities $p_1,dots,p_K$.

#pause
Outcome $x_i$ carries surprise

$ -log_2 p_i. $

#pause
Before observing $X$, its expected surprise is

#result[$H(X)=-sum_(i=1)^K p_i log_2 p_i$.]

#pause
This is Shannon entropy, measured in bits.

== Entropy is an expectation

$ H(X)=EE[-log_2 p(X)]. $

#pause
The random quantity is the surprise of whichever outcome occurs.

#pause
High-probability outcomes receive larger weights in this average, even though each carries less surprise individually.

== Fair coin

$ P(H)=P(T)=1/2. $

#pause
$ H(X)=-1/2 log_2(1/2)-1/2 log_2(1/2). $

#pause
Since $log_2(1/2)=-1$,

$ H(X)=1/2+1/2=1 " bit". $

#pause
One fair flip produces one bit of uncertainty.

== A biased coin

Let $P(H)=0.9$ and $P(T)=0.1$.

$ H(X)=-0.9log_2 0.9-0.1log_2 0.1. $

#pause
$ H(X) approx 0.137+0.332=0.469 " bits". $

#pause
The outcome is not known in advance, but it is much more predictable than a fair coin.

== A certain coin

If $P(H)=1$ and $P(T)=0$,

$ H(X)=-(1 log_2 1+0 log_2 0)=0. $

#pause
We use the continuous limit

$ lim_(p arrow.r 0^+) p log p=0. $

#pause
A deterministic source has zero entropy.

== Fair die

For six equally likely faces,

$ H(X)=-sum_(i=1)^6 1/6 log_2(1/6). $

#pause
$ H(X)=log_2 6 approx 2.585 " bits". $

#pause
The value need not be an integer: entropy is an ideal *average* over many outcomes.

== Weather entropy

For probabilities $(0.75,0.20,0.05)$,

$ H(X)=0.75(0.415)+0.20(2.322)+0.05(4.322). $

#pause
$ H(X) approx 0.311+0.464+0.216=0.991 " bits". $

#pause
The three-outcome source has less than one bit of average uncertainty because “sun” dominates.

== Entropy and average questions

If outcomes are equally likely, a balanced tree uses exactly $log_2 K$ questions when $K$ is a power of two.

#pause
For unequal probabilities, use shorter paths for common outcomes and longer paths for rare ones.

#pause
Entropy is the theoretical lower limit on average binary description length, up to coding details developed in L23.

// ═══════════════════ 4 · binary entropy ═══════════════════
= How uncertainty changes with bias

== The binary entropy function

For a Bernoulli random variable with $P(X=1)=p$,

$ H_2(p)=-p log_2 p-(1-p)log_2(1-p). $

#pause
It satisfies

$ H_2(p)=H_2(1-p), $

because renaming heads and tails does not change uncertainty.

== The complete curve #V

#fig("/lecture22/figures/binary_entropy.svg", w: 63%)

#pause
Entropy is maximal at $p=1/2$ and approaches zero as $p$ approaches $0$ or $1$.

== Differentiate to find the maximum

It is cleaner to use natural logs and convert at the end:

$ H_2(p)=-(p ln p+(1-p)ln(1-p))/(ln 2). $

#pause
$ H_2'(p)=(ln(1-p)-ln p)/(ln 2). $

#pause
Set $H_2'(p)=0$:

$ ln(1-p)=ln p quad arrow.r.double quad p=1/2. $

== Check the curvature

$ H_2''(p)=-1/(ln 2)(1/p+1/(1-p))<0 $

for $0<p<1$.

#pause
So the stationary point $p=1/2$ is the unique maximum.

#pause
#result[A fair coin is the most uncertain Bernoulli source.]

== Ten flips

For independent flips $X_1,dots,X_10$,

$ H(X_1,dots,X_10)=sum_(i=1)^10 H(X_i). $

#pause
+ fair coin: $10 times 1=10$ bits,
+ coin with $p=0.9$: $10 times 0.469=4.69$ bits.

#pause
Independence makes joint entropy additive, matching the additive-surprise requirement.

== Question: compare two coins #Q

#mcq(
  [Which Bernoulli source has greater entropy?],
  [$p=0.01$],
  [$p=0.20$],
  [$p=0.50$],
  [$p=0.99$],
)

== Answer #A

#mcq-answer("C", [$p=0.50$], [The binary entropy curve reaches its unique maximum of one bit at the fair coin.])

// ═══════════════════ 5 · maximum entropy ═══════════════════
= The least committed distribution

== Fixed number of outcomes

Among all distributions over $K$ outcomes, which one has the largest entropy?

$ max_bold(p) quad -sum_i p_i ln p_i $
$ "subject to" quad sum_i p_i=1, quad p_i>=0. $

#pause
L19 solved the $K=3$ version. We now complete the general derivation.

== ⭐ Form the Lagrangian #D

Assume an interior solution with $p_i>0$:

$ cal(L)(bold(p),lambda)=-sum_i p_i ln p_i-lambda(sum_i p_i-1). $

#pause
For each coordinate,

$ partial cal(L)/partial p_i=-(ln p_i+1)-lambda=0. $

== ⭐ Every probability is equal #D

Stationarity gives

$ ln p_i=-1-lambda $

and hence

$ p_i=exp(-1-lambda). $

#pause
The right side does not depend on $i$, so all $p_i$ have the same value $c$.

== ⭐ Normalize and evaluate #D

$ sum_(i=1)^K p_i=K c=1 quad arrow.r.double quad c=1/K. $

#pause
The entropy is

$ H=-sum_(i=1)^K 1/K log_2(1/K)=log_2 K. $

#pause
#result[$H(X)<=log_2 K$, with equality for the uniform distribution.]

#pause
The Hessian is diagonal with entries $-1/(p_i ln 2)<0$ in the simplex interior, so entropy is strictly concave. Therefore this stationary point is the unique global maximum; boundary values follow by continuity.

== Example: four outcomes

#table(
  columns: (1.4fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([distribution], [entropy]),
  [$(1,0,0,0)$], [$0$ bits],
  [$(0.7,0.1,0.1,0.1)$], [$1.357$ bits],
  [$(0.4,0.3,0.2,0.1)$], [$1.846$ bits],
  [$(0.25,0.25,0.25,0.25)$], [*$2$ bits*],
)

== Known constraints change the answer

Uniform is maximum entropy only when normalization is the sole constraint.

#pause
If we also know an expected value,

$ sum_i p_i a_i=mu, $

the maximum-entropy solution has exponential form

$ p_i prop exp(-beta a_i). $

#pause
The constraints determine $beta$ and the normalizing constant.

== Why Gaussians recur #OPT

Among continuous distributions with fixed mean and variance, the Gaussian has maximum differential entropy.

#pause
Interpretation: specifying only mean and variance, the Gaussian adds the least extra structure.

#pause
This is one reason the Gaussian appeared so often in L12–L15. Differential entropy has subtleties that discrete entropy does not; we use the statement here without proof.

// ═══════════════════ 6 · estimate from data ═══════════════════
= Entropy of an observed source

== Empirical probabilities

Given counts $n_1,dots,n_K$ from $N$ observations, estimate

$ hat(p)_i=n_i/N. $

#pause
The plug-in entropy estimate is

$ hat(H)=-sum_i hat(p)_i log_2 hat(p)_i. $

#pause
This is the entropy of the empirical distribution—the MLE from L14 inserted into the entropy formula.

== A short symbol sequence

Sequence: `A A A A B B C D`

#pause
Counts and probabilities:

$ hat(bold(p))=(4/8,2/8,1/8,1/8). $

#pause
Entropy:

$ hat(H)=-(1/2)(-1)-(1/4)(-2)-2(1/8)(-3)=1.75 " bits/symbol". $

== Course-plan letter frequencies #V

#fig("/lecture22/figures/course_letter_entropy.svg", w: 70%)

#pause
The calculation treats letters as independent draws. Natural language has dependencies, so a model that uses context can predict—and compress—better than this unigram estimate.

== Compute it directly

#codebox[```python
from collections import Counter
from math import log2

text = open("lecture-plan-detailed.md").read().lower()
counts = Counter(ch for ch in text if ch.isascii() and ch.isalpha())
N = sum(counts.values())
p = [n/N for n in counts.values()]
H = -sum(pi*log2(pi) for pi in p)
print(H)  # about 4.24 bits/letter
```]

== Finite-sample caution

The plug-in entropy tends to underestimate the entropy of a large alphabet when data are scarce.

#pause
Why?

+ unseen outcomes receive estimated probability zero,
+ rare outcomes have noisy frequency estimates,
+ the logarithm magnifies errors near zero.

#pause
For this course, the main lesson is to report the sample size and alphabet along with the estimate.

== Independence changes the scale

If letters were independent with entropy $4.24$ bits/letter, a 1000-letter text would carry about

$ 4240 " bits". $

#pause
But English letters are not independent: after `q`, the next letter is often `u`; spaces and words add more structure.

#pause
Conditional models reduce uncertainty by using context. Markov chains in L25 make this explicit.

// ═══════════════════ 7 · close ═══════════════════
= One scale for uncertainty

== Entropy and prediction

Low entropy means a source is predictable:

+ a coin with $p=0.99$,
+ a sensor stuck at one value,
+ a highly repetitive text.

#pause
High entropy means outcomes are spread more evenly:

+ a fair coin,
+ a uniform die,
+ a password drawn uniformly from a large set.

== Entropy is not “disorder” by itself

Entropy is defined relative to:

+ a random variable,
+ its possible outcomes,
+ and a probability distribution.

#pause
Changing the representation can change the entropy under discussion.

#pause
Use the formula and specify the source; do not rely on the word “disorder” alone.

== Summary

#result[Information is surprise; entropy is its expected value.]

#pause
+ Self-information: $I(x)=-log p(x)$.
+ Independent surprises add because logarithms turn products into sums.
+ Entropy: $H(X)=EE[-log p(X)]$.
+ A fair Bernoulli source has one bit of entropy.
+ Over $K$ outcomes, the uniform distribution has maximum entropy $log_2 K$.
+ Empirical entropy is computed from observed frequencies, with finite-sample caution.

== Practice

1. Find the surprise of an event with probability $1/32$.

#pause
2. Compute the entropy of $(1/2,1/4,1/8,1/8)$.

#pause
3. Which has higher entropy: $(0.6,0.4)$ or $(0.9,0.1)$? Explain without calculation, then check.

#pause
4. Find the maximum entropy of a distribution over 100 equally possible labels.

== Practice answers #A

1. $-log_2(1/32)=5$ bits.

2. $1/2(1)+1/4(2)+2(1/8)(3)=1.75$ bits.

3. $(0.6,0.4)$ is closer to uniform and has larger entropy: about $0.971$ versus $0.469$ bits.

4. $log_2 100 approx 6.644$ bits, attained by the uniform distribution.

== Next: turn probabilities into descriptions

Entropy states an ideal average number of bits. We still need a constructive code.

#pause
Next lecture:

+ prefix-free binary codes,
+ Huffman's greedy algorithm,
+ expected code length versus entropy.

#focus-slide[Probability determines surprise; averaging surprise gives entropy.]
