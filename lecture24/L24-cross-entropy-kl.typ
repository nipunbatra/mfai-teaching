// Cross-Entropy and KL Divergence — Lecture 24 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture24/L24-cross-entropy-kl.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Cross-Entropy and KL Divergence],
  subtitle: [The cost of using the wrong probability model],
)

#title-slide()

// ═══════════════════ 1 · wrong code ═══════════════════
= A code built for the wrong weather

== True weather and forecast model

Suppose the actual distribution is

$ p=(0.75,0.20,0.05) $

for sun, cloud, and rain.

#pause
A forecasting model reports

$ q=(0.45,0.35,0.20). $

#pause
If code lengths are built from $q$, outcome $x$ receives ideal length

$ -log_2 q(x). $

== Model probabilities determine lengths #V

#fig("/lecture24/figures/wrong_weather_model.svg", w: 76%)

#pause
The average uses true frequencies $p$, but the lengths come from model probabilities $q$.

== Average cost under the true source

Define cross-entropy

$ H(p,q)=-sum_x p(x)log_2 q(x). $

#pause
For this weather example,

$ H(p,q)=-0.75log_2 0.45-0.20log_2 0.35-0.05log_2 0.20. $

#pause
$ H(p,q) approx 1.283 " bits/outcome". $

== Compare with the true entropy

L22 computed

$ H(p)=-sum_x p(x)log_2 p(x) approx 0.992 " bits/outcome". $

#pause
Using $q$ instead of $p$ costs

$ 1.283-0.992=0.291 " extra bits/outcome". $

#pause
That excess is the KL divergence from $p$ to $q$.

== Learning outcomes

By the end of this lecture you will be able to:

+ compute cross-entropy and KL divergence for discrete distributions,
+ derive ⭐ $H(p,q)=H(p)+D_"KL"(p||q)$,
+ prove KL non-negativity using Jensen's inequality,
+ explain KL asymmetry and support failures,
+ show that minimizing cross-entropy is maximum likelihood,
+ reduce one-hot classification cross-entropy to $-log q_y$,
+ convert average log loss to perplexity,
+ and interpret mutual information as dependence.

// ═══════════════════ 2 · decomposition ═══════════════════
= Entropy plus an overcharge

== Three related quantities

Entropy of the true source:

$ H(p)=-sum_x p(x)log p(x). $

#pause
Cross-entropy of model $q$ under source $p$:

$ H(p,q)=-sum_x p(x)log q(x). $

#pause
KL divergence:

$ D_"KL"(p||q)=sum_x p(x)log((p(x))/(q(x))). $

== ⭐ Derive the identity #D

Start from cross-entropy and multiply inside the logarithm by $p(x)/p(x)$:

$ H(p,q)=-sum_x p(x)log q(x). $

#pause
$ =-sum_x p(x)log p(x)+sum_x p(x)log((p(x))/(q(x))). $

#pause
#result[$H(p,q)=H(p)+D_"KL"(p||q)$.]

== Read each term

$ underbrace(H(p,q), "actual average length using model q") $
$ =underbrace(H(p), "unavoidable source uncertainty") $
$ +underbrace(D_"KL"(p||q), "extra cost from mismatch"). $

#pause
For a fixed source $p$, only the KL term depends on $q$.

== Three models, one source #V

#fig("/lecture24/figures/cross_entropy_decomposition.svg", w: 63%)

#pause
The source entropy is unchanged. A poorer model increases only the mismatch term.

== Weather KL calculation

$ D_"KL"(p||q)=sum_x p(x)log_2((p(x))/(q(x))). $

#pause
$ =0.75log_2(0.75/0.45)+0.20log_2(0.20/0.35)+0.05log_2(0.05/0.20). $

#pause
$ approx 0.291 " bits". $

#pause
Some individual terms are negative; their weighted sum is never negative.

== Question: perfect model #Q

#mcq(
  [If $q=p$, what are $D_"KL"(p||q)$ and $H(p,q)$?],
  [$0$ and $0$],
  [$0$ and $H(p)$],
  [$H(p)$ and $0$],
  [$H(p)$ and $2H(p)$],
)

== Answer #A

#mcq-answer("B", [$0$ and $H(p)$], [A perfect model removes the mismatch cost, but it does not remove the source's intrinsic entropy.])

// ═══════════════════ 3 · KL properties ═══════════════════
= A divergence, not a distance

== KL is non-negative

For distributions $p$ and $q$ on the same outcome set,

$ D_"KL"(p||q)>=0, $

with equality if and only if $p=q$ almost everywhere.

#pause
This makes cross-entropy no smaller than entropy:

$ H(p,q)>=H(p). $

== ⭐ Jensen proof #D

If $q(x)=0$ for an outcome with $p(x)>0$, then $D_"KL"(p||q)=infinity$ and the claim already holds.

#pause
Otherwise let $S$ be the support of $p$. Since $-log z$ is convex,

$ D_"KL"(p||q)=-sum_(x in S) p(x)log((q(x))/(p(x))). $

#pause
Jensen's inequality gives

$ >=-log(sum_(x in S) p(x)(q(x))/(p(x))) $

#pause
$ =-log(sum_(x in S) q(x))>=-log 1=0. $

== ⭐ Equality conditions #D

Equality requires $q(x)/p(x)$ to be constant on $S$ and $q$ to put no mass outside $S$. Normalization then makes that constant one, so $q=p$.

== KL is asymmetric

Usually

$ D_"KL"(p||q) eq.not D_"KL"(q||p). $

#pause
Example:

$ p=(0.9,0.1), quad q=(0.5,0.5). $

#pause
$ D_"KL"(p||q) approx 0.531 " bits", $
$ D_"KL"(q||p) approx 0.737 " bits". $

#pause
Swapping the arguments changes which distribution supplies the average.

== KL does not obey the triangle inequality

A metric distance $d$ must be symmetric and satisfy

$ d(p,r)<=d(p,q)+d(q,r). $

#pause
KL is asymmetric and does not generally satisfy this triangle inequality.

#pause
Therefore call it a *divergence*, not a distance.

== Missing support gives infinite cost

If $p(x)>0$ but $q(x)=0$, then

$ -log q(x)=infinity $

and

$ D_"KL"(p||q)=infinity. $

#pause
A model that declares a possible observation impossible receives infinite ideal log loss.

#pause
This is one reason probability smoothing matters.

== Direction matters under approximation #OPT

#fig("/lecture24/figures/kl_directions.svg", w: 77%)

#pause
When a single Gaussian approximates a bimodal target, the two KL directions penalize different failures. This is a property of the approximation problem, not a general synonym for “wide” versus “narrow.”

== Which direction appears in supervised learning?

Data are drawn from an unknown source $p$. A model supplies $q_theta$.

#pause
Expected log loss is

$ EE_(X tilde p)[-log q_theta(X)]=H(p,q_theta). $

#pause
Therefore the relevant mismatch is

$ D_"KL"(p||q_theta). $

#pause
The source distribution supplies the average.

// ═══════════════════ 4 · maximum likelihood ═══════════════════
= Cross-entropy is negative log-likelihood

== Dataset negative log-likelihood

For independent observations $x_1,dots,x_n$ under model $q_theta$,

$ L(theta)=product_(i=1)^n q_theta(x_i). $

#pause
Negative log-likelihood is

$ "NLL"(theta)=-sum_(i=1)^n log q_theta(x_i). $

#pause
Divide by $n$ to compare datasets of different sizes.

== Group observations by symbol

Let $n_x$ be the count of symbol $x$, and $hat(p)(x)=n_x/n$.

#pause
$ 1/n "NLL"(theta)=-1/n sum_x n_x log q_theta(x). $

#pause
$ =-sum_x hat(p)(x)log q_theta(x). $

#pause
#result[$1/n "NLL"(theta)=H(hat(p),q_theta)$.]

== The complete equivalence

$ arg min_theta H(hat(p),q_theta) $

#pause
$ =arg min_theta [H(hat(p))+D_"KL"(hat(p)||q_theta)] $

#pause
$ =arg min_theta D_"KL"(hat(p)||q_theta) $

#pause
$ =arg max_theta sum_i log q_theta(x_i). $

#pause
The empirical entropy is constant with respect to model parameters.

== Bernoulli example: eight heads, two tails

The empirical distribution is

$ hat(p)=(0.8,0.2). $

#pause
A Bernoulli model uses

$ q_theta=(theta,1-theta). $

#pause
Mean NLL, or binary cross-entropy, is

$ -0.8ln theta-0.2ln(1-theta). $

== The loss curve #V

#fig("/lecture24/figures/bernoulli_cross_entropy.svg", w: 64%)

#pause
The minimum occurs at $theta=0.8$, exactly the Bernoulli MLE from L14.

== Differentiate the Bernoulli loss

$ cal(L)(theta)=-0.8ln theta-0.2ln(1-theta). $

#pause
$ cal(L)'(theta)=-0.8/theta+0.2/(1-theta). $

#pause
Set the derivative to zero:

$ 0.2theta=0.8(1-theta) quad arrow.r.double quad theta=0.8. $

#pause
The second derivative is positive on $(0,1)$, so this is the global minimum.

// ═══════════════════ 5 · classification loss ═══════════════════
= Categorical cross-entropy

== One-hot target

For $K$ classes, let target vector $bold(y)$ be one-hot and prediction $bold(q)$ be a probability vector.

$ "CE"(bold(y),bold(q))=-sum_(k=1)^K y_k log q_k. $

#pause
If the correct class is $c$, then only $y_c=1$:

#result[$"CE"=-log q_c$.]

== Numerical example

Correct class: cat.

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([model], [$q("cat")$], [loss $-ln q("cat")$]),
  [A], [$0.90$], [$0.105$],
  [B], [$0.60$], [$0.511$],
  [C], [$0.10$], [$2.303$],
)

#pause
Confident correct predictions receive small loss; confident errors receive large loss.

== Soft labels

If the target is a distribution $bold(p)$ rather than one-hot,

$ "CE"(bold(p),bold(q))=-sum_k p_k log q_k. $

#pause
This occurs with:

+ label smoothing,
+ distillation from a teacher model,
+ probabilistic annotations,
+ mixture targets.

#pause
The same decomposition $H(p,q)=H(p)+D_"KL"(p||q)$ still applies.

== Softmax and numerical stability

Logits $z_k$ produce

$ q_k=(exp z_k)/(sum_j exp z_j). $

#pause
Compute log probabilities using log-sum-exp:

$ log q_k=z_k-"logsumexp"(bold(z)). $

#pause
This avoids overflow and underflow from L2. Practical libraries combine `log_softmax` with NLL loss.

== Gradient with respect to logits #OPT

For one-hot target $bold(y)$ and softmax probabilities $bold(q)$,

$ partial "CE"/partial z_k=q_k-y_k. $

#pause
The gradient has a direct interpretation:

+ lower the logits of over-predicted classes,
+ raise the logit of the under-predicted correct class.

#pause
Reverse-mode autodiff in L11 computes this at scale.

== Question: one-hot loss #Q

#mcq(
  [The correct class has predicted probability $0.25$. What is cross-entropy loss in bits?],
  [$0.25$],
  [$1$],
  [$2$],
  [$4$],
)

== Answer #A

#mcq-answer("C", [$2$ bits], [$-log_2 0.25=2$. The loss is the ideal code length assigned to the observed class.])

// ═══════════════════ 6 · perplexity ═══════════════════
= Exponentiate average log loss

== Definition

If average cross-entropy is measured in bits per token,

$ "perplexity"=2^(H_"bits"). $

#pause
If measured in nats per token,

$ "perplexity"=exp(H_"nats"). $

#pause
The base must match the logarithm used in the loss.

== Interpret uniform predictions

If a model predicts uniformly over $K$ choices,

$ H=log_2 K " bits". $

#pause
Then

$ "perplexity"=2^(log_2 K)=K. $

#pause
Perplexity recovers the number of equally plausible choices.

== Non-integer effective choices

If a language model has cross-entropy $1.5$ bits/token,

$ "perplexity"=2^(1.5) approx 2.828. $

#pause
This does not mean exactly 2.828 tokens are possible. It is the equivalent branching factor of a uniform source with the same average log loss.

== Sequence cross-entropy

For tokens $x_1,dots,x_n$,

$ 1/n sum_(t=1)^n -log_2 q(x_t|x_(<t)) $

is bits per token.

#pause
Its exponential is token-level perplexity.

#pause
Comparisons require the same tokenization and evaluation data; otherwise the unit “token” has changed.

== Perplexity is not accuracy

Accuracy checks whether the top predicted class is correct.

#pause
Perplexity uses the full probability assigned to every observed token.

#pause
Two models can have the same top-1 accuracy but different calibration and therefore different cross-entropy or perplexity.

// ═══════════════════ 7 · mutual information ═══════════════════
= How much does one variable reveal about another?

== Dependence as a KL divergence #OPT

Mutual information compares the joint distribution with the product of marginals:

$ I(X;Y)=D_"KL"(p(x,y)||p(x)p(y)). $

#pause
If $X$ and $Y$ are independent, $p(x,y)=p(x)p(y)$ and $I(X;Y)=0$.

#pause
If observing $Y$ changes predictions about $X$, mutual information is positive.

== Entropy reduction #OPT

The same quantity can be written

$ I(X;Y)=H(X)-H(X|Y). $

#pause
It is the average reduction in uncertainty about $X$ after observing $Y$.

#pause
Mutual information is symmetric:

$ I(X;Y)=I(Y;X). $

== Two binary examples #OPT

Let $X$ be a fair bit.

#pause
1. If $Y=X$, then $H(X)=1$ bit and $H(X|Y)=0$, so $I(X;Y)=1$ bit.

#pause
2. If $Y$ is an independent fair bit, then $H(X|Y)=H(X)=1$, so $I(X;Y)=0$.

#pause
One variable either reveals the bit completely or reveals nothing.

// ═══════════════════ 8 · close ═══════════════════
= The loss used by probabilistic models

== One identity joins three modules

$ H(p,q)=H(p)+D_"KL"(p||q). $

#pause
+ *Probability:* $p$ describes the data source.
+ *Information:* $-log q(x)$ is model-assigned code length.
+ *Estimation:* minimizing sample cross-entropy is maximizing likelihood.
+ *Optimization:* gradients change $q_theta$ to reduce mismatch.

#pause
The same calculation is read in four different ways.

== Summary

#result[Cross-entropy is the cost of coding data from $p$ with probabilities from $q$; KL is the excess cost.]

#pause
+ $H(p,q)=H(p)+D_"KL"(p||q)$.
+ KL is non-negative, asymmetric, and may be infinite.
+ Mean NLL is empirical cross-entropy.
+ Minimizing cross-entropy is maximum likelihood and minimizes forward KL to the empirical distribution.
+ One-hot categorical cross-entropy is $-log$ probability of the correct class.
+ Perplexity exponentiates average log loss.

== Practice

1. Compute $H(p,q)$ for $p=(1/2,1/2)$ and $q=(3/4,1/4)$ in bits.

#pause
2. Use the identity to obtain $D_"KL"(p||q)$.

#pause
3. A correct class receives probability $0.01$. Find its loss in nats and bits.

#pause
4. A model reports $2$ nats/token. Find perplexity.

== Practice answers #A

1. $H(p,q)=-1/2log_2(3/4)-1/2log_2(1/4) approx 1.208$ bits.

2. $H(p)=1$ bit, so $D_"KL"(p||q) approx 0.208$ bits.

3. $-ln 0.01 approx 4.605$ nats; $-log_2 0.01 approx 6.644$ bits.

4. $exp(2) approx 7.389$.

== Next: dependence across time

So far, most examples treated observations as independent. Sequences have structure:

$ p(x_1,dots,x_n)=p(x_1)product_(t=2)^n p(x_t|x_(<t)). $

#pause
A Markov model keeps only the previous state:

$ p(x_t|x_(<t)) approx p(x_t|x_(t-1)). $

#pause
Next lecture turns this assumption into a transition matrix.

#focus-slide[Minimize cross-entropy, maximize likelihood, and reduce coding cost: the same optimization.]
