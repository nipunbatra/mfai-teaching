// Learning Sequences and Course Finale — Lecture 26 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture26/L26-learning-sequences-finale.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Learning Sequence Models],
  subtitle: [Counts, smoothing, generation, and the course finale],
)

#title-slide()

// ═══════════════════ 1 · count transitions ═══════════════════
= Estimate the transition matrix from data

== An observed sequence

Suppose the two states A and B produce

`A A B A B B B A`

#pause
There are seven observed transitions:

$ A arrow.r A, A arrow.r B, B arrow.r A, A arrow.r B, $
$ B arrow.r B, B arrow.r B, B arrow.r A. $

#pause
Count each ordered pair.

== Transition count table

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 9pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([current], [next A], [next B], [row total]),
  [A], [$1$], [$2$], [$3$],
  [B], [$2$], [$2$], [$4$],
)

#pause
Rows correspond to current states; columns correspond to next states, matching L25.

== Normalize each row

For current state A:

$ hat(P)_(A A)=1/3, quad hat(P)_(A B)=2/3. $

#pause
For current state B:

$ hat(P)_(B A)=2/4=1/2, quad hat(P)_(B B)=2/4=1/2. $

#pause
Thus

#result[$hat(P)=mat(1/3,2/3;1/2,1/2)$.]

== Why row normalization is plausible

Given the current state $i$, the next state must be one of the columns $j$.

#pause
The empirical conditional frequency is

$ ("number of observed" i arrow.r j " transitions")/("number of transitions leaving" i). $

#pause
The calculation is also the maximum-likelihood estimator. We now derive it.

== Learning outcomes

By the end of this lecture you will be able to:

+ estimate a transition matrix from counts,
+ derive ⭐ $hat(P)_(i j)=n_(i j)/n_(i dot)$ using Lagrange multipliers,
+ apply Dirichlet smoothing and distinguish prediction from MAP,
+ build a character $n$-gram model,
+ score held-out text in bits per character and perplexity,
+ sample new sequences from a fitted model,
+ identify how every course module appears in this one artifact,
+ and locate the next mathematical steps toward ML and DL.

// ═══════════════════ 2 · MLE derivation ═══════════════════
= Maximum likelihood for one row

== Likelihood of transition counts

Let $n_(i j)$ count transitions from state $i$ to state $j$.

#pause
Ignoring the initial-state term, the path likelihood is

$ L(P)=product_i product_j P_(i j)^(n_(i j)). $

#pause
The log-likelihood is

$ ell(P)=sum_i sum_j n_(i j)log P_(i j). $

== Constraints live row by row

Every row is a probability distribution:

$ sum_j P_(i j)=1, quad P_(i j)>=0. $

#pause
The log-likelihood separates across rows. For one fixed current state $i$, solve

$ max_(P_(i 1),dots,P_(i K)) quad sum_j n_(i j)log P_(i j) $
$ "subject to" quad sum_j P_(i j)=1. $

== ⭐ Form the row Lagrangian #D

$ cal(L)_i=sum_j n_(i j)log P_(i j)-lambda_i(sum_j P_(i j)-1). $

#pause
Differentiate with respect to $P_(i j)$:

$ partial cal(L)_i/(partial P_(i j))=(n_(i j))/(P_(i j))-lambda_i=0. $

#pause
Therefore

$ P_(i j)=(n_(i j))/(lambda_i). $

== ⭐ Use the row-sum constraint #D

$ sum_j P_(i j)=sum_j (n_(i j))/(lambda_i)=1. $

#pause
Define the row total

$ n_(i dot)=sum_j n_(i j). $

#pause
Then

$ lambda_i=n_(i dot). $

== ⭐ The transition MLE #D

Substitute $lambda_i=n_(i dot)$:

#result[$hat(P)_(i j)=(n_(i j))/(n_(i dot))$.]

#pause
The maximum-likelihood transition probability is the empirical fraction of departures from $i$ that arrive at $j$.

== Return to the A/B sequence

$ n_(A dot)=3, quad n_(B dot)=4. $

#pause
$ hat(P)_(A A)=1/3, quad hat(P)_(A B)=2/3, $
$ hat(P)_(B A)=2/4, quad hat(P)_(B B)=2/4. $

#pause
The intuitive row-normalization calculation and the constrained MLE agree.

== The initial distribution

The full sequence likelihood also contains $p(X_1)$.

#pause
Possible choices:

+ estimate an initial-state distribution from many independent sequences,
+ fix the observed first state,
+ initialize from a known external distribution,
+ use the stationary distribution.

#pause
Transition estimation and initialization answer different questions.

== Code from a sequence

#codebox(size: 13pt)[```python
import numpy as np

path = np.array([0, 0, 1, 0, 1, 1, 1, 0])
counts = np.zeros((2, 2), dtype=int)

for current, nxt in zip(path[:-1], path[1:]):
    counts[current, nxt] += 1

P = counts / counts.sum(axis=1, keepdims=True)
print(counts)  # [[1 2], [2 2]]
print(P)       # [[.333 .667], [.5 .5]]
```]

== Question: estimate a row #Q

#mcq(
  [From state C, observed next-state counts are $(6,3,1)$. What is the MLE row?],
  [$(6,3,1)$],
  [$(0.6,0.3,0.1)$],
  [$(1/3,1/3,1/3)$],
  [$(7/13,4/13,2/13)$],
)

== Answer #A

#mcq-answer("B", [$(0.6,0.3,0.1)$], [Divide each count by the row total $6+3+1=10$.])

// ═══════════════════ 3 · character model ═══════════════════
= A bigram language model

== Characters are states

Let each character be a state: space plus `a` through `z`.

#pause
A character bigram model assumes

$ p(x_t|x_1,dots,x_(t-1)) approx p(x_t|x_(t-1)). $

#pause
The transition matrix has $27 times 27$ entries. Row `t` stores probabilities for the character following `t`.

== Counts from the course plan #V

#fig("/lecture26/figures/bigram_counts.svg", w: 64%)

#pause
The heatmap shows a subset of the alphabet. Common spelling and spacing patterns appear as large counts.

== One learned row #V

#fig("/lecture26/figures/next_after_t.svg", w: 65%)

#pause
The model learns from data that `t` is often followed by a space, `h`, or another common continuation.

== Build character counts

#codebox(size: 13pt)[```python
from collections import defaultdict, Counter

text = clean(open("lecture-plan-detailed.md").read())
counts = defaultdict(Counter)

for current, nxt in zip(text[:-1], text[1:]):
    counts[current][nxt] += 1

row_t = counts["t"]
total_t = sum(row_t.values())
p_t = {ch: n/total_t for ch, n in row_t.items()}
```]

== Bigram likelihood

For text $x_1,dots,x_T$,

$ p(x_1,dots,x_T)=p(x_1)product_(t=2)^T P_(x_(t-1),x_t). $

#pause
The training MLE simply counts every adjacent pair and normalizes rows.

#pause
No gradient descent is needed because the constrained optimum has a closed form.

== Higher-order $n$-grams

+ unigram: $p(x_t)$ — no context,
+ bigram: $p(x_t|x_(t-1))$ — one previous character,
+ trigram: $p(x_t|x_(t-2),x_(t-1))$ — two previous characters,
+ 4-gram: three previous characters.

#pause
A higher-order model is still first-order Markov after treating the whole context as the state.

== Parameter growth

With alphabet size $K$ and context length $m$, a dense table contains

$ K^m(K-1) $

free transition parameters.

#pause
For $K=27$:

+ bigram ($m=1$): $702$ free parameters,
+ trigram ($m=2$): $18,954$,
+ 4-gram ($m=3$): $511,758$.

#pause
Longer context needs much more data.

// ═══════════════════ 4 · smoothing ═══════════════════
= Unseen does not mean impossible

== The zero-probability problem

Suppose one row has counts

$ (8,2,0,0). $

#pause
MLE gives

$ (0.8,0.2,0,0). $

#pause
If either unseen transition occurs in test data, its log loss is infinite:

$ -log 0=infinity. $

== Add-one smoothing #V

#fig("/lecture26/figures/smoothing.svg", w: 64%)

#pause
Add-one prediction changes $(8,2,0,0)$ to

$ (9/14,3/14,1/14,1/14). $

== Dirichlet prior

For a transition row $bold(P)_i$, choose

$ bold(P)_i tilde "Dirichlet"(alpha_1,dots,alpha_K). $

#pause
After counts $n_(i 1),dots,n_(i K)$, conjugacy gives

$ bold(P)_i|"data" tilde "Dirichlet"(n_(i 1)+alpha_1,dots,n_(i K)+alpha_K). $

#pause
This is the multinomial analogue of L15's Beta–Bernoulli update.

== Posterior predictive probabilities

The probability of the next transition, averaged over the posterior, is

#result[$P(X_"next"=j|X_"current"=i,"data")=(n_(i j)+alpha_j)/(n_(i dot)+sum_k alpha_k)$.]

#pause
For symmetric $alpha_j=alpha$, this is add-$alpha$ smoothing.

== Predictive mean is not always MAP

For $alpha_j>1$, the row-wise MAP is

$ (n_(i j)+alpha_j-1)/(n_(i dot)+sum_k alpha_k-K). $

#pause
The posterior predictive mean uses $n_(i j)+alpha_j$ instead.

#pause
“Add-one smoothing” usually refers to predictive probabilities under a uniform Dirichlet prior, not the MAP for that prior.

== Choose the smoothing strength

Large $alpha$ pulls rows strongly toward the prior mean.

#pause
Small $alpha$ stays closer to counts but may assign extremely small probabilities to unseen transitions.

#pause
Choose $alpha$ on validation data using held-out log loss, not training likelihood alone.

// ═══════════════════ 5 · evaluate ═══════════════════
= Test on text not used for counting

== Held-out cross-entropy

For a test sequence of length $T$, bits per character is

$ "BPC"=-1/(T-m) sum_(t=m+1)^T log_2 q(x_t|x_(t-m),dots,x_(t-1)). $

#pause
Lower BPC means the model assigns higher probability to the observed continuation.

#pause
Character-level perplexity is

$ 2^"BPC". $

== Train/test split

For the course lecture plan:

+ clean to lowercase letters and spaces,
+ first 80%: estimate counts,
+ final 20%: evaluate,
+ use add-$0.1$ smoothing,
+ never update counts on the test segment.

#pause
Chronological splitting better reflects sequence prediction than random character shuffling.

== Context reduces held-out coding cost #V

#fig("/lecture26/figures/ngram_evaluation.svg", w: 68%)

#pause
Held-out BPC falls from $4.18$ for a unigram to $2.69$ for a 4-gram on this corpus.

== Convert BPC to effective choices

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([model], [BPC], [perplexity]),
  [unigram], [$4.181$], [$18.1$],
  [bigram], [$3.581$], [$12.0$],
  [trigram], [$2.991$], [$7.9$],
  [4-gram], [$2.687$], [$6.4$],
)

#pause
The longer context narrows the effective next-character choice set.

== More context is not automatically better

A higher-order model can have lower training loss but worse test loss because many contexts are rare.

#pause
Control the trade-off with:

+ smoothing,
+ backoff or interpolation with shorter contexts,
+ validation data,
+ parameter sharing.

#pause
Neural language models replace explicit tables with learned shared representations.

== Question: compare models fairly #Q

#mcq(
  [Which comparison makes character-level BPC meaningful?],
  [different test corpora],
  [same corpus and same character alphabet],
  [training loss for one model and test loss for another],
  [different log bases without conversion],
)

== Answer #A

#mcq-answer("B", [same corpus and same character alphabet], [Cross-entropy is meaningful only when the evaluated data and unit of a symbol are held fixed.])

// ═══════════════════ 6 · generate ═══════════════════
= Sample the fitted model

== Generation is repeated conditional sampling

Given current context $c_t$:

1. retrieve the smoothed row $q(dot|c_t)$,
2. sample the next character $x_(t+1)$,
3. append it to the output,
4. update the context,
5. repeat.

#pause
The probabilities used for scoring are the same probabilities used for generation.

== Bigram sampler

#codebox(size: 13pt)[```python
rng = np.random.default_rng(8)
context = "t"
output = [context]

for _ in range(300):
    probs = smoothed_row(context)
    nxt = rng.choice(alphabet, p=probs)
    output.append(nxt)
    context = nxt

print("".join(output))
```]

== What a bigram learns

Sample from the course-plan bigram model:

#codebox[```text
fulothecanaie s denivala chenen ch gers vex herde inghar
ne d heg pe tion ceadeximaty de itangsinim sata lly bin...
```]

#pause
Local letter pairs look plausible; words and sentences do not. One previous character is insufficient for language structure.

== A 4-gram sample

#codebox[```text
the exchand compostep newton derivation lives x ent inputed
bound float does ch dernoully ipynb invert convectors manythis...
```]

#pause
Short fragments from the course vocabulary appear, but long-range syntax remains weak.

== Swap in a larger corpus

The same notebook can replace the course plan with Tiny Shakespeare:

#codebox[```python
text = open("input.txt", encoding="utf-8").read()
model = fit_ngram(text, order=3, alpha=0.1)
print(sample(model, n=1000, seed=0))
```]

#pause
The algorithm does not change. Corpus size, alphabet, and context order change the fitted probabilities.

== Sampling temperature #OPT

Given probabilities $q_j$, temperature $tau>0$ defines

$ q_j^(tau) prop exp((log q_j)/tau). $

#pause
+ $tau<1$: sharper, more repetitive samples,
+ $tau=1$: original model,
+ $tau>1$: flatter, more varied, more errors.

#pause
Temperature changes generation; it does not improve held-out likelihood.

// ═══════════════════ 7 · the course in one model ═══════════════════
= Every module appears here

== Machine numbers and linear algebra

#table(
  columns: (0.7fr, 1.3fr, 1.6fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([lecture], [object], [use in the sequence model]),
  [L2], [floating point], [sum log probabilities; never multiply thousands of tiny terms],
  [L3], [vectors], [a state distribution is a probability vector],
  [L4], [matrices], [$bold(mu)_(t+1)=bold(mu)_t P$],
  [L5], [eigenvectors], [stationary distribution has eigenvalue one],
  [L6], [low rank], [large transition tables may admit compressed structure],
)

== Calculus and automatic differentiation

#table(
  columns: (0.7fr, 1.3fr, 1.6fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([lecture], [object], [use beyond the count model]),
  [L7], [local linearization], [optimize parameterized sequence models],
  [L8], [gradients], [direction of cross-entropy change],
  [L9], [Jacobians/Hessians], [sensitivity and curvature of model outputs],
  [L10], [finite/forward diff], [check small derivatives],
  [L11], [reverse mode], [backpropagate sequence loss through shared parameters],
)

== Probability and estimation

#table(
  columns: (0.7fr, 1.3fr, 1.6fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([lecture], [object], [use in the sequence model]),
  [L12–13], [distributions], [represent uncertainty and dependence],
  [L14], [MLE], [row-normalized transition counts],
  [L15], [MAP / conjugacy], [Dirichlet smoothing of each row],
)

#pause
Counting, normalization, and smoothing are estimation procedures—not ad hoc text tricks.

== Optimization

#table(
  columns: (0.7fr, 1.3fr, 1.6fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([lecture], [object], [use in the sequence model]),
  [L16], [convexity], [row log-likelihood has one global optimum],
  [L17–18], [first/second order], [fit models without a count closed form],
  [L19], [Lagrange multipliers], [derive row-normalized MLE],
  [L20], [KKT], [handle probability non-negativity and active zeros],
  [L21], [structured solvers], [recognize special training objectives],
)

== Information and sequences

#table(
  columns: (0.7fr, 1.3fr, 1.6fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([lecture], [object], [use in the sequence model]),
  [L22], [entropy], [irreducible uncertainty of a source],
  [L23], [coding], [probability becomes description length],
  [L24], [cross-entropy / KL], [held-out BPC and model mismatch],
  [L25], [Markov chains], [transition matrix, stationary distribution, PageRank],
  [L26], [learning sequences], [estimate, smooth, score, and sample],
)

== From $n$-grams to neural language models

An $n$-gram model stores a separate probability row for every discrete context.

#pause
A neural language model:

+ maps tokens and contexts to vectors,
+ shares parameters across related contexts,
+ produces logits and softmax probabilities,
+ minimizes the same cross-entropy,
+ samples from the same conditional factorization.

#pause
The probability and loss framework remains; the parameterization becomes much richer.

// ═══════════════════ 8 · close ═══════════════════
= What you can now build and check

== A compact sequence-model workflow

1. Define states or symbols and context length.
2. Split training, validation, and test sequences.
3. Count transitions or choose a parameterized model.
4. Normalize with appropriate smoothing.
5. Evaluate held-out cross-entropy in a clear unit.
6. Compare against simpler baselines.
7. Inspect generated samples and failure modes.
8. Record assumptions about stationarity and memory.

== Summary

#result[Transition-count MLE is row normalization; smoothing makes prediction finite; cross-entropy evaluates the fitted sequence model.]

#pause
+ $hat(P)_(i j)=n_(i j)/n_(i dot)$ follows from constrained maximum likelihood.
+ A Dirichlet prior yields add-$alpha$ posterior predictive probabilities.
+ Longer context can reduce BPC but increases parameter and data requirements.
+ Generation repeatedly samples the same conditional probabilities used for scoring.
+ The character model uses every mathematical module in the course.

== Practice

1. Estimate $P$ from transitions $A arrow.r A$ twice, $A arrow.r B$ once, $B arrow.r A$ three times, and $B arrow.r B$ once.

#pause
2. Apply add-one smoothing to the A row.

#pause
3. A test sequence receives probabilities $(1/2,1/4,1/8)$. Find total bits and average BPC.

#pause
4. Why can a 10-gram have worse held-out BPC than a 4-gram?

== Practice answers #A

1. $hat(P)=mat(2/3,1/3;3/4,1/4)$.

2. Counts $(2,1)$ become predictive probabilities $(3/5,2/5)$.

3. Total $=1+2+3=6$ bits; average $=2$ BPC.

4. Most long contexts are rare or unseen, so their estimates have high variance unless data sharing, smoothing, or backoff is strong enough.

== What follows this course

In a machine-learning course, these foundations support:

+ linear and logistic regression,
+ regularization and model selection,
+ support vector machines and kernels,
+ tree ensembles and probabilistic models.

#pause
In a deep-learning course, they support:

+ neural networks and backpropagation,
+ embeddings and attention,
+ convolutional and sequence models,
+ generative models and large language models.

== Final checklist

You can now ask of an AI method:

+ What is represented as a vector or matrix?
+ Which probability model produced the loss?
+ Which derivative is being computed, and in what shape?
+ What local approximation justifies the update?
+ Which constraint or numerical scale can break it?
+ What does its cross-entropy say in bits?
+ Which assumptions connect observations across time?

#focus-slide[You have fitted, evaluated, and sampled a language model; larger models change the parameterization, not the mathematical questions.]
