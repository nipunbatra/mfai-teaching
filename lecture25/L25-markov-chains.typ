// Markov Chains — Lecture 25 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture25/L25-markov-chains.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Markov Chains],
  subtitle: [Sequences, transition matrices, and long-run behaviour],
)

#title-slide()

// ═══════════════════ 1 · sequence model ═══════════════════
= Tomorrow depends on today

== A two-state weather model

Each day is either sunny $S$ or rainy $R$.

#pause
Assume:

$ P(S_(t+1)|S_t)=0.8, quad P(R_(t+1)|S_t)=0.2, $
$ P(S_(t+1)|R_t)=0.4, quad P(R_(t+1)|R_t)=0.6. $

#pause
Tomorrow is random, but today's state changes its probabilities.

== State diagram #V

#fig("/lecture25/figures/weather_chain.svg", w: 66%)

#pause
Outgoing probabilities from each state sum to one.

== The Markov assumption

A first-order Markov model assumes

#result[$P(X_(t+1)|X_t,X_(t-1),dots,X_1)=P(X_(t+1)|X_t)$.]

#pause
Given the present state, earlier states add no further information about the next state.

#pause
This is a modeling assumption, not a universal property of sequences.

== What the assumption removes

Without a Markov assumption, predicting $X_(t+1)$ may require a table for every possible history.

#pause
For $K$ states, a length-$t$ history has $K^t$ possibilities.

#pause
A first-order homogeneous chain uses only a $K times K$ transition table, shared across time.

#pause
The reduction is substantial; the lost memory may also matter.

== Examples and non-examples

Reasonable first approximations:

+ weather category tomorrow from weather today,
+ a board-game position after one move,
+ user state after one interaction,
+ next character from the current character.

#pause
Often insufficient:

+ seasonal weather without the month,
+ language when only one previous character is kept,
+ medical history summarized by an incomplete state.

== Learning outcomes

By the end of this lecture you will be able to:

+ factor a sequence probability using the Markov assumption,
+ represent a chain by a row-stochastic transition matrix,
+ propagate a state distribution by matrix multiplication,
+ compute multi-step transition probabilities,
+ derive ⭐ a stationary distribution by equations and eigenvectors,
+ distinguish irreducibility from aperiodicity,
+ run power iteration,
+ and derive PageRank as a damped random walk on a web graph.

// ═══════════════════ 2 · transition matrix ═══════════════════
= The entire chain is a matrix

== Fix an orientation

Order states as $(S,R)$. Put current state in rows and next state in columns:

$ P=mat(0.8,0.2;0.4,0.6). $

#pause
Entry $P_(i j)$ means

$ P_(i j)=P(X_(t+1)=j|X_t=i). $

#pause
Every row sums to one and every entry is non-negative. Such a matrix is *row-stochastic*.

== Read the matrix both ways

$ P=mat(0.8,0.2;0.4,0.6). $

#pause
Row view:

+ if today is sunny, tomorrow is $(0.8,0.2)$ over $(S,R)$,
+ if today is rainy, tomorrow is $(0.4,0.6)$.

#pause
Column view:

+ the sunny column collects all ways to arrive at sunny,
+ the rainy column collects all ways to arrive at rainy.

== A distribution over states

Let

$ bold(mu)_t=(P(X_t=S),P(X_t=R)) $

be a row vector.

#pause
The next distribution is

#result[$bold(mu)_(t+1)=bold(mu)_t P$.]

#pause
This is the law of total probability written as matrix multiplication.

== One-day prediction

Suppose today is certainly sunny:

$ bold(mu)_0=(1,0). $

#pause
$ bold(mu)_1=(1,0)mat(0.8,0.2;0.4,0.6)=(0.8,0.2). $

#pause
The first row of $P$ is the next-day distribution.

== Two-day prediction

$ bold(mu)_2=bold(mu)_1P=(0.8,0.2)mat(0.8,0.2;0.4,0.6). $

#pause
$ P(X_2=S)=0.8(0.8)+0.2(0.4)=0.72. $

$ P(X_2=R)=0.8(0.2)+0.2(0.6)=0.28. $

#pause
Thus

$ bold(mu)_2=(0.72,0.28). $

== $k$ steps use a matrix power

Repeated substitution gives

$ bold(mu)_k=bold(mu)_0 P^k. $

#pause
Entry $(P^k)_(i j)$ is the probability of moving from state $i$ to state $j$ in $k$ steps.

#pause
This is L4's repeated linear map, now applied to probabilities.

== Verify in NumPy

#codebox[```python
import numpy as np

P = np.array([[.8, .2],
              [.4, .6]])
mu0 = np.array([1., 0.])

print(mu0 @ P)                         # [0.8  0.2 ]
print(mu0 @ np.linalg.matrix_power(P, 2))  # [0.72 0.28]
print(P.sum(axis=1))                   # [1. 1.]
```]

== Question: one matrix step #Q

#mcq(
  [If $bold(mu)_t=(0.25,0.75)$, what is $P(X_(t+1)=S)$?],
  [$0.25$],
  [$0.40$],
  [$0.50$],
  [$0.75$],
)

== Answer #A

#mcq-answer("C", [$0.50$], [$0.25(0.8)+0.75(0.4)=0.20+0.30=0.50$.])

// ═══════════════════ 3 · sequence probabilities ═══════════════════
= Probability of an entire path

== Chain rule first

For any sequence,

$ p(x_1,dots,x_T)=p(x_1)product_(t=2)^T p(x_t|x_1,dots,x_(t-1)). $

#pause
This is the probability chain rule; no Markov assumption yet.

== Apply the Markov assumption

Replace the full history by the previous state:

#result[$p(x_1,dots,x_T)=p(x_1)product_(t=2)^T P_(x_(t-1),x_t)$.]

#pause
A path probability is the initial-state probability times the transition probabilities along its arrows.

== Worked path: $S,R,R,S$

Let $P(X_1=S)=0.6$ and $P(X_1=R)=0.4$.

#pause
$ P(S,R,R,S)=P(S)P(R|S)P(R|R)P(S|R). $

#pause
$ =0.6 times 0.2 times 0.6 times 0.4. $

#pause
$ =0.0288. $

== Compare two paths

$ P(S,S,S,S)=0.6(0.8)^3=0.3072. $

#pause
$ P(S,R,R,S)=0.0288. $

#pause
Both paths are possible, but the first follows high-probability transitions throughout.

#pause
The log path probability is a sum, exactly as in L14 and L24.

== Log probability of a path

$ log p(x_1,dots,x_T)=log p(x_1)+sum_(t=2)^T log P_(x_(t-1),x_t). $

#pause
Negative average log probability is a cross-entropy rate for sequences.

#pause
This is how L26 will score a fitted character model.

== Sample a chain

#codebox(size: 13pt)[```python
rng = np.random.default_rng(4)
state = 0                      # 0=sunny, 1=rainy
path = [state]

for _ in range(12):
    state = rng.choice(2, p=P[state])
    path.append(state)

print(path)
```]

#pause
Sampling uses the row corresponding to the current state.

// ═══════════════════ 4 · stationary distribution ═══════════════════
= A distribution unchanged by one step

== Definition

A stationary distribution $bold(pi)$ satisfies

$ bold(pi)=bold(pi)P, quad sum_i pi_i=1, quad pi_i>=0. $

#pause
If $X_t$ has distribution $bold(pi)$, then $X_(t+1)$ has the same distribution.

#pause
Stationary does not mean the state stops changing; it means the distribution over states is unchanged.

== ⭐ Solve the weather equations #D

Let $bold(pi)=(pi_S,pi_R)$.

$ (pi_S,pi_R)=(pi_S,pi_R)mat(0.8,0.2;0.4,0.6). $

#pause
The sunny coordinate gives

$ pi_S=0.8pi_S+0.4pi_R. $

#pause
Therefore

$ 0.2pi_S=0.4pi_R quad arrow.r.double quad pi_S=2pi_R. $

== ⭐ Normalize #D

$ pi_S+pi_R=1 $

and $pi_S=2pi_R$ imply

$ 3pi_R=1. $

#pause
#result[$bold(pi)=(2/3,1/3)$.]

#pause
In the long-run balance, sunny is twice as likely as rainy.

== Convergence from a sunny start #V

#fig("/lecture25/figures/weather_convergence.svg", w: 65%)

#pause
The initial condition is gradually forgotten, and the distribution approaches $(2/3,1/3)$.

== Stationarity is an eigenvector equation

$ bold(pi)=bold(pi)P $

means $bold(pi)$ is a *left eigenvector* of $P$ with eigenvalue $1$.

#pause
Equivalently,

$ P^top bold(pi)^top=bold(pi)^top. $

#pause
This is the eigenvalue-one direction from L5, normalized to sum to one.

== Why eigenvalue one exists

Since every row of $P$ sums to one,

$ P bold(1)=bold(1). $

#pause
Thus $bold(1)$ is a right eigenvector with eigenvalue one.

#pause
The stationary distribution is the corresponding left eigenvector. Under suitable conditions it is unique.

== Power iteration

Choose any initial distribution and repeat

$ bold(mu)_(t+1)=bold(mu)_t P. $

#pause
This is L5's power iteration, now constrained to probability vectors.

#codebox[```python
mu = np.array([1., 0.])
for _ in range(30):
    mu = mu @ P
print(mu)  # [0.66666667 0.33333333]
```]

== The second eigenvalue sets the rate #OPT

The weather matrix has eigenvalues $1$ and $0.4$.

#pause
The non-stationary component shrinks by a factor of $0.4$ per step:

$ bold(mu)_t-bold(pi) prop 0.4^t. $

#pause
An eigenvalue closer to one would mean slower mixing.

// ═══════════════════ 5 · when convergence fails ═══════════════════
= Connectivity and periodicity

== Irreducibility

A finite chain is *irreducible* if every state can eventually reach every other state with positive probability.

#pause
Then the chain has one communicating class and a unique stationary distribution.

#pause
If the chain is reducible, different closed groups can retain memory of the starting group.

== Aperiodicity

A state has period $d$ if returns to it can occur only at times sharing greatest common divisor $d$.

#pause
A chain is aperiodic if its states have period one.

#pause
In an irreducible chain, one positive self-loop is enough to make every state aperiodic.

== Two failure patterns #V

#fig("/lecture25/figures/chain_failures.svg", w: 78%)

#pause
Irreducibility concerns reachability. Aperiodicity concerns synchronized cycles.

== Unique stationary does not imply convergence

For the two-state alternator

$ P=mat(0,1;1,0), $

the stationary distribution is $(1/2,1/2)$.

#pause
But starting from $(1,0)$ gives

$ (1,0),(0,1),(1,0),(0,1),dots $

#pause
The chain is irreducible but periodic, so the distributions do not converge.

== A standard convergence statement

For a finite, irreducible, aperiodic Markov chain:

+ a unique stationary distribution exists,
+ every initial distribution converges to it,
+ long-run state frequencies converge to stationary probabilities.

#pause
This is the operating theorem behind PageRank and many sampling algorithms.

// ═══════════════════ 6 · PageRank ═══════════════════
= A random walk on the web

== Links define transitions

Imagine a user on a web page choosing one outgoing link uniformly.

#pause
For four pages:

+ A links to B and C,
+ B links to C,
+ C links to A,
+ D links to C.

#pause
The undamped transition matrix is

$ S=mat(0,1/2,1/2,0;0,0,1,0;1,0,0,0;0,0,1,0). $

== The random-surfer problem

The link chain may be reducible, periodic, or contain dangling pages with no outgoing links.

#pause
PageRank adds teleportation:

$ P=alpha S+(1-alpha)1/4 bold(1)bold(1)^top. $

#pause
With probability $alpha$, follow a link. With probability $1-alpha$, jump uniformly to any page.

== Why teleportation helps

For $0<alpha<1$, every entry of $P$ is positive.

#pause
Therefore:

+ every page can reach every other page in one step,
+ every page has a positive self-transition probability,
+ the chain is irreducible and aperiodic.

#pause
The stationary distribution is unique and power iteration converges.

== Four-page PageRank #V

#fig("/lecture25/figures/pagerank.svg", w: 79%)

#pause
With $alpha=0.85$, the stationary scores are approximately

$ (0.373,0.196,0.394,0.038) $

for $(A,B,C,D)$.

== Interpret the scores

Page C receives links from A, B, and D, so its stationary probability is largest.

#pause
Page A receives from C, which itself has high rank.

#pause
Page D has no incoming web links; it receives only teleportation mass.

#pause
PageRank values are visitation probabilities under a specific random-surfer model, not a universal measure of page quality.

== Power iteration for PageRank

#codebox(size: 13pt)[```python
S = np.array([[0, .5, .5, 0],
              [0,  0,  1, 0],
              [1,  0,  0, 0],
              [0,  0,  1, 0]], dtype=float)

alpha = 0.85
P = alpha*S + (1-alpha)*np.ones((4, 4))/4
r = np.ones(4)/4
for _ in range(100):
    r = r @ P
print(r)
```]

== Sparse computation at web scale #OPT

A web transition matrix is enormous but sparse: each page links to only a small fraction of all pages.

#pause
Power iteration needs matrix–vector products, not a dense matrix inverse.

#pause
The same principle appeared throughout the course:

+ solve systems without forming inverses,
+ apply structured matrices without storing zeros,
+ exploit repeated local operations.

// ═══════════════════ 7 · model scope ═══════════════════
= What the state must remember

== A higher-order chain

If the last two states matter, define

$ P(X_(t+1)|X_t,X_(t-1)). $

#pause
This second-order model can be converted to a first-order chain by defining an expanded state

$ Z_t=(X_(t-1),X_t). $

#pause
The Markov property depends on how the state is defined.

== State design is modeling

A useful state should contain enough information to predict the next step for the intended task.

#pause
Too small:

+ important dependence remains in the history.

#pause
Too large:

+ transition tables become sparse,
+ more parameters must be estimated,
+ data requirements grow quickly.

== Time-homogeneous versus changing chains

We assumed one matrix $P$ for every time step:

$ P(X_(t+1)=j|X_t=i)=P_(i j). $

#pause
Seasonal or intervention-driven systems may require $P_t$ that changes with time.

#pause
The matrix framework remains useful, but stationarity may no longer be the right question.

// ═══════════════════ 8 · close ═══════════════════
= A sequence becomes linear algebra

== Summary

#result[A finite Markov chain is a stochastic matrix; its long-run distribution is an eigenvector with eigenvalue one.]

#pause
+ The Markov assumption keeps only the current state when predicting the next.
+ Sequence probabilities multiply transition probabilities along a path.
+ State distributions evolve as $bold(mu)_(t+1)=bold(mu)_t P$.
+ A stationary distribution satisfies $bold(pi)=bold(pi)P$.
+ Irreducibility gives one communicating class; aperiodicity prevents synchronized oscillation.
+ PageRank is a damped random walk solved by power iteration.

== Practice

1. For $P=mat(0.9,0.1;0.3,0.7)$, find the stationary distribution.

#pause
2. Starting from state 1, compute the probability of state 2 after two steps.

#pause
3. Find the probability of path $(1,1,2,2)$ if $P(X_1=1)=0.6$.

#pause
4. Explain whether $P=mat(0,1;1,0)$ is irreducible and aperiodic.

== Practice answers #A

1. Solve $0.1pi_1=0.3pi_2$ and normalize: $bold(pi)=(0.75,0.25)$.

2. $(P^2)_(12)=0.9(0.1)+0.1(0.7)=0.16$.

3. $0.6(0.9)(0.1)(0.7)=0.0378$.

4. It is irreducible because each state reaches the other; it is periodic with period two, so it is not aperiodic.

== Next: estimate and sample

So far $P$ was given. Data usually provide only observed transitions.

#pause
Next lecture:

+ estimate rows of $P$ by maximum likelihood,
+ smooth counts with a Dirichlet prior,
+ score held-out sequences by cross-entropy,
+ sample characters from a fitted model.

#focus-slide[Repeated transition-matrix multiplication turns local rules into long-run behaviour.]
