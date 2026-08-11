// Duality and KKT — Lecture 20 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture20/L20-duality-kkt.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Duality and KKT Conditions],
  subtitle: [Bounds, certificates, and inequality constraints],
)

#title-slide()

// ═══════════════════ 1 · inequalities ═══════════════════
= When a constraint may or may not bind

== How do you certify "no better solution exists"?

An optimizer returns a feasible $bold(x)$ with $f(bold(x))=10.003$. Is it optimal? Checking every other feasible point is impossible.

#pause
This lecture builds a second problem — the *dual* — whose value is a guaranteed lower bound on the best possible objective.

#pause
+ If the dual reports $10.000$, no feasible point can beat $10.000$.
+ The pair certifies: $bold(x)$ is within $0.003$ of optimal.

#pause
#notebox[Support vector machines are trained through exactly this dual problem. The bound needs inequality constraints done right — we start there.]

== One boundary, two outcomes

Compare these problems:

$ min_x (x-0.2)^2 quad "subject to" quad x<=1, $
$ min_x (x-2)^2 quad "subject to" quad x<=1. $

#pause
The same inequality appears in both. In the first problem it does nothing; in the second it determines the answer.

== Active and inactive constraints #V

#fig("/lecture20/figures/active_inactive.svg", w: 78%)

#pause
+ *Inactive:* $x^*<1$; there is positive slack.
+ *Active:* $x^*=1$; the boundary is reached.

== Put every inequality in one form

We will write a minimization problem as

$ min_bold(x) f(bold(x)) quad "subject to" quad g_i (bold(x))<=0, quad h_j (bold(x))=0. $

#pause
For $x<=1$, use

$ g(x)=x-1<=0. $

#pause
Its non-negative slack is

$ s=-g(x)=1-x>=0. $

== The multiplier must be non-negative

For $g(bold(x))<=0$, use

$ cal(L)(bold(x),lambda)=f(bold(x))+lambda g(bold(x)), quad lambda>=0. $

#pause
At any feasible $bold(x)$,

$ g(bold(x))<=0 quad arrow.r.double quad lambda g(bold(x))<=0. $

#pause
Therefore

$ cal(L)(bold(x),lambda)<=f(bold(x)). $

#pause
This simple inequality is the source of the dual lower bound.

== Prices for inequalities

An active constraint can carry a positive multiplier: relaxing it can improve the optimum.

#pause
An inactive constraint has no local effect, so its multiplier should be zero.

#pause
These two cases are summarized by

$ lambda>=0, quad -g(bold(x))>=0, quad lambda g(bold(x))=0. $

#pause
The product equation is *complementary slackness*.

== Complementary slackness #V

#fig("/lecture20/figures/complementary_slackness.svg", w: 62%)

#pause
Either the constraint has slack, or it has a price. Their product must vanish.

== Learning outcomes

By the end of this lecture you will be able to:

+ construct the Lagrange dual function,
+ prove ⭐ weak duality,
+ read a duality gap as an optimality certificate,
+ state and interpret the four KKT conditions,
+ verify KKT on an active and an inactive constraint,
+ and solve a small non-negative allocation problem from its KKT equations.

// ═══════════════════ 2 · the dual problem ═══════════════════
= A lower bound from the Lagrangian

== Start with the active example

$ p^*=min_x (x-2)^2 quad "subject to" quad x<=1. $

#pause
We already know the constrained answer:

$ x^*=1, quad p^*=1. $

#pause
Write $g(x)=x-1$ and

$ cal(L)(x,lambda)=(x-2)^2+lambda(x-1), quad lambda>=0. $

== Fix $lambda$ and minimize over $x$

For a chosen $lambda>=0$, define the dual function

$ q(lambda)=inf_x cal(L)(x,lambda). $

#pause
Differentiate with respect to $x$:

$ 2(x-2)+lambda=0 quad arrow.r.double quad x(lambda)=2-lambda/2. $

#pause
Substitute back:

$ q(lambda)=lambda-lambda^2/4. $

== Every $q(lambda)$ is a lower bound

Take any feasible $x$. Since $lambda>=0$ and $g(x)<=0$,

$ cal(L)(x,lambda)<=f(x). $

#pause
The infimum over all $x$ cannot be larger than the Lagrangian at that feasible point:

$ q(lambda)=inf_z cal(L)(z,lambda)<=cal(L)(x,lambda)<=f(x). $

#pause
In particular,

$ q(lambda)<=p^*. $

== ⭐ Weak duality #D

For any primal-feasible $bold(x)$ and dual-feasible $bold(lambda)>=bold(0)$,

$ q(bold(lambda))<=f(bold(x)). $

#pause
Taking the best primal point on the right gives

$ q(bold(lambda))<=p^*. $

#pause
Taking the best lower bound on the left gives

#result[$d^*=max_(bold(lambda)>=bold(0)) q(bold(lambda)) <= p^*$.]

#pause
This is weak duality. It holds even when the primal problem is non-convex.

== Choose the best lower bound #V

#fig("/lecture20/figures/dual_bound.svg", w: 63%)

#pause
Here $q(lambda)$ reaches $1$ at $lambda=2$. The dual lower bound meets the primal optimum.

== Solve the dual directly

$ q(lambda)=lambda-lambda^2/4, quad lambda>=0. $

#pause
$ q'(lambda)=1-lambda/2=0 quad arrow.r.double quad lambda^*=2. $

#pause
$ q(lambda^*)=2-2^2/4=1=p^*. $

#pause
The associated minimizer of the Lagrangian is

$ x(lambda^*)=2-(lambda^*)/2=1. $

== The duality gap

For a primal-feasible point $bold(x)$ and a dual-feasible multiplier $bold(lambda)$,

$ "gap"=f(bold(x))-q(bold(lambda))>=0. $

#pause
Example: choose $x=0.8$ and $lambda=1$.

$ f(0.8)=1.44, quad q(1)=0.75, quad "gap"=0.69. $

#pause
At $x=1,lambda=2$, the gap is zero. No feasible point can have a smaller objective than the matching lower bound.

== A certificate of optimality

Suppose an algorithm returns:

+ a feasible point with objective $10.003$,
+ and a dual-feasible point with bound $10.000$.

#pause
Then the unknown optimum lies in

$ 10.000<=p^*<=10.003. $

#pause
The solution is within $0.003$ of optimal, even if we do not know $p^*$ exactly.

== Question: interpret the bound #Q

#mcq(
  [For a minimization problem, a dual-feasible point gives…],
  [an upper bound on $p^*$],
  [a lower bound on $p^*$],
  [always the exact optimizer],
  [a learning rate],
)

== Answer #A

#mcq-answer("B", [a lower bound on $p^*$], [Weak duality gives $q(bold(lambda))<=p^*$ for every dual-feasible multiplier.])

// ═══════════════════ 3 · KKT ═══════════════════
= Four conditions at once

== The general Lagrangian

For

$ min_bold(x) f(bold(x)) $
$ "subject to" quad g_i (bold(x))<=0, quad h_j (bold(x))=0, $

define

$ cal(L)(bold(x),bold(lambda),bold(nu))=f(bold(x))+sum_i lambda_i g_i (bold(x))+sum_j nu_j h_j (bold(x)). $

#pause
Inequality multipliers satisfy $lambda_i>=0$. Equality multipliers $nu_j$ may have either sign.

== KKT 1 — primal feasibility

The candidate must obey the original problem:

$ g_i (bold(x)^*)<=0, quad h_j (bold(x)^*)=0. $

#pause
No amount of stationarity can rescue an infeasible point.

#pause
For the running example, primal feasibility is simply

$ x^*-1<=0. $

== KKT 2 — dual feasibility

Each inequality multiplier must obey

$ lambda_i^*>=0. $

#pause
This sign makes the dual function a valid lower bound for a minimization problem.

#pause
There is no sign restriction on equality multipliers.

== KKT 3 — stationarity

The Lagrangian must be stationary in the primal variables:

$ nabla f(bold(x)^*)+sum_i lambda_i^* nabla g_i (bold(x)^*)+sum_j nu_j^* nabla h_j (bold(x)^*)=bold(0). $

#pause
The objective gradient is balanced by normals from the active constraints.

#pause
This is L19's gradient-alignment equation with more possible normals.

== KKT 4 — complementary slackness

For every inequality,

$ lambda_i^* g_i (bold(x)^*)=0. $

#pause
Two permitted cases:

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*state*], [$g_i (bold(x)^*)$], [$lambda_i^*$]),
  [inactive], [$<0$], [$=0$],
  [active], [$=0$], [$>=0$],
)

== The four conditions on one slide

#table(
  columns: (1.1fr, 1.9fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  [*primal feasibility*], [$g_i (bold(x)^*)<=0, quad h_j (bold(x)^*)=0$],
  [*dual feasibility*], [$lambda_i^*>=0$],
  [*stationarity*], [$nabla_bold(x) cal(L)(bold(x)^*,bold(lambda)^*,bold(nu)^*)=bold(0)$],
  [*complementary slackness*], [$lambda_i^* g_i (bold(x)^*)=0$],
)

#pause
#notebox[Feasible here, feasible there, gradients balance, and inactive constraints cost zero.]

== Necessary or sufficient?

Under a suitable constraint qualification, a local optimum satisfies KKT: the conditions are *necessary*.

#pause
For a convex problem—convex $f$ and $g_i$, affine $h_j$—any KKT point is globally optimal: the conditions are *sufficient*.

#pause
Under Slater's condition (a strictly feasible point exists), strong duality also holds:

$ d^*=p^*. $

#pause
We use these results; a proof belongs in a convex optimization course.

// ═══════════════════ 4 · verify the running example ═══════════════════
= KKT by hand

== The problem and its Lagrangian

$ min_x (x-2)^2 quad "subject to" quad x-1<=0. $

#pause
$ cal(L)(x,lambda)=(x-2)^2+lambda(x-1). $

#pause
KKT requires

$ x-1<=0, quad lambda>=0, $
$ 2(x-2)+lambda=0, quad lambda(x-1)=0. $

== Case 1 — assume the constraint is inactive

If $x-1<0$, complementary slackness forces $lambda=0$.

#pause
Stationarity then gives

$ 2(x-2)=0 quad arrow.r.double quad x=2. $

#pause
But $x=2$ violates $x<=1$.

#pause
#alertbox[The inactive case is inconsistent with primal feasibility.]

== Case 2 — the constraint is active

Set $x-1=0$, so $x=1$.

#pause
Stationarity gives

$ 2(1-2)+lambda=0 quad arrow.r.double quad lambda=2. $

#pause
Dual feasibility holds because $lambda=2>=0$.

#pause
#result[$x^*=1, quad lambda^*=2, quad p^*=1$.]

== Verify all four receipts

#table(
  columns: (1.2fr, 1.6fr, 0.6fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*condition*], [*at $(x,lambda)=(1,2)$*], [*check*]),
  [primal feasibility], [$1-1=0<=0$], [✓],
  [dual feasibility], [$2>=0$], [✓],
  [stationarity], [$2(1-2)+2=0$], [✓],
  [complementary slackness], [$2(1-1)=0$], [✓],
)

#pause
The problem is convex, so this KKT point is the global optimum.

== The multiplier again measures sensitivity

Replace the boundary $x<=1$ by $x<=b$ near $b=1$.

$ V(b)=(b-2)^2 quad "for" b<2. $

#pause
$ V'(b)=2(b-2), quad V'(1)=-2. $

#pause
Our multiplier is $lambda^*=2$. With $g(x;b)=x-b$,

$ (dif V)/(dif b)=-lambda^*. $

#pause
Increasing the allowed boundary by $0.01$ decreases the optimum by about $0.02$.

== Check the inactive example

$ min_x (x-0.2)^2 quad "subject to" quad x-1<=0. $

#pause
The unconstrained minimizer $x^*=0.2$ is feasible with slack $0.8$.

#pause
Complementary slackness therefore gives $lambda^*=0$.

#pause
Stationarity checks:

$ 2(0.2-0.2)+0=0. $

#pause
Moving the boundary slightly has no effect on the optimum.

== L19's example, now with an inequality budget

L19 maximized $x y$ on the line $x+y=1$. Keep the budget but allow spending less of it:

$ max_(x,y) quad x y quad "subject to" quad x+y<=1, quad x>=0, quad y>=0. $

#pause
In this deck's standard form: minimize $f(x,y)=-x y$ with

$ g_1=x+y-1<=0, quad g_2=-x<=0, quad g_3=-y<=0. $

#pause
$ cal(L)=-x y+lambda(x+y-1)-mu_1 x-mu_2 y, quad lambda,mu_1,mu_2>=0. $

#pause
#notebox[L19 wrote $cal(L)=x y-lambda(x+y-1)$ for the maximization. Minimizing $-x y$ with $+lambda$ is the same bookkeeping; the numbers below agree.]

== Does the budget bind?

Stationarity:

$ -y+lambda-mu_1=0, quad -x+lambda-mu_2=0. $

#pause
Suppose $x>0$ and $y>0$. Complementary slackness gives $mu_1=mu_2=0$, so

$ x=y=lambda. $

#pause
+ Budget inactive: $lambda=0$ forces $x=y=0$, contradicting $x>0$.
+ Budget active: $x+y=1$ gives $x=y=1/2$ and $lambda=1/2$.

#pause
The product is $x y=1/4$, exactly L19's answer — the inequality binds, so the equality solution carries over.

== Verify KKT at $(1\/2,1\/2)$

#table(
  columns: (1.2fr, 1.7fr, 0.5fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*condition*], [*at $(x,y)=(1/2,1/2)$, $lambda=1/2$, $bold(mu)=bold(0)$*], [*check*]),
  [primal feasibility], [$1/2+1/2-1=0<=0$, $quad -1/2<=0$], [✓],
  [dual feasibility], [$lambda=1/2>=0$, $quad mu_1=mu_2=0$], [✓],
  [stationarity], [$(-1/2+1/2, -1/2+1/2)=(0,0)$], [✓],
  [complementary slackness], [$lambda dot 0=0$, $quad mu_i x_i=0$], [✓],
)

#pause
The budget's price is $lambda=1/2$, the same shadow price L19 computed: raising the budget by $delta$ raises the best product by about $delta\/2$.

#pause
#notebox[$(0,0)$ also satisfies KKT with every multiplier zero — $-x y$ is not convex, so KKT points are candidates. Compare values: $1\/4$ beats $0$.]

== Question: an inactive constraint #Q

#mcq(
  [If $g_i (bold(x)^*)<0$, what must complementary slackness imply?],
  [$lambda_i^*<0$],
  [$lambda_i^*=0$],
  [$lambda_i^*=1$],
  [$nabla f=bold(0)$ always],
)

== Answer #A

#mcq-answer("B", [$lambda_i^*=0$], [The product $lambda_i^* g_i (bold(x)^*)$ can vanish with nonzero slack only when its multiplier is zero.])

// ═══════════════════ 5 · allocation ═══════════════════
= A non-negative allocation problem

== Project preferred amounts onto a budget

Three channels prefer amounts

$ bold(a)=(3,1,0.5). $

Choose a non-negative allocation of total $B=2.5$ that stays closest to these preferences:

$ min_bold(x) quad 1/2 sum_(i=1)^3 (x_i-a_i)^2 $
$ "subject to" quad sum_i x_i=2.5, quad x_i>=0. $

#pause
This is Euclidean projection onto a simplex-like set.

== Write the inequalities in standard form

Use $-x_i<=0$ with multipliers $mu_i>=0$, and multiplier $nu$ for the equality:

$ cal(L)=1/2 sum_i (x_i-a_i)^2 + nu(sum_i x_i-B)-sum_i mu_i x_i. $

#pause
Stationarity in coordinate $i$ gives

$ x_i-a_i+nu-mu_i=0. $

== Positive coordinates

If $x_i>0$, the inequality $-x_i<=0$ is inactive.

#pause
Complementary slackness gives $mu_i=0$, so

$ x_i=a_i-nu. $

#pause
Every positive coordinate is reduced by the same threshold $nu$.

== Zero coordinates

If $x_i=0$, stationarity gives

$ mu_i=nu-a_i. $

#pause
Dual feasibility $mu_i>=0$ requires $nu>=a_i$.

#pause
Combining the two cases:

#result[$x_i^*=max(a_i-nu,0)$.]

== Determine the threshold

We still need $sum_i x_i=2.5$:

$ max(3-nu,0)+max(1-nu,0)+max(0.5-nu,0)=2.5. $

#pause
Try $0.5<nu<1$. Then only the first two coordinates are positive:

$ (3-nu)+(1-nu)=2.5. $

#pause
$ nu=0.75. $

== The allocation #V

#fig("/lecture20/figures/water_filling.svg", w: 56%)

#pause
$bold(x)^*=(2.25,0.25,0)$ — the third coordinate is clipped at zero because its preferred amount lies below the common threshold.

== Verify KKT numerically

For $nu=0.75$,

$ bold(mu)=max(nu bold(1)-bold(a),bold(0))=(0,0,0.25). $

#pause
+ primal: $bold(x)>=0$ and $sum_i x_i=2.5$,
+ dual: $bold(mu)>=0$,
+ stationarity: $bold(x)-bold(a)+nu bold(1)-bold(mu)=bold(0)$,
+ complementary slackness: $mu_i x_i=0$ for each $i$.

== The computation is a threshold search

#codebox[```python
import numpy as np

a = np.array([3.0, 1.0, 0.5])
B = 2.5
lo, hi = 0.0, a.max()

for _ in range(60):
    nu = (lo + hi)/2
    if np.maximum(a-nu, 0).sum() > B:
        lo = nu
    else:
        hi = nu

x = np.maximum(a-hi, 0)
print(x, x.sum())  # [2.25 0.25 0.  ] 2.5
```]

// ═══════════════════ 6 · strong duality and use ═══════════════════
= When the certificate is exact

== Weak and strong duality

Weak duality always gives

$ d^*<=p^*. $

#pause
Strong duality means the best bound is exact:

$ d^*=p^*. $

#pause
For convex problems satisfying a constraint qualification such as Slater's condition, strong duality holds.

#pause
#notebox[Convexity turns KKT from a candidate test into a global optimality certificate.]

== Slater's condition, operationally

For a convex problem, look for a point satisfying

$ g_i (bold(x))<0 quad "for every inequality," $
$ h_j (bold(x))=0 quad "for every equality." $

#pause
Such a point is *strictly feasible*.

#pause
In the running problem $x<=1$, the point $x=0$ is strictly feasible. Strong duality is therefore unsurprising.

== Why solve a dual problem?

The dual can be useful because it may:

+ have fewer variables,
+ expose separable subproblems,
+ provide a numerical lower bound at every iteration,
+ give interpretable prices,
+ or reveal structure hidden in the primal.

#pause
Support vector machines are a standard ML example: data points receive dual weights, and only support vectors keep nonzero weights.

== A practical stopping rule

Many solvers track three quantities:

+ primal residual: violation of original constraints,
+ dual residual: violation of stationarity,
+ duality gap: distance between primal value and dual bound.

#pause
Small residuals plus a small gap give a checkable stopping criterion.

== Common sign errors

Before doing any algebra, record:

#pause
1. Is this a minimization or maximization problem?
#pause
2. Did we write inequalities as $g_i<=0$ or $g_i>=0$?
#pause
3. Which sign appears in the Lagrangian?
#pause
4. Which multipliers must be non-negative?

#pause
#alertbox[Do not memorize a multiplier sign separately from the chosen standard form.]

== Summary

#result[The dual gives a bound; KKT explains when the bound and the feasible solution meet.]

#pause
+ Weak duality: $d^*<=p^*$ for minimization.
+ A zero duality gap certifies global optimality.
+ KKT combines primal feasibility, dual feasibility, stationarity, and complementary slackness.
+ An inactive inequality has zero multiplier; an active one may have a positive price.
+ For convex problems under regularity, KKT is sufficient and strong duality holds.

== Practice

1. Solve $min_x (x+1)^2$ subject to $x>=0$ using KKT.

#pause
2. For $min_x x^2$ subject to $x<=2$, state whether the constraint is active and find its multiplier.

#pause
3. For the running problem, evaluate $q(0)$, $q(1)$, $q(2)$, and $q(3)$.

#pause
4. Project $bold(a)=(2,1)$ onto $x_1+x_2=1$, $x_i>=0$.

== Practice answers #A

1. Write $-x<=0$. The solution is $x^*=0$ with $lambda^*=2$.

2. $x^*=0$ is strictly feasible; the constraint is inactive and $lambda^*=0$.

3. $q(lambda)=lambda-lambda^2/4$, so the values are $0,0.75,1,0.75$.

4. $nu=1$, hence $bold(x)^*=max(bold(a)-nu bold(1),bold(0))=(1,0)$.

== Next: recognizable problem classes

Lagrange multipliers and KKT are general conditions. Some objectives and constraints have enough structure for highly reliable solvers.

#pause
Next lecture:

+ linear objective + linear constraints $arrow.r$ linear program,
+ quadratic objective + linear constraints $arrow.r$ quadratic program.

#pause
#notebox[*Read before L21* — Boyd & Vandenberghe Ch 5 (guided: 5.1–5.5, pictures and boxed statements first); MML §7.2.]

#focus-slide[
  KKT is Lagrange with inequalities and receipts.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Linear & Quadratic Programming* — the two constrained shapes with industrial-strength solvers.
]
