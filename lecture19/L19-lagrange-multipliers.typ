// Lagrange Multipliers — Lecture 19 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture19/L19-lagrange-multipliers.typ
//   typst compile --root . --input handout=true lecture19/L19-lagrange-multipliers.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Lagrange Multipliers],
  subtitle: [Optimizing while an equality must hold],
)

#title-slide()

// ═══════════════════ 1 · one constrained problem ═══════════════════
= A fixed budget

== Two quantities, one budget

Choose two non-negative quantities $x$ and $y$. Their product is the score, but the total is fixed:

$ max_(x,y) quad x y quad "subject to" quad x + y = 1, quad x >= 0, y >= 0. $

#pause
Some feasible choices:

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([$x$], [$y$], [$x y$]),
  [$0$], [$1$], [$0$],
  [$0.2$], [$0.8$], [$0.16$],
  [$0.5$], [$0.5$], [*$0.25$*],
  [$0.8$], [$0.2$], [$0.16$],
  [$1$], [$0$], [$0$],
)

#pause
The equal split looks best. We need a method that still works when substitution is not so easy.

== Where this form appears in ML

#table(
  columns: (1.1fr, 1.25fr, 1.25fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*choice*], [*objective*], [*equality*]),
  [probabilities $p_i$], [entropy or likelihood], [$sum_i p_i = 1$],
  [portfolio weights $w_i$], [predicted return], [$sum_i w_i = 1$],
  [model parameters $bold(theta)$], [training loss], [$A bold(theta) = bold(b)$],
  [resource allocation $x_i$], [utility], [$sum_i x_i = B$],
)

#pause
#notebox[The objective says what we prefer. The constraint says what is allowed.]

== Unconstrained and constrained stationarity

Without a constraint, a differentiable interior optimum satisfies

$ nabla f(bold(x)^*) = bold(0). $

#pause
With $h(bold(x)) = 0$, the optimum may have $nabla f(bold(x)^*) eq.not bold(0)$.

#pause
For $f(x,y)=x y$ at $(0.5,0.5)$,

$ nabla f = mat(y; x) = mat(0.5; 0.5) eq.not mat(0; 0). $

#pause
The gradient points uphill, but that direction leaves the line $x+y=1$.

== The feasible set is part of the problem #V

#fig("/lecture19/figures/budget_contours.svg", w: 60%)

#pause
The orange line contains every feasible point. The best contour that touches it does so at $(0.5,0.5)$.

== What should a general method return?

For the budget problem it should give:

#pause
+ the point $x^*=y^*=0.5$,
#pause
+ the value $f^*=0.25$,
#pause
+ a certificate that no small *feasible* move improves the objective,
#pause
+ and, if the budget changes, the rate at which the best value changes.

#pause
#result[One extra scalar, the Lagrange multiplier, supplies both stationarity and sensitivity.]

== Learning outcomes

By the end of this lecture you will be able to:

+ derive ⭐ $nabla f = lambda nabla h$ from feasible directions,
+ form a Lagrangian and solve its stationarity equations,
+ interpret $lambda^*$ as the local value of relaxing a constraint,
+ handle several equality constraints,
+ derive the maximum-entropy distribution on a finite set,
+ and distinguish a stationary candidate from a constrained optimum.

// ═══════════════════ 2 · geometry ═══════════════════
= Geometry of an equality constraint

== The constraint is a level set

Write the budget as

$ h(x,y) = x+y-1 = 0. $

#pause
The feasible set is the zero-level set of $h$.

#pause
Its gradient is

$ nabla h = mat(1;1). $

#pause
From L8, a gradient is normal to a level set. Therefore $nabla h$ points across the feasible line, not along it.

== Feasible directions are tangent directions

At a feasible point, let $bold(v)$ be a small direction that stays on the constraint to first order.

$ h(bold(x)+epsilon bold(v)) approx h(bold(x)) + epsilon nabla h(bold(x))^top bold(v). $

#pause
Since $h(bold(x))=0$, first-order feasibility requires

$ nabla h(bold(x))^top bold(v)=0. $

#pause
For $x+y=1$, one tangent direction is $bold(v)=(1,-1)$ because $(1,1)^top(1,-1)=0$.

== Stationarity only along the feasible set

At a constrained optimum, neither $bold(v)$ nor $-bold(v)$ may improve the objective to first order.

$ f(bold(x)+epsilon bold(v)) approx f(bold(x)) + epsilon nabla f(bold(x))^top bold(v). $

#pause
Hence every feasible tangent direction must satisfy

$ nabla f(bold(x)^*)^top bold(v)=0. $

#pause
So $nabla f(bold(x)^*)$ is also normal to the feasible set.

== ⭐ Two normals to the same curve #D

For one regular constraint, the normal space has one direction: $nabla h$.

#pause
We just found that $nabla f$ belongs to the same normal space.

#pause
Therefore some scalar $lambda$ satisfies

#result[$nabla f(bold(x)^*) = lambda^* nabla h(bold(x)^*)$.]

#pause
This is the Lagrange multiplier condition. “Regular” means $nabla h(bold(x)^*) eq.not bold(0)$.

== The running example at the optimum #V

#fig("/lecture19/figures/gradient_alignment.svg", w: 60%)

#pause
At $(0.5,0.5)$,

$ nabla f=(0.5,0.5), quad nabla h=(1,1), quad "so" quad lambda^*=0.5. $

== Check the tangent calculation directly

The feasible tangent direction is $bold(v)=(1,-1)$.

#pause
At the candidate $(0.5,0.5)$,

$ nabla f^top bold(v) = (0.5,0.5)^top(1,-1)=0. $

#pause
At the feasible point $(0.2,0.8)$,

$ nabla f^top bold(v) = (0.8,0.2)^top(1,-1)=0.6. $

#pause
Moving a little toward increasing $x$ and decreasing $y$ improves the product there.

== A one-variable check

The constraint gives $y=1-x$. Substitute it into the objective:

$ phi(x)=x(1-x)=x-x^2. $

#pause
$ phi'(x)=1-2x=0 quad arrow.r.double quad x=1/2, quad y=1/2. $

#pause
$ phi''(x)=-2<0 $, so this candidate is a maximum.

#pause
#notebox[Substitution is an excellent check. Lagrange multipliers become useful when there are many variables or several coupled constraints.]

== Question: which statement is necessary? #Q

#mcq(
  [At a regular equality-constrained optimum, which statement must hold?],
  [$nabla f=bold(0)$],
  [$nabla f$ is parallel to $nabla h$],
  [$f=h$],
  [$nabla f$ is tangent to $h=0$],
)

== Answer #A

#mcq-answer("B", [$nabla f$ is parallel to $nabla h$], [Both gradients are normal to the feasible level set at a regular constrained optimum.])

// ═══════════════════ 3 · the Lagrangian ═══════════════════
= The Lagrangian

== Put the condition into one function

For a maximization problem

$ max_bold(x) f(bold(x)) quad "subject to" quad h(bold(x))=0, $

define

$ cal(L)(bold(x),lambda)=f(bold(x))-lambda h(bold(x)). $

#pause
Stationarity with respect to both arguments gives

$ nabla_bold(x) cal(L)=nabla f-lambda nabla h=bold(0), $
$ partial cal(L) / partial lambda=-h(bold(x))=0. $

#pause
The last equation puts the solution back on the constraint.

== Sign conventions

Both definitions occur in books:

$ cal(L)=f-lambda h quad "or" quad cal(L)=f+lambda h. $

#pause
Changing the sign changes the reported multiplier, not the point $bold(x)^*$.

#pause
#notebox[In this lecture, maximization examples use $cal(L)=f-lambda h$. State the convention before interpreting the sign of $lambda$.]

== ⭐ Step 1 — write the running Lagrangian #D

$ f(x,y)=x y, quad h(x,y)=x+y-1. $

#pause
$ cal(L)(x,y,lambda)=x y-lambda(x+y-1). $

#pause
There are now three unknowns and three stationarity equations.

== ⭐ Step 2 — differentiate #D

$ partial cal(L) / partial x = y-lambda=0, $
$ partial cal(L) / partial y = x-lambda=0, $
$ partial cal(L) / partial lambda = -(x+y-1)=0. $

#pause
The first two equations say

$ x=y=lambda. $

== ⭐ Step 3 — enforce feasibility #D

Insert $x=y=lambda$ into $x+y=1$:

$ 2 lambda=1 quad arrow.r.double quad lambda^*=1/2. $

#pause
Therefore

#result[$x^*=1/2, quad y^*=1/2, quad f(x^*,y^*)=1/4$.]

#pause
The same calculation produced the optimizer and the multiplier.

== The equations as a nonlinear system

The stationary point solves

$ F(x,y,lambda)=mat(y-lambda; x-lambda; x+y-1)=bold(0). $

#pause
This viewpoint matters computationally: for harder problems we solve the coupled equations numerically.

#codebox[```python
from scipy.optimize import root

def F(z):
    x, y, lam = z
    return [y-lam, x-lam, x+y-1]

root(F, [0.3, 0.7, 0.0]).x
# array([0.5, 0.5, 0.5])
```]

== A minimum: closest point on a line

Find the point of smallest squared norm on $x+2y=4$:

$ min_(x,y) quad 1/2(x^2+y^2) quad "subject to" quad x+2y-4=0. $

#pause
Use the minimization convention

$ cal(L)=1/2(x^2+y^2)+lambda(x+2y-4). $

#pause
Stationarity gives

$ x+lambda=0, quad y+2lambda=0, quad x+2y=4. $

== Solve the closest-point example

$ x=-lambda, quad y=-2lambda. $

#pause
The constraint gives

$ -lambda-4lambda=4 quad arrow.r.double quad lambda=-4/5. $

#pause
#result[$(x^*,y^*)=(4/5,8/5)=(0.8,1.6)$.]

#pause
The point vector $(0.8,1.6)$ is parallel to the line normal $(1,2)$, as the geometry predicts.

== Verify the answer numerically

#codebox[```python
import numpy as np

x = np.array([0.8, 1.6])
print(x @ x / 2)              # 1.6
print(np.array([1, 2]) @ x)   # 4.0

# nearby feasible direction v=(2,-1)
for t in [-0.2, 0.0, 0.2]:
    z = x + t*np.array([2, -1])
    print(t, z @ z / 2)
# -0.2 1.7; 0.0 1.6; 0.2 1.7
```]

== Question: what does $partial_lambda cal(L)=0$ do? #Q

#mcq(
  [In an equality-constrained Lagrangian, the multiplier equation…],
  [minimizes the objective],
  [sets the learning rate],
  [enforces the original constraint],
  [tests positive definiteness],
)

== Answer #A

#mcq-answer("C", [enforces the original constraint], [Differentiating with respect to the multiplier returns the constraint equation, up to sign.])

// ═══════════════════ 4 · sensitivity ═══════════════════
= What the multiplier measures

== Let the budget change

Replace $x+y=1$ by $x+y=b$:

$ V(b)=max_(x,y) quad x y quad "subject to" quad x+y=b. $

#pause
The same stationarity equations give

$ x^*(b)=y^*(b)=b/2. $

#pause
Therefore the best attainable value is

$ V(b)=b^2/4. $

== The derivative of the optimal value #V

#fig("/lecture19/figures/shadow_price.svg", w: 62%)

#pause
At $b=1$,

$ V'(1)=1/2. $

The Lagrange calculation gave $lambda^*=1/2$ as well.

== Why the same number appears

Use

$ cal(L)(x,y,lambda;b)=x y-lambda(x+y-b). $

#pause
At the optimum, the derivatives through $x^*(b)$ and $y^*(b)$ vanish by stationarity. The remaining direct derivative is

$ (dif V)/(dif b) = partial cal(L)/partial b = lambda^*. $

#pause
#result[The multiplier is the local change in optimal value per unit relaxation of the constraint.]

== Read $lambda=0.5$ in units

Increase the budget from $1$ to $1.1$.

#pause
The first-order prediction is

$ V(1.1) approx V(1)+lambda^*(0.1)=0.25+0.5(0.1)=0.30. $

#pause
The exact value is

$ V(1.1)=1.1^2/4=0.3025. $

#pause
The difference $0.0025$ is a second-order effect.

== Shadow prices in ML

#table(
  columns: (1.2fr, 1fr, 1.4fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*constraint*], [*multiplier*], [*local reading*]),
  [memory $=B$], [$lambda_B$], [value of one more unit of memory],
  [expected cost $=C$], [$lambda_C$], [price of tightening the cost target],
  [normalization $sum p_i=1$], [$lambda$], [normalization offset in the optimum],
  [fairness statistic $=0$], [$lambda_F$], [loss paid for local enforcement],
)

#pause
The sign depends on how the constraint was written; the magnitude and units carry the interpretation.

== Question: predict a relaxed budget #Q

#mcq(
  [At $b=1$, $V=0.25$ and $lambda^*=0.5$. What is the first-order estimate of $V(1.02)$?],
  [$0.2501$],
  [$0.26$],
  [$0.27$],
  [$0.75$],
)

== Answer #A

#mcq-answer("B", [$0.26$], [$V(1+Delta b) approx V(1)+lambda^* Delta b=0.25+0.5(0.02)$.])

// ═══════════════════ 5 · probability simplex ═══════════════════
= A probability distribution with maximum entropy

== Three probabilities live on a triangle #V

#fig("/lecture19/figures/probability_simplex.svg", w: 58%)

#pause
Every distribution over three outcomes satisfies

$ p_1+p_2+p_3=1, quad p_i>=0. $

The equality removes one degree of freedom, so the feasible set is two-dimensional.

== Entropy rewards spread

For a distribution $bold(p)$ over $K$ outcomes,

$ H(bold(p))=-sum_(i=1)^K p_i log p_i. $

#pause
Compare three distributions for $K=3$ (natural logarithm):

#table(
  columns: (1.3fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([$bold(p)$], [$H(bold(p))$]),
  [$(1,0,0)$], [$0$],
  [$(0.8,0.1,0.1)$], [$0.639$],
  [$(1/3,1/3,1/3)$], [$log 3 approx 1.099$],
)

== Set up maximum entropy

$ max_bold(p) quad -sum_i p_i log p_i quad "subject to" quad sum_i p_i=1. $

#pause
For the moment assume $p_i>0$, so the optimum is inside the simplex. Define

$ cal(L)(bold(p),lambda)=-sum_i p_i log p_i-lambda(sum_i p_i-1). $

#pause
The inequality $p_i>=0$ will be handled systematically by KKT conditions in L20.

== Differentiate one coordinate

For each $i$,

$ partial cal(L)/partial p_i = -(log p_i+1)-lambda=0. $

#pause
Hence

$ log p_i=-1-lambda $

and therefore

$ p_i=exp(-1-lambda). $

#pause
The right-hand side is the same for every $i$.

== Enforce normalization

All $K$ probabilities are equal. Write each one as $c$.

$ sum_(i=1)^K p_i=K c=1 quad arrow.r.double quad c=1/K. $

#pause
#result[Maximum entropy on $K$ outcomes: $p_i^*=1/K$, with $H(bold(p)^*)=log K$.]

#pause
For $K=3$, the solution is the centre of the triangle in the previous figure.

== Check the entropy calculation in code

#codebox[```python
import numpy as np

def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return -(p*np.log(p)).sum()

for p in ([1, 0, 0], [.8, .1, .1], [1/3]*3):
    print(round(entropy(p), 3))
# 0.0
# 0.639
# 1.099
```]

== Why this example matters

The probability simplex appears whenever a model produces categorical probabilities.

#pause
+ *Softmax* maps logits to the interior of this simplex.
#pause
+ *Cross-entropy* compares a predicted distribution with a target.
#pause
+ *Maximum entropy* chooses the least concentrated distribution consistent with known constraints.
#pause
+ Extra moment constraints lead to exponential-family distributions.

#pause
This calculation will return in the information-theory module.

== Add one expected-value constraint #OPT

Suppose outcomes have values $a_i$, and we also require $sum_i p_i a_i=mu$.

$ cal(L)=-sum_i p_i log p_i-lambda(sum_i p_i-1)-beta(sum_i p_i a_i-mu). $

#pause
Stationarity gives

$ p_i=exp(-1-lambda-beta a_i) prop exp(-beta a_i). $

#pause
Normalization determines the missing constant. This is the basic exponential-family form.

// ═══════════════════ 6 · several equalities ═══════════════════
= Several equality constraints

== One multiplier per constraint

For

$ min_bold(x) f(bold(x)) quad "subject to" quad h_j(bold(x))=0, quad j=1,dots,m, $

define

$ cal(L)(bold(x),bold(lambda))=f(bold(x))+sum_(j=1)^m lambda_j h_j(bold(x)). $

#pause
Stationarity becomes

$ nabla f(bold(x)^*)+sum_j lambda_j^* nabla h_j(bold(x)^*)=bold(0), $
$ h_j(bold(x)^*)=0 quad "for every" j. $

== The gradient lies in a span

With one equality, $nabla f$ is parallel to one constraint gradient.

#pause
With several equalities,

$ -nabla f(bold(x)^*) in "span"{nabla h_1(bold(x)^*),dots,nabla h_m(bold(x)^*)}. $

#pause
The multipliers are the coordinates of $-nabla f$ in this normal space.

#pause
#notebox[The feasible tangent space is orthogonal to every constraint gradient. At stationarity, the objective has zero component in that tangent space.]

== Linear equalities in matrix form

For $A bold(x)=bold(b)$,

$ cal(L)(bold(x),bold(lambda))=f(bold(x))+bold(lambda)^top(A bold(x)-bold(b)). $

#pause
The stationarity equations are

$ nabla f(bold(x))+A^top bold(lambda)=bold(0), $
$ A bold(x)=bold(b). $

#pause
These equations are called the *KKT system* for an equality-constrained problem. L20 adds inequalities.

== Projection onto linear constraints

Project a point $bold(z)$ onto $A bold(x)=bold(b)$:

$ min_bold(x) quad 1/2 norm(bold(x)-bold(z))^2 quad "subject to" quad A bold(x)=bold(b). $

#pause
Its Lagrangian is

$ cal(L)=1/2 norm(bold(x)-bold(z))^2+bold(lambda)^top(A bold(x)-bold(b)). $

== ⭐ Solve the projection equations #D

Stationarity in $bold(x)$ gives

$ bold(x)-bold(z)+A^top bold(lambda)=bold(0), $

so

$ bold(x)=bold(z)-A^top bold(lambda). $

#pause
Insert this into $A bold(x)=bold(b)$:

$ A bold(z)-A A^top bold(lambda)=bold(b). $

== ⭐ The projection formula #D

If $A A^top$ is invertible,

$ bold(lambda)=(A A^top)^(-1)(A bold(z)-bold(b)). $

#pause
Therefore

#result[$bold(x)^*=bold(z)-A^top(A A^top)^(-1)(A bold(z)-bold(b))$.]

#pause
The correction lies in the row space of $A$—normal to the feasible affine set.

== Numbers: project $(2,0)$ onto $x+y=1$

$ bold(z)=mat(2;0), quad A=mat(1,1), quad b=1. $

#pause
$ A bold(z)-b=1, quad A A^top=2. $

#pause
$ bold(x)^*=mat(2;0)-mat(1;1)(1/2)(1)=mat(1.5;-0.5). $

#pause
Check: $1.5+(-0.5)=1$, and the correction $(-0.5,-0.5)$ is normal to the line.

== The same projection in NumPy

#codebox[```python
import numpy as np

z = np.array([2., 0.])
A = np.array([[1., 1.]])
b = np.array([1.])

lam = np.linalg.solve(A @ A.T, A @ z - b)
x = z - A.T @ lam
print(x)        # [ 1.5 -0.5]
print(A @ x)    # [1.]
```]

#pause
As in L4, solve the linear system; do not form a matrix inverse in code.

// ═══════════════════ 7 · what the equations do not guarantee ═══════════════════
= Candidates, not guarantees

== Stationarity is necessary, not sufficient

Consider

$ max_(x,y) quad x^2 quad "subject to" quad x^2+y^2=1. $

#pause
The Lagrange equations return four stationary points:

$ (plus.minus 1,0) quad "and" quad (0,plus.minus 1). $

#pause
But their objective values are different:

$ f(plus.minus 1,0)=1 quad "(maxima)," quad f(0,plus.minus 1)=0 quad "(minima)." $

#pause
We still have to compare candidates or inspect second-order behaviour along feasible directions.

== Regularity can fail

The geometric derivation assumed $nabla h(bold(x)^*) eq.not bold(0)$.

#pause
For the constraint $h(x)=x^2=0$, the only feasible point is $x=0$, but

$ h'(0)=0. $

#pause
The feasible set is perfectly clear; the gradient description is not regular there.

#pause
#notebox[Constraint qualifications state when multiplier conditions are valid. L20 uses one of them when inequalities become active.]

== Equality constraints do not describe boundaries

Many ML problems also contain inequalities:

$ p_i>=0, quad norm(bold(w))<=R, quad "loss"_g<=epsilon, quad x_i>=0. $

#pause
An inequality may be:

+ inactive, with room to spare, or
+ active, behaving like an equality at the solution.

#pause
KKT conditions decide which case applies. That is L20.

== A compact procedure

For a smooth equality-constrained problem:

#pause
1. Write every equality as $h_j(bold(x))=0$.
#pause
2. Form $cal(L)=f+sum_j lambda_j h_j$; record the sign convention.
#pause
3. Solve $nabla_bold(x) cal(L)=bold(0)$ and $h_j(bold(x))=0$ together.
#pause
4. Check feasibility and compare all candidates.
#pause
5. Use curvature, convexity, or direct comparison to classify them.
#pause
6. Interpret $lambda_j^*$ only after fixing the units and sign.

== Summary

#result[At a regular equality-constrained optimum, the objective gradient lies in the span of the constraint gradients.]

#pause
+ The Lagrangian turns that geometric condition and the constraints into one system of equations.
+ A multiplier measures local sensitivity of the optimal value to its constraint.
+ One multiplier is introduced for each equality.
+ The probability simplex gives a central ML example: entropy is maximized by the uniform distribution.
+ Lagrange equations produce stationary candidates; they do not by themselves prove optimality.

== Practice

1. Minimize $x^2+y^2$ subject to $x+y=6$. Give $(x^*,y^*)$ and $lambda^*$ under $cal(L)=f+lambda(x+y-6)$.

#pause
2. Maximize $x y z$ subject to $x+y+z=12$ and $x,y,z>0$.

#pause
3. Project $bold(z)=(3,1)$ onto $2x-y=0$ using the matrix projection formula.

#pause
4. For maximum entropy with $K=5$, find $bold(p)^*$ and $H(bold(p)^*)$.

== Practice answers #A

1. $(x^*,y^*)=(3,3)$ and $lambda^*=-6$.

2. $(x^*,y^*,z^*)=(4,4,4)$ and the maximum product is $64$.

3. $A=(2,-1)$, so $bold(x)^*=bold(z)-A^top(A A^top)^(-1)A bold(z)=(0.6,1.2)$.

4. $p_i^*=1/5$ and $H=log 5$ nats.

== Next: inequalities

The probability simplex already contained $p_i>=0$, which we temporarily set aside.

#pause
Next lecture:

$ "primal feasibility" + "dual feasibility" + "stationarity" + "complementary slackness". $

#pause
Together these are the Karush–Kuhn–Tucker conditions.

#focus-slide[Equality constraints add normals. Inequalities add a decision: active or inactive.]
