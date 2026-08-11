// Linear and Quadratic Programming — Lecture 21 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture21/L21-linear-quadratic-programming.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Linear and Quadratic Programming],
  subtitle: [Recognize the form, build the model, use a solver],
)

#title-slide()

// ═══════════════════ 1 · recognition ═══════════════════
= Two common problem classes

== A planning problem

A small workshop makes two products. One unit of product 1 earns 3, and one unit of product 2 earns 2.

#pause
Resource limits are

$ x_1+x_2<=4, quad x_1<=2, quad x_2<=3, quad x_1,x_2>=0. $

#pause
The decision is

$ max_(x_1,x_2) quad 3x_1+2x_2. $

#pause
Every expression is linear. This is a *linear program*.

== Recognition matters more than hand calculation

Reliable solvers already exist for:

+ linear programs with millions of variables,
+ convex quadratic programs,
+ least-squares problems,
+ network flows, cone programs, and related structured classes.

#pause
Boyd and Vandenberghe call least squares, LP, and QP a _technology_: reliable, mature, ready to use. Our job is to decide:

1. what the variables mean,
2. what quantity should be optimized,
3. which statements must be constraints,
4. and whether the resulting class is convex.

== Linear program

One common LP form is

$ min_bold(x) quad bold(c)^top bold(x) $
$ "subject to" quad A bold(x)<=bold(b), quad E bold(x)=bold(d). $

#pause
The objective is linear. Every constraint is affine.

#pause
Variable bounds such as $ell_i<=x_i<=u_i$ are also linear inequalities.

== Quadratic program

A convex QP has the form

$ min_bold(x) quad 1/2 bold(x)^top P bold(x)+bold(q)^top bold(x) $
$ "subject to" quad A bold(x)<=bold(b), quad E bold(x)=bold(d), $

where $P$ is positive semidefinite ($P>=0$ in matrix order).

#pause
The feasible set is still described by linear constraints. The objective now has curvature.

== Where they appear in AI and data science

#table(
  columns: (1.1fr, 1.3fr, 1.2fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*problem*], [*objective*], [*class*]),
  [diet / allocation], [linear cost or reward], [LP],
  [transport / matching relaxation], [linear cost], [LP],
  [least squares with bounds], [quadratic error], [QP],
  [mean–variance portfolio], [risk minus return], [QP],
  [hard-margin SVM], [squared norm], [QP],
)

#pause
#notebox[The support vector machine you will meet in ML is "just" the last row: its training problem is a convex QP, and the solver for it already exists.]

== Learning outcomes

By the end of this lecture you will be able to:

+ recognize an LP or convex QP from its objective and constraints,
+ model a two-food diet problem as an LP,
+ verify a small LP optimum by checking vertices,
+ derive and interpret the dual of that LP,
+ model projection and portfolio choice as QPs,
+ express both problem classes in CVXPY,
+ and read solver status, residuals, and dual values critically.

// ═══════════════════ 2 · LP geometry ═══════════════════
= Linear objectives over polytopes

== The feasible set is a polytope

For the workshop,

$ x_1+x_2<=4, quad x_1<=2, quad x_2<=3, quad x_1,x_2>=0. $

#pause
Each inequality defines a half-plane. Their intersection is a polygon in two dimensions, called a *polytope* in general dimensions.

#pause
Its vertices are

$ (0,0), (2,0), (2,2), (1,3), (0,3). $

== Move a linear contour #V

#two(r: (56%, 44%),
  fig("/lecture21/figures/lp_vertices.svg", w: 88%),
  [Contours of $3x_1+2x_2$ are parallel lines.

  #pause
  Moving the line toward larger values makes final contact at $(2,2)$.],
)

== Check every vertex #D

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*vertex*], [$3x_1+2x_2$]),
  [$(0,0)$], [$0$],
  [$(2,0)$], [$6$],
  [$(2,2)$], [*$10$*],
  [$(1,3)$], [$9$],
  [$(0,3)$], [$6$],
)

#pause
For this bounded LP, a vertex attains the optimum.

== Why a vertex is enough #D

Every point in a bounded polytope is a convex combination of its vertices:

$ bold(x)=sum_k alpha_k bold(v)_k, quad alpha_k>=0, quad sum_k alpha_k=1. $

#pause
For a linear objective,

$ bold(c)^top bold(x)=sum_k alpha_k bold(c)^top bold(v)_k. $

#pause
This weighted average cannot exceed the best vertex value. Hence some vertex is at least as good as $bold(x)$.

== Three possible LP outcomes

An LP may be:

#pause
+ *optimal:* a finite best value exists,
#pause
+ *infeasible:* no point satisfies all constraints,
#pause
+ *unbounded:* feasible points exist and the objective can improve without limit.

#pause
These are solver statuses, not numerical inconveniences.

== Multiple optima

If an objective contour is parallel to a binding edge, every point on that edge can be optimal.

#pause
Example over the workshop polytope:

$ max quad x_1+x_2. $

#pause
The constraint $x_1+x_2<=4$ is tight along the segment from $(1,3)$ to $(2,2)$, so every point on that segment has value $4$.

== Question: where can an LP optimum occur? #Q

#mcq(
  [A bounded feasible LP always has at least one optimum at…],
  [the centre of the polytope],
  [a vertex of the polytope],
  [the origin],
  [a point with $nabla f=bold(0)$],
)

== Answer #A

#mcq-answer("B", [a vertex of the polytope], [Every feasible point is a convex combination of vertices, so its linear-objective value cannot exceed the best vertex value.])

// ═══════════════════ 3 · modeling ═══════════════════
= From words to an LP

== A two-food diet model

Choose servings of oats $x$ and dal $y$.

#table(
  columns: (1.15fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([], [*oats*], [*dal*]),
  [cost], [$2$], [$3$],
  [protein], [$5$], [$10$],
  [fibre], [$4$], [$3$],
)

#pause
Requirements: at least 20 units of protein and 12 units of fibre.

== Step 1 — choose variables and units

$ x="servings of oats", quad y="servings of dal". $

#pause
Write units beside the first draft:

+ cost: currency / serving,
+ protein: units / serving,
+ fibre: units / serving.

#pause
Then $5x+10y$ has units of protein, as required.

== Step 2 — write the objective

We want the least cost:

$ min_(x,y) quad 2x+3y. $

#pause
The coefficient of a variable is the change in objective per one unit of that variable.

#pause
If a statement cannot be assigned consistent units, the model is probably wrong.

== Step 3 — translate requirements

“At least” becomes $>=$:

$ 5x+10y>=20 quad "(protein)," $
$ 4x+3y>=12 quad "(fibre)." $

#pause
Servings cannot be negative:

$ x>=0, quad y>=0. $

== Solve the two-variable model by vertices

Candidate intersections on the lower boundary:

+ $y=0$ gives $x=4$, cost $8$;
+ $x=0$ gives $y=4$, cost $12$;
+ intersect $5x+10y=20$ — i.e. $x+2y=4$ — with $4x+3y=12$.

#pause
The intersection is

$ x=2.4, quad y=0.8, quad "cost"=2(2.4)+3(0.8)=7.2. $

#pause
The mixed diet is cheaper than either axis solution.

== A modeling checklist

Before calling a solver:

1. State every variable in words and units.
2. Confirm the objective direction: minimize or maximize.
3. Translate “at most” as $<=$ and “at least” as $>=$.
4. Add physical bounds such as non-negativity.
5. Test one hand-constructed feasible point.
6. Estimate the expected scale of the answer.

== A common modeling error

Suppose protein must be at least 20. Writing

$ 5x+10y<=20 $

describes a maximum allowance, not a minimum requirement.

#pause
The zero-food point would then be feasible and would minimize cost.

#pause
#alertbox[If the solver returns a suspiciously perfect solution, inspect the model before blaming the solver.]

// ═══════════════════ 4 · LP dual ═══════════════════
= Resource prices in the dual

== The workshop LP in matrix form

Primal:

$ max quad bold(c)^top bold(x) quad "subject to" quad A bold(x)<=bold(b), quad bold(x)>=bold(0), $

with

$ A=mat(1,1;1,0;0,1), quad bold(b)=mat(4;2;3), quad bold(c)=mat(3;2). $

#pause
The three rows of $A$ are the three resource limits.

== The corresponding dual

For this maximization form, the dual is

$ min_bold(lambda) quad bold(b)^top bold(lambda) $
$ "subject to" quad A^top bold(lambda)>=bold(c), quad bold(lambda)>=bold(0). $

#pause
Explicitly,

$ min quad 4lambda_1+2lambda_2+3lambda_3 $
$ "subject to" quad lambda_1+lambda_2>=3, quad lambda_1+lambda_3>=2. $

== Read the dual variables as prices

$ lambda_1,lambda_2,lambda_3$ price the right-hand sides $4,2,3$.

#pause
The dual constraints require the total price of resources consumed by each product to cover its unit profit:

+ product 1 uses resources 1 and 2: $lambda_1+lambda_2>=3$,
+ product 2 uses resources 1 and 3: $lambda_1+lambda_3>=2$.

#pause
The dual minimizes the total priced capacity.

== Solve the dual from complementary slackness

At the primal optimum $(x_1,x_2)=(2,2)$:

+ constraints 1 and 2 are active,
+ constraint 3 has slack $3-2=1$.

#pause
Hence $lambda_3=0$.

#pause
Both primal variables are positive, so both dual inequalities are tight:

$ lambda_1+lambda_2=3, quad lambda_1+lambda_3=2. $

== Matching primal and dual values

From $lambda_3=0$,

$ lambda_1=2, quad lambda_2=1. $

#pause
The dual value is

$ 4(2)+2(1)+3(0)=10. $

#pause
The primal value at $(2,2)$ is also

$ 3(2)+2(2)=10. $

#pause
#result[Primal value = dual value = 10: a certificate of optimality.]

== Sensitivity from the dual

Near the current solution, one extra unit of resource 1 is worth about $lambda_1=2$ objective units.

#pause
One extra unit of resource 2 is worth about $lambda_2=1$.

#pause
Resource 3 has slack, so $lambda_3=0$: a small increase has no value until another constraint becomes active.

#pause
This is L19's shadow-price interpretation and L20's inactive-multiplier rule in an LP.

// ═══════════════════ 5 · QP geometry ═══════════════════
= A curved objective over the same polytope

== Projection is a QP

Given a point $bold(z)$, find its closest feasible point:

$ min_bold(x) quad 1/2 norm(bold(x)-bold(z))_2^2 $
$ "subject to" quad A bold(x)<=bold(b). $

#pause
Expanding the objective:

$ 1/2 bold(x)^top bold(x)-bold(z)^top bold(x)+1/2 bold(z)^top bold(z). $

#pause
The final term is constant, so this is a QP with $P=I$ and $bold(q)=-bold(z)$.

== A QP optimum need not be a vertex #V

#two(r: (52%, 48%),
  fig("/lecture21/figures/qp_solution.svg", w: 80%),
  [For $bold(z)=(3,1.4)$, the projection is $(2,1.4)$ — inside an edge, not at a vertex.],
)

== LP and QP geometry

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([], [*LP*], [*convex QP*]),
  [objective contours], [parallel hyperplanes], [ellipses / cylinders],
  [curvature], [none], [$P$ positive semidefinite],
  [typical optimum], [vertex, for a bounded polytope], [interior, edge, or vertex],
  [constraints], [affine], [affine],
)

== Why positive semidefinite $P$ matters

The Hessian of

$ 1/2 bold(x)^top P bold(x)+bold(q)^top bold(x) $

is $P$ when $P$ is symmetric.

#pause
If $P$ is positive semidefinite, the objective is convex. Combined with affine constraints, every local optimum is global.

#pause
If $P$ has a negative eigenvalue, the QP is non-convex and the same guarantees no longer apply.

== Least squares with linear constraints

$ min_bold(x) quad 1/2 norm(A bold(x)-bold(b))_2^2 quad "subject to" quad C bold(x)<=bold(d). $

#pause
Expand:

$ 1/2 bold(x)^top A^top A bold(x)-bold(b)^top A bold(x)+"constant". $

#pause
Thus

$ P=A^top A, quad bold(q)=-A^top bold(b), quad "and" quad P " is positive semidefinite". $

#pause
Least squares with affine equality or inequality constraints is a convex QP.

// ═══════════════════ 6 · portfolio QP ═══════════════════
= Balancing return and risk

== Portfolio variables

Let $w_i$ be the fraction invested in asset $i$.

$ sum_i w_i=1, quad w_i>=0. $

#pause
If expected returns are $bold(mu)$ and the covariance matrix is $Sigma$:

+ expected portfolio return: $bold(mu)^top bold(w)$,
+ portfolio variance: $bold(w)^top Sigma bold(w)$.

== Two common formulations

Minimum variance for a target return:

$ min_bold(w) quad bold(w)^top Sigma bold(w) $
$ "subject to" quad bold(mu)^top bold(w)>=r, quad bold(1)^top bold(w)=1, quad bold(w)>=bold(0). $

#pause
Or trade risk against return directly:

$ min_bold(w) quad 1/2 bold(w)^top Sigma bold(w)-gamma bold(mu)^top bold(w). $

#pause
Both are convex QPs when $Sigma$ is positive semidefinite.

== The risk–return curve #V

#fig("/lecture21/figures/portfolio_frontier.svg", w: 66%)

#pause
Changing the target return moves the solution along the feasible risk–return curve.

== What the model omits

A useful mathematical model is still a simplification.

#pause
The basic mean–variance QP omits:

+ transaction costs,
+ uncertain estimates of $bold(mu)$ and $Sigma$,
+ liquidity and position limits,
+ tail risk and non-Gaussian returns,
+ integer purchase sizes.

#pause
Some omissions remain convex after modeling; others change the problem class.

// ═══════════════════ 7 · CVXPY ═══════════════════
= Express the model, then verify the result

== Workshop LP in CVXPY

#codebox[```python
import cvxpy as cp
import numpy as np

x = cp.Variable(2, nonneg=True)
c = np.array([3., 2.])
A = np.array([[1., 1.], [1., 0.], [0., 1.]])
b = np.array([4., 2., 3.])

problem = cp.Problem(cp.Maximize(c @ x), [A @ x <= b])
problem.solve()
print(x.value, problem.value)  # [2. 2.], 10.0
```]

== Read status before reading values

#codebox[```python
print(problem.status)
# 'optimal', 'infeasible', 'unbounded',
# or an '..._inaccurate' variant
```]

#pause
Only interpret variable values after checking the status.

#pause
For an `optimal_inaccurate` result, inspect residuals, scaling, and solver tolerances.

== Verify feasibility yourself

#codebox[```python
x_hat = x.value
print(A @ x_hat - b)   # every entry should be <= tolerance
print(x_hat.min())     # should be >= -tolerance
print(c @ x_hat)       # recompute the objective
```]

#pause
The solver is a numerical component. Its output should satisfy the mathematical model to an appropriate tolerance.

== Inspect dual values

#codebox[```python
constraint = (A @ x <= b)
problem = cp.Problem(cp.Maximize(c @ x), [constraint])
problem.solve()
print(constraint.dual_value)
# approximately [2., 1., 0.]
```]

#pause
The last resource has slack and zero price, exactly as the hand calculation predicted.

== Projection QP in CVXPY

#codebox[```python
z = np.array([3.0, 1.4])
x = cp.Variable(2)

problem = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(x-z)),
    [A @ x <= b, x >= 0],
)
problem.solve()
print(x.value)  # approximately [2.0, 1.4]
```]

== Use expressions that reveal structure

Prefer

```python
cp.sum_squares(A @ x - b)
cp.quad_form(w, Sigma)
cp.norm(x, 2)
```

over manual algebra that obscures convexity.

#pause
CVXPY checks disciplined convex programming rules before choosing a solver.

== Scaling affects numerical accuracy

If one coefficient is $10^9$ and another is $10^(-7)$, residuals and search directions become difficult to balance.

#pause
Useful steps:

+ choose consistent units,
+ scale variables to comparable magnitudes,
+ avoid redundant constraints,
+ use solver tolerances appropriate to the application,
+ and verify the unscaled answer afterward.

== Question: classify the problem #Q

#mcq(
  [Minimize $norm(A bold(x)-bold(b))_2^2$ subject to $bold(x)>=bold(0)$. What class is it?],
  [LP],
  [convex QP],
  [non-convex QP always],
  [unconstrained problem],
)

== Answer #A

#mcq-answer("B", [convex QP], [The squared least-squares objective has positive-semidefinite Hessian $2A^top A$, and non-negativity is a linear constraint.])

// ═══════════════════ 8 · algorithms ═══════════════════
= What the solver does

== Choosing a solver

Choice depends on:

+ LP or QP structure,
+ dense versus sparse matrices,
+ problem size,
+ desired accuracy,
+ warm starts and repeated solves,
+ available commercial or open-source software.

#pause
Modeling systems can select a compatible solver; production work should record the solver and settings.

== Two paths through an LP #OPT

#fig("/lecture21/figures/solver_paths.svg", w: 76%)

#pause
The paths differ, but both exploit the LP's geometry and return the same optimum.

== Simplex method, one-slide view #OPT

The simplex method:

1. starts at a feasible vertex,
2. chooses an adjacent edge that improves the objective,
3. moves to the next vertex,
4. stops when no adjacent improving edge remains.

#pause
Its worst-case complexity is exponential, but it is highly effective on many practical LPs.

== Interior-point method, one-slide view #OPT

An interior-point method adds a barrier against constraint boundaries, for example

$ -mu sum_i log(b_i-bold(a)_i^top bold(x)). $

#pause
It solves a sequence of smooth problems while reducing $mu$.

#pause
The iterates move through the interior and approach the optimal boundary as the barrier weakens.

// ═══════════════════ 9 · close ═══════════════════
= Model, solve, check

== A complete workflow

1. Define variables, units, and physical meaning.
2. Write the objective and constraints from the application.
3. Classify the problem: LP, convex QP, or something else.
4. Construct and test a small feasible instance.
5. Solve with an appropriate numerical solver.
6. Check status, feasibility, objective, and dual values.
7. Perform sensitivity checks before using the decision.

== Summary

#result[Linear and convex quadratic programs are standard numerical technologies; the main skill is building the right model.]

#pause
+ LP: linear objective and affine constraints; over a bounded polytope, some vertex is optimal.
+ Convex QP: positive-semidefinite quadratic objective over affine constraints.
+ Dual variables attach marginal prices to constraints.
+ CVXPY expresses the model; the solver status and residuals still require inspection.
+ Scaling and units are part of correctness.

== Practice

1. Formulate an LP that maximizes calories subject to cost and protein limits.

#pause
2. For the workshop LP, find all optima of $max x_1+x_2$.

#pause
3. Classify $min norm(bold(x))_2^2$ subject to $A bold(x)=bold(b)$.

#pause
4. Add upper bounds $w_i<=0.4$ to the portfolio QP. Do they change convexity?

== Practice answers #A

1. Use food servings as non-negative variables; calories form a linear objective and cost/protein statements form linear constraints.

2. Every point on $x_1+x_2=4$ between $(1,3)$ and $(2,2)$.

3. A convex equality-constrained QP with $P=2I$.

4. No. Upper bounds are linear inequalities, so the problem remains a convex QP.

== Module 4 in one view

#table(
  columns: (0.8fr, 1.3fr, 1.5fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([*lecture*], [*question*], [*tool*]),
  [L16], [is local information trustworthy?], [convexity],
  [L17], [how do slopes move us?], [gradient descent],
  [L18], [how does curvature change the step?], [Newton / Gauss–Newton],
  [L19], [what if an equality must hold?], [Lagrange multipliers],
  [L20], [what if a boundary may bind?], [duality / KKT],
  [L21], [can a solver exploit the form?], [LP / QP],
)

== Next: information

Optimization asks which feasible choice is best. The next module asks how uncertainty itself can be measured.

#pause
We begin with a quantity already seen in L19:

$ H(bold(p))=-sum_i p_i log p_i. $

#pause
Next lecture: surprise, bits, and entropy.

#notebox[*Reading:* Boyd & Vandenberghe, _Convex Optimization_ §4.3–4.4 (LP and QP) · the CVXPY example library at cvxpy.org.]

#focus-slide[Don't solve optimization problems — recognize them.]
