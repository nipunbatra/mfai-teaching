// Why Math for AI + the Course Map — Lecture 1 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture1/L1-why-math-for-ai.typ
//   typst compile --root . --input handout=true lecture1/L1-why-math-for-ai.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Why Math for AI],
  subtitle: [Three examples, one destination, and the course map],
)

#title-slide()

= Three motivating examples

== Example 1: a softmax loss becomes `nan` #V

You train a model. The loss falls beautifully for 617 steps. Then —

#fig("/lecture1/figures/nan_loss.svg", w: 72%)

#pause
No error message. No crash. Just `nan`, forever. *What happened?*

== Reproduce the numerical failure in three lines

The model's last layer computes a *softmax* — it turns scores into probabilities:

#codebox[```python
>>> import numpy as np
>>> z = np.array([1000., 1001., 1002.])   # perfectly reasonable scores
>>> np.exp(z) / np.exp(z).sum()
array([nan, nan, nan])
```]

#pause
$e^1000$ is larger than *any number this machine can store* — so it gives up.

#pause
#notebox[Real numbers and machine numbers are *not the same thing*. The gap between them is *Lecture 2* — the very next class.]

== Example 2: one learning rate, two outcomes #V

Gradient-based methods train many machine-learning models. Same update rule, same learning rate $eta = 0.8$ — on two different problems:

#fig("/lecture1/figures/lr_two_problems.svg", w: 80%)

#pause
It crawls on one and explodes on the other. *Why?* Resolved in *L17* (conditioning).

== Example 3: arithmetic with word embeddings #V

A word-embedding model stores each vocabulary item as a vector of learned numbers. In one such space, this approximate relation appears:

$ "king" - "man" + "woman" approx "queen" $

#pause
#fig("/lecture1/figures/embedding_arithmetic.svg", w: 52%)

#pause
How can *subtracting lists of numbers* capture _gender_? Resolved in *L3* (vectors).

== Three examples, three mathematical explanations

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 9pt,
  table.header([*Example*], [*The mathematics used to explain it*], [*Lecture*]),
  [the loss goes `nan`], [floating-point numbers], [*L2* — next class!],
  [the same $eta$ crawls _and_ explodes], [conditioning, curvature], [*L17*],
  [king − man + woman ≈ queen], [vector geometry], [*L3*],
)

#pause
#result[Each example points to a mathematical question that the course will answer. The mathematics is the explanation, not decoration.]

= The AI stack

== A useful five-stage view of an AI system #V

#fig("/lecture1/figures/ai_stack.svg", w: 96%)

== The stack, stage by stage

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Stage*], [*The math it runs on*], [*Where we build it*]),
  [*Data*], [vectors, matrices, norms], [L3–L6 (stored as floats: L2)],
  [*Model*], [linear maps, probability distributions], [L4–L6, L12–L13, L25],
  [*Loss*], [likelihood, cross-entropy], [L14, L22–L24],
  [*Optimizer*], [gradients, autodiff, convexity], [L7–L11, L16–L21],
  [*Deployment*], [floating point, compression], [L2, L23],
)

#pause
#result[This table *is* the course: five stages, six modules, one destination.]

= The destination

== By Lecture 26, you will train this

A *character-level language model*, from scratch, on the complete works of Shakespeare. Its actual output:

#codebox[```text
DUKE VINCENTIO:
Well, your wit is in the care of side and that.

CLARENCE:
And so the sight the world and the more,
I shall be my lord the king of hearth.
```]

#pause
This is much smaller than GPT. What the two share is precise: both assign probabilities to text, fit parameters from data, use a cross-entropy objective, and train with gradient-based computation in floating point.

== What each module contributes

What that little language model needs from each module:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Module*], [*What the language model needs from it*]),
  [*0 · Machine numbers*], [probabilities underflow → log-space, log-sum-exp],
  [*1 · Linear algebra*], [transition matrices, stationary distributions],
  [*2 · Calculus + autodiff*], [gradients of the loss, backprop],
  [*3 · Probability + MLE*], [the model _is_ a distribution; fitting = MLE],
  [*4 · Optimization*], [gradient descent actually finds the parameters],
  [*5 · Information theory*], [cross-entropy loss, perplexity],
  [*6 · Markov chains*], [the model class itself],
)

== We check the destination at every module boundary

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*After module…*], [*…you can build this much of it*]),
  [0 · machine numbers], [store its probabilities without underflow],
  [1 · linear algebra], [run its transition matrix, find steady states],
  [2 · calculus + autodiff], [differentiate its loss automatically],
  [3 · probability + MLE], [define its loss (and know where losses come from)],
  [4 · optimization], [actually train it],
  [5 · information theory], [evaluate it (perplexity, compression)],
  [6 · Markov chains], [assemble the whole thing — and generate Shakespeare],
)

#pause
#result[At each module boundary, we can point to one more part of the final model that you can already explain or implement.]

= The teaching contract

== How every idea in this course will arrive

#pause
+ *Picture* — a geometric intuition you can _see_
#pause
+ *Numbers* — a worked example small enough to check by hand
#pause
+ *Code* — the same example in \~5 lines of NumPy
#pause
+ *Symbols* — only _now_, the general definition

#pause
#alertbox[If you ever meet a formula in this course and think "where did that come from?" — that is a bug in the course. Report it.]

#pause
Let's demo the contract right now, on a concept from *L5*: _eigenvectors_.

== Step 1 · picture: the directions that don't turn #V

A matrix moves every vector. Watch 16 directions, before and after:

#fig("/lecture1/figures/eigenvector_demo.svg", w: 76%)

#pause
Almost every direction gets *turned*. Two directions refuse.

== Step 2 · numbers: check it by hand

$ A = mat(2, 1; 1, 2) $

#pause
*The special direction* $bold(v) = (1, 1)$:

$ A bold(v) = mat(2 dot 1 + 1 dot 1; 1 dot 1 + 2 dot 1) = mat(3; 3) = 3 bold(v) quad #text(fill: MUTED)[same line, stretched ×3] $

#pause
*An ordinary direction* $bold(u) = (1, 0)$:

$ A bold(u) = mat(2; 1) quad #text(fill: MUTED)[knocked off its line — turned] $

== Step 3 · code: three lines of NumPy

#codebox[```python
>>> A = np.array([[2., 1.], [1., 2.]])
>>> vals, vecs = np.linalg.eig(A)
>>> vals
array([3., 1.])
>>> vecs[:, 0]                    # the (1,1) direction, normalized
array([0.70710678, 0.70710678])
```]

#pause
The machine found both special directions — and how much each one stretches.

== Step 4 · symbols: only now, the definition

A nonzero vector $bold(v)$ is an *eigenvector* of $A$, with *eigenvalue* $lambda$, if

$ A bold(v) = lambda bold(v) $

— applying $A$ only _scales_ $bold(v)$; it never turns it.

#pause
#result[You just previewed *L5* — and seen how *every* concept will arrive. \ Picture first. Symbols last. Always.]

= The fine print

== Prerequisites · an honest accounting

#two[
  *We assume (ES 113 + ES 114)*
  - Python + NumPy — loops, arrays, plotting
  - discrete probability — coins, dice, Bayes on events
  - descriptive statistics — mean, variance
  - school calculus — $dif/(dif x) x^n$, one variable
][
  *We do NOT assume*
  - any machine learning
  - linear algebra beyond 12th grade
  - multivariable calculus
  - continuous distributions / densities
  - optimization of any kind
]

#pause
#notebox[If you can read a `for` loop and reason about a coin flip, you have everything you need on day one.]

== How the course runs

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*What*], [*How much*], [*Notes*]),
  [Lectures], [26 × 80 min], [intuition first, with an AI example in every lecture],
  [Tutorials], [13 × 80 min], [*hybrid*: \~40 min worksheet + \~40 min notebook],
  [Credit], [3–1–0–4], [tutorials are formative — effort counts, not marks],
  [Assessment], [quizzes + midsem + endsem], [a quiz after (nearly) every module],
)

#pause
Worksheets build pen-and-paper fluency (exam style); notebooks rebuild the _same idea_ computationally. You need both hands.

== The textbooks · all excellent, all free

#table(
  columns: (1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Book*], [*Role in this course*]),
  [Deisenroth, Faisal, Ong — _Mathematics for ML_], [the backbone (Modules 1–4)],
  [Boyd & Vandenberghe — _Convex Optimization_], [Module 4 (optimization)],
  [MacKay — _Information Theory, Inference & Learning_], [Module 5 (information theory)],
  [Solomon — _Numerical Algorithms_], [Module 0 + numerics throughout],
)

#pause
#notebox[All four are *legally free PDFs* from the authors: #link("https://mml-book.github.io")[mml-book.github.io] · #link("https://web.stanford.edu/~boyd/cvxbook/")[stanford.edu/\~boyd/cvxbook] · #link("http://www.inference.org.uk/itila/")[inference.org.uk/itila] · Solomon's _Numerical Algorithms_. Reading pointers close every lecture.]

== The course site · everything in one place

- *Slides* — PDF for every lecture, posted before class
#pause
- *Worksheets* — pen-and-paper problem sets, with solutions
#pause
- *Notebooks* — every tutorial's computational half, Colab-ready
#pause
- *Interactives* — browser toys for the big ideas (eigenvectors, gradient descent, IEEE-754, …)

#pause
#notebox[*Tutorial 1* lands right after L2: IEEE-754 by hand (worksheet) + floating-point experiments in NumPy (notebook). Tutorials always follow the lecture that feeds them.]

= The map

== The course at a glance #V

#fig("/lecture1/figures/course_arc.svg", w: 96%)

== The course sequence

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Lectures*], [*Arc*], [*The question it answers*]),
  [L1–L6], [*numbers & shapes*], [how do machines store data — and transform it?],
  [L7–L11], [*change*], [how do we measure sensitivity — and automate it?],
  [L12–L15], [*uncertainty*], [how do we model data with distributions — and fit them?],
  [L16–L21], [*search*], [how do we find the best parameters, free and constrained?],
  [L22–L26], [*communication*], [entropy, compression, and sequence models],
)

#pause
#result[In *L14*, negative log-likelihood explains an important family of losses, including cross-entropy. Other objectives, such as hinge loss and explicit regularizers, have different origins.]

== Checkpoint: calibrate yourself #Q

No marks — just checking that the prerequisites are alive:

#mcq([Three warm-ups. Commit to answers for all three before the next slide.],
  [You flip a fair coin twice. P(two heads)?],
  [What does `np.dot([1, 2], [3, 4])` return?],
  [What is the slope of $f(x) = x^2$ at $x = 3$?],
  [(no fourth question — breathe)],
)

== Answers: calibrate yourself #A

*A.* $1/2 times 1/2 = 1/4$ — independent events multiply (ES 114). By L12 you'll do the _continuous_ version.

#pause
*B.* $1 dot 3 + 2 dot 4 = 11$ — the dot product. In L3 it becomes _geometry_: length, angle, similarity.

#pause
*C.* $f'(x) = 2x$, so $6$ — school calculus. In L7 the derivative becomes _the best local linear model_.

#pause
#notebox[If all three felt easy, you are fully prepared. If one wobbled, skim your ES 113/114 notes this week — these are the only prerequisites we lean on.]

== Lecture 1 — summary

- *AI fails (and works) in mathematically explainable ways* — NaN losses (L2), diverging optimizers (L17), embedding arithmetic (L3).
- *The stack*: data → model → loss → optimizer → deployment — each stage runs on specific math.
- *The destination*: a character-level language model, trained from scratch in L26.
- *The contract*: picture → numbers → code → symbols — demoed today on eigenvectors.
- *Prerequisites*: ES 113/114 only; no ML assumed. Four textbooks, all free.

#pause
#notebox[*Read before L2* — nothing required. If curious: Goldberg, _What Every Computer Scientist Should Know About Floating-Point Arithmetic_ (1991), §1.]

#focus-slide[
  An AI system combines several mathematical components — by L26 you'll have built a small one from scratch.
  #v(12pt)
  #set text(size: 22pt)
  Next: *floating point* — how machines store numbers, why $e^1000$ overflows, and how stable softmax avoids it.
]
