# Mathematical Foundations for AI — 26-Lecture Plan (80 min each)

**Instructor:** Prof. Nipun Batra, IIT Gandhinagar
**Structure:** 3-1-0-4 · 26 lectures + 13 tutorials (80 min each)
**Prerequisites:** ES 113 (Data Centric Computing) + ES 114 (Probability, Statistics and Data Visualisation) — students know Python/NumPy, basic discrete probability, descriptive statistics. They have **not** done machine learning.

## Design Principles

- **Proposal-faithful** — follows the approved course proposal: refresher → multivariate calculus + autodiff → probability & estimation → optimization → information theory → Markov chains. (The Spring-2025 run went deep on linear algebra and dropped autodiff/info-theory/Markov chains; we deliberately restore the proposal's balance.)
- **Intuition first, 3Blue1Brown-style** — every core concept arrives as picture → numeric example → code → formalism, in that order. One interactive per major concept.
- **Textbook-anchored** — Deisenroth *Mathematics for Machine Learning* (MML) is the backbone; Boyd & Vandenberghe for optimization, MacKay ITILA for information theory, Solomon *Numerical Algorithms* for numerics.
- **One destination, every road leads there** — the final lecture trains a character-level n-gram language model: a Markov chain, fit by MLE, with cross-entropy loss, minimized by gradient descent, differentiated automatically, computed in floating point. Every module is a prerequisite for that one artifact.
- **AI examples before the math** — each lecture opens with a real AI failure or capability that the day's math explains (NaN losses, LoRA, PageRank, diffusion noise schedules, LLM compression).

---

## Module 0: Motivation & Machine Numbers (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 1 | **Why Math for AI + Course Map** | The mathematical stack of modern AI (data → model → loss → optimizer → deployment, and the math each stage needs); live demo of things breaking without the math (NaN loss, exploding gradients, singular matrix); how the course works |
| 2 | **Floating Point: How Machines See Numbers** | Binary representation, IEEE-754 (sign/exponent/mantissa), machine epsilon, rounding, underflow/overflow, catastrophic cancellation, log-sum-exp trick, float32 vs float16/bfloat16 in deep learning |

## Module 1: Linear Algebra Refresher (4 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 3 | **Vectors: Geometry of Data** | Vectors as data points, norms, dot product, cosine similarity, projections; embeddings as the AI motivation ("king − man + woman") |
| 4 | **Matrices as Linear Maps** | Matrix × vector = transformation (rotate/scale/shear), composition, rank, invertibility, solving Ax = b, least squares preview |
| 5 | **Eigendecomposition** | Eigenvectors as invariant directions, geometric intuition, power iteration, spectral view of repeated application; PageRank as the running example |
| 6 | **SVD & PCA** | Every matrix = rotate·scale·rotate, low-rank approximation, image compression demo, PCA as best-fit subspace; LoRA teaser |

> **Quiz 1** (Modules 0–1)

## Module 2: Multivariate Calculus & Automatic Differentiation (5 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 7 | **Univariate Calculus & Taylor Series** | Derivative as sensitivity / best local linear model, chain rule, Taylor series as "local polynomial world-view"; why loss curves look like parabolas near minima |
| 8 | **Gradients & the Geometry of Surfaces** | Partial derivatives, gradient, contour plots, gradient ⊥ level sets, directional derivatives, multivariate chain rule |
| 9 | **Jacobian, Hessian & Multivariate Taylor** | Vector-valued derivatives, Jacobian, Hessian as curvature, quadratic forms, positive (semi-)definiteness, multivariate Taylor expansion |
| 10 | **Differentiation on a Computer I** | Symbolic vs numeric vs automatic; finite differences and their failure modes (truncation vs rounding error — floating point returns!); forward-mode autodiff, dual numbers |
| 11 | **Differentiation on a Computer II: Backprop** | Computation graphs, reverse-mode autodiff, build micrograd live, PyTorch autograd; why reverse mode wins for ML (one output, many parameters) |

> **Quiz 2** (Module 2)

## Module 3: Probability & Estimation (4 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 12 | **Continuous Distributions** | Densities vs probabilities, expectation/variance, uniform/exponential/Gaussian/Beta, change of variables, sampling by inverse CDF |
| 13 | **Multivariate Gaussians** | Joint/marginal/conditional, covariance, the Gaussian ellipse, Mahalanobis distance, why Gaussians are everywhere (CLT + max entropy) |
| 14 | **Maximum Likelihood Estimation** | Likelihood as a function of parameters, log-likelihood, MLE for Bernoulli/Gaussian/linear regression; **negative log-likelihood = loss function** — the course's keystone identity |
| 15 | **MAP & Conjugate Priors** | Priors, posterior, MAP for the coin (Beta-Bernoulli), Gaussian-Gaussian, sequential Bayesian updating; regularization = prior (ridge teaser) |

> **Quiz 3 / Midsem** (Modules 0–3)

## Module 4: Optimization (6 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 16 | **The Optimization Landscape** | Optimization as the engine of learning; stationary points, minima/saddles, convexity (definitions, recognizing convex functions), why convexity buys global guarantees |
| 17 | **Gradient Descent** | Derivation from Taylor series, learning rate as trust region, convergence behavior on quadratics (condition number!), momentum |
| 18 | **Second-Order Methods** | Newton's method from the quadratic model, Gauss-Newton for least squares, Levenberg-Marquardt; curve-fitting as the running example; cost of second order at scale |
| 19 | **Constrained Optimization: Lagrange Multipliers** | Constraints in ML (norm bounds, simplex, fairness), geometric intuition (gradient alignment at optima), Lagrangian, worked examples |
| 20 | **Duality & KKT** | Primal/dual, weak/strong duality, KKT conditions as the unifying optimality certificate; economic interpretation of multipliers |
| 21 | **Linear & Quadratic Programming** | LP/QP standard forms, modeling with cvxpy, classic examples (diet/allocation, portfolio, SVM as QP teaser) |

> **Quiz 4** (Module 4)

## Module 5: Information Theory ↔ Machine Learning (3 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 22 | **Surprise & Entropy** | Self-information, entropy as expected surprise, the 20-questions game, entropy of English text, maximum entropy distributions |
| 23 | **Compression & Optimal Codes** | Source coding, prefix codes, optimal code lengths = −log p, Huffman coding (build one live), noisy channels in brief; "LLMs are compressors" |
| 24 | **KL Divergence & the Unification** | Cross-entropy, KL divergence, the identity chain **minimize cross-entropy = maximize likelihood = compress optimally**, mutual information in brief |

## Module 6: Markov Chains & the Payoff (2 lectures)

| # | Topic | Key Content |
|---|-------|-------------|
| 25 | **Markov Chains** | Sequential modeling, joint probability factorization, transition matrices, state diagrams, stationary distributions (eigenvector callback — PageRank resolved!) |
| 26 | **Learning Sequences + Course Finale** | MLE for transition probabilities, character n-gram language model on real text — generate Shakespeare; the full-circle recap: every module inside one tiny language model; what comes next (ML, DL courses) |

> **Quiz 5 / Endsem** (Modules 5–6, cumulative)

---

## Lecture Allocation Summary

| Module | Lectures | % of course |
|--------|----------|-------------|
| Motivation + Machine Numbers | 2 | 7.7% |
| Linear Algebra Refresher | 4 | 15.4% |
| Multivariate Calculus + Autodiff | 5 | 19.2% |
| Probability & Estimation | 4 | 15.4% |
| Optimization | 6 | 23.1% |
| Information Theory | 3 | 11.5% |
| Markov Chains + Finale | 2 | 7.7% |

## Tutorial Plan (13 × 80 min, hybrid: ~40 min worksheet + ~40 min notebook)

| # | After | Worksheet (pen & paper) | Notebook (laptop) |
|---|-------|------------------------|-------------------|
| T1 | L2 | Binary/IEEE-754 by hand, epsilon reasoning | Floating-point experiments: `0.1+0.2`, cancellation, log-sum-exp |
| T2 | L4 | Vector/matrix computations, rank, solving small systems | Images as matrices, broadcasting, transformations animated |
| T3 | L6 | Eigen/SVD by hand on 2×2/3×3 | PCA on faces/MNIST, image compression via SVD |
| T4 | L8 | Derivatives, gradients, chain rule drills | Contour plots, gradient fields, quiver plots |
| T5 | L11 | Backprop by hand on a small graph | Build micrograd step by step; check vs PyTorch |
| T6 | L13 | Density/expectation/change-of-variables problems | Sampling, histograms → densities, 2D Gaussians |
| T7 | L15 | MLE/MAP derivations (Bernoulli, Gaussian) | Coin-flip Bayesian updating dashboard |
| T8 | L17 | GD convergence on quadratics, condition number | Optimizer race on 2D landscapes |
| T9 | L18 | Newton/Gauss-Newton steps by hand | Nonlinear curve fitting (exponential decay, logistic) |
| T10 | L20 | Lagrange/KKT worked problems | Constrained optimization visualized |
| T11 | L21 | Formulating LPs/QPs from word problems | cvxpy: diet, portfolio, tiny SVM |
| T12 | L24 | Entropy/Huffman/KL computations | Build a Huffman compressor; compress real text |
| T13 | L26 | Markov chain steady states, sequence likelihoods | n-gram language model: train, sample, evaluate perplexity |

## Narrative Arc

- **L1–L6**: numbers and shapes — how machines represent data and transform it
- **L7–L11**: change — measuring sensitivity, and teaching machines to differentiate themselves
- **L12–L15**: uncertainty — modeling data with distributions, and fitting them (the keystone: NLL = loss)
- **L16–L21**: search — finding the best parameters, free and constrained
- **L22–L26**: communication — entropy, compression, sequences; everything converges in one tiny language model

## Inspirations

- **Approved course proposal** (Batra, Dasgupta, Singh, Raman): module structure, topic list, textbook set
- **MML book (Deisenroth et al.)**: chapter backbone for Modules 1–4
- **3Blue1Brown**: geometric intuition-first presentation (Essence of Linear Algebra / Calculus)
- **MacKay ITILA + video lectures**: information theory as inference, the compression ↔ learning unification
- **Boyd & Vandenberghe**: constrained optimization, duality, LP/QP
- **Spring-2025 colleague run**: cadence calibration (quiz frequency, worksheet style, PCA/SVD emphasis retained as LA capstone)
