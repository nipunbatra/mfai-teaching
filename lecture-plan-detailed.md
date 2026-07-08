# MFAI — Detailed Lecture Plan (26 × 80 min + 13 tutorials)

**Companion to** `lecture-plan.md` (the map) and `course-design.md` (the pillars). This file is the per-lecture spec used to build each deck.

## Textbook backbone

| Code | Book | Role |
|---|---|---|
| **MML** | Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* ([free PDF](https://mml-book.github.io/)) | Primary: Modules 1–4 |
| **Boyd** | Boyd & Vandenberghe — *Convex Optimization* ([free PDF](https://stanford.edu/~boyd/cvxbook/)) | Module 4 (constrained, duality, LP/QP) |
| **MacKay** | MacKay — *Information Theory, Inference and Learning Algorithms* ([free PDF](https://www.inference.org.uk/itprnn/book.pdf)) | Module 5–6 |
| **Solomon** | Solomon — *Numerical Algorithms* ([free PDF](https://people.csail.mit.edu/jsolomon/share/book/numerical_book.pdf)) | Numerics thread (L2, L10, L18) |

All four are free PDFs — say this loudly in L01.

## Per-lecture conventions

- **Picture → numbers → code → symbols**, in that order (course-design.md Pillar 2).
- **AI hook** opens every lecture; resolved by the end of the lecture (or explicitly deferred with a pointer).
- **One reproducible derivation** per lecture (⭐); everything else is plausibility. Star ratings live in `teaching-guides/lecNN.md`.
- **Insight line** closes every deck and feeds the insight-ledger page.
- **Reuse** column tells the deck author where existing material lives (see repos: `ml-teaching`, `pml-teaching`, `psdv-teaching`, `dl-teaching`, `~/git/interactive`).

## Assessment summary (advertised on syllabus)

- 4–5 in-slot quizzes at module boundaries (best n−1), midsem after Module 3, endsem cumulative — mirrors ES 335/667 pattern.
- 3 take-home assignments: A1 micrograd (after L11), A2 constrained optimization with CVXPY (after L21), A3 Huffman compressor + n-gram language model (after L26 window opens at L24).
- Tutorials are formative (attendance/effort only).

---

## Module 0 · Motivation & Machine Numbers (L1–L2)

### L1 · Why Math for AI + Course Map
- **Spine:** every stage of an AI system is a piece of math; this course builds the whole stack and ends by training a tiny language model from scratch.
- **AI hook:** three live failures — NaN loss, a learning rate that diverges on one problem and crawls on another, `king − man + woman ≈ queen`.
- **Beats:** the AI stack (data → model → loss → optimizer) annotated with modules · the destination demo (n-gram Shakespeare output, "you will build this") · the picture→numbers→code→symbols contract, meta-demonstrated on eigenvectors · prerequisites honesty (ES 113/114 assumed; no ML assumed) · logistics, free textbooks, tutorial format · module map + narrative arc.
- **Derivation ⭐:** none — contract-setting lecture.
- **Reading:** MML Ch 1; course site tour.
- **Reuse:** fresh build (this repo). Destination demo reuses T13/A3 notebook output.
- **Insight line:** *Every AI system is a stack of math — by L26 you'll have built one from scratch.*

### L2 · Floating Point: How Machines See Numbers
- **Spine:** computers represent finitely many, unevenly spaced numbers; knowing where they are and how they round explains half of ML's mysterious crashes.
- **AI hook:** softmax returns NaN mid-training; `0.1 + 0.2 != 0.3`.
- **Beats:** fixed vs floating point · IEEE-754 float32 anatomy with worked encodings (6.25, 0.1) · the uneven number line, machine epsilon, `x+1==x` · inf/NaN semantics · overflow (`exp(89)` in float32) / underflow · catastrophic cancellation (quadratic formula; variance two ways) · fixes: log-space, log-sum-exp derivation, stable softmax · float16 vs bfloat16 (range over precision) · ⭐⭐⭐ Kahan summation, subnormals.
- **Derivation ⭐:** log-sum-exp correctness: log Σ eˣⁱ = m + log Σ eˣⁱ⁻ᵐ.
- **Reading:** Solomon Ch 2 (Numerics and Error Analysis).
- **Reuse:** [numerical-tricks interactive](https://nipunbatra.github.io/interactive-articles/) (stability half); IEEE-754 internals built fresh (gap identified in inventory). Solomon's failure-gallery opening (6.S955 pattern).
- **Insight line:** *Computers don't do real numbers — they do ~4 billion of them, unevenly spaced.*

**→ T1:** binary/IEEE-754 by hand + floating-point experiments notebook.

## Module 1 · Linear Algebra Refresher (L3–L6)

### L3 · Vectors: the Geometry of Data
- **Spine:** data points are vectors; similarity is geometry (dot products, angles, projections).
- **AI hook:** word embeddings — `king − man + woman ≈ queen` resolved; recommender "people like you."
- **Beats:** vectors as arrows AND as data rows · norms (L1/L2, unit balls) · dot product three ways (algebraic, geometric, projection) · cosine similarity on real embeddings · projections · orthogonality · basis and span (3b1b visual language).
- **Derivation ⭐:** cos θ = xᵀy / (‖x‖‖y‖) from the law of cosines.
- **Reading:** MML Ch 2.1–2.4, Ch 3.1–3.3; CS229 [linear algebra notes](https://cs229.stanford.edu/section/cs229-linalg.pdf) as reference handout.
- **Reuse:** psdv `embeddings-angle.ipynb`, `random-vector.ipynb`; 3b1b Essence of LA ch 1–3 as pre-watch.
- **Insight line:** *Similarity is an angle.*

### L4 · Matrices as Linear Maps
- **Spine:** a matrix is a function that moves space; multiplication is composition; rank is how much of space survives.
- **AI hook:** every neural-network layer is `Wx + b` — a move of space; image transforms.
- **Beats:** matrix × vector as transformation (rotate/scale/shear, animated figures) · matrix × matrix as composition · column view (Strang: Ax = combination of columns; rank-1 sums) · rank, null space intuition · invertibility · solving Ax = b (and why we never invert numerically) · least squares preview.
- **Derivation ⭐:** rank of outer product uvᵀ is 1; column picture of Ax.
- **Reading:** MML Ch 2.5–2.8; [matrixmultiplication.xyz](http://matrixmultiplication.xyz/) in class.
- **Reuse:** ml `maths/slides/mathematical-ml.tex` (rank, row×col views) — convert; dimension-annotated color-coded matrices (CMU Gormley pattern).
- **Insight line:** *A matrix is a verb, not a table.*

### L5 · Eigendecomposition
- **Spine:** eigenvectors are the directions a matrix cannot turn; repeated application of a matrix is governed entirely by them.
- **AI hook:** how Google ranked the web (PageRank teased; fully resolved L25).
- **Beats:** "directions that don't turn" (interactive first) · characteristic polynomial on 2×2 only · geometric meaning of eigenvalues (stretch factors) · diagonalization = change to the eigen-basis · power iteration live (converges to top eigenvector) · symmetric matrices: real eigenvalues, orthogonal eigenvectors (spectral theorem, stated) · Markov matrix teaser.
- **Derivation ⭐:** power iteration converges to the top eigenvector (2×2, eigenbasis expansion).
- **Reading:** MML Ch 4.1–4.2; [Setosa eigenvectors explorable](https://setosa.io/ev/eigenvectors-and-eigenvalues/).
- **Reuse:** fresh deck (inventory gap); Solomon Ch 6; 3b1b ch 14 as pre-watch.
- **Insight line:** *Eigenvectors are the directions a matrix can't turn.*

### L6 · SVD & PCA
- **Spine:** every matrix is rotate·stretch·rotate; keeping only the big stretches gives the best low-rank story of your data.
- **AI hook:** LoRA fine-tunes a 7B-parameter model by learning only low-rank updates; image compression.
- **Beats:** SVD geometrically (unit circle → ellipse) · singular values as importance · Eckart–Young ("learning = approximation") · image compression demo with rank-k error curve (IITGn campus photo — Strang's demo) · PCA as best-fit subspace / max-variance directions · eigenfaces or MNIST · PCA = SVD of centered data · ⭐⭐⭐ randomized SVD exists (fast.ai pointer).
- **Derivation ⭐:** best rank-1 approximation via SVD (Eckart–Young, sketch).
- **Reading:** MML Ch 4.4–4.5, Ch 10 (PCA); Solomon Ch 7.
- **Reuse:** psdv `pca.tex` (convert), ml `pca.ipynb` + tutorials (SVD/eigen route already there).
- **Insight line:** *Every matrix is a rotation, a stretch, and a rotation.*

**→ T2 (after L4), T3 (after L6).** **Quiz 1** after L6.

## Module 2 · Multivariate Calculus & Autodiff (L7–L11)

### L7 · Univariate Calculus & Taylor Series
- **Spine:** a derivative is the best local linear replacement of a function; Taylor series is the systematic upgrade to polynomial replacements.
- **AI hook:** why every loss curve looks like a parabola if you zoom in near the minimum.
- **Beats:** derivative as sensitivity (∆output per ∆input) · derivative as local line · chain rule as ratio bookkeeping · maxima/minima, second derivative as curvature · Taylor: linear → quadratic → the cos(x) build-up · convergence radius intuition (where the local story breaks).
- **Derivation ⭐:** second-order Taylor of cos(x) at 0 with error visualization.
- **Reading:** MML Ch 5.1; 3b1b Essence of Calculus ch 1–4 + Taylor video as pre-watch.
- **Reuse:** ml `gradient-descent.tex` Taylor frames; ml `taylor-series.ipynb`.
- **Insight line:** *Calculus is the art of replacing a function with a line.*

### L8 · Gradients & the Geometry of Surfaces
- **Spine:** the gradient collects all partial sensitivities and points straight uphill, perpendicular to the contours.
- **AI hook:** loss landscapes — the pictures behind every "training curve".
- **Beats:** functions of 2 variables as surfaces AND contour maps · partial derivatives · gradient as vector of partials · gradient ⊥ level sets (picture proof, then algebra) · directional derivative · multivariate chain rule (composition diagrams) · gradient fields (quiver plots).
- **Derivation ⭐:** gradient ⊥ contours via directional derivative = ∇f·u.
- **Reading:** MML Ch 5.2–5.3.
- **Reuse:** ml `ml-maths-2-contour.tex` (strong geometric deck — convert), `contour.ipynb`, `meshgrid-contour-explanation.ipynb`.
- **Insight line:** *The gradient points where the contour lines crowd.*

### L9 · Jacobian, Hessian & Multivariate Taylor
- **Spine:** vector-in/vector-out functions have matrix derivatives (Jacobian); curvature becomes a matrix (Hessian); Taylor still works, one degree at a time.
- **AI hook:** a neural network is ℝ^(millions) → ℝ; what does its "derivative" even mean?
- **Beats:** ℝ²→ℝ² everywhere first (difficulty valve) · Jacobian as local linear map (little squares → parallelograms; determinant = area scaling — 3b1b) · Hessian as curvature matrix · quadratic forms xᵀAx, positive (semi-)definite geometry (bowls/saddles/cylinders) · multivariate Taylor to 2nd order · shapes/conventions bookkeeping (numerator vs denominator layout — pick one, stick to it).
- **Derivation ⭐:** ∇(xᵀAx) = (A + Aᵀ)x.
- **Reading:** MML Ch 5.4–5.7; [Parr & Howard matrix calculus](https://explained.ai/matrix-calculus/); [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) as reference.
- **Reuse:** pml `calculus-terms.tex` (Jacobian/Hessian worked examples — convert); ml `mathematical-ml.tex` quadratic forms.
- **Insight line:** *Curvature is a matrix, and its sign is the shape of the bowl.*

### L10 · Differentiation on a Computer I: Finite Differences → Forward Mode
- **Spine:** numerical differentiation fights a two-front war (truncation vs rounding — floating point returns!); forward-mode autodiff wins it by computing derivatives exactly, one input at a time.
- **AI hook:** gradient checking — how PyTorch tests itself.
- **Beats:** symbolic vs numeric vs automatic (the trichotomy) · forward difference: error analysis, the U-shaped error-vs-h curve (worked live — connects to L2!) · central differences · why finite differences can't scale to 10⁶ parameters (one function eval per input) · dual numbers / forward mode: carry (value, derivative) pairs · forward mode costs one pass per INPUT.
- **Derivation ⭐:** optimal step size h* ≈ √ε for forward differences, from truncation + rounding terms.
- **Reading:** MML Ch 5.6 (start); Baydin et al. autodiff survey §2–3 (skim).
- **Reuse:** interactive `autograd` article (contrasts all three); ml `autodiff.ipynb` (numerical part).
- **Insight line:** *Finite differences fight floating point — and both lose.*

### L11 · Differentiation on a Computer II: Reverse Mode & Backprop
- **Spine:** reverse mode computes the gradient of one output w.r.t. a million inputs in one backward pass — that asymmetry is why deep learning is possible.
- **AI hook:** `loss.backward()` — the single most-executed line in modern AI.
- **Beats:** computation graphs · chain rule on the graph, forward values then backward adjoints (worked numeric on a 5-node graph) · reverse mode costs one pass per OUTPUT · build micrograd live (~60 lines: Value class, backward, topological order) · same numbers from PyTorch autograd · what `zero_grad` is for · ⭐⭐⭐ JVP/VJP framing, Hessian-vector products.
- **Derivation ⭐:** full backward pass by hand on the 5-node graph (matches T5 worksheet).
- **Reading:** MML Ch 5.6; Karpathy micrograd video (assigned); [JAX autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) ⭐⭐⭐.
- **Reuse:** ml `autograd.tex`, `autograd-from-scratch.ipynb`; dl `01-micrograd-mlp.ipynb`; principles-ai `L05_autodiff.ipynb`.
- **Insight line:** *Backprop is just the chain rule with memoization.*

**→ T4 (after L8), T5 (after L11).** **Quiz 2** after L11. **A1 (micrograd)** goes out.

## Module 3 · Probability & Estimation (L12–L15)

### L12 · Continuous Distributions
- **Spine:** for continuous quantities probability lives in areas under a density; transformations of variables must pay a Jacobian toll.
- **AI hook:** why generative models sample noise and transform it (the reparameterization idea, teased).
- **Beats:** density ≠ probability (P(X = x) = 0; densities can exceed 1) · CDF ↔ PDF · expectation/variance as integrals · the zoo: uniform, exponential, Gaussian, Beta (star of L15) · change of variables in 1D with the |dx/dy| factor · inverse-CDF sampling (uniform → anything).
- **Derivation ⭐:** change of variables for Y = aX + b; inverse-CDF sampling correctness.
- **Reading:** MML Ch 6.1–6.4, 6.7; [Distribution Explorer](https://distribution-explorer.github.io/); [Seeing Theory](https://seeing-theory.brown.edu/).
- **Reuse:** pml `Prob-refresher.tex`, `distributions.ipynb`, `inverse-cdf.ipynb`; psdv notebook suite (`pdf-continuous`, `cdf`, …).
- **Insight line:** *Continuous probability is area, and transformations charge a Jacobian toll.*

### L13 · Multivariate Gaussians
- **Spine:** the multivariate Gaussian is geometry — a mean point and a covariance ellipse; conditioning and marginalizing are just slicing and projecting it.
- **AI hook:** why "Gaussian" appears in every ML paper (CLT + max entropy + closed forms).
- **Beats:** joint/marginal/conditional pictures · covariance matrix as shape (isotropic → diagonal → full, gallery) · Mahalanobis distance ("how many standard deviations, direction-aware") · the ellipse = eigenvectors of Σ (L5 callback!) · CLT demo · max-entropy statement (proved in L22 module) · sampling via Cholesky ⭐⭐⭐.
- **Derivation ⭐:** contours of the 2D Gaussian are ellipses aligned with Σ's eigenvectors.
- **Reading:** MML Ch 6.5–6.6; CS229 [probability notes](https://cs229.stanford.edu/section/cs229-prob.pdf) handout.
- **Reuse:** pml `mvn.tex`; psdv `bivariate-derivation.tex` (full derivation), `integration.tex`; interactive `multivariate-normal`.
- **Insight line:** *A Gaussian is a point and an ellipse.*

### L14 · Maximum Likelihood Estimation
- **Spine:** flip the model around — fix the data, vary the parameters, climb the likelihood. Every standard loss function falls out of this one move.
- **AI hook:** where does cross-entropy loss come from? (resolved fully in L24; MSE resolved today).
- **Beats:** likelihood as function of θ (animated: slide the Gaussian over the data) · log-likelihood (products → sums; underflow — L2 callback) · MLE for Bernoulli (coin), Gaussian (mean/variance), linear regression → **MSE = Gaussian NLL** (the keystone) · properties in brief (consistency; biased σ² as honest footnote) · NLL = loss table (Bernoulli→BCE, Categorical→CE, Gaussian→MSE).
- **Derivation ⭐:** MLE for Bernoulli via log-likelihood; MSE from Gaussian noise model.
- **Reading:** MML Ch 8.3 (via 8.1–8.2 skim); MacKay Ch 2.
- **Reuse:** pml `MLE.tex`, `MLE_Lin_Log_Reg.tex`, mle notebook suite; dl `lec00` Marp deck (drop-in, re-pitch for pre-ML audience); interactive `mle-map-coin`.
- **Insight line:** *Every loss function is a negative log-likelihood in disguise.*

### L15 · MAP & Conjugate Priors
- **Spine:** beliefs before data are priors; MAP is MLE with a prior's vote; conjugate pairs make the update a one-line parameter bump.
- **AI hook:** 3 heads in 3 flips — is the coin rigged? MLE says θ̂=1; your intuition says no. Priors are the fix.
- **Beats:** Bayes rule as belief update · MAP vs MLE on the coin · Beta-Bernoulli conjugacy: posterior = Beta(α+heads, β+tails), animated as flips stream in · pseudo-counts intuition · Gaussian-Gaussian (known σ) · sequential updating = online learning · regularization = prior (ridge teaser, cashed in ML course) · when priors wash out (data swamps prior).
- **Derivation ⭐:** Beta-Bernoulli posterior update.
- **Reading:** MML Ch 6.6.1 + 8.3.2; MacKay Ch 3.
- **Reuse:** pml `MAP.tex` (43 frames, exactly this); dl `lec00b` Marp; interactives `bayesian-posterior`, `mle-map-coin`; Seeing Theory Bayesian page.
- **Insight line:** *Every regularizer is a prior in disguise.*

**→ T6 (after L13), T7 (after L15).** **Quiz 3 / Midsem** after L15.

## Module 4 · Optimization (L16–L21)

### L16 · The Optimization Landscape
- **Spine:** learning = minimizing a function you can only probe locally; convexity is the property that makes local information globally trustworthy.
- **AI hook:** why does training a linear model "just work" while deep nets need tricks? (convex vs non-convex).
- **Beats:** optimization as the engine (fitting = minimizing NLL — Module 3 handshake) · stationary points, minima/maxima/saddles (Hessian eigenvalue signs — L9 callback) · convex sets and functions (pictures: chords above graph) · recognizing convexity (composition rules, examples: x², eˣ, log-sum-exp!) · convex ⇒ local = global (proof sketch) · least squares is convex.
- **Derivation ⭐:** convexity of least squares via Hessian AᵀA ⪰ 0.
- **Reading:** MML Ch 7.3 (intro); Boyd Ch 2–3 (skim, guided).
- **Reuse:** ml `convexity.tex` (definitions + proofs — convert).
- **Insight line:** *Convexity is a promise: what you see locally is the truth globally.*

### L17 · Gradient Descent
- **Spine:** trust the linear model for one small step, repeat; the learning rate is the radius of your trust, and conditioning decides your fate.
- **AI hook:** same code, two datasets: one converges in 20 steps, one takes 20,000 — why? (condition number).
- **Beats:** GD derived from first-order Taylor (L7 payoff) · learning-rate trichotomy (crawl/converge/diverge, animated) · exact analysis on quadratics: contraction factor, condition number as the villain (ill-conditioned bowl figure) · zig-zag pathology · momentum as a moving average / heavy ball (Distill widgets in class) · feature scaling as preconditioning · SGD exists ⭐⭐⭐ (pointer to ML/DL courses).
- **Derivation ⭐:** GD on f(x)=½x²κ: convergence rate (1−η)ᵏ and the κ dependence.
- **Reading:** MML Ch 7.1; [Distill: Why Momentum Really Works](https://distill.pub/2017/momentum/) (assigned).
- **Reuse:** ml `gradient-descent.tex` (72 frames — the flagship conversion), GD notebook suite; interactive `optimizer-race`; dl `lec04` momentum treatment.
- **Insight line:** *The learning rate is how far you trust a linear approximation.*

### L18 · Second-Order Methods: Newton & Gauss-Newton
- **Spine:** if a linear model buys one step, a quadratic model buys a jump to its bowl's bottom; Gauss-Newton gets the curvature almost for free when the loss is a sum of squares.
- **AI hook:** calibrating a sensor / fitting an exponential decay — scipy's `curve_fit` is Levenberg–Marquardt; what's inside?
- **Beats:** Newton from second-order Taylor (jump to the quadratic's minimum) · one-step-on-quadratics demo vs GD's crawl · Newton's costs and dangers (Hessian at scale; saddles; non-descent) · nonlinear least squares setting · Gauss-Newton: linearize residuals, JᵀJ ≈ Hessian · Levenberg–Marquardt = Gauss-Newton with a trust dial · why deep learning still uses first-order ⭐⭐⭐.
- **Derivation ⭐:** Newton step = −H⁻¹∇f from minimizing the quadratic model.
- **Reading:** MML Ch 7.1.3 context; Solomon Ch 9 (the reference treatment of Gauss-Newton).
- **Reuse:** ml `optimization/tutorials/optimization.tex` (Newton/Gauss-Newton); ml `convexity-hessian-irls.pdf`.
- **Insight line:** *Newton doesn't step downhill — it teleports to the bottom of its local bowl.*

### L19 · Constrained Optimization: Lagrange Multipliers
- **Spine:** at a constrained optimum you cannot improve without leaving the feasible set — which forces the objective and constraint gradients into alignment; the multiplier is the exchange rate.
- **AI hook:** "maximize engagement subject to fairness"; project weights onto a norm ball; probabilities must sum to 1 (the simplex is everywhere).
- **Beats:** constraints in ML gallery · geometric picture first: contours kissing the constraint curve (∇f ∥ ∇g) · the Lagrangian as bookkeeping for that alignment · worked example run of the module: maximize xy s.t. x+y=1 · second worked: max-entropy distribution under simplex constraint (Module 5 handshake!) · multiplier = shadow price (economic reading) · multiple constraints.
- **Derivation ⭐:** ∇f = λ∇g at the constrained optimum, geometrically then algebraically.
- **Reading:** MML Ch 7.2; Boyd Ch 5.1 (gently).
- **Reuse:** ml `constrained-1/2.tex` exist only as scanned handwritten notes — this deck typesets them at last (flagged in inventory).
- **Insight line:** *At the optimum, the objective and the constraint pull in the same direction.*

### L20 · Duality & KKT
- **Spine:** every constrained problem has a shadow problem whose answer bounds yours; KKT conditions are the receipts both must show at the optimum.
- **AI hook:** how do you *certify* "no better solution exists"? (duality gap = certificate); SVMs are trained through their dual (teaser for ML course).
- **Beats:** inequality constraints: active vs inactive (pictures) · the dual as best lower bound · weak duality (one-line proof) · strong duality for convex problems (stated) · KKT conditions one by one, each with its picture (stationarity, feasibility, dual feasibility, complementary slackness = "either the constraint bites or its price is zero") · continue the xy running example with inequality version · water-filling or box-constrained example ⭐⭐.
- **Derivation ⭐:** weak duality; KKT verified on the running example.
- **Reading:** Boyd Ch 5 (guided: 5.1–5.5); MML Ch 7.2.
- **Reuse:** ml `kkt-conditions.tex` (step-by-step + worked example — convert); ml SVM dual decks as instructor background only.
- **Insight line:** *KKT is Lagrange with inequalities and receipts.*

### L21 · Linear & Quadratic Programming
- **Spine:** LP and QP are the two problem shapes the world has industrial-strength solvers for; the skill is recognizing and modeling, not solving (Boyd's mantra).
- **AI hook:** portfolio optimization, diet planning, scheduling — and the SVM you'll meet in ML is "just" a QP.
- **Beats:** LP standard form; feasible polytopes; optima at vertices (picture) · modeling drills: diet problem, allocation (from EE364a archives) · QP: quadratic bowl over a polytope · CVXPY live: 10 lines per problem · duality of LP (prices again) · what solvers do inside ⭐⭐⭐ (simplex/interior-point, one slide each) · "least squares, LP, QP are technology" framing.
- **Derivation ⭐:** none heavy — modeling lecture; verify a small LP's optimum at a vertex by hand.
- **Reading:** Boyd Ch 4.3–4.4; [CVXPY docs](https://www.cvxpy.org/) examples.
- **Reuse:** ml `svm-cvxopt.ipynb` (QP via cvxopt — modernize to cvxpy); LP is fresh (inventory gap).
- **Insight line:** *Don't solve optimization problems — recognize them.*

**→ T8 (after L17), T9 (after L18), T10 (after L20), T11 (after L21).** **Quiz 4** after L21. **A2 (CVXPY)** goes out.

## Module 5 · Information Theory ↔ Machine Learning (L22–L24)

### L22 · Surprise & Entropy
- **Spine:** information is surprise, measured in bits; entropy is the average surprise of a source — and it is a hard floor on how briefly you can describe it.
- **AI hook:** "GPT-4 has ~1 bit per character of English uncertainty" — what does that even mean?
- **Beats:** MacKay's games: 12-coins puzzle, 20 questions / submarine battleship (information gain) · self-information −log p (axioms: rare = surprising, independent surprises add) · entropy as expected surprise · entropy of biased coins (the ∩ curve), dice, English letters (computed live from a corpus) · maximum entropy: uniform (and Gaussian for fixed variance — L13 debt paid, via L19's Lagrange!) · entropy as "questions needed" (guessing-game equivalence).
- **Derivation ⭐:** H(X) maximized by the uniform distribution (Lagrange multipliers — Module 4 payoff).
- **Reading:** MacKay Ch 2 + Ch 4 (guided); MacKay video lecture 1 (assigned — it's wonderful).
- **Reuse:** pml `Information-Theory.tex` (first half), dl `lec00c` Marp deck (surprise→entropy arc); interactive `info-theory`; ml `entropy.ipynb`.
- **Insight line:** *Information is surprise; entropy is its average.*

### L23 · Compression & Optimal Codes
- **Spine:** to compress is to bet on probabilities: give short names to likely things. The optimal bet spends −log p bits, and Huffman's algorithm builds it greedily.
- **AI hook:** "language models are compressors" — a 7B LLM beats gzip on text; why is prediction the same job as compression?
- **Beats:** codes, prefix-free property (decodability) · code lengths ↔ probabilities (Kraft inequality, light) · source coding theorem (stated, plausibility via typicality cartoon) · Huffman construction live on "MATHEMATICS" letter counts · achieved bits/char vs entropy on real text · noisy channels in brief: repetition code vs Hamming(7,4) on a corrupted image (MacKay Ch 1 demo) · channel capacity exists (stated) ⭐⭐⭐.
- **Derivation ⭐:** Huffman tree construction + its expected length vs entropy on the worked example.
- **Reading:** MacKay Ch 4–5 (Huffman: 5); Ch 1 for the channel demo.
- **Reuse:** pml `Information-Theory.tex` (Huffman frames — the only existing Huffman asset); pml `information-theory.ipynb`.
- **Insight line:** *A good predictor and a good compressor are the same object.*

### L24 · KL Divergence & the Grand Unification
- **Spine:** cross-entropy is the price of betting with the wrong distribution; KL is the overcharge; minimizing it is exactly maximum likelihood. One identity unifies Modules 3 and 5.
- **AI hook:** the loss curve of every language model ever trained is measured in cross-entropy (nats/bits per token) — now you can read it.
- **Beats:** wrong-code story: encode source p with code built for q → pay H(p) + KL(p‖q) (Olah's visual treatment) · KL properties: ≥ 0 (Gibbs), = 0 iff p=q, asymmetric (both directions visualized — mode-seeking vs mass-covering) · cross-entropy H(p,q) · **the identity chain: min CE = min KL = max likelihood** (derived on one slide, slowly) · perplexity = 2^H · mutual information in brief (channel callback) · course-wide loss table revisited and completed.
- **Derivation ⭐:** minimizing cross-entropy over q equals MLE; Gibbs' inequality via Jensen ⭐⭐.
- **Reading:** MacKay Ch 2.6, Ch 8 (MI, skim); [Olah — Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (assigned).
- **Reuse:** dl `lec00c` (KL/CE half); interactive `info-theory` (drag distributions, watch KL).
- **Insight line:** *Cross-entropy is the price of believing the wrong distribution.*

**→ T12 (after L24).** **A3 (compressor + n-gram LM)** goes out.

## Module 6 · Markov Chains & the Payoff (L25–L26)

### L25 · Markov Chains
- **Spine:** when the next state depends only on the current one, the entire process is a matrix; its long-run behavior is that matrix's top eigenvector.
- **AI hook:** PageRank, finally resolved (the web as a Markov chain); also: weather models, board games, text.
- **Beats:** sequences and the Markov assumption (memorylessness, honest limits) · joint probability factorizes: p(x₁..xₙ) = p(x₁)Πp(xᵢ|xᵢ₋₁) · transition matrix + state diagram (Setosa explorable in class) · evolution = repeated matrix multiplication (L4/L5 full circle) · stationary distribution = eigenvector with λ=1 (power iteration = "just run the chain") · conditions in brief (irreducible/aperiodic, pictures not proofs) · PageRank worked on a 4-page web with the damping factor.
- **Derivation ⭐:** stationary distribution of a 2-state chain, two ways (solve πP = π; power iteration).
- **Reading:** MacKay pointers; MML Ch 4 callback; [Setosa Markov chains](https://setosa.io/ev/markov-chains/).
- **Reuse:** pml `MCMC.tex` (Markov-properties half — extract, drop the MCMC); pml `markov-chain.ipynb`; Imperial's PageRank assignment design (serves L5 AND L25).
- **Insight line:** *A Markov chain is a matrix; its destiny is an eigenvector.*

### L26 · Learning Sequences + Course Finale
- **Spine:** fitting a Markov chain is counting (that's MLE); scoring it is cross-entropy; sampling it generates text. You have now trained a language model — everything after this is scale.
- **AI hook:** train the n-gram model on Shakespeare live; generate; laugh; then show the same loss curve from a real LLM run.
- **Beats:** MLE for transition probabilities = normalized counts (derived — Module 3 full circle, with Lagrange for the simplex constraint — Module 4 cameo) · Dirichlet smoothing = conjugate prior (L15 cameo) · character bigram/trigram LM on real text (makemore-style) · evaluate: perplexity (L24 cameo) · numerically: log-space everything (L2 cameo) · sample and generate · **the recap slide: one artifact, every module** (the Pillar-1 table, now all checked off) · what's next: ES 335 ML, ES 667 DL — "attention is just a smarter memory than one-step Markov" · course insight ledger, all 26 lines on one slide.
- **Derivation ⭐:** MLE for transition matrix via Lagrange on row-simplex constraints.
- **Reading:** MacKay Ch 2 revisited; Karpathy makemore part 1 (optional video).
- **Reuse:** pml `markov-chain.ipynb`; makemore bigram pattern; the T13/A3 notebook is this lecture's live artifact.
- **Insight line:** *You just trained a language model — everything else is scale.*

**→ T13 (after L26).** **Quiz 5 / Endsem** window.

---

## What this plan deliberately cuts

- **Nonlinear dimensionality reduction** (t-SNE/UMAP from the Spring-2025 run) — belongs to the ML course; PCA is our stopping point.
- **MCMC/sampling algorithms beyond inverse-CDF** — rejection/importance sampling demoted to ⭐⭐⭐ pointers in L12; full treatment belongs to PML.
- **HMMs** — one teaser slide in L26; Rabiner stays a reference.
- **Arithmetic coding, channel coding theorems in depth** — MacKay pointers only.
- **EM algorithm** (colleague covered it) — needs latent-variable models we don't build; deferred to ML/PML.
- **Measure theory, proofs of CLT/consistency** — stated with pictures, cited, not proved.
- **SGD and deep-net optimizers** — one ⭐⭐⭐ slide in L17; that story belongs to ES 667.

## Continuity ledger (planted → paid off)

| Planted | Paid off |
|---|---|
| L2 log-sum-exp | L10 step-size war, L14 log-likelihood, L26 log-space LM |
| L5 eigenvectors / PageRank tease | L13 Gaussian ellipses, L25 stationary distributions |
| L6 low-rank / LoRA tease | ES 667 |
| L7 Taylor | L16–L18 (GD, Newton derived from it) |
| L14 NLL = loss | L24 CE = MLE identity, L26 LM training |
| L15 conjugacy | L26 Dirichlet smoothing |
| L19 max-entropy example | L22 uniform/Gaussian max-entropy |
| L22–L24 codes/CE | L26 perplexity |
