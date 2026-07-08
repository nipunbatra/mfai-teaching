# MFAI Course Design — Making Math Land for Pre-ML Undergrads

**Audience:** undergrads fresh from ES 113 (Data Centric Computing) and ES 114 (Probability, Statistics & Data Visualisation). They know Python/NumPy, plotting, basic discrete probability. They have **not** seen machine learning, gradients of vector functions, or optimization. This course is their bridge into every CS/AI course that follows.

**Design goal:** the course that makes students *see* the math (3Blue1Brown standard), grounded in real AI artifacts, sustainable to teach every year by multiple instructors.

---

## Pillar 1 — One destination: the tiny language model

The last lecture trains a character-level n-gram language model and generates Shakespeare. Working backwards, every module is a strict prerequisite:

| Module | What the language model needs from it |
|---|---|
| Machine numbers | probabilities underflow → log-space, log-sum-exp |
| Linear algebra | transition matrices, stationary distributions (eigenvectors) |
| Calculus + autodiff | gradients of the loss, backprop |
| Probability + MLE | the model *is* a distribution; fitting = MLE |
| Optimization | gradient descent actually finds the parameters |
| Information theory | cross-entropy loss, perplexity, "LLMs are compressors" |
| Markov chains | the model class itself |

State this in L01 with a "course map" slide and revisit it at every module boundary ("you can now build *this much* of the language model"). A course with a visible destination feels designed, not assembled.

## Pillar 2 — Picture → numbers → code → symbols (the 3b1b contract)

Every core concept is introduced in this order, no exceptions:

1. **Picture**: a geometric/visual intuition (animated figure or interactive)
2. **Numbers**: a worked numeric example small enough to verify by hand (2×2 matrices, 3-state chains, 2-parameter models)
3. **Code**: the same example in ~5 lines of NumPy/PyTorch on the slide
4. **Symbols**: only now the general definition/theorem

The proposal's topics are standard; this ordering is the course's identity. Concretely: eigenvectors arrive as "directions that don't turn" with an interactive before the word "characteristic polynomial" appears; KL divergence arrives as "extra bits paid for using the wrong code" before its integral.

## Pillar 3 — An AI hook opens every lecture

Each lecture starts with a real AI capability or failure that the day's math explains, resolved by the end:

- L2: a training loss goes NaN live → floating point
- L5: how Google ranked the web → eigenvectors (resolved fully in L25)
- L6: how LoRA fine-tunes a 7B model with 0.1% of the parameters → low rank
- L11: how PyTorch differentiates a million-parameter function in one backward pass
- L14: where cross-entropy loss actually comes from → MLE
- L17: why the same learning rate diverges on one problem and crawls on another → conditioning
- L23: why a 7B LLM is also a state-of-the-art file compressor → optimal codes

## Pillar 4 — Difficulty valves

Cliffs for this audience and the fix:

- **L9 (Jacobian/Hessian)**: heaviest notation of the course. Valve: everything on ℝ²→ℝ² examples first; the general case is a "same picture, more rows" slide. One reproducible derivation only (Hessian of a quadratic form).
- **L19–L20 (Lagrange/duality/KKT)**: keep *one* fully worked geometric example (maximize xy subject to x+y=1) running through both lectures; duality demoted to "certificate" intuition with the SVM payoff deferred to the ML course.
- **General rule: one reproducible derivation per lecture**, everything else is plausibility. Star ratings (⭐ core / ⭐⭐ should / ⭐⭐⭐ optional) in teaching guides, as in dl-teaching.

## Pillar 5 — Reuse ledger (sustainability)

The content is ~60% assembled from existing repos; conversion, not creation, is most of the work:

| Source | What we take | Cost |
|---|---|---|
| dl-teaching (Marp) | lec00 MLE recap, lec00b Bayes/MAP/KL, lec00c info theory, optimizer decks | Drop-in, re-theme + re-pitch for pre-ML audience |
| pml-teaching (beamer) | MLE.tex, MAP.tex (conjugates), Information-Theory.tex (Huffman!), Sampling, MCMC.tex (Markov chains), calculus-terms.tex | Convert beamer → Marp |
| ml-teaching (beamer) | gradient-descent.tex (72 frames), convexity.tex, kkt-conditions.tex, contour/Taylor decks, mathematical-ml.tex | Convert beamer → Marp |
| psdv-teaching | pca.tex, probability notebook suite, bivariate Gaussian derivation | Convert / link notebooks directly |
| ~/git/interactive (72 articles) | autograd, numerical-tricks, mle-map-coin, bayesian-posterior, info-theory, optimizer-race, multivariate-normal, … | Link/embed as-is |
| **Build fresh** | IEEE-754 internals (L2), dedicated eigen (L5) & SVD (L6) decks, LP (L21), standalone Markov chains (L25–26), Lagrange typeset (L19–20 exist only as scans) | New decks |

## Pillar 6 — Tutorials are the practice engine

13 × 80 min, hybrid by design: ~40 min pen-and-paper worksheet (exam-style fluency) + ~40 min notebook (the same idea computationally). Worksheets are the colleague-run's proven format; notebooks are ours. Both published on the site; tutorials are formative (attendance/effort credit only), assessment mirrors Nipun's usual quizzes + midsem + endsem pattern.

## Pillar 7 — The insight ledger

One-sentence takeaway closing every lecture, collected on one page:

- L2: *Computers don't do real numbers — they do ~4 billion of them, unevenly spaced.*
- L5: *Eigenvectors are the directions a matrix can't turn.*
- L6: *Every matrix is a rotation, a stretch, and a rotation.*
- L7: *Calculus is the art of replacing a function with a line.*
- L11: *Backprop is just the chain rule with memoization.*
- L14: *Every loss function is a negative log-likelihood in disguise.*
- L15: *Every regularizer is a prior in disguise.*
- L17: *The learning rate is how far you trust a linear approximation.*
- L20: *KKT is Lagrange with inequalities and receipts.*
- L23: *A good predictor and a good compressor are the same object.*
- L24: *Cross-entropy is the price of believing the wrong distribution.*
- L26: *You just trained a language model — everything else is scale.*

---

## Execution order (priority)

1. **Lecture plan + site skeleton** — done first so the semester has a visible spine.
2. **Fresh-build decks** (L2 floating point, L5/L6 eigen/SVD, L25/26 Markov) — these have no existing source.
3. **Beamer → Marp conversions** (probability, MLE/MAP, optimization core) — mechanical, high volume.
4. **Tutorials T1–T13** — worksheet + notebook pairs, one week ahead of delivery.
5. **Interactives**: link existing 72; build the few missing (IEEE-754 explorer, eigenvector "doesn't turn" toy, Huffman builder).
6. **Teaching guides** per lecture, dl-teaching template (spine / 80-min plan / stumbles / closing line).
