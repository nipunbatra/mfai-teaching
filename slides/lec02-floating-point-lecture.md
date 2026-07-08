---
marp: true
theme: mfai
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Floating Point

## Lecture 2 · How Machines See Numbers

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# Last time, we left a body at the crime scene

617 steps of beautiful training, then `nan` forever:

![w:800px](figures/lec01/svg/nan_loss.svg)

Today we solve it — and learn the fix used inside **every** ML framework you will ever touch.

---

# Exhibit B · a one-line scandal

Before the big mystery, a small one. Type this into any Python prompt:

```python
>>> 0.1 + 0.2 == 0.3
False

>>> 0.1 + 0.2
0.30000000000000004
```

Not a Python bug. The same happens in C, Java, Rust, JavaScript, Excel, your calculator app…

<div class="popquiz">

**Guess before we continue:** is `0.5 + 0.25 == 0.75` also `False`?
Commit to an answer — we'll resolve it (and you'll see *exactly why*) within the hour.

</div>

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why fixed-size **integers and fixed point** can't serve ML — and what dynamic range is.
2. Decode and encode **IEEE-754 float32** by hand (sign / exponent / mantissa).
3. Reason about the **unevenly spaced** float number line: machine epsilon, `x + 1 == x`.
4. Predict **overflow / underflow**: why `exp(89)` breaks float32.
5. Spot **catastrophic cancellation** and rewrite formulas to dodge it.
6. Derive and apply the **log-sum-exp trick** and the numerically stable softmax.
7. Choose between **float32 / float16 / bfloat16** — and say why DL picked bfloat16.

---

<!-- _class: section-divider -->

### PART 1

# Before floats

Integers, fixed point, and why ML breaks both

---

# Integers · exact, but boxed in

A 32-bit integer stores any whole number in $[-2^{31},\ 2^{31}-1] \approx \pm 2.1 \times 10^9$ — **exactly**.

```python
>>> 2**31 - 1          # largest int32
2147483647
```

- Arithmetic on integers is *perfect*: no rounding, ever.
- But there are no fractions — and beyond the box, classic C/NumPy `int32` **wraps around** to negative numbers.

**Q.** So how do we get decimals? First idea: just *fix* where the point goes.

---

# Fixed point · a decimal point bolted in place

Split 32 bits: 16 for the whole part, 16 for the fraction ("16.16"):

<div class="math-box">

- Smallest step: $2^{-16} \approx 0.000015$ — everywhere on the line
- Largest value: $2^{15} \approx 32{,}768$
- Example: $6.25 = 0000000000000110.0100000000000000_2$

</div>

Simple, fast, and used in DSP chips and retro game consoles. **But look at that range**: nothing above 33k, nothing (nonzero) below 0.000015 — and both limits are hard walls.

---

# Why ML kills fixed point · dynamic range

One training run of one neural network routinely contains, **simultaneously**:

| Quantity | Typical magnitude |
|---|---|
| probability of a rare token | $10^{-30}$ and below |
| a small gradient | $10^{-8}$ |
| a learning rate | $10^{-3}$ |
| activations, logits | $1$ to $10^{4}$ |
| a loss spike, a softmax numerator | $10^{10}$ and beyond |

<div class="keypoint">

That's **40+ orders of magnitude** in one program. A fixed decimal point serves ~9 of them. We need a number format where the point **floats**.

</div>

---

# The rescue idea is 400 years old · scientific notation

Chemists never write Avogadro's number as $602{,}214{,}076{,}000{,}000{,}000{,}000{,}000$. They write:

$$6.022 \times 10^{23}$$

- **A few significant digits** (the *mantissa*: 6.022) — the precision
- **A magnitude dial** (the *exponent*: 23) — the range

Same digit budget describes atoms ($10^{-27}$ kg) or stars ($10^{30}$ kg): the point *floats* to where it's needed.

<div class="insight">

A floating-point number is scientific notation, in base 2, squeezed into 32 bits. That is the whole idea — the rest is bookkeeping.

</div>

---

<!-- _class: section-divider -->

### PART 2

# IEEE-754

The 32 bits, dissected

---

# The anatomy of a float32

![w:1080px](figures/lec02/svg/float32_anatomy.svg)

---

# The decoding rule

<div class="math-box">

$$\text{value} = (-1)^{\text{sign}} \times \underbrace{1.\text{mantissa}}_{\text{24 significant bits}} \times\ 2^{\,\text{exponent} - 127}$$

- **Sign** (1 bit): 0 is positive, 1 is negative
- **Exponent** (8 bits): stored with a **bias of 127**, so the field $[1, 254]$ covers powers $2^{-126}$ to $2^{127}$ — no separate sign needed for the exponent
- **Mantissa** (23 bits): binary normalized numbers always start "1.", so the leading 1 is **not stored** — a free 24th bit of precision

</div>

Two exponent patterns are reserved: all-zeros (zero & *subnormals*, later ⭐⭐⭐) and all-ones (`inf` & `nan`, soon).

---

# Worked encoding · 6.25, step by step

<div class="math-box">

**Step 1 — binary.** $6.25 = 4 + 2 + 0.25 = 110.01_2$

**Step 2 — normalize.** $110.01_2 = 1.1001_2 \times 2^{2}$

**Step 3 — sign.** positive → $s = 0$

**Step 4 — exponent.** $2 + 127 = 129 = 10000001_2$

**Step 5 — mantissa.** drop the leading "1." → $10010000000000000000000$

$$\boxed{\ 0\ \ 10000001\ \ 10010000000000000000000\ } \; = \texttt{0x40C80000}$$

</div>

Tutorial 1's worksheet has you do this by hand for $-0.75$, $13.5$, and one nasty one…

---

# Worked encoding · 0.1 — and the wheels come off

Multiply by 2, harvest the integer bit, repeat:

<div class="math-box">

$$0.1 \xrightarrow{\times 2} \underline{0}.2 \xrightarrow{\times 2} \underline{0}.4 \xrightarrow{\times 2} \underline{0}.8 \xrightarrow{\times 2} \underline{1}.6 \xrightarrow{\times 2} \underline{1}.2 \xrightarrow{\times 2} \underline{0}.4 \cdots$$

$$0.1 = 0.0\overline{0011}_2 = 0.00011001100110011\ldots_2 \quad \text{— repeats forever!}$$

</div>

Like $1/3 = 0.333\ldots$ in decimal: a perfectly clean fraction, an **infinite** expansion in the wrong base. The mantissa holds 23 bits, so the machine must **round**:

```python
>>> from decimal import Decimal
>>> Decimal(float(np.float32(0.1)))
Decimal('0.100000001490116119384765625')
```

There is **no 0.1** in floating point. There never was.

---

# Mystery of `0.1 + 0.2`, resolved

What float64 *actually stores* (every digit below is exact):

| You typed | The machine stored |
|---|---|
| `0.1` | 0.1000000000000000055511151231257827… |
| `0.2` | 0.2000000000000000111022302462515654… |
| `0.1 + 0.2` | 0.3000000000000000444089209850062616… |
| `0.3` | 0.2999999999999999888977697537484345… |

Two *different* doubles → `==` says `False`. Case closed.

<div class="warning">

Never compare floats with `==`. Use `np.isclose(a, b)` / `math.isclose(a, b)` — every test suite you write in this course will.

</div>

---

<!-- _class: section-divider -->

### PART 3

# The float number line

~4 billion numbers, unevenly spaced

---

# The number line, as the machine sees it

![w:900px](figures/lec02/svg/float_number_line.svg)

---

# Machine epsilon · the gap next to 1.0

<div class="math-box">

**Machine epsilon** $\varepsilon$ = the gap between $1.0$ and the next representable float.

$$\varepsilon_{\text{float32}} = 2^{-23} \approx 1.19 \times 10^{-7} \qquad \varepsilon_{\text{float64}} = 2^{-52} \approx 2.22 \times 10^{-16}$$

</div>

Anything smaller than about $\varepsilon/2$, added to 1, simply **vanishes**:

```python
>>> np.float32(1.0) + np.float32(1e-8) == np.float32(1.0)
True                       # 1e-8 fell into the gap

>>> 1.0 + 1e-8 == 1.0      # float64's gap is much smaller
False
```

Rule of thumb: float32 carries **~7 reliable decimal digits**; float64 carries ~16.

---

# At $2^{24}$, adding 1 does nothing

The gap doubles at every power of 2. By $2^{24} = 16{,}777{,}216$ the gap between neighboring float32s is **2** — so $+1$ rounds right back:

```python
>>> x = np.float32(16_777_216)     # 2**24
>>> x + np.float32(1) == x
True
>>> np.spacing(x)                  # the gap at x
2.0
```

<div class="warning">

A float32 counter that increments by 1 **stops counting at 16.7 million** — silently. Count in integers; accumulate long sums (losses, metrics) in float64.

</div>

---

# Rounding · one tiny lie per operation

Each arithmetic result is computed exactly, then **rounded** to the nearest float:

<div class="math-box">

$$\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le \tfrac{\varepsilon}{2}$$

</div>

- Default mode: **round to nearest, ties to even** — ties go to the even last bit, so up/down rounds don't accumulate a bias
- Other IEEE modes exist (toward $0$, toward $\pm\infty$) — used for interval arithmetic, not our concern

<div class="keypoint">

One operation = one tiny, *bounded* lie ($10^{-8}$ relative in float32). The drama is never one lie — it's how lies **compound**. That's Part 4.

</div>

---

# When arithmetic escapes the range · `inf` and `nan`

IEEE-754 doesn't crash on impossible arithmetic — it returns special values:

| Expression (NumPy) | Result | Meaning |
|---|---|---|
| `1.0 / 0.0` | `inf` | overflow / true limit |
| `-1.0 / 0.0` | `-inf` | |
| `0.0 / 0.0` | `nan` | "not a number" — no defensible value |
| `inf - inf`, `inf / inf` | `nan` | |
| `np.log(-1.0)`, `np.sqrt(-1.0)` | `nan` | |

<div class="warning">

**NaN is contagious**: any operation touching `nan` returns `nan`. And `nan == nan` is `False` — the only value not equal to itself (`x != x` is the classic NaN test; prefer `np.isnan`). One NaN at step 618 → *every* weight is NaN by step 619.

</div>

---

# Overflow arrives shockingly early

The range boundary isn't some astronomical corner case. For `exp`:

| Computation | float32 (max ≈ 3.4×10³⁸) | float64 (max ≈ 1.8×10³⁰⁸) |
|---|---|---|
| `exp(88)` | 1.65×10³⁸ ✓ | fine |
| `exp(89)` | **inf** | fine |
| `exp(709)` | inf | 8.2×10³⁰⁷ ✓ |
| `exp(710)` | inf | **inf** |
| `exp(-104)` | **0.0** (underflow) | fine |

<div class="keypoint">

A score of **89** is all it takes to overflow float32. Your model *will* produce one — remember Mystery 1's logits were ~1000. The fix is coming in Part 5.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Catastrophic cancellation

When subtraction eats your digits

---

# The idea · subtraction of near-equals is a digit shredder

Work in 8 significant decimal digits (like a float32-ish machine):

<div class="math-box">

$$1.2345678 - 1.2345677 = 0.0000001$$

Both inputs had **8** correct digits. The result has **1** — and if the inputs were themselves rounded (they always are), that surviving digit is *pure noise*.

</div>

<div class="keypoint">

Adding same-sign numbers keeps relative error tame. **Subtracting nearly equal numbers** cancels the trustworthy leading digits and promotes the rounded-off garbage to the front. Nothing "overflows" — the answer is just quietly wrong.

</div>

---

# Worked crime 1 · the quadratic formula

Solve $x^2 - 10000x + 1 = 0$ in float32. The small root, two algebraically identical ways:

<div class="math-box">

**Naive** (subtract near-equals: $b \approx \sqrt{b^2 - 4}$):

$$x = \frac{b - \sqrt{b^2 - 4}}{2} = \frac{10000 - 9999.9998\ldots}{2} \;\xrightarrow{\text{float32}}\; \mathbf{0.0}$$

**Stable** (rationalize — same numbers, now *added*):

$$x = \frac{2}{b + \sqrt{b^2 - 4}} = \frac{2}{19999.9998} \;\xrightarrow{\text{float32}}\; \mathbf{1.0000\times 10^{-4}} \; ✓$$

</div>

The naive formula returned **zero** — a 100% error — with no warning of any kind.

---

# The two formulas, head to head

![w:860px](figures/lec02/svg/cancellation_error.svg)

Same math on paper. In float32, one drifts to 100% error; the other sits at machine epsilon.

---

# Worked crime 2 · variance, two ways

Textbooks give two equal formulas for variance. Try both on `[10000.0, 10000.1, 10000.2]` (true variance ≈ **0.00667**), in float32:

<div class="math-box">

**One-pass** $\ \mathbb{E}[X^2] - (\mathbb{E}[X])^2$: two huge, nearly equal numbers —

$$100001992.0 - 100002011.7 = \mathbf{-19.7} \quad\text{(a *negative* variance!)}$$

**Two-pass** $\ \mathbb{E}[(X - \bar{X})^2]$: subtract *first*, while numbers are small —

$$\text{mean of } \{(-0.1)^2,\ 0^2,\ (0.1)^2\} = \mathbf{0.00668} \; ✓$$

</div>

<div class="keypoint">

Near $10^8$, float32's grid spacing is **8** — the true answer 0.00667 is a thousand times *smaller than the gap between representable numbers* there. No algorithm that visits $10^8$ can see it. Keep intermediate quantities small.

</div>

---

<!-- _class: section-divider -->

### PART 5

# The ML fixes

Log-space, log-sum-exp, and the stable softmax

---

# Fix 1 · probabilities live in log-space

A language model scores a 1000-character text; each character has probability ≈ 0.03:

<div class="math-box">

$$P(\text{text}) = \prod_{i=1}^{1000} p_i \approx 0.03^{1000} \approx 10^{-1523} \;\xrightarrow{\text{even float64}}\; \mathbf{0.0}$$

**In log-space** — products become sums, tiny becomes ordinary:

$$\log P(\text{text}) = \sum_{i=1}^{1000} \log p_i \approx 1000 \times (-3.51) = -3507 \quad ✓ \text{ (a boring, safe float)}$$

</div>

<div class="keypoint">

Multiply probabilities → **add log-probabilities**. This is why every loss you will ever meet is a *log*-likelihood — floating point demands it (and L14 will show the deeper reason).

</div>

---

# Fix 2 · log-sum-exp, derived in three lines

In log-space we still must *normalize*: compute $\log \sum_i e^{x_i}$ — but the $e^{x_i}$ overflow! Pull out the max $m = \max_i x_i$:

<div class="math-box">

$$\log \sum_i e^{x_i} \;=\; \log \sum_i e^{m} \, e^{x_i - m} \;=\; \log \Big( e^m \sum_i e^{x_i - m} \Big) \;=\; m + \log \sum_i e^{x_i - m}$$

Every exponent $x_i - m \le 0$, so every $e^{x_i - m} \in (0, 1]$: **nothing can overflow**, and the largest term is exactly $1$ — no total underflow either.

**Check** on Mystery 1's scores $[1000, 1001, 1002]$:

$$\mathrm{LSE} = 1002 + \log(e^{-2} + e^{-1} + e^{0}) = 1002 + \log(1.503) = 1002.41 \; ✓$$

</div>

---

# Fix 3 · the stable softmax

Softmax is **shift-invariant** — multiply top and bottom by $e^{-c}$:

$$\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} \quad \text{for any } c$$

Choose $c = \max_j z_j$. On the crime-scene scores $\mathbf{z} = [1000, 1001, 1002]$:

<div class="math-box">

| | $z_i$ | $z_i - 1002$ | $e^{z_i - 1002}$ | probability |
|---|---|---|---|---|
| naive | $e^{1000} =$ `inf` → | | `inf / inf` | **nan** |
| stable | | $-2, -1, 0$ | $0.135,\ 0.368,\ 1.0$ | $0.090,\ 0.245,\ \mathbf{0.665}$ ✓ |

</div>

Identical mathematics. One works on real hardware; the other burned down a training run.

---

# The fix, in pictures

![w:1000px](figures/lec02/svg/softmax_stability.svg)

---

<!-- _class: code-heavy -->

# The fixes, in code

```python
def softmax_naive(z):
    e = np.exp(z)                # inf for z ~ 89+  → nan
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())      # largest exponent is now 0
    return e / e.sum()           # same answer, can't overflow

def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())
```

```python
>>> softmax_stable(np.array([1000., 1001., 1002.]))
array([0.09003057, 0.24472847, 0.66524096])
```

Five lines you will reuse in Tutorial 1, in L14 (MLE), in L24 (cross-entropy), and in L26's language model.

---

# Mystery 1 · case closed

What actually happened at step 618:

1. The model got confident → logits grew past **±89** in float32
2. `exp(logit)` → `inf` → `inf / inf` → **nan** in the softmax
3. NaN loss → NaN gradients → NaN weights, **everywhere, permanently**

<div class="realworld">

This is why PyTorch's `CrossEntropyLoss` takes **raw logits**, not probabilities: it fuses log-softmax into the loss and applies *exactly* the subtract-the-max trick you just derived. The fix for a billion-dollar training run is one line of first-year algebra.

</div>

---

<!-- _class: section-divider -->

### PART 6

# Precision in deep learning

float64 · float32 · float16 · bfloat16

---

# The four formats that run the AI world

| Format | Bits | Exponent | Mantissa | Max value | Machine ε | ~Digits |
|---|---|---|---|---|---|---|
| float64 | 64 | 11 | 52 | 1.8×10³⁰⁸ | 2.2×10⁻¹⁶ | 15.9 |
| float32 | 32 | 8 | 23 | 3.4×10³⁸ | 1.2×10⁻⁷ | 7.2 |
| float16 | 16 | 5 | 10 | **65 504** | 9.8×10⁻⁴ | 3.3 |
| bfloat16 | 16 | 8 | 7 | 3.4×10³⁸ | 7.8×10⁻³ | 2.4 |

Halving the bits doubles the training throughput and halves the memory — modern GPUs are *built* around 16-bit math. But look at float16's max value…

---

# Range vs precision · the 16-bit dilemma

![w:1000px](figures/lec02/svg/fp_formats.svg)

---

# Why deep learning picked bfloat16

float16 spent its bits on precision and starved the exponent — `exp(12)` already overflows its 65,504 ceiling. **bfloat16 keeps float32's 8 exponent bits** and pays with mantissa:

| | float16 | bfloat16 |
|---|---|---|
| `exp(12)` ≈ 162,755 | **inf** — training dies | 1.63×10⁵ ✓ |
| copy a float32's range? | no — overflow risk everywhere | yes — same exponent, just chop |
| digits of precision | 3.3 | 2.4 |

<div class="keypoint">

DL chose **range over precision**: overflow makes NaNs (fatal, as we saw); low-precision noise mostly averages out across millions of SGD updates. Losing digits is survivable. Losing the exponent is not.

</div>

---

# Pop quiz · you are now floating-point literate

<div class="popquiz">

**Q1.** `np.exp(np.float16(12.0))` returns…?

**Q2.** `np.float32(16_777_216) + np.float32(1.0)` returns…?

**Q3.** From the start of the lecture: is `0.5 + 0.25 == 0.75` `True` or `False` — and why?

Commit before the next slide.

</div>

---

# Pop quiz · answers

<div class="math-box">

**A1.** `inf` — $e^{12} \approx 162{,}755 > 65{,}504$, float16's ceiling. (bfloat16 shrugs: $\approx 1.6 \times 10^5$.)

**A2.** `16777216.0` — unchanged. At $2^{24}$ the float32 gap is 2, so $+1$ rounds back down.

**A3.** **`True`!** $0.5 = 2^{-1}$, $0.25 = 2^{-2}$, $0.75 = 2^{-1} + 2^{-2}$ — all *exactly* representable in binary. Floats aren't sloppy; they are **exact binary fractions**. Trouble only starts when your decimal (0.1) has no finite binary form.

</div>

---

# ⭐⭐⭐ Optional · Kahan summation

Add `np.float32(0.1)` ten million times. True answer: $10^6$. A naive loop returns…

<div class="math-box">

$$\texttt{1087937.0} \quad \text{— off by } 8.8\%$$

Once the sum reaches $10^6$, the grid gap is $0.0625$: each $+0.1$ rounds badly, ten million times, *in the same direction*.

</div>

```python
s, c = 0.0, 0.0
for x in xs:
    y = x - c            # re-inject the error we still owe
    t = s + y            # big + small: low digits of y get dropped...
    c = (t - s) - y      # ...(t - s) - y recovers exactly what was dropped
    s = t
```

Kahan's compensated sum tracks the rounding error in `c` and repays it — error stays ~ε **independent of n**. (`np.sum` fights the same war with pairwise summation.)

---

# ⭐⭐⭐ Optional · subnormals: the fine print near zero

The smallest *normal* float32 is $2^{-126} \approx 1.18 \times 10^{-38}$. Without further tricks there'd be a **moat around zero**, and `x - y == 0` could be true for `x != y`(!).

<div class="math-box">

**Subnormals**: when the exponent field is all zeros, drop the implicit leading 1 and let the mantissa alone carry the value — filling the moat with evenly spaced numbers down to $2^{-149} \approx 1.4 \times 10^{-45}$ ("gradual underflow").

</div>

<div class="realworld">

The catch: some hardware handles subnormals in microcode — **orders of magnitude slower**. GPUs and DL runtimes often enable *flush-to-zero* modes that trade this correctness for speed. If a kernel inexplicably slows down as values shrink toward zero: suspect subnormals.

</div>

---

# Practice problems

Try on paper; verify in the Tutorial 1 notebook.

<div class="math-box">

**P1.** Encode $-0.75$ as float32 (sign, exponent field, first mantissa bits).

**P2.** What is the largest odd integer float32 represents exactly? (Hint: where does the gap reach 2?)

**P3.** Compute $\mathrm{logsumexp}([-1000, -1001])$ by hand. What would the naive computation return?

**P4.** Explain why `(0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)` in float64 — what algebraic law do floats break?

**P5.** Predict the sign and rough size of the error when the one-pass variance formula meets `[1e6, 1e6 + 1, 1e6 + 2]` in float32.

**P6.** Your float16 training run dies at `exp(12)`. A colleague proposes bfloat16. What exactly changes, and what do you give up?

</div>

---

<!-- _class: summary-slide compact -->

# Lecture 2 — summary

- **Fixed formats fail ML**: one program spans 40+ orders of magnitude → the point must float.
- **IEEE-754 float32** = sign + biased exponent (127) + mantissa with a free leading 1; 6.25 encodes exactly, 0.1 *cannot* — hence `0.1 + 0.2 != 0.3`.
- **The float line is unevenly spaced**: gaps double at each power of 2; $\varepsilon_{32} \approx 10^{-7}$; at $2^{24}$, $x + 1 == x$.
- **Special values**: overflow → `inf`, indeterminate → `nan`; NaN propagates and `nan != nan`.
- **Catastrophic cancellation**: subtracting near-equals shreds digits — rewrite the formula (quadratic roots, two-pass variance).
- **The ML fixes**: log-space probabilities, log-sum-exp, subtract-the-max softmax — the exact code inside every framework's loss.
- **bfloat16** = float32's range at half the bits: DL trades precision for range, because overflow (NaN) is fatal and noise is not.

### Read before Lecture 3

Goldberg (1991), §1–2 · Solomon, *Numerical Algorithms*, Ch. 1–2 (free PDF).

### Next lecture

Mystery 3: how "king − man + woman ≈ queen" can possibly work — vectors as the geometry of data.

---

# The one-sentence takeaway

<div class="insight">

**Computers don't do real numbers — they do ~4 billion of them, unevenly spaced.**

</div>

<div class="notebook">

**Tutorial 1** (this week): IEEE-754 encoding by hand · then break `0.1 + 0.2`, cancellation, and softmax yourself in NumPy — and fix them.

</div>

*Next (L3) · Vectors: the geometry of data — where lists of numbers acquire length, angle, and meaning.*
