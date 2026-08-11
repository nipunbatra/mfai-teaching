// Floating Point: How Machines See Numbers — Lecture 2 · Mathematical Foundations for AI
// Compile from the repo root:
//   typst compile --root . lecture2/L2-floating-point.typ
//   typst compile --root . --input handout=true lecture2/L2-floating-point.typ
// Theme, palette, helpers live in common/metropolis.typ; chalkdust in common/mldiag.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Floating Point],
  subtitle: [How machines see numbers],
)

#title-slide()

= A softmax computation that returns `nan`

== The numerical failure from Lecture 1 #V

617 steps of beautiful training, then `nan` forever:

#fig("/lecture1/figures/nan_loss.svg", w: 68%)

#pause
Today we solve it — and derive the stable log-softmax computation used by major ML frameworks.

== A decimal calculation worth checking

First, type this into any Python prompt:

#codebox[```python
>>> 0.1 + 0.2 == 0.3
False

>>> 0.1 + 0.2
0.30000000000000004
```]

#pause
Not a Python bug. The same happens in C, Java, Rust, JavaScript, Excel, your calculator app…

#pause
#notebox[*Guess before we continue:* is `0.5 + 0.25 == 0.75` also `False`? Commit to an answer — we'll resolve it (and you'll see _exactly why_) within the hour.]

== Learning outcomes

By the end of this lecture you will be able to:

+ Explain why one fixed-point scale cannot cover a training run's full *dynamic range*, and how quantized ML works around that limit.
+ Decode and encode *IEEE-754 float32* by hand (sign / exponent / mantissa).
+ Reason about the *unevenly spaced* float number line: machine epsilon, `x + 1 == x`.
+ Predict *overflow / underflow*: why `exp(89)` breaks float32.
+ Spot *catastrophic cancellation* and rewrite formulas to dodge it.
+ Derive and apply the *log-sum-exp trick* and the numerically stable softmax.
+ Choose between *float32 / float16 / bfloat16* and explain why many accelerators favour bfloat16 or mixed precision.

= Before floats: integers and fixed point

== Integers · exact, but boxed in

A 32-bit integer stores any whole number in $[-2^31, 2^31 - 1] approx plus.minus 2.1 times 10^9$ — *exactly*.

#codebox[```python
>>> 2**31 - 1          # largest int32
2147483647
```]

#pause
- Arithmetic on integers is _perfect_: no rounding, ever.
#pause
- But there are no fractions — and beyond the box, NumPy `int32` wraps modulo $2^32$. Languages differ: signed overflow in C is undefined, so never rely on wrapping unless the type specifies it.

#pause
*Q.* So how do we get decimals? First idea: just _fix_ where the point goes.

== Fixed point · a decimal point bolted in place

Split 32 bits: 16 for the whole part, 16 for the fraction ("16.16"):

- Smallest step: $2^(-16) approx 0.000015$ — everywhere on the line
- Largest value: $2^15 approx 32,768$
- Example: $6.25 = 0000000000000110.0100000000000000_2$

#pause
Simple, fast, and used in DSP chips and retro game consoles. *But look at that range*: nothing above 33k, nothing (nonzero) below 0.000015 — and both limits are hard walls.

== Why one fixed-point scale is not enough

One training run of one neural network routinely contains, *simultaneously*:

#table(
  columns: (1fr, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Quantity*], [*Typical magnitude*]),
  [probability of a rare token], [$10^(-30)$ and below],
  [a small gradient], [$10^(-8)$],
  [a learning rate], [$10^(-3)$],
  [activations, logits], [$1$ to $10^4$],
  [a loss spike, a softmax numerator], [$10^10$ and beyond],
)

#pause
#result[That's *40+ orders of magnitude* in one program. One 32-bit fixed-point scale covers only a small window, so general-purpose training needs a format whose scale can move.]

== Quantized ML uses many local scales

Fixed point remains useful when the program does *not* rely on one global scale:

- assign a scale and zero-point to each tensor or channel,
- perform fast integer matrix products,
- rescale between layers using calibration data or learned quantization parameters.

#pause
#result[Quantized inference works around fixed point's narrow range by using many locally chosen scales. General training still usually relies on floating point or mixed precision.]

== The rescue idea is 400 years old · scientific notation

Chemists never write Avogadro's number as $602,214,076,000,000,000,000,000$. They write:

$ 6.022 times 10^23 $

#pause
- *A few significant digits* (the _mantissa_: 6.022) — the precision
- *A magnitude field* (the _exponent_: 23) — controls the range

#pause
Same digit budget describes atoms ($10^(-27)$ kg) or stars ($10^30$ kg): the point _floats_ to where it's needed.

#pause
#result[A floating-point number is scientific notation, in base 2, squeezed into 32 bits. That is the whole idea — the rest is bookkeeping.]

= IEEE-754: the 32 bits, dissected

== The anatomy of a float32 #V

#fig("/lecture2/figures/float32_anatomy.svg", w: 92%)

== The decoding rule

$ "value" = (-1)^"sign" times underbrace(1."mantissa", "24 significant bits") times 2^("exponent" - 127) $

#pause
- *Sign* (1 bit): 0 is positive, 1 is negative
#pause
- *Exponent* (8 bits): stored with a *bias of 127*, so the field $[1, 254]$ covers powers $2^(-126)$ to $2^127$ — no separate sign needed for the exponent
#pause
- *Mantissa* (23 bits): binary normalized numbers always start "1.", so the leading 1 is *not stored* — a free 24th bit of precision

#pause
#notebox[Two exponent patterns are reserved: all-zeros (zero & _subnormals_, later #sym.star.filled#sym.star.filled#sym.star.filled) and all-ones (`inf` & `nan`, soon).]

== Worked encoding · 6.25, step by step #D

*Step 1 — binary.* $6.25 = 4 + 2 + 0.25 = 110.01_2$

#pause
*Step 2 — normalize.* $110.01_2 = 1.1001_2 times 2^2$

#pause
*Step 3 — sign.* positive → $s = 0$

#pause
*Step 4 — exponent.* $2 + 127 = 129 = 10000001_2$

#pause
*Step 5 — mantissa.* drop the leading "1." → $10010000000000000000000$

#pause
#result[$0 quad 10000001 quad 10010000000000000000000 = #raw("0x40C80000")$]

#pause
Tutorial 1's worksheet has you do this by hand for $-0.75$, $13.5$, and one nasty one…

== Worked encoding · 0.1 — and the wheels come off #D

Multiply by 2, harvest the integer bit, repeat:

$ 0.1 arrow.r^(times 2) underline(0).2 arrow.r^(times 2) underline(0).4 arrow.r^(times 2) underline(0).8 arrow.r^(times 2) underline(1).6 arrow.r^(times 2) underline(1).2 arrow.r^(times 2) underline(0).4 dots.c $

#pause
$ 0.1 = 0.0overline(0011)_2 = 0.00011001100110011..._2 quad #text(fill: RED)[— repeats forever!] $

#pause
Like $1/3 = 0.333...$ in decimal: a perfectly clean fraction, an *infinite* expansion in the wrong base. The mantissa holds 23 bits, so the machine must *round*:

#codebox[```python
>>> from decimal import Decimal
>>> Decimal(float(np.float32(0.1)))
Decimal('0.100000001490116119384765625')
```]

#pause
#result[There is *no 0.1* in floating point. There never was.]

== Why `0.1 + 0.2 != 0.3`

What float64 _actually stores_ (every digit below is exact):

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*You typed*], [*The machine stored*]),
  [`0.1`], [0.1000000000000000055511151231257827…],
  [`0.2`], [0.2000000000000000111022302462515654…],
  [`0.1 + 0.2`], [0.3000000000000000444089209850062616…],
  [`0.3`], [0.2999999999999999888977697537484345…],
)

#pause
The two calculations produce different adjacent floating-point values, so `==` returns `False`.

#pause
#alertbox[Do not use `==` when you mean "equal up to rounding" — for example, after inexact decimal arithmetic. Use `np.isclose(a, b)` / `math.isclose(a, b)`. Exact binary values, integer-valued floats, and sentinels can still be compared exactly.]

= The float number line

== The number line, as the machine sees it #V

#fig("/lecture2/figures/float_number_line.svg", w: 72%)

== Machine epsilon · the gap next to 1.0

*Machine epsilon* $epsilon$ = the gap between $1.0$ and the next representable float.

$ epsilon_"float32" = 2^(-23) approx 1.19 times 10^(-7) quad quad epsilon_"float64" = 2^(-52) approx 2.22 times 10^(-16) $

#pause
Anything smaller than about $epsilon\/2$, added to 1, simply *vanishes*:

#codebox[```python
>>> np.float32(1.0) + np.float32(1e-8) == np.float32(1.0)
True                       # 1e-8 fell into the gap

>>> 1.0 + 1e-8 == 1.0      # float64's gap is much smaller
False
```]

#pause
Rule of thumb: float32 carries *\~7 reliable decimal digits*; float64 carries \~16.

== At $2^24$, adding 1 does nothing

The gap doubles at every power of 2. By $2^24 = 16,777,216$ the gap between neighboring float32s is *2* — so $+1$ rounds right back:

#codebox[```python
>>> x = np.float32(16_777_216)     # 2**24
>>> x + np.float32(1) == x
True
>>> np.spacing(x)                  # the gap at x
2.0
```]

#pause
#alertbox[A float32 counter that increments by 1 *stops counting at 16.7 million* — silently. Count in integers; accumulate long sums (losses, metrics) in float64.]

== Rounding · one tiny lie per operation

Each arithmetic result is computed exactly, then *rounded* to the nearest float:

$ "fl"(a compose b) = (a compose b)(1 + delta), quad abs(delta) <= epsilon / 2 $

This standard relative-error model applies when the exact result is a finite normal number; underflow, overflow, and subnormals need separate treatment.

#pause
- Default mode: *round to nearest, ties to even* — ties go to the even last bit, so up/down rounds don't accumulate a bias
#pause
- Other IEEE modes exist (toward $0$, toward $plus.minus infinity$) — used for interval arithmetic, not our concern

#pause
#result[One operation = one tiny, _bounded_ lie ($10^(-8)$ relative in float32). The drama is never one lie — it's how lies *compound*.]

== When arithmetic escapes the range · `inf` and `nan`

IEEE-754 doesn't crash on impossible arithmetic — it returns special values:

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Expression (NumPy)*], [*Result*], [*Meaning*]),
  [`np.float64(1.0) / 0.0`], [`inf`], [division by zero / limiting value],
  [`np.float64(-1.0) / 0.0`], [`-inf`], [],
  [`np.float64(0.0) / 0.0`], [`nan`], ["not a number" — no defensible value],
  [`inf - inf`, `inf / inf`], [`nan`], [],
  [`np.log(-1.0)`, `np.sqrt(-1.0)`], [`nan`], [],
)

#pause
#alertbox[Most arithmetic with `nan` produces `nan`, so one invalid value can spread quickly through a computation graph. Comparisons are different: `nan == nan` is `False` (`x != x` is the classic test; prefer `np.isnan`).]

== Overflow arrives shockingly early

The range boundary isn't some astronomical corner case. For `exp`:

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([*Computation*], [*float32 (max ≈ 3.4×10³⁸)*], [*float64 (max ≈ 1.8×10³⁰⁸)*]),
  [`exp(88)`], [1.65×10³⁸ ✓], [fine],
  [`exp(89)`], [#text(fill: RED)[*inf*]], [fine],
  [`exp(709)`], [inf], [8.2×10³⁰⁷ ✓],
  [`exp(710)`], [inf], [#text(fill: RED)[*inf*]],
  [`exp(-104)`], [#text(fill: RED)[*0.0* (underflow)]], [fine],
)

#pause
#result[In float32, $e^z$ overflows for $z$ near $89$; the opening logits around $1000$ therefore overflow.]

= Catastrophic cancellation

== Subtraction of near-equals is a digit shredder

Work in 8 significant decimal digits (like a float32-ish machine):

$ 1.2345678 - 1.2345677 = 0.0000001 $

#pause
Both inputs had *8* correct digits. The result has *1* — and if the inputs were themselves rounded (they always are), that surviving digit is _pure noise_.

#pause
#result[Adding same-sign numbers keeps relative error tame. *Subtracting nearly equal numbers* cancels the trustworthy leading digits and promotes the rounded-off garbage to the front.]

#pause
Nothing "overflows" — the answer is just quietly wrong.

== Worked example 1: the quadratic formula #D

Solve $x^2 - 10000x + 1 = 0$ in float32. The small root, two algebraically identical ways:

#pause
*Naive* (subtract near-equals: $b approx sqrt(b^2 - 4)$):

$ x = (b - sqrt(b^2 - 4)) / 2 = (10000 - 9999.9998...) / 2 arrow.r.double_("float32") #text(fill: RED)[$bold(0.0)$] $

#pause
*Stable* (rationalize — same numbers, now _added_):

$ x = 2 / (b + sqrt(b^2 - 4)) = 2 / 19999.9998 arrow.r.double_("float32") #text(fill: GREEN)[$bold(1.0000 times 10^(-4))$ ✓] $

#pause
The naive formula returned *zero* — a 100% error — with no warning of any kind.

== The two formulas, head to head #V

#fig("/lecture2/figures/cancellation_error.svg", w: 74%)

#pause
Same math on paper. In float32, one drifts to 100% error; the other sits at machine epsilon.

== Worked example 2: two variance formulas

Textbooks give two equal formulas for variance. Try both on `[10000.0, 10000.1, 10000.2]` (true variance ≈ *0.00667*), in float32:

#pause
*One-pass* $EE[X^2] - (EE[X])^2$: two huge, nearly equal numbers —

$ 100001992.0 - 100002011.7 = #text(fill: RED)[$bold(-19.7)$] quad #text(fill: RED)[(a _negative_ variance!)] $

#pause
*Two-pass* $EE[(X - macron(X))^2]$: subtract _first_, while numbers are small —

$ "mean of" {(-0.1)^2, 0^2, (0.1)^2} = #text(fill: GREEN)[$bold(0.00668)$ ✓] $

#pause
#result[Near $10^8$, float32's grid spacing is *8* — the true answer is a thousand times _smaller than the gap between representable numbers_ there. Keep intermediate quantities small.]

= The ML fixes

== Fix 1 · probabilities live in log-space

A language model scores a 1000-character text; each character has probability ≈ 0.03:

$ P("text") = product_(i=1)^1000 p_i approx 0.03^1000 approx 10^(-1523) arrow.r.double_("even float64") bold(0.0) $

#pause
*In log-space* — products become sums, tiny becomes ordinary:

$ log P("text") = sum_(i=1)^1000 log p_i approx 1000 times (-3.51) = -3507 quad #text(fill: GREEN)[✓ a boring, safe float] $

#pause
#result[Multiply probabilities → *add log-probabilities*. Many probabilistic objectives are negative log-likelihoods (L14); other losses need not be.]

== Fix 2 · log-sum-exp, derived in three lines #D

In log-space we still must _normalize_: compute $log sum_i e^(x_i)$ — but the $e^(x_i)$ overflow! Pull out the max $m = max_i x_i$:

#pause
$ log sum_i e^(x_i) = log sum_i e^m e^(x_i - m) = log (e^m sum_i e^(x_i - m)) = m + log sum_i e^(x_i - m) $

#pause
Every exponent $x_i - m <= 0$, so every $e^(x_i - m) in (0, 1]$: *nothing can overflow* — and the largest term is exactly $1$, so no total underflow either.

#pause
*Check* on the opening scores $[1000, 1001, 1002]$:

$ "LSE" = 1002 + log(e^(-2) + e^(-1) + e^0) = 1002 + log(1.503) = 1002.41 quad ✓ $

== Fix 3 · the stable softmax

Softmax is *shift-invariant* — multiply top and bottom by $e^(-c)$:

$ "softmax"(bold(z))_i = e^(z_i) / (sum_j e^(z_j)) = e^(z_i - c) / (sum_j e^(z_j - c)) quad "for any" c $

#pause
Choose $c = max_j z_j$. For $bold(z) = [1000, 1001, 1002]$:

#table(
  columns: (auto, auto, auto, auto, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [$z_i$], [$z_i - 1002$], [$e^(z_i - 1002)$], [*probability*]),
  [naive], [$e^1000 = $ `inf`], [], [`inf / inf`], [#text(fill: RED)[*nan*]],
  [stable], [], [$-2, -1, 0$], [$0.135, 0.368, 1.0$], [#text(fill: GREEN)[$0.090, 0.245, bold(0.665)$ ✓]],
)

#pause
Identical mathematics. One works on real hardware; the other burned down a training run.

== The fix, in pictures #V

#fig("/lecture2/figures/softmax_stability.svg", w: 84%)

== The fixes, in code

#codebox[```python
def softmax_naive(z):
    e = np.exp(z)                # inf for z ~ 89+  → nan
    return e / e.sum()

def softmax_stable(z):
    e = np.exp(z - z.max())      # largest exponent is now 0
    return e / e.sum()           # same answer, can't overflow

def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())
```]

#pause
#codebox[```python
>>> softmax_stable(np.array([1000., 1001., 1002.]))
array([0.09003057, 0.24472847, 0.66524096])
```]

#pause
Five lines you will reuse in Tutorial 1, in L14 (MLE), in L24 (cross-entropy), and in L26's language model.

== Stable softmax avoids the overflow

What actually happened at step 618:

#pause
+ The model got confident → logits grew past *±89* in float32
#pause
+ `exp(logit)` → `inf` → `inf / inf` → *nan* in the softmax
#pause
+ NaN loss → NaN gradients → NaN weights, *everywhere, permanently*

#pause
#notebox[This is why PyTorch's `CrossEntropyLoss` takes *raw logits*, not probabilities: it fuses log-softmax into the loss and applies _exactly_ the subtract-the-max trick you just derived. The fix for a billion-dollar training run is one line of first-year algebra.]

= Precision in deep learning

== The four formats that run the AI world

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 7pt,
  table.header([*Format*], [*Bits*], [*Exp*], [*Mant*], [*Max value*], [*Machine ε*], [*\~Digits*]),
  [float64], [64], [11], [52], [1.8×10³⁰⁸], [2.2×10⁻¹⁶], [15.9],
  [float32], [32], [8], [23], [3.4×10³⁸], [1.2×10⁻⁷], [7.2],
  [float16], [16], [5], [10], [#text(fill: RED)[*65 504*]], [9.8×10⁻⁴], [3.3],
  [bfloat16], [16], [8], [7], [3.4×10³⁸], [7.8×10⁻³], [2.4],
)

#pause
Halving storage cuts tensor memory and bandwidth; suitable accelerators can also execute 16-bit matrix operations much faster than float32. The exact speedup depends on hardware and workload. But look at float16's max value…

== Range vs precision · the 16-bit dilemma #V

#fig("/lecture2/figures/fp_formats.svg", w: 84%)

== Why many accelerators use bfloat16

float16 spent its bits on precision and starved the exponent — `exp(12)` already overflows its 65,504 ceiling. *bfloat16 keeps float32's 8 exponent bits* and pays with mantissa:

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt + MUTED.lighten(40%),
  inset: 8pt,
  table.header([], [*float16*], [*bfloat16*]),
  [`exp(12)` ≈ 162,755], [#text(fill: RED)[*inf* — training dies]], [1.63×10⁵ ✓],
  [copy a float32's range?], [no — overflow risk everywhere], [yes — same exponent, just chop],
  [digits of precision], [3.3], [2.4],
)

== Mixed precision keeps critical work in float32

Low-precision training is not "make every number bfloat16":

- multiply large tensors in float16 or bfloat16 for speed,
- accumulate dot products and keep selected state in float32,
- use loss scaling when float16 gradients risk underflow.

#pause
#result[bfloat16 keeps float32's range; float16 keeps more fraction bits. Mixed precision chooses where each trade-off is acceptable.]

== Checkpoint: you are now floating-point literate #Q

#mcq([Predict all three, then commit.],
  [`np.exp(np.float16(12.0))` returns…?],
  [`np.float32(16_777_216) + np.float32(1.0)` returns…?],
  [From the start of class: is `0.5 + 0.25 == 0.75` `True` or `False`?],
  [(bonus) why is `nan == nan` `False`?],
)

== Answers: floating-point literacy #A

*A.* `inf` — $e^12 approx 162,755 > 65,504$, float16's ceiling. (bfloat16 shrugs: $approx 1.6 times 10^5$.)

#pause
*B.* `16777216.0` — unchanged. At $2^24$ the float32 gap is 2, so $+1$ rounds back down.

#pause
*C.* *`True`!* $0.5 = 2^(-1)$, $0.25 = 2^(-2)$, $0.75 = 2^(-1) + 2^(-2)$ — all _exactly_ representable in binary. Floats aren't sloppy; they are *exact binary fractions*. Trouble only starts when your decimal (0.1) has no finite binary form.

#pause
*D.* NaN means "no defensible value" — two undefined results can't be declared equal. `x != x` is the classic NaN test.

== Practice problems

Try on paper; verify in the Tutorial 1 notebook.

+ Encode $-0.75$ as float32 (sign, exponent field, first mantissa bits).
+ What is the largest odd integer float32 represents exactly? (Hint: where does the gap reach 2?)
+ Compute $"logsumexp"([-1000, -1001])$ by hand. What would the naive computation return?
+ Explain why `(0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)` in float64 — what algebraic law do floats break?
+ Predict the sign and rough size of the error when the one-pass variance formula meets `[1e6, 1e6 + 1, 1e6 + 2]` in float32.
+ Your float16 training run dies at `exp(12)`. A colleague proposes bfloat16. What exactly changes, and what do you give up?

== ⭐⭐⭐ Kahan summation #OPT

Add `np.float32(0.1)` ten million times. True answer: $10^6$. A naive loop returns…

$ #raw("1087937.0") quad #text(fill: RED)[— off by 8.8%] $

#pause
Once the sum reaches $10^6$, the grid gap is $0.0625$: each $+0.1$ rounds badly, ten million times, _in the same direction_.

#pause
#codebox[```python
s, c = 0.0, 0.0
for x in xs:
    y = x - c            # re-inject the error we still owe
    t = s + y            # big + small: low digits of y get dropped...
    c = (t - s) - y      # ...(t - s) - y recovers exactly what was dropped
    s = t
```]

== What compensated summation guarantees

Kahan's variable `c` carries low-order bits that the running sum lost. This greatly reduces first-order accumulation error, especially when many same-sign terms are added.

#pause
The error still depends on the conditioning of the sum and on higher-order terms; it is not literally independent of $n$. NumPy commonly reduces error with a different method, *pairwise summation*.

== ⭐⭐⭐ Subnormals: the fine print near zero #OPT

The smallest _normal_ float32 is $2^(-126) approx 1.18 times 10^(-38)$. Without further tricks there'd be a *moat around zero*, and `x - y == 0` could be true for `x != y`(!).

#pause
#notebox[*Subnormals*: when the exponent field is all zeros, drop the implicit leading 1 and let the mantissa alone carry the value — filling the moat with evenly spaced numbers down to $2^(-149) approx 1.4 times 10^(-45)$ ("gradual underflow").]

#pause
#alertbox[The catch: some hardware handles subnormals in microcode — *orders of magnitude slower*. GPUs and DL runtimes often enable _flush-to-zero_ modes that trade this correctness for speed. If a kernel inexplicably slows down as values shrink toward zero: suspect subnormals.]

== Lecture 2 — summary

- *Fixed formats fail ML*: one program spans 40+ orders of magnitude → the point must float.
- *IEEE-754 float32* = sign + biased exponent (127) + mantissa with a free leading 1; 6.25 encodes exactly, 0.1 _cannot_ — hence `0.1 + 0.2 != 0.3`.
- *The float line is unevenly spaced*: gaps double at each power of 2; $epsilon_32 approx 10^(-7)$; at $2^24$, $x + 1 == x$.
- *Special values*: overflow → `inf`, indeterminate → `nan`; NaN propagates and `nan != nan`.
- *Catastrophic cancellation*: subtracting near-equals shreds digits — rewrite the formula.
- *The ML fixes*: log-space probabilities, log-sum-exp, subtract-the-max softmax.
- *bfloat16* = float32's range at half the bits: DL trades precision for range.

== Before L3 and Tutorial 1

- *Reading*: Goldberg (1991), §1–2; Solomon, _Numerical Algorithms_, Ch. 1–2.
- *Worksheet*: decode IEEE-754 values by hand.
- *Notebook*: reproduce decimal error, cancellation, and softmax overflow in NumPy, then implement the stable versions.

#focus-slide[
  Computers don't do real numbers — they do \~4 billion of them, unevenly spaced.
  #v(12pt)
  #set text(size: 22pt)
  Next: *vectors* — how direction and displacement explain `king − man + woman ≈ queen`.
]
