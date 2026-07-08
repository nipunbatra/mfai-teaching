# Teaching guide · L02 — Floating Point · 80-min plan

Audience: they've written Python for a year but have never asked what a `float` is. Everything here is new and slightly shocking — use that.

## The spine (say this in 2 sentences)

Computers represent finitely many, unevenly spaced numbers, and every arithmetic result gets rounded to the nearest one. Once you know where the numbers are (dense near 0, sparse far out) and what rounding does to subtraction, all of ML's numerical folklore — NaN losses, log-probs, bfloat16 — becomes obvious.

## Where it sits

- **Builds on:** L1's NaN hook (resolve it today).
- **Sets up:** L10 (finite-difference step-size war — the U-curve reappears), L14 (log-likelihood), L26 (log-space LM). Say each forward pointer out loud.

## 80-minute plan

- ⭐ (10 min) Hooks: `0.1 + 0.2 == 0.3` → False; the NaN-softmax from L1, now resolved by the end.
- ⭐ (15 min) IEEE-754 float32 anatomy; encode 6.25 fully by hand; show 0.1 is periodic in binary.
- ⭐ (15 min) The uneven number line: spacing doubles per binade; machine epsilon; `x+1==x` at 1e16. This is THE picture of the lecture.
- ⭐ (15 min) Catastrophic cancellation: variance two ways (naive goes negative!), quadratic formula. Worked numerics.
- ⭐ (15 min) The ML fixes: log-space, log-sum-exp derivation (the ⭐ derivation), stable softmax. Recompute the L1 NaN — now finite.
- ⭐⭐ (7 min) dtype tour: float64/32/16/bfloat16 table; why bf16 keeps fp32's exponent.
- ⭐⭐ (3 min) Pop quiz + insight line.
- ⭐⭐⭐ (0 min, slides only) Kahan summation, subnormals, rounding modes.

**Do live:** the epsilon-discovery while-loop, and stable vs naive softmax side by side.

## Teach it like this

Physicalize the number line — "4 billion marked points, half of them between −2 and 2." Cancellation lands best as *subtraction annihilates shared leading digits*; show digits vanishing in color on the slide.

## Heads-up for YOU

- Exponent bias trips everyone; do 6.25 slowly, don't improvise a second encoding on the board.
- `np.float32(0.1)` printing tricks: repr may hide the error — use `%.20f`.
- T1's toy 8-bit format (1-4-3, bias 7) matches this lecture — reference it so tutorial feels continuous.

## Where students stumble (and the fix)

- "Why not just use more bits?" → dynamic range vs memory/bandwidth; bf16 exists *because* 32 bits is too many at scale.
- Epsilon vs smallest positive number confusion → epsilon is a *gap near 1*, not a smallest value.
- Thinking NaN is an error state that halts → it propagates silently; that's why loss curves flatline.

## If a student asks…

- "Does this matter with float64?" → `exp(710)`, and naive variance still fails; float64 delays, never fixes.
- "What about integers/quantization?" → int8 inference exists; pointer to ES 667.

## If you're short on time

Cut the dtype tour to the table slide only. Never cut cancellation or log-sum-exp.

## Closing line

"Computers don't do real numbers — they do ~4 billion of them, unevenly spaced."
