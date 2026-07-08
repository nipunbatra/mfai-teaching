# Teaching guide · L01 — Why Math for AI + Course Map · 80-min plan

Audience: 2nd-years out of ES 113/114. Zero ML. High curiosity, uneven math confidence — this lecture's job is motivation and contract-setting, not content.

## The spine (say this in 2 sentences)

Every stage of an AI system — data, model, loss, optimizer — is a piece of math, and mysterious AI failures are mysterious only until you know which piece broke. By the last lecture you will train a small language model from scratch, and every module in between exists because that artifact needs it.

## Where it sits

- **Builds on:** ES 113 (Python/NumPy), ES 114 (probability basics). Cite their actual assignments if possible.
- **Sets up:** everything; explicitly plants the PageRank (L5/L25), NaN (L2), conditioning (L17) hooks.

## 80-minute plan

- ⭐ (10 min) Three live failures: NaN loss · LR diverges-vs-crawls · king−man+woman. Don't explain them — promise the explanations, with lecture numbers.
- ⭐ (15 min) The AI stack slide → module map. One pass, slow.
- ⭐ (15 min) The destination: show generated Shakespeare from the finished T13 notebook. State Pillar-1 table: what the LM needs from each module.
- ⭐ (15 min) The teaching contract, meta-demonstrated on eigenvectors (picture → 2×2 numbers → 3 lines numpy → Av=λv). Say out loud: "this ordering is how every concept will arrive."
- ⭐⭐ (10 min) Logistics: 26+13 structure, hybrid tutorials, assessment, all-free textbooks, the site tour.
- ⭐⭐ (10 min) Pop quiz (ungraded, clickers/show of hands) on prerequisites — calibrates you AND them.
- ⭐⭐⭐ (5 min) Narrative arc slide (numbers/shapes → change → uncertainty → search → communication).

**Do live:** the NaN demo (`np.exp(1000)` inside a softmax) — typing it beats a screenshot.

## Teach it like this

Sell the destination hard. The course's identity is "one artifact, every module" — students should leave able to repeat that sentence. Under-explain the failures deliberately; unresolved tension is the hook.

## Heads-up for YOU

- Have the T13/A3 notebook output ready (or a saved sample) — the Shakespeare demo cannot fail on day 1.
- The eigenvector meta-demo must stay under 15 min; it's a trailer, not L5.

## Where students stumble (and the fix)

- "Is this a math course or a CS course?" → "It's the math course that makes the CS courses easy. Every topic ships with code."
- Anxiety about proofs → point at the one-derivation-per-lecture rule and the star system.

## If a student asks…

- "Why not just learn PyTorch?" → PyTorch is L11's punchline; frameworks change, the math beneath them hasn't in 50 years.
- "Will this cover ML models?" → No — ES 335 does; we build what ES 335 assumes.

## If you're short on time

Cut the narrative-arc slide and compress logistics to 5 min. Never cut the destination demo or the contract demo.

## Closing line

"Every AI system is a stack of math — by L26 you'll have built one from scratch."
