// Compression and Optimal Codes — Lecture 23 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture23/L23-compression-huffman.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Compression and Optimal Codes],
  subtitle: [Prefix codes, code length, and Huffman's algorithm],
)

// ── Huffman build for "MATHEMATICS": one fletcher forest per merge step.
//    Node positions are fixed once, so every step draws the same layout.
#let hf-pos = (
  C: (0, 3), E: (1, 3), H: (2, 3), I: (3, 3),
  S: (4, 3), A: (5, 3), M: (6, 3), T: (7, 3),
  n1: (0.5, 2), n2: (2.5, 2), n3: (4.5, 2), n4: (6.5, 2),
  n5: (1.5, 1), n6: (5.5, 1), n7: (3.5, 0),
)
#let hf-leaves = (C: 1, E: 1, H: 1, I: 1, S: 1, A: 2, M: 2, T: 2)
#let hf-merges = (
  ("n1", 2, "C", "E"), ("n2", 2, "H", "I"), ("n3", 3, "S", "A"),
  ("n4", 4, "M", "T"), ("n5", 4, "n1", "n2"), ("n6", 7, "n3", "n4"),
  ("n7", 11, "n5", "n6"),
)
#let hf-codes = (C: "000", E: "001", H: "010", I: "011", S: "100", A: "101", M: "110", T: "111")
#let hf-forest(k, codes: false) = align(center, diagram(
  spacing: (12mm, 8mm), node-stroke: 0.9pt + INK, node-fill: white,
  {
    let hot = if k > 0 { hf-merges.at(k - 1).slice(2) + (hf-merges.at(k - 1).at(0),) } else { () }
    for i in range(k) {
      let (key, w, l, r) = hf-merges.at(i)
      let new = i == k - 1
      let ecol = if new { ACC } else { MUTED }
      for (child, bit) in ((l, "0"), (r, "1")) {
        edge(hf-pos.at(key), hf-pos.at(child), stroke: (if new { 1.2pt } else { 0.8pt }) + ecol,
          label: if codes { text(size: 11pt, fill: BLUE, bit) } else { none }, label-sep: 2pt)
      }
      node(hf-pos.at(key), text(size: 13pt, weight: 700, fill: if new { ACC } else { INK })[#w],
        radius: 4.2mm, stroke: 0.9pt + (if new { ACC } else { INK }))
    }
    for (key, w) in hf-leaves {
      let new = key in hot
      node(hf-pos.at(key), stack(spacing: 2.5pt,
          text(size: 13.5pt, weight: 700, fill: if new { ACC } else { INK }, key),
          text(size: 10pt, fill: MUTED, str(w))),
        radius: 4.8mm, stroke: 0.9pt + (if new { ACC } else { INK }))
      if codes {
        let (x, y) = hf-pos.at(key)
        node((x, y + 0.8), text(size: 11.5pt, fill: TEAL, raw(hf-codes.at(key))),
          stroke: none, fill: none)
      }
    }
  },
))

#title-slide()

// ═══════════════════ 1 · the AI hook ═══════════════════
= Language models are compressors

== gzip versus a language model

`enwik9` is the first $10^9$ bytes of English Wikipedia — a standard compression benchmark.

#pause
#table(
  columns: (1.4fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([compressor], [compressed size]),
  [gzip], [$32.3%$ of the raw bytes],
  [LZMA2 (7-zip)], [$23.0%$],
  [Chinchilla 70B + arithmetic coding], [$8.3%$],
)

#pause
#notebox[Delétang et al., _Language Modeling Is Compression_, ICLR 2024. The comparison charges nothing for the model itself: sender and receiver must already share the 70B weights. Counted honestly, the network only pays off on large corpora.]

== Prediction is doing the work

The language model never sees a file format. It does one thing: assign a probability to the next token.

#pause
A coder (arithmetic coding, later this lecture) turns those probabilities into bits: high probability, few bits.

#pause
To compress is to bet on probabilities: give short names to the outcomes you expect.

#pause
This lecture builds that link from the ground up — codes, code lengths, and the optimal code.

// ═══════════════════ 2 · variable-length codes ═══════════════════
= Give short names to common outcomes

== A five-symbol source

Suppose a source emits

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([symbol], [probability]),
  [A], [$0.40$],
  [B], [$0.20$],
  [C], [$0.15$],
  [D], [$0.15$],
  [E], [$0.10$],
)

#pause
A fixed-length binary code needs three bits per symbol because $2^2<5<=2^3$.

== Fixed-length code

One valid assignment is

$ A arrow.r 000, quad B arrow.r 001, quad C arrow.r 010, $
$ D arrow.r 011, quad E arrow.r 100. $

#pause
Every symbol costs three bits, so the average length is

$ L=3 " bits/symbol". $

#pause
The code ignores the fact that A is four times as common as E.

== Variable-length code

Try

$ A arrow.r 0, quad B arrow.r 111, quad C arrow.r 101, $
$ D arrow.r 110, quad E arrow.r 100. $

#pause
The expected length is

$ L=0.40(1)+0.20(3)+0.15(3)+0.15(3)+0.10(3). $

#pause
$ L=2.20 " bits/symbol". $

== Encode one message

Message: `A B A E`

#pause
Fixed-length code:

`000 001 000 100` — 12 bits.

#pause
Variable-length code:

`0 111 0 100` — 8 bits.

#pause
The decoder must be able to determine where one codeword ends and the next begins. The code structure must make those boundaries recoverable from the bit stream.

== An ambiguous code

Suppose

$ A arrow.r 0, quad B arrow.r 01, quad C arrow.r 10. $

#pause
The bit string `010` can be decoded as

+ `A C` = `0 | 10`, or
+ `B A` = `01 | 0`.

#pause
The problem is that the codeword for A is a prefix of the codeword for B.

== Prefix-free codes

A code is *prefix-free* if no codeword is the prefix of another codeword.

#pause
Then decoding can proceed from left to right without separators.

#pause
Prefix-free is sufficient, but not necessary, for unique decoding. For example, $A arrow.r 0, B arrow.r 01$ is uniquely decodable but not prefix-free; prefix-free codes are preferred because they decode immediately from left to right.

#pause
In a binary tree:

+ each left edge writes 0,
+ each right edge writes 1,
+ symbols occupy leaves only.

== A prefix tree #V

#fig("/lecture23/figures/prefix_tree.svg", w: 70%)

#pause
Reaching a leaf completes one symbol; decoding restarts at the root.

== Learning outcomes

By the end of this lecture you will be able to:

+ distinguish fixed-length, uniquely decodable, and prefix-free codes,
+ relate probability to ideal code length $-log_2 p$, with Kraft as the feasibility check,
+ state Shannon's source coding theorem,
+ construct ⭐ a Huffman tree by repeated merging,
+ compare expected code length with entropy, on a worked example and on real text,
+ contrast compression with error-correcting codes,
+ and connect probabilistic prediction with compression.

// ═══════════════════ 2 · probability and length ═══════════════════
= Code length follows surprise

== Ideal length

L22 assigned surprise

$ I(x)=-log_2 p(x). $

#pause
This suggests an ideal code length

$ ell(x)=-log_2 p(x). $

#pause
Examples:

$ p=1/2 arrow.r ell=1, quad p=1/4 arrow.r ell=2, quad p=1/8 arrow.r ell=3. $

== Ideal lengths need not be integers

For $p=0.40$,

$ -log_2 0.40 approx 1.322 " bits". $

#pause
A single binary codeword cannot contain 1.322 bits.

#pause
Integer code lengths introduce a rounding cost. Coding long blocks or using arithmetic coding can approach fractional average lengths.

== Integer lengths #V

#fig("/lecture23/figures/ideal_lengths.svg", w: 64%)

#pause
Choosing $ell_i=ceil(-log_2 p_i)$ adds less than one bit to each ideal length.

== The Kraft inequality

Integer lengths $ell_1,dots,ell_K$ can form a binary prefix code if and only if

#result[$sum_(i=1)^K 2^(-ell_i)<=1$.]

#pause
Each codeword at depth $ell_i$ occupies a fraction $2^(-ell_i)$ of the leaves at a common deeper level.

== Kraft check: a valid set of lengths

For lengths $(1,3,3,3,3)$,

$ 2^(-1)+4 times 2^(-3)=1/2+4/8=1. $

#pause
The tree is exactly filled. This is the length pattern used for the five-symbol variable code.

== Kraft check: an impossible set

For lengths $(1,2,2,2)$,

$ 2^(-1)+3 times 2^(-2)=1/2+3/4=1.25>1. $

#pause
There is not enough room in a binary tree for one one-bit leaf and three two-bit leaves.

== Expected code length

For probabilities $p_i$ and lengths $ell_i$,

$ L=sum_i p_i ell_i. $

#pause
Entropy is the corresponding average ideal length:

$ H(X)=sum_i p_i (-log_2 p_i). $

#pause
For any binary prefix code,

$ L>=H(X). $

#pause
We use this bound today and prove the size of the gap in L24 (Gibbs' inequality).

== Shannon lengths give an upper bound

Choose

$ ell_i=ceil(-log_2 p_i). $

#pause
Since $ceil(a)<a+1$,

$ ell_i<-log_2 p_i+1. $

#pause
Average both sides:

#result[$H(X)<=L<H(X)+1$.]

== Shannon's source coding theorem

Coding one symbol at a time can waste up to one bit on rounding. Encoding long blocks removes even that.

#pause
#result[Source coding theorem (stated): $n$ i.i.d. symbols from $X$ can be encoded in about $n H(X)$ bits with error probability $arrow.r 0$ as $n arrow.r infinity$; below $H(X)$ bits per symbol, errors are unavoidable.]

#pause
Entropy is not just a lower bound — it is the exact price of description. We state the theorem and make it plausible; MacKay Ch 4 proves it.

== Why entropy is the boundary

Flip a coin with $p("heads")=0.9$, $n=1000$ times. Here $H approx 0.469$ bits/flip.

#pause
Almost every sequence that actually occurs has close to $900$ heads. These _typical_ sequences number about

$ 2^(n H)=2^469, $

a vanishing fraction of the $2^1000$ possible strings.

#pause
Give each typical sequence a serial number: about $469$ bits describe $1000$ flips.

#pause
Counting typical sequences is the whole idea; making "almost never" precise is the work of the proof.

== Question: check Kraft #Q

#mcq(
  [Which set of binary code lengths can satisfy the Kraft inequality?],
  [$(1,1,2)$],
  [$(1,2,3,3)$],
  [$(1,2,2,2)$],
  [$(1,1,1)$],
)

== Answer #A

#mcq-answer("B", [$(1,2,3,3)$], [$1/2+1/4+1/8+1/8=1$. Every other option has Kraft sum greater than one.])

// ═══════════════════ 3 · Huffman construction ═══════════════════
= Merge the two least frequent symbols

== The greedy rule

Huffman's algorithm operates on symbol weights:

1. take the two smallest weights,
2. merge them into one node whose weight is their sum,
3. return the merged node to the collection,
4. repeat until one root remains,
5. label left and right edges 0 and 1.

#pause
Less frequent symbols are pushed deeper in the tree.

== Why merge the two smallest?

In an optimal full binary tree, two deepest leaves can be made siblings.

#pause
Placing the two least probable symbols at those deepest positions cannot increase expected length.

#pause
Merge those siblings into a single pseudo-symbol. The remaining problem has one fewer symbol and the same form.

#pause
This exchange argument plus induction proves optimality.

== Counts in “MATHEMATICS”

`M A T H E M A T I C S` contains 11 letters.

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 7pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  [A], [$2$], [M], [$2$],
  [T], [$2$], [C], [$1$],
  [E], [$1$], [H], [$1$],
  [I], [$1$], [S], [$1$],
)

#pause
There are eight distinct symbols, so a fixed code needs three bits per letter.

== ⭐ Merge 1: C and E #D

The smallest weights are the five letters of count 1 — a tie. Ties can be broken arbitrarily; every choice gives the same expected length. Fix one order and take C and E first.

#hf-forest(1)

#pause
Pool: $"CE" 2$ and letters $"H" 1, "I" 1, "S" 1, "A" 2, "M" 2, "T" 2$.

== ⭐ Merge 2: H and I #D

Two of the three remaining count-1 letters merge next.

#hf-forest(2)

#pause
Pool: $"S" 1$ and $"CE" 2, "HI" 2, "A" 2, "M" 2, "T" 2$.

== ⭐ Merge 3: S and A #D

S is the only weight left at 1; its partner is any of the weight-2 nodes. Take the letter A.

#hf-forest(3)

#pause
Pool: $"SA" 3$ and $"CE" 2, "HI" 2, "M" 2, "T" 2$.

== ⭐ Merge 4: M and T #D

The two smallest weights are now both 2.

#hf-forest(4)

#pause
Pool: $"CE" 2, "HI" 2, "SA" 3, "MT" 4$.

== ⭐ Merge 5: CE and HI #D

The two smallest weights are the merged pairs CE and HI. Merged nodes re-enter the pool like any symbol.

#hf-forest(5)

#pause
Pool: $"SA" 3, "CEHI" 4, "MT" 4$.

== ⭐ Merge 6: SA and MT #D

Take the 3 and one of the tied 4s.

#hf-forest(6)

#pause
Pool: $"CEHI" 4, "SAMT" 7$.

== ⭐ Merge 7: the root #D

The last merge joins everything: weight $4+7=11$, the total letter count.

#hf-forest(7)

#pause
Seven merges for eight symbols — each merge removes one node from the pool.

== The resulting tree #V

Label left edges 0 and right edges 1; read each codeword from root to leaf.

#hf-forest(7, codes: true)

#pause
Here every letter happens to receive length three. Huffman cannot improve on the fixed-length code for this short frequency table.

== Verify the encoded word

Reading the tree gives

$ A arrow.r 101, quad C arrow.r 000, quad E arrow.r 001, quad H arrow.r 010, $
$ I arrow.r 011, quad M arrow.r 110, quad S arrow.r 100, quad T arrow.r 111. $

#pause
Every codeword has length three, so “MATHEMATICS” uses

$ 11 times 3=33 " bits". $

#pause
The example is still useful: an optimal code need not beat fixed length on every finite sample.

== ⭐ Expected length versus entropy #D

The total bit count can be read off the build: each merge pushes its letters one level deeper, so total bits $=$ sum of merged weights,

$ 2+2+3+4+4+7+11=33, quad L=33/11=3.000 " bits/letter". $

#pause
The empirical entropy of the letter counts is

$ H=-3 dot 2/11 log_2 2/11 - 5 dot 1/11 log_2 1/11 approx 2.914 " bits/letter". $

#pause
The bound holds:

#result[$H approx 2.914 <= L_"Huffman" = 3.000 < H+1$.]

== A source with a useful imbalance

Return to probabilities

$ (0.40,0.20,0.15,0.15,0.10). $

#pause
The merges are

$ 0.10+0.15 arrow.r 0.25, quad 0.15+0.20 arrow.r 0.35, $
$ 0.25+0.35 arrow.r 0.60, quad 0.40+0.60 arrow.r 1.00. $

#pause
A alone stays shallow: lengths $(1,3,3,3,3)$, one assignment

$ A arrow.r 0, B arrow.r 111, C arrow.r 101, D arrow.r 110, E arrow.r 100. $

== Compare entropy and lengths #V

#fig("/lecture23/figures/length_comparison.svg", w: 63%)

#pause
$ H=2.146<=L_"Huffman"=2.200<3=L_"fixed". $

== Huffman on real letter counts #V

L22 estimated letter probabilities from this course's plan file: $22560$ letters, alphabet of $26$, $hat(H) approx 4.238$ bits/letter. Build Huffman on those counts:

#align(center, bars((5.0, 4.265, 4.238),
  labels: ([fixed 5-bit], [Huffman $L$], [entropy $hat(H)$]),
  digits: 3, horizontal: true, bar: 11mm, span: 105mm))

#pause
Huffman lands within $0.03$ bits of the entropy: `e` ($hat(p)=0.110$) gets 3 bits, `j` ($hat(p) approx 0.002$) gets 9.

== Huffman is optimal within a class

Huffman minimizes expected length among binary prefix codes for a known symbol distribution.

#pause
It does not mean:

+ the encoded file is always shorter after storing the tree,
+ one symbol at a time reaches entropy exactly,
+ the assumed probabilities match future data,
+ context has been used.

#pause
Those are separate modeling and engineering questions.

== Question: three symbols #Q

#mcq(
  [A source has counts $(5,2,1)$. Which code lengths does Huffman produce?],
  [$(1,2,2)$],
  [$(1,1,2)$],
  [$(2,2,2)$],
  [$(1,2,3)$],
)

== Answer #A

#mcq-answer("A", [$(1,2,2)$], [Merge $1+2 arrow.r 3$, then $3+5 arrow.r 8$: the two rare symbols become depth-2 siblings. $L=(5 dot 1+2 dot 2+1 dot 2)\/8=1.375$ bits versus $H approx 1.30$. Option $(1,1,2)$ fails Kraft: $1/2+1/2+1/4>1$.])

// ═══════════════════ 4 · implement and decode ═══════════════════
= A small compressor

== A priority queue implements the merges

#codebox(size: 13pt)[```python
import heapq, itertools, collections

counts = collections.Counter("MATHEMATICS")
serial = itertools.count()
heap = [(count, next(serial), symbol)
        for symbol, count in sorted(counts.items())]
heapq.heapify(heap)
while len(heap) > 1:
    wa, _, a = heapq.heappop(heap)
    wb, _, b = heapq.heappop(heap)
    heapq.heappush(heap, (wa+wb, next(serial), (a, b)))
tree = heap[0][2]
```]

#pause
The serial number breaks ties deterministically; sorting first reproduces the merge order we drew.

== Traverse the tree to assign codes

#codebox(size: 13pt)[```python
def make_codes(node, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(node, str):
        out[node] = prefix or "0"
    else:
        left, right = node
        make_codes(left,  prefix + "0", out)
        make_codes(right, prefix + "1", out)
    return out
```]

== Encode by table lookup

#codebox[```python
def encode(text, code):
    return "".join(code[ch] for ch in text)

bits = encode("MATHEMATICS", code)
print(bits[:12])   # 110101111010...
print(len(bits))   # 33
```]

#pause
Real implementations pack bits into bytes; a Python string of `'0'` and `'1'` is only a teaching representation.

== Decode by walking the tree

#codebox(size: 13pt)[```python
def decode(bits, tree):
    output, node = [], tree
    for bit in bits:
        node = node[0] if bit == "0" else node[1]
        if isinstance(node, str):
            output.append(node)
            node = tree
    assert node is tree       # no unfinished codeword
    return "".join(output)
```]

== What must be stored?

A standalone compressed file needs enough metadata to reconstruct the code:

+ the tree or canonical code lengths,
+ the number of valid data bits in the final byte,
+ the original alphabet or symbol ordering,
+ and often checksums or format information.

#pause
For short messages, this overhead can exceed the saved bits.

== Canonical Huffman codes #OPT

The decoder does not need the exact tree shape if it receives the code length of each symbol.

#pause
Canonical Huffman coding assigns codewords deterministically from sorted lengths.

#pause
Benefits:

+ compact header,
+ fast table-based decoding,
+ reproducible code assignment.

Many file formats use this representation.

// ═══════════════════ 5 · prediction and context ═══════════════════
= Better probabilities give shorter descriptions

== Code with the wrong frequency model

If a compressor expects distribution $q$ but data come from $p$, it chooses lengths near

$ -log_2 q(x). $

#pause
Its average cost under the true source is

$ EE_[X tilde p][-log_2 q(X)]. $

#pause
This is cross-entropy. L24 will show exactly how much larger it is than $H(p)$.

== Independent-letter model

L22 estimated letter frequencies without context.

#pause
Such a model assigns one probability to each letter regardless of its neighbors:

$ q(x_1,dots,x_n)=product_(t=1)^n q(x_t). $

#pause
It cannot exploit that `u` is likely after `q`, or that common words recur.

== Conditional prediction

A context model uses

$ q(x_1,dots,x_n)=product_(t=1)^n q(x_t | x_(<t)). $

#pause
If the context makes the next symbol predictable, $q(x_t | x_(<t))$ is larger and

$ -log_2 q(x_t | x_(<t)) $

is shorter.

#pause
Compression improves when prediction improves.

== Arithmetic coding #OPT

Huffman assigns an integer number of bits to each symbol. Arithmetic coding represents an entire sequence as an interval.

#pause
It can approach the fractional sequence length

$ -log_2 q(x_1,dots,x_n) $

much more closely, especially when some probabilities are near one.

#pause
Modern learned compressors combine a probability model with an arithmetic or range coder.

== Language models report coding cost

A language model outputs a probability distribution for the next token.

#pause
For observed tokens $x_1,dots,x_n$, its total ideal code length is

$ -sum_(t=1)^n log_2 q(x_t | x_(<t)). $

#pause
Divide by $n$ to obtain bits per token. Exponentiate to obtain perplexity in L24.

#pause
This resolves the opening comparison: Chinchilla's $8.3%$ on `enwik9` is a measurement of its next-token predictions, turned into bits by an arithmetic coder. gzip's dictionary model predicts worse, so it pays more bits.

// ═══════════════════ 6 · noisy channels ═══════════════════
= Compression and error correction solve different problems

== Compression removes predictable bits

Source coding uses probability to reduce average description length.

#pause
Channel coding adds structured redundancy so corrupted bits can be detected or corrected.

#pause
The two goals point in opposite directions:

+ compress before transmission,
+ then add carefully designed redundancy for the channel.

== Repetition code

Encode one bit as three:

$ 0 arrow.r 000, quad 1 arrow.r 111. $

#pause
If at most one bit flips, majority vote recovers the original:

$ 111 arrow.r 101 arrow.r "decode as" 1. $

#pause
The price is rate $1/3$: three transmitted bits per information bit.

== Hamming(7,4): three parity bits protect four data bits

Send $(d_1,d_2,d_3,d_4)$ together with

$ p_1=d_1 plus.o d_2 plus.o d_4, quad p_2=d_1 plus.o d_3 plus.o d_4, quad p_3=d_2 plus.o d_3 plus.o d_4. $

#pause
Each of the seven possible single-bit flips makes a distinct subset of the three checks fail, so the failure pattern points at the flipped position. Flip it back.

#pause
Rate $4/7 approx 0.57$ — better than $1/3$, and still every single error per block is corrected.

== Repetition versus Hamming at 10% noise

Simulate a channel that flips each transmitted bit with probability $f=0.1$:

#table(
  columns: (1.2fr, 0.6fr, 1fr),
  inset: 8pt,
  stroke: 0.5pt + MUTED.lighten(40%),
  table.header([code], [rate], [bit error after decoding]),
  [none], [$1$], [$0.100$],
  [repetition R3], [$1\/3$], [$0.028$],
  [Hamming(7,4)], [$4\/7$], [$approx 0.07$],
)

#pause
R3 reaches the lowest error but pays three transmitted bits per data bit. Hamming protects almost twice as much data per transmitted bit. MacKay Ch 1 opens with exactly this trade-off.

== The corrupted image #V

#fig("/lecture23/figures/channel_demo.svg", w: 92%)

#align(center, text(size: 16pt, fill: MUTED)[Every panel decodes the same bitmap after the same noisy channel (simulated, seed fixed).])

== A limit also exists for noisy channels #OPT

Shannon's channel coding theorem states that reliable communication is possible below a channel's capacity and impossible above it, under precise assumptions.

#pause
Entropy governs source description; mutual information and capacity govern transmission.

#pause
We will use mutual information only briefly in L24.

// ═══════════════════ 7 · close ═══════════════════
= From entropy to an actual code

== Summary

#result[Frequent symbols receive short codewords; Huffman's greedy merges produce the optimal binary prefix tree.]

#pause
+ Prefix-free codes decode without separators.
+ Kraft's inequality characterizes feasible prefix-code lengths.
+ Ideal length is $-log_2 p$; entropy is the average ideal length, and the source coding theorem makes it the exact price.
+ Huffman coding minimizes expected length among binary prefix codes.
+ For a source distribution, $H<=L_"Huffman"<H+1$.
+ A better conditional predictor gives a shorter description.

== Practice

1. Check Kraft's inequality for lengths $(2,2,2,2)$ and $(1,3,3,3)$.

#pause
2. Build a Huffman code for weights $(8,4,2,1,1)$.

#pause
3. Compute the expected length of codes $A:0$, $B:10$, $C:110$, $D:111$ under probabilities $(1/2,1/4,1/8,1/8)$.

#pause
4. Explain why storing a Huffman tree can make a tiny file larger.

== Practice answers #A

1. The Kraft sums are $1$ and $1/2+3/8=7/8$; both are feasible.

2. One optimal code has lengths $(1,2,3,4,4)$; the two weight-1 symbols are deepest siblings.

3. $L=1/2(1)+1/4(2)+1/8(3)+1/8(3)=1.75$ bits/symbol, equal to the entropy.

4. The tree or length table, padding, alphabet, and format metadata create overhead independent of message length.

== Next: the cost of a wrong model

Entropy is the average cost under the true distribution. A model usually supplies a different distribution.

#pause
Next lecture derives

$ H(p,q)=H(p)+D_"KL"(p || q). $

#pause
The excess code length becomes KL divergence, and minimizing cross-entropy becomes maximum likelihood.

#pause
#notebox[*Reading:* MacKay, _Information Theory, Inference, and Learning Algorithms_ — Ch 5 (Huffman), Ch 4 (source coding theorem), Ch 1 (repetition vs Hamming). Delétang et al., _Language Modeling Is Compression_, ICLR 2024.]

#focus-slide[A good predictor and a good compressor are the same object.]
