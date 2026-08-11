// Compression and Optimal Codes — Lecture 23 · Mathematical Foundations for AI
// Compile from the repository root:
//   typst compile --root . lecture23/L23-compression-huffman.typ

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Compression and Optimal Codes],
  subtitle: [Prefix codes, code length, and Huffman's algorithm],
)

#title-slide()

// ═══════════════════ 1 · variable-length codes ═══════════════════
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
+ relate probability to ideal code length $-log_2 p$,
+ use the Kraft inequality as a feasibility check,
+ construct ⭐ a Huffman tree by repeated merging,
+ calculate expected code length and compare it with entropy,
+ explain why Huffman coding is optimal among binary prefix codes,
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

$ H(X)=sum_i p_i(-log_2 p_i). $

#pause
For any binary prefix code,

$ L>=H(X). $

== Shannon lengths give an upper bound

Choose

$ ell_i=ceil(-log_2 p_i). $

#pause
Since $ceil(a)<a+1$,

$ ell_i<-log_2 p_i+1. $

#pause
Average both sides:

#result[$H(X)<=L<H(X)+1$.]

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

== The merge sequence

Ties can be resolved in several ways; all optimal trees have the same expected length.

#pause
One deterministic sequence of weights is

$ 1+1 arrow.r 2, $
$ 1+1 arrow.r 2, $
$ 1+2 arrow.r 3, $
$ 2+2 arrow.r 4, $
$ 2+2 arrow.r 4, $
$ 3+4 arrow.r 7, $
$ 4+7 arrow.r 11. $

== The resulting tree #V

#fig("/lecture23/figures/huffman_mathematics.svg", w: 76%)

#pause
Here every letter happens to receive length three. Huffman cannot improve on the fixed-length code for this short frequency table.

== Verify the encoded word

One generated code is

$ A arrow.r 101, quad C arrow.r 000, quad E arrow.r 001, quad H arrow.r 010, $
$ I arrow.r 011, quad M arrow.r 110, quad S arrow.r 100, quad T arrow.r 111. $

#pause
Every codeword has length three, so “MATHEMATICS” uses

$ 11 times 3=33 " bits". $

#pause
The example is still useful: an optimal code need not beat fixed length on every finite sample.

== A source with a useful imbalance

Return to probabilities

$ (0.40,0.20,0.15,0.15,0.10). $

#pause
Huffman produces lengths

$ (1,3,3,3,3) $

and one code assignment

$ A arrow.r 0, B arrow.r 111, C arrow.r 101, D arrow.r 110, E arrow.r 100. $

== Compare entropy and lengths #V

#fig("/lecture23/figures/length_comparison.svg", w: 63%)

#pause
$ H=2.146<=L_"Huffman"=2.200<3=L_"fixed". $

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

// ═══════════════════ 4 · implement and decode ═══════════════════
= A small compressor

== A priority queue implements the merges

#codebox(size: 13pt)[```python
import heapq, itertools

serial = itertools.count()
heap = [(count, next(serial), symbol)
        for symbol, count in counts.items()]
heapq.heapify(heap)

while len(heap) > 1:
    wa, _, a = heapq.heappop(heap)
    wb, _, b = heapq.heappop(heap)
    heapq.heappush(heap, (wa+wb, next(serial), (a, b)))

tree = heap[0][2]
```]

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
print(bits)
print(len(bits))
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
+ Ideal length is $-log_2 p$; entropy is the average ideal length.
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

#focus-slide[Probabilities determine ideal lengths; a code turns those lengths into bits.]
