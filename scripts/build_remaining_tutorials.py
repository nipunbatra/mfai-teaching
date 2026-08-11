#!/usr/bin/env python3
"""Generate T2–T13 worksheet QMDs and executable companion notebooks.

Run from the repository root. The tutorial-specific content lives below; this
script only removes repetitive QMD and ipynb scaffolding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def block(text: str) -> str:
    return dedent(text).strip() + "\n"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": block(source).splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": block(source).splitlines(keepends=True),
    }


def latex_math(text: str) -> str:
    """Translate the compact Typst-like math used in the content to MathJax."""

    def convert(match: re.Match[str]) -> str:
        expression = match.group(1)
        expression = re.sub(
            r"\[\[[^\[\]]+(?:\],\[[^\[\]]+)+\]\]",
            lambda m: r"\begin{bmatrix}"
            + r" \\ ".join(row.replace(",", " & ") for row in m.group(0)[2:-2].split("],["))
            + r"\end{bmatrix}",
            expression,
        )
        expression = re.sub(r"_\(([^()]*)\)", r"_{\1}", expression)
        expression = re.sub(r"\^\(([^()]*)\)", r"^{\1}", expression)
        expression = re.sub(r"hat\s*\(([^()]*)\)", r"\\hat{\1}", expression)
        expression = re.sub(r"\bhat\s+([A-Za-z]+)", r"\\hat{\1}", expression)
        expression = re.sub(r"\bbar\s+([A-Za-z]+)", r"\\bar{\1}", expression)
        expression = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", expression)
        expression = re.sub(r"\bsqrt\s*([0-9A-Za-z.]+)", r"\\sqrt{\1}", expression)
        expression = re.sub(r"\|\|([^|]+)\|\|", r"\\lVert \1 \\rVert", expression)
        expression = re.sub(r"\bR\^", r"\\mathbb{R}^", expression)
        for source, target in {
            "nabla": r"\nabla",
            "partial": r"\partial",
            "lambda": r"\lambda",
            "delta": r"\delta",
            "theta": r"\theta",
            "pi": r"\pi",
            "Sigma": r"\Sigma",
            "sim": r"\sim",
            "approx": r"\approx",
            "infinity": r"\infty",
            "times": r"\times",
            "sum": r"\sum",
            "prod": r"\prod",
            "arrow.r": r"\to",
            " dot ": r" \cdot ",
            "sin": r"\sin",
            "cos": r"\cos",
            "exp": r"\exp",
            "det": r"\det",
        }.items():
            if source.strip().isalnum():
                expression = re.sub(
                    rf"\b{re.escape(source)}\b", lambda _match, value=target: value, expression
                )
            else:
                expression = expression.replace(source, target)
        expression = re.sub(r"\bproj_", r"\\operatorname{proj}_", expression)
        expression = re.sub(r"\bspan\b", r"\\operatorname{span}", expression)
        expression = re.sub(r"\bVar\b", r"\\operatorname{Var}", expression)
        expression = re.sub(r"\b to \b", r" \\to ", expression)
        expression = re.sub(r"\bsum_", r"\\sum_", expression)
        expression = re.sub(r"\bpartial_", r"\\partial_", expression)
        expression = re.sub(r"(?<![A-Za-z\\])approx", r"\\approx ", expression)
        expression = re.sub(r"(?<![A-Za-z\\])cos", r"\\cos", expression)
        expression = expression.replace("<=", r"\le ").replace(">=", r"\ge ")
        return f"${expression}$"

    return re.sub(r"\$([^$]+)\$", convert, text)


TUTORIALS = [
    {
        "n": 2,
        "slug": "vectors-matrices",
        "title": "Vectors & Matrices",
        "after": "L4",
        "intro": "Represent data as vectors, read matrix multiplication geometrically, and solve a small least-squares problem.",
        "problems": [
            ("Cosine and projection", "For $x=(1,2,2)$ and $y=(2,0,1)$, compute $x^Ty$, both norms, cosine similarity, and the projection of $x$ onto $y$."),
            ("A linear map", "For $A=[[2,1],[0,1]]$, map $e_1$, $e_2$, and the unit square. State the area scale and whether the map is invertible."),
            ("Column space", "Solve $Ax=b$ for $A=[[1,0],[0,1],[1,1]]$ and $b=(1,2,4)$ in least-squares form. Write the normal equations."),
            ("Rank one", "Show that every column of $uv^T$ is a multiple of $u$. What is its rank when $u,v$ are nonzero?"),
            ("Conditioning", "Compare solving $Ax=b$ when the columns of $A$ are orthogonal versus nearly parallel. Which geometry amplifies noise?"),
        ],
        "solutions": [
            "$x^Ty=4$, $||x||=3$, $||y||=sqrt(5)$, cosine $4/(3sqrt(5))$, and $proj_y x=(4/5)y$.",
            "$Ae_1=(2,0)$, $Ae_2=(1,1)$; the square becomes their parallelogram. $det A=2$, so area doubles and the map is invertible.",
            "$A^TA=[[2,1],[1,2]]$, $A^Tb=(5,6)$, hence $hat x=(4/3,7/3)$.",
            "Column $j$ equals $v_j u$, so the column space is the line through $u$ and the rank is one.",
            "Nearly parallel columns make the inverse problem ill-conditioned; small perturbations can require large coefficient changes.",
        ],
        "cells": [
            md("""## 1 · Similarity and projection\n\nPredict the cosine and projection before running the cell."""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\nx = np.array([1., 2., 2.])\ny = np.array([2., 0., 1.])\ncosine = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))\nprojection = (x @ y) / (y @ y) * y\nprint("cosine:", cosine)\nprint("projection:", projection)"""),
            md("""## 2 · Watch a matrix move a grid"""),
            code("""A = np.array([[2., 1.], [0., 1.]])\nt = np.linspace(-2, 2, 9)\nfig, ax = plt.subplots(figsize=(6, 5))\nfor value in t:\n    horizontal = np.vstack([np.linspace(-2, 2, 100), np.full(100, value)])\n    vertical = np.vstack([np.full(100, value), np.linspace(-2, 2, 100)])\n    for line in (horizontal, vertical):\n        moved = A @ line\n        ax.plot(moved[0], moved[1], color="#2C7A7B", alpha=.45)\nax.set_aspect("equal"); ax.set_title(f"det(A) = {np.linalg.det(A):.1f}"); ax.grid(alpha=.15);"""),
            md("""## 3 · Least squares as projection\n\nThe residual should be orthogonal to every column of $A$."""),
            code("""A = np.array([[1., 0.], [0., 1.], [1., 1.]])\nb = np.array([1., 2., 4.])\nx_hat, *_ = np.linalg.lstsq(A, b, rcond=None)\nr = b - A @ x_hat\nprint("x_hat:", x_hat)\nprint("residual:", r)\nprint("A.T @ residual:", A.T @ r)"""),
        ],
    },
    {
        "n": 3,
        "slug": "eigen-svd-pca",
        "title": "Eigen, SVD & PCA",
        "after": "L6",
        "intro": "Use repeated multiplication to find a dominant eigenvector, then compress data with SVD and PCA.",
        "problems": [
            ("Eigenpairs", "Find the eigenvalues/eigenvectors of $[[3,1],[1,3]]$. What happens to a generic vector after repeated multiplication and normalization?"),
            ("Power iteration", "Starting at $(1,0)$, perform three normalized power-iteration steps for the same matrix."),
            ("SVD reconstruction", "For singular values $(5,2,0.5)$, compute rank-1 and rank-2 squared Frobenius errors."),
            ("Centre before PCA", "Explain why PCA on uncentred data can point toward the mean rather than the direction of variation."),
            ("Variance", "For covariance $[[4,3],[3,4]]$, find principal directions and explained-variance fractions."),
        ],
        "solutions": [
            "Eigenpairs: $4$ with $(1,1)$ and $2$ with $(1,-1)$. The normalized iterate approaches $(1,1)/sqrt(2)$.",
            "The unnormalized iterates are $(3,1)$, $(10,6)$, $(36,28)$; normalize each to see alignment.",
            "Rank-1 error $=2^2+0.5^2=4.25$; rank-2 error $=0.5^2=0.25$.",
            "PCA is about deviations from the mean; centring moves the origin to that mean.",
            "Directions $(1,1)$ and $(1,-1)$ have variances $7$ and $1$, explaining $7/8$ and $1/8$.",
        ],
        "cells": [
            md("""## 1 · Power iteration"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\nA = np.array([[3., 1.], [1., 3.]])\nx = np.array([1., 0.])\nfor k in range(8):\n    x = A @ x\n    x /= np.linalg.norm(x)\n    print(k + 1, x)"""),
            md("""## 2 · Low-rank image reconstruction\n\nThe synthetic image keeps the experiment self-contained."""),
            code("""u = np.linspace(-2, 2, 80)\nX, Y = np.meshgrid(u, u)\nimage = np.exp(-(X**2 + Y**2)) + .45*np.exp(-((X-1)**2 + (Y+.7)**2)/.25)\nU, s, Vt = np.linalg.svd(image, full_matrices=False)\nfig, axes = plt.subplots(1, 4, figsize=(12, 3))\nfor ax, rank in zip(axes, [1, 2, 5, 20]):\n    approx = (U[:, :rank] * s[:rank]) @ Vt[:rank]\n    ax.imshow(approx, cmap="gray"); ax.set_title(f"rank {rank}"); ax.axis("off")\nprint("first ten singular values:", np.round(s[:10], 3))"""),
            md("""## 3 · PCA from the SVD of centred data"""),
            code("""rng = np.random.default_rng(3)\nZ = rng.normal(size=(500, 2)) @ np.array([[2.2, .3], [1.4, .5]]).T + [3, -2]\nZc = Z - Z.mean(axis=0)\n_, s, Vt = np.linalg.svd(Zc, full_matrices=False)\nprint("principal directions:\\n", Vt)\nprint("explained fractions:", s**2 / np.sum(s**2))\nplt.scatter(*Zc.T, s=8, alpha=.25)\nfor scale, direction in zip(s[:2] / 8, Vt):\n    plt.arrow(0, 0, *(scale * direction), width=.02, color="#EB811B")\nplt.axis("equal"); plt.grid(alpha=.15);"""),
        ],
    },
    {
        "n": 4,
        "slug": "gradients-contours",
        "title": "Derivatives, Gradients & Contours",
        "after": "L8",
        "intro": "Move between formulas, contour geometry, directional derivatives, and numerical gradient checks.",
        "problems": [
            ("Gradient", "For $f=x^2+xy+2y^2$, compute $nabla f$ and evaluate it at $(1,-1)$."),
            ("Directional derivative", "At $(1,-1)$, compute the derivative in unit direction $(3,4)/5$."),
            ("Contour tangent", "Find a direction tangent to the contour through $(1,-1)$ and verify its dot product with the gradient is zero."),
            ("Chain rule", "For $z=sin(x^2+3y)$, compute both partial derivatives."),
            ("Finite difference", "Derive the $O(h^2)$ central-difference formula and explain why very small $h$ eventually fails in floating point."),
        ],
        "solutions": [
            "$nabla f=(2x+y,x+4y)$, hence $(1,-3)$.",
            "$(1,-3)^T(3,4)/5=-9/5$.",
            "One tangent is $(3,1)$ because $(1,-3)^T(3,1)=0$.",
            "$partial_x z=2x cos(x^2+3y)$ and $partial_y z=3cos(x^2+3y)$.",
            "Taylor expansions cancel even powers; rounding/cancellation dominates when $h$ is too small.",
        ],
        "cells": [
            md("""## 1 · Contours and the gradient field"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\ndef f(x, y): return x*x + x*y + 2*y*y\ndef grad(x, y): return np.array([2*x + y, x + 4*y])\n\nx = np.linspace(-2, 2, 120); X, Y = np.meshgrid(x, x); Z = f(X, Y)\nplt.figure(figsize=(6, 5)); plt.contour(X, Y, Z, levels=14)\nq = np.linspace(-1.8, 1.8, 13); QX, QY = np.meshgrid(q, q)\nG = grad(QX, QY); plt.quiver(QX, QY, G[0], G[1], color="#EB811B", alpha=.65)\nplt.axis("equal"); plt.grid(alpha=.12);"""),
            md("""## 2 · Directional derivative versus a small step"""),
            code("""point = np.array([1., -1.]); direction = np.array([3., 4.]) / 5\nexact = grad(*point) @ direction\nfor h in [1e-1, 1e-3, 1e-5]:\n    estimate = (f(*(point + h*direction)) - f(*point)) / h\n    print(h, estimate, "error", abs(estimate-exact))\nprint("exact:", exact)"""),
            md("""## 3 · Gradient checking and the U-shaped error curve"""),
            code("""hs = np.logspace(-16, -1, 80)\nerrors = []\nfor h in hs:\n    gx = (f(point[0]+h, point[1]) - f(point[0]-h, point[1]))/(2*h)\n    gy = (f(point[0], point[1]+h) - f(point[0], point[1]-h))/(2*h)\n    errors.append(np.linalg.norm([gx, gy] - grad(*point)))\nplt.loglog(hs, errors); plt.xlabel("h"); plt.ylabel("gradient error"); plt.grid(True, which="both", alpha=.2);"""),
        ],
    },
    {
        "n": 5,
        "slug": "backprop-micrograd",
        "title": "Backprop by Hand + micrograd",
        "after": "L11",
        "intro": "Run reverse-mode differentiation on a graph, then implement the small scalar engine behind it.",
        "problems": [
            ("Graph forward pass", "For $a=xy$, $b=a+x$, $L=b^2$ at $(x,y)=(2,-3)$, compute every node value."),
            ("Backward pass", "Starting with $partial L/partial L=1$, compute adjoints for $b,a,x,y$. Remember that $x$ has two paths."),
            ("Topological order", "Explain why parents must receive gradients only after all children have contributed."),
            ("Accumulation", "Give a graph where assigning rather than adding gradients returns a wrong result."),
            ("Vector–Jacobian product", "For $f:R^n to R^m$, state the shapes of a seed row vector and the resulting VJP."),
        ],
        "solutions": [
            "$a=-6$, $b=-4$, $L=16$.",
            "$bar b=-8$, $bar a=-8$, $bar x=bar b+bar a y=16$, $bar y=bar a x=-16$.",
            "Reverse mode must sum every downstream contribution before applying a node's local rule.",
            "$x+x$ or any shared subexpression needs two contributions accumulated at $x$.",
            "Seed has shape $1 times m$ and VJP has shape $1 times n$.",
        ],
        "cells": [
            md("""## 1 · A scalar reverse-mode engine"""),
            code("""class Value:\n    def __init__(self, data, parents=(), backward=lambda: None):\n        self.data, self.grad = float(data), 0.0\n        self.parents, self._backward = parents, backward\n    def __add__(self, other):\n        other = other if isinstance(other, Value) else Value(other)\n        out = Value(self.data + other.data, (self, other))\n        def back(): self.grad += out.grad; other.grad += out.grad\n        out._backward = back; return out\n    __radd__ = __add__\n    def __mul__(self, other):\n        other = other if isinstance(other, Value) else Value(other)\n        out = Value(self.data * other.data, (self, other))\n        def back(): self.grad += other.data*out.grad; other.grad += self.data*out.grad\n        out._backward = back; return out\n    __rmul__ = __mul__\n    def __pow__(self, power):\n        out = Value(self.data**power, (self,))\n        def back(): self.grad += power*self.data**(power-1)*out.grad\n        out._backward = back; return out\n    def backward(self):\n        topo, seen = [], set()\n        def visit(v):\n            if id(v) not in seen:\n                seen.add(id(v))\n                for parent in v.parents: visit(parent)\n                topo.append(v)\n        visit(self); self.grad = 1.0\n        for v in reversed(topo): v._backward()"""),
            md("""## 2 · Reproduce the hand calculation"""),
            code("""x, y = Value(2), Value(-3)\na = x*y\nb = a+x\nloss = b**2\nloss.backward()\nprint("values:", a.data, b.data, loss.data)\nprint("gradients dx, dy:", x.grad, y.grad)"""),
            md("""## 3 · Compare with finite differences"""),
            code("""def objective(x, y): return (x*y + x)**2\nh = 1e-6\ndx = (objective(2+h, -3)-objective(2-h, -3))/(2*h)\ndy = (objective(2, -3+h)-objective(2, -3-h))/(2*h)\nprint(dx, dy)"""),
        ],
    },
    {
        "n": 6,
        "slug": "densities-gaussians",
        "title": "Densities & Gaussians",
        "after": "L13",
        "intro": "Connect density, area, sampling, transformations, and covariance geometry.",
        "problems": [
            ("Density versus probability", "For $X sim Uniform(0,0.2)$, find the density and $P(0.05<X<0.08)$. Why may the density exceed one?"),
            ("Exponential waiting", "For rate $lambda=2$, find $P(X>1)$ and $E[X]$."),
            ("Linear transformation", "If $X sim N(2,9)$ and $Y=3X-1$, find the mean and variance of $Y$."),
            ("Covariance ellipse", "For $Sigma=[[4,3],[3,4]]$, find ellipse axes and standard deviations."),
            ("Mahalanobis", "Compare squared Mahalanobis distances of $(1,1)$ and $(1,-1)$ under that covariance."),
        ],
        "solutions": [
            "Density is $5$; probability is rectangle area $5(0.03)=0.15$. Density is probability per unit, not probability.",
            "$P(X>1)=e^{-2}$ and $E[X]=1/2$.",
            "$E[Y]=5$ and $Var(Y)=9Var(X)=81$.",
            "Eigenvectors are diagonal/anti-diagonal with eigenvalues $7,1$, so standard deviations are $sqrt7,1$.",
            "They are $2/7$ and $2$ respectively; the low-variance direction makes deviations more unusual.",
        ],
        "cells": [
            md("""## 1 · Density is area, not height"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy import stats\n\nx = np.linspace(-.05, .25, 400)\npdf = stats.uniform(0, .2).pdf(x)\nplt.plot(x, pdf); plt.fill_between(x, 0, pdf, where=(x>.05)&(x<.08), alpha=.35)\nprint("area:", stats.uniform(0, .2).cdf(.08)-stats.uniform(0, .2).cdf(.05))"""),
            md("""## 2 · Transform samples"""),
            code("""rng = np.random.default_rng(6)\nX = rng.normal(2, 3, 100_000); Y = 3*X - 1\nprint("X mean/var", X.mean(), X.var())\nprint("Y mean/var", Y.mean(), Y.var())"""),
            md("""## 3 · A covariance ellipse from eigenvectors"""),
            code("""Sigma = np.array([[4., 3.], [3., 4.]])\nZ = rng.multivariate_normal([0, 0], Sigma, 3000)\nvalues, vectors = np.linalg.eigh(Sigma)\nplt.scatter(*Z.T, s=5, alpha=.15)\nfor value, vector in zip(values, vectors.T):\n    plt.arrow(0, 0, *(2*np.sqrt(value)*vector), width=.025, color="#EB811B")\nplt.axis("equal"); plt.grid(alpha=.15)\nprint(values, vectors)"""),
        ],
    },
    {
        "n": 7,
        "slug": "mle-map-bayes",
        "title": "MLE, MAP & Bayesian Updating",
        "after": "L15",
        "intro": "Estimate a coin bias, update a Beta prior, and see regularization as prior information.",
        "problems": [
            ("Bernoulli MLE", "For 3 heads and 1 tail, write the likelihood/log-likelihood and derive the MLE."),
            ("Beta posterior", "With prior Beta$(2,2)$, find the posterior after those flips and its mean."),
            ("MAP", "Find the interior MAP of Beta$(5,3)$ and compare it with the mean."),
            ("Sequential update", "Show that updating after batches (2H,1T) then (1H,0T) matches one update with all data."),
            ("Prior as counts", "Interpret Beta$(10,2)$ using pseudo-counts, and explain when data will dominate it."),
        ],
        "solutions": [
            "$L(theta)=theta^3(1-theta)$; derivative of log likelihood gives $hat theta=3/4$.",
            "Posterior Beta$(5,3)$ with mean $5/8$.",
            "MAP $(5-1)/(5+3-2)=2/3$; mean is $5/8$.",
            "Both yield Beta$(5,3)$ because sufficient statistics add.",
            "The prior contributes 9 head-like and 1 tail-like MAP pseudo-count; enough observations overwhelm fixed prior counts.",
        ],
        "cells": [
            md("""## 1 · Likelihood and posterior on one plot"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy import stats\n\ntheta = np.linspace(.001, .999, 600)\nh, t = 3, 1\nlikelihood = theta**h * (1-theta)**t\nprior = stats.beta(2, 2).pdf(theta)\nposterior = stats.beta(2+h, 2+t).pdf(theta)\nfor y, label in [(likelihood/likelihood.max(), "likelihood (scaled)"), (prior/prior.max(), "prior"), (posterior/posterior.max(), "posterior")]:\n    plt.plot(theta, y, label=label)\nplt.axvline(h/(h+t), color="#EB811B", ls="--", label="MLE")\nplt.legend(); plt.xlabel("coin bias"); plt.grid(alpha=.15);"""),
            md("""## 2 · Stream flips and update online"""),
            code("""flips = [1, 1, 0, 1, 0, 1, 1, 1]\na, b = 2, 2\nfor n, flip in enumerate(flips, 1):\n    a += flip; b += 1-flip\n    print(n, f"Beta({a},{b})", "mean", round(a/(a+b), 3))"""),
            md("""## 3 · MLE, MAP, and posterior mean"""),
            code("""h, t, a0, b0 = 3, 1, 2, 2\nmle = h/(h+t)\na, b = a0+h, b0+t\nmap_est = (a-1)/(a+b-2)\nmean = a/(a+b)\nprint({"MLE": mle, "MAP": map_est, "posterior mean": mean})"""),
        ],
    },
    {
        "n": 8,
        "slug": "gradient-descent-conditioning",
        "title": "Gradient Descent & Conditioning",
        "after": "L17",
        "intro": "Run gradient descent on quadratic bowls and isolate the roles of learning rate, condition number, scaling, and momentum.",
        "problems": [
            ("Scalar recurrence", "For $f(x)=x^2/2$, derive the GD recurrence and the range of learning rates that converges."),
            ("Quadratic directions", "For $f=(x^2+50y^2)/2$, write the two coordinate recurrences."),
            ("Safe step", "Find the largest stable constant learning rate and explain why the flat direction then moves slowly."),
            ("Scaling", "Rescale $z=sqrt(50)y$. What happens to the Hessian condition number?"),
            ("Momentum", "Explain why a velocity average can reduce zig-zagging in the steep direction while retaining motion along the valley."),
        ],
        "solutions": [
            "$x_(k+1)=(1-eta)x_k$; convergence requires $0<eta<2$.",
            "$x^+=(1-eta)x$ and $y^+=(1-50eta)y$.",
            "$eta<2/50=0.04$; then the $x$ contraction is near one.",
            "The Hessian becomes identity and condition number becomes one.",
            "Alternating steep-direction gradients cancel in the moving average; consistent valley gradients accumulate.",
        ],
        "cells": [
            md("""## 1 · Learning-rate regimes"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\ndef run(eta, steps=30):\n    x = 2.0; history=[]\n    for _ in range(steps): history.append(x); x -= eta*x\n    return history\nfor eta in [.1, .8, 1.8, 2.1]: plt.plot(run(eta), label=f"eta={eta}")\nplt.axhline(0, color="black", lw=.7); plt.legend(); plt.xlabel("step"); plt.grid(alpha=.15);"""),
            md("""## 2 · Conditioning creates a zig-zag"""),
            code("""H = np.diag([1., 50.])\nx = np.array([2., .8]); path=[x.copy()]\nfor _ in range(100):\n    x -= .035*(H @ x); path.append(x.copy())\npath=np.array(path)\nxx, yy=np.meshgrid(np.linspace(-2.2,2.2,160), np.linspace(-1,1,160))\nplt.contour(xx,yy,.5*(xx**2+50*yy**2),levels=18)\nplt.plot(path[:,0],path[:,1],"o-",ms=2,color="#EB811B"); plt.axis("equal");"""),
            md("""## 3 · Momentum on the same bowl"""),
            code("""def optimize(momentum):\n    x=np.array([2.,.8]); v=np.zeros(2); path=[]\n    for _ in range(100):\n        v=momentum*v + H@x; x=x-.02*v; path.append(x.copy())\n    return np.array(path)\nfor beta in [0, .8]:\n    p=optimize(beta); plt.semilogy(np.linalg.norm(p,axis=1),label=f"beta={beta}")\nplt.legend(); plt.xlabel("step"); plt.ylabel("distance to optimum"); plt.grid(alpha=.15);"""),
        ],
    },
    {
        "n": 9,
        "slug": "newton-curve-fitting",
        "title": "Newton & Curve Fitting",
        "after": "L18",
        "intro": "Compare gradient and Newton steps, then fit a nonlinear decay model with Gauss–Newton structure.",
        "problems": [
            ("Newton step", "For $f(x)=x-ln x$, derive Newton's update and take two steps from $x_0=0.5$."),
            ("Quadratic", "Show that Newton reaches the minimizer of $f(x)=a x^2/2+b x+c$ in one step when $a>0$."),
            ("Least-squares Jacobian", "For residual $r_i(a,b)=a exp(-bt_i)-y_i$, compute both Jacobian columns."),
            ("Gauss–Newton", "Write the normal equation for the parameter step and state when $J^TJ$ may be singular."),
            ("Damping", "Explain how $(J^TJ+lambda I)delta=-J^Tr$ interpolates between Gauss–Newton and a small gradient-like step."),
        ],
        "solutions": [
            "$x^+=x-f'/f''=2x-x^2$; $0.5 to 0.75 to 0.9375$.",
            "$f'=ax+b$, $f''=a$, so $x^+=x-(ax+b)/a=-b/a$.",
            "$partial_a r_i=e^{-bt_i}$ and $partial_b r_i=-a t_i e^{-bt_i}$.",
            "$(J^TJ)delta=-J^Tr$; singularity occurs when columns are dependent or parameters are locally unidentifiable.",
            "Large $lambda$ gives $delta approx-(1/lambda)J^Tr$; zero damping recovers Gauss–Newton.",
        ],
        "cells": [
            md("""## 1 · Newton's digits double near the solution"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\nx=.5\nfor k in range(7):\n    print(k, x, "error", abs(x-1))\n    x=2*x-x*x"""),
            md("""## 2 · Fit exponential decay"""),
            code("""from scipy.optimize import least_squares\nrng=np.random.default_rng(9)\nt=np.linspace(0,4,18); y=2.4*np.exp(-.55*t)+rng.normal(0,.035,len(t))\ndef residual(params):\n    a,b=params; return a*np.exp(-b*t)-y\ndef jac(params):\n    a,b=params; e=np.exp(-b*t); return np.column_stack([e,-a*t*e])\nfit=least_squares(residual,[1,.2],jac=jac,method="lm")\nprint(fit.x, "steps", fit.nfev)\nplt.scatter(t,y); plt.plot(t,fit.x[0]*np.exp(-fit.x[1]*t),color="#EB811B");"""),
            md("""## 3 · Inspect the Gauss–Newton curvature"""),
            code("""J=jac(fit.x); print("J.T J =\\n",J.T@J); print("condition number",np.linalg.cond(J.T@J))"""),
        ],
    },
    {
        "n": 10,
        "slug": "lagrange-kkt",
        "title": "Lagrange Multipliers & KKT",
        "after": "L20",
        "intro": "Solve equality and inequality constrained problems, then verify every KKT condition numerically.",
        "problems": [
            ("Equality", "Maximize $xy$ subject to $x+y=1$. Solve for the point and multiplier using $L=xy-lambda(x+y-1)$."),
            ("Sensitivity", "Replace 1 by budget $b$. Derive the optimal value and confirm that its derivative equals the multiplier."),
            ("Active inequality", "Minimize $(x-2)^2$ subject to $x<=1$. Solve both complementary-slackness cases."),
            ("Inactive inequality", "Minimize $(x-.2)^2$ subject to $x<=1$. Find the multiplier and slack."),
            ("Simplex projection", "Project $(3,1,.5)$ onto non-negative vectors summing to 2.5. Find the threshold and result."),
        ],
        "solutions": [
            "$x=y=lambda=1/2$, objective $1/4$.",
            "$x=y=b/2$, $V(b)=b^2/4$, and $V'(b)=b/2=lambda$.",
            "Inactive gives infeasible $x=2$; active gives $x=1$, $lambda=2$.",
            "$x=.2$, slack $.8$, multiplier zero.",
            "Threshold $.75$ gives $(2.25,.25,0)$.",
        ],
        "cells": [
            md("""## 1 · Contours meeting a constraint"""),
            code("""import numpy as np\nimport matplotlib.pyplot as plt\n\nx=np.linspace(0,1,200); X,Y=np.meshgrid(x,x)\nplt.contour(X,Y,X*Y,levels=10); plt.plot(x,1-x,color="#EB811B",lw=3)\nplt.scatter([.5],[.5],color="#D64550"); plt.axis("equal");"""),
            md("""## 2 · Verify KKT residuals"""),
            code("""x, lam = 1.0, 2.0\nchecks={\n    "primal g(x)<=0": x-1,\n    "dual lambda>=0": lam,\n    "stationarity": 2*(x-2)+lam,\n    "complementarity": lam*(x-1),\n}\nprint(checks)"""),
            md("""## 3 · Project onto the simplex by thresholding"""),
            code("""def project_simplex(a, budget):\n    lo, hi = 0.0, np.max(a)\n    for _ in range(80):\n        nu=(lo+hi)/2\n        if np.maximum(a-nu,0).sum()>budget: lo=nu\n        else: hi=nu\n    return np.maximum(a-hi,0), hi\nx,nu=project_simplex(np.array([3.,1.,.5]),2.5)\nprint(x,"threshold",nu,"sum",x.sum())"""),
        ],
    },
    {
        "n": 11,
        "slug": "lp-qp-cvxpy",
        "title": "LP/QP with CVXPY",
        "after": "L21",
        "intro": "Model an allocation LP and projection QP, solve them, and inspect primal and dual checks.",
        "problems": [
            ("LP model", "Maximize $3x+2y$ subject to $x+y<=4$, $x<=2$, $y<=3$, $x,y>=0$. List vertices and solve."),
            ("LP dual", "Write the three-variable dual and find prices matching the primal optimum."),
            ("QP projection", "Project $(3,1.4)$ onto the same polytope. Explain why the answer need not be a vertex."),
            ("Convexity", "For QP Hessian $P$, state the condition that makes the objective convex."),
            ("Status", "Explain what `infeasible`, `unbounded`, and `optimal_inaccurate` require you to do before using variable values."),
        ],
        "solutions": [
            "Best vertex is $(2,2)$ with value 10.",
            "Dual prices $(2,1,0)$ give value $4(2)+2(1)=10$.",
            "The projection is $(2,1.4)$, in the interior of the edge $x=2$.",
            "$P$ must be positive semidefinite.",
            "Treat status as part of the result; inspect constraints, scaling, tolerances, and residuals before interpreting values.",
        ],
        "cells": [
            md("""## 1 · Solve the LP with SciPy and inspect slacks"""),
            code("""import numpy as np\nfrom scipy.optimize import linprog, minimize\n\nc=np.array([3.,2.]); A=np.array([[1.,1.],[1.,0.],[0.,1.]]); b=np.array([4.,2.,3.])\nres=linprog(-c,A_ub=A,b_ub=b,bounds=[(0,None)]*2,method="highs")\nprint(res.x, "maximum", -res.fun, "slacks", b-A@res.x)\nprint("inequality marginals for minimized -profit",res.ineqlin.marginals)"""),
            md("""## 2 · Solve the projection QP"""),
            code("""z=np.array([3.,1.4])\nobjective=lambda x:.5*np.sum((x-z)**2)\nconstraints=[{"type":"ineq","fun":lambda x:b-A@x}]\nqp=minimize(objective,[1.,1.],bounds=[(0,None)]*2,constraints=constraints,method="SLSQP")\nprint(qp.x, qp.fun, "max violation",np.max(A@qp.x-b))"""),
            md("""## 3 · The same models in CVXPY when installed"""),
            code("""try:\n    import cvxpy as cp\n    x=cp.Variable(2,nonneg=True); con=A@x<=b\n    prob=cp.Problem(cp.Maximize(c@x),[con]); prob.solve()\n    print("CVXPY LP",x.value,prob.value,"dual",con.dual_value)\n    q=cp.Variable(2); prob2=cp.Problem(cp.Minimize(.5*cp.sum_squares(q-z)),[A@q<=b,q>=0]); prob2.solve()\n    print("CVXPY QP",q.value,prob2.value)\nexcept ImportError:\n    print("Optional: install cvxpy to run this comparison; SciPy results above remain complete.")"""),
        ],
    },
    {
        "n": 12,
        "slug": "entropy-huffman-kl",
        "title": "Entropy, Huffman & KL",
        "after": "L24",
        "intro": "Compute surprise and entropy, build a Huffman code, and measure the excess cost of a wrong model.",
        "problems": [
            ("Entropy", "Compute entropy of $(.4,.2,.15,.15,.1)$ in bits."),
            ("Kraft", "Check whether lengths $(1,3,3,3,3)$ form a complete binary prefix tree."),
            ("Huffman", "Run the merge weights for probabilities proportional to $(40,20,15,15,10)$."),
            ("Cross-entropy", "For $p=(.75,.2,.05)$ and $q=(.45,.35,.2)$, compute $H(p)$, $H(p,q)$, and KL."),
            ("Perplexity", "Convert 2.5 bits/symbol and 2.5 nats/symbol to their matching perplexities."),
        ],
        "solutions": [
            "$H approx2.146$ bits.",
            "$1/2+4/8=1$, so it exactly fills a binary tree.",
            "Merge $10+15$, $15+20$, $25+35$, then $40+60$ (ties may change codewords, not expected length).",
            "$H(p)approx.992$, $H(p,q)approx1.283$, KL $approx.291$ bits.",
            "$2^{2.5}approx5.657$ and $e^{2.5}approx12.182$.",
        ],
        "cells": [
            md("""## 1 · Entropy, cross-entropy, and KL"""),
            code("""import math, heapq, itertools\nimport numpy as np\n\ndef entropy(p):\n    p=np.asarray(p,float); return -np.sum(p[p>0]*np.log2(p[p>0]))\ndef cross_entropy(p,q): return -np.sum(np.asarray(p)*np.log2(q))\np=np.array([.75,.2,.05]); q=np.array([.45,.35,.2])\nprint("H",entropy(p),"CE",cross_entropy(p,q),"KL",cross_entropy(p,q)-entropy(p))"""),
            md("""## 2 · Build a deterministic Huffman code"""),
            code("""weights={"A":40,"B":20,"C":15,"D":15,"E":10}; serial=itertools.count()\nheap=[(w,next(serial),s) for s,w in weights.items()]; heapq.heapify(heap)\nwhile len(heap)>1:\n    wa,_,a=heapq.heappop(heap); wb,_,b=heapq.heappop(heap)\n    heapq.heappush(heap,(wa+wb,next(serial),(a,b)))\ntree=heap[0][2]\ndef codes(node,prefix="",out=None):\n    out={} if out is None else out\n    if isinstance(node,str): out[node]=prefix\n    else: codes(node[0],prefix+"0",out); codes(node[1],prefix+"1",out)\n    return out\ncodebook=codes(tree); print(codebook)\nL=sum(weights[s]*len(codebook[s]) for s in weights)/sum(weights.values())\nprint("average length",L)"""),
            md("""## 3 · Encode and decode"""),
            code("""message="ABACADABRA".replace("R","E")\nbits="".join(codebook[ch] for ch in message)\nreverse={v:k for k,v in codebook.items()}; decoded=[]; prefix=""\nfor bit in bits:\n    prefix+=bit\n    if prefix in reverse: decoded.append(reverse[prefix]); prefix=""\nprint(message,bits,"".join(decoded),len(bits))"""),
        ],
    },
    {
        "n": 13,
        "slug": "markov-ngram",
        "title": "Markov Chains & the n-gram LM",
        "after": "L26",
        "intro": "Compute a stationary distribution, fit smoothed character transitions, score held-out text, and generate a sample.",
        "problems": [
            ("Stationary distribution", "For $P=[[.8,.2],[.4,.6]]$, solve $pi=pi P$ and normalize."),
            ("Path probability", "With initial $(.6,.4)$, find $P(S,R,R,S)$."),
            ("Transition MLE", "Derive the MLE row from counts $n_(ij)$ under the row-sum constraint."),
            ("Smoothing", "Apply add-one prediction to counts $(8,2,0,0)$."),
            ("BPC", "If test transitions receive probabilities $(1/2,1/4,1/8)$, compute total bits, BPC, and perplexity."),
        ],
        "solutions": [
            "$pi=(2/3,1/3)$.",
            "$0.6(.2)(.6)(.4)=.0288$.",
            "$hat P_(ij)=n_(ij)/sum_j n_(ij)$.",
            "$(9/14,3/14,1/14,1/14)$.",
            "Total 6 bits, BPC 2, perplexity 4.",
        ],
        "cells": [
            md("""## 1 · Power iteration finds the stationary distribution"""),
            code("""import collections, math, string\nimport numpy as np\n\nP=np.array([[.8,.2],[.4,.6]]); mu=np.array([1.,0.])\nfor _ in range(30): mu=mu@P\nprint(mu, "residual",np.linalg.norm(mu@P-mu))"""),
            md("""## 2 · Fit a smoothed character bigram"""),
            code("""from pathlib import Path

def clean(text):
    text=text.lower(); return " ".join("".join(ch if ch in string.ascii_lowercase else " " for ch in text).split())
source = Path("lecture-plan-detailed.md")
if not source.exists(): source = Path("../lecture-plan-detailed.md")
text=clean(source.read_text())
split=int(.8*len(text)); train,test=text[:split],text[split:]
alphabet=" "+string.ascii_lowercase; K=len(alphabet); alpha=.1
counts=collections.defaultdict(collections.Counter)
for a,b in zip(train[:-1],train[1:]): counts[a][b]+=1
def row(context):
    total=sum(counts[context].values())+alpha*K
    return np.array([(counts[context][ch]+alpha)/total for ch in alphabet])
loss=[]
for a,b in zip(test[:-1],test[1:]): loss.append(-math.log2(row(a)[alphabet.index(b)]))
print("held-out BPC",np.mean(loss),"perplexity",2**np.mean(loss))"""),
            md("""## 3 · Generate from the same rows"""),
            code("""rng=np.random.default_rng(13); current="t"; output=[current]\nfor _ in range(400):\n    current=rng.choice(list(alphabet),p=row(current)); output.append(current)\nprint("".join(output))"""),
            md("""## 4 · Closing experiment\n\nChange `alpha`, the training fraction, or the context order. Report held-out BPC before judging samples by eye."""),
        ],
    },
]


def qmd_for(tutorial: dict) -> str:
    n = tutorial["n"]
    problem_text = []
    for index, (title, prompt) in enumerate(tutorial["problems"], 1):
        problem_text.append(f"### Problem {index} — {title}\n\n{latex_math(prompt)}")
    solution_text = []
    for index, solution in enumerate(tutorial["solutions"], 1):
        solution_text.append(f"**{index}.** {latex_math(solution)}")
    return (
        "---\n"
        f'title: "Tutorial {n} · {tutorial["title"]}"\n'
        f'subtitle: "After {tutorial["after"]} · 80 minutes"\n'
        "---\n\n"
        f'{tutorial["intro"]}\n\n'
        "## Part A · Worksheet (~40 min, pen and paper)\n\n"
        f'{(chr(10) * 2).join(problem_text)}\n\n'
        '::: {.callout-note collapse="true"}\n'
        "## Solutions — open only after attempting everything\n\n"
        f'{(chr(10) * 2).join(solution_text)}\n'
        ":::\n\n"
        "## Part B · Notebook (~40 min, laptop)\n\n"
        f'Open [`../notebooks/tut{n:02d}-{tutorial["slug"]}.ipynb`](../notebooks/tut{n:02d}-{tutorial["slug"]}.ipynb). '
        "Predict each result, run the cell, and explain any disagreement.\n"
    )


def notebook_for(tutorial: dict) -> dict:
    n = tutorial["n"]
    opening = md(
        f"""
        # Tutorial {n} · {tutorial['title']} — Part B

        **~40 minutes · after the worksheet**

        {tutorial['intro']}

        Work in pairs. Before each code cell, write down the qualitative result you expect. The notebook is designed to run top-to-bottom in a fresh kernel.
        """
    )
    closing = md(
        f"""
        ## Closing check

        Write three sentences: one numerical result you verified, one geometric/probabilistic interpretation, and one failure mode you would now test in a larger implementation.
        """
    )
    cells = [opening, *tutorial["cells"], closing]
    for index, cell in enumerate(cells):
        cell["id"] = f"t{n:02d}-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for tutorial in TUTORIALS:
        n = tutorial["n"]
        qmd_path = ROOT / "tutorials" / f"tut{n:02d}-{tutorial['slug']}.qmd"
        ipynb_path = ROOT / "notebooks" / f"tut{n:02d}-{tutorial['slug']}.ipynb"
        qmd_path.write_text(qmd_for(tutorial), encoding="utf-8")
        ipynb_path.write_text(json.dumps(notebook_for(tutorial), indent=1) + "\n", encoding="utf-8")
        print(qmd_path.relative_to(ROOT), ipynb_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
