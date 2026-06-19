"""Shared helpers: data loading, a small MLP, SGD, and an energy-distance score."""
from pathlib import Path

import numpy as np

from autograd import Tensor

ROOT = Path(__file__).resolve().parents[1]


def load():
    return np.loadtxt(ROOT / "data" / "samples.csv", delimiter=",", skiprows=1)


class MLP:
    """Configurable MLP with tanh hidden activations and linear output."""
    def __init__(self, sizes, seed=0, out_act=None):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(Tensor(rng.standard_normal((n_in, n_out)) * np.sqrt(1.0 / n_in)))
            self.b.append(Tensor(np.zeros(n_out)))
        self.out_act = out_act

    def params(self):
        return self.W + self.b

    def forward(self, X):
        h = X if isinstance(X, Tensor) else Tensor(X)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h.matmul(W) + b
            if i < len(self.W) - 1:
                h = h.tanh()
        if self.out_act == "sigmoid":
            h = h.sigmoid()
        return h


def adam(params):
    state = [{"m": np.zeros_like(p.data), "v": np.zeros_like(p.data)} for p in params]
    t = [0]

    def step(lr=2e-3, b1=0.9, b2=0.999):
        t[0] += 1
        for p, s in zip(params, state):
            g = p.grad
            s["m"] = b1 * s["m"] + (1 - b1) * g
            s["v"] = b2 * s["v"] + (1 - b2) * g * g
            mh = s["m"] / (1 - b1 ** t[0])
            vh = s["v"] / (1 - b2 ** t[0])
            p.data -= lr * mh / (np.sqrt(vh) + 1e-8)
    return step


def zero_grad(params):
    for p in params:
        p.grad = np.zeros_like(p.data)


def energy_distance(X, Y, rng, m=400):
    """Lower is better: 0 ⇔ same distribution. Energy distance on subsamples."""
    X = X[rng.choice(len(X), min(m, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(m, len(Y)), replace=False)]

    def pdist_mean(A, B):
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) + 1e-12)
        return d.mean()
    return float(2 * pdist_mean(X, Y) - pdist_mean(X, X) - pdist_mean(Y, Y))

