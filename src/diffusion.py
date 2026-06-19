"""Denoising Diffusion Probabilistic Model (DDPM) on the 2-D ring.

Forward process gradually adds Gaussian noise over T steps; a network ε_θ(x_t, t)
learns to predict the noise. Sampling runs the learned reverse process from pure
noise back to data. Diffusion is slower to sample but is the most faithful at
covering all modes (no collapse, no over-smoothing).
"""
import numpy as np

from autograd import Tensor
from common import MLP, adam, zero_grad, load

HID, STEPS, BATCH, T = 96, 5000, 256, 40


def schedule():
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar


def train(seed=0):
    rng = np.random.default_rng(seed)
    X = load()
    _, _, abar = schedule()
    net = MLP([3, HID, HID, 2], seed=seed)   # input = [x_t (2), t/T]
    step = adam(net.params())
    for s in range(STEPS):
        idx = rng.choice(len(X), BATCH, replace=False)
        x0 = X[idx]
        t = rng.integers(0, T, BATCH)
        ab = abar[t][:, None]
        eps = rng.standard_normal((BATCH, 2))
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        inp = np.concatenate([xt, (t / T)[:, None]], axis=1)
        zero_grad(net.params())
        pred = net.forward(inp)
        diff = pred - Tensor(eps)
        loss = (diff * diff).sum() * (1.0 / BATCH)
        loss.backward()
        step(lr=2e-3)
    return net


def sample(net, n, rng):
    betas, alphas, abar = schedule()
    x = rng.standard_normal((n, 2))
    for t in reversed(range(T)):
        inp = np.concatenate([x, np.full((n, 1), t / T)], axis=1)
        eps = net.forward(inp).data
        a, ab, b = alphas[t], abar[t], betas[t]
        mean = (x - (b / np.sqrt(1 - ab)) * eps) / np.sqrt(a)
        if t > 0:
            x = mean + np.sqrt(b) * rng.standard_normal((n, 2))
        else:
            x = mean
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    net = train()
    print("Diffusion trained; sample shape", sample(net, 5, rng).shape)
