"""Variational Autoencoder on the 2-D ring.

Encoder maps x -> (mu, logvar); we reparameterize z = mu + σ·ε and decode back
to x. Loss = reconstruction MSE + KL(q(z|x) ‖ N(0,I)). Generation samples
z ~ N(0,I) and decodes. VAEs cover all modes but tend to *over-smooth*.
"""
import numpy as np

from autograd import Tensor
from common import MLP, adam, zero_grad, load

ZDIM, HID, EPOCHS, BATCH = 2, 64, 1500, 256


def train(seed=0):
    rng = np.random.default_rng(seed)
    X = load()
    enc = MLP([2, HID, 2 * ZDIM], seed=seed)        # -> [mu | logvar]
    dec = MLP([ZDIM, HID, 2], seed=seed + 1)
    params = enc.params() + dec.params()
    step = adam(params)
    for ep in range(EPOCHS):
        idx = rng.choice(len(X), BATCH, replace=False)
        xb = X[idx]
        zero_grad(params)
        stats = enc.forward(xb)
        mu = _slice(stats, 0, ZDIM)
        logvar = _slice(stats, ZDIM, 2 * ZDIM)
        eps = rng.standard_normal((BATCH, ZDIM))
        std = (logvar * 0.5).exp()
        z = mu + std * Tensor(eps)
        recon = dec.forward(z)
        diff = recon - Tensor(xb)
        recon_loss = (diff * diff).sum() * (1.0 / BATCH)
        kl = (((mu * mu) + logvar.exp() - logvar + (-1.0)) * 0.5).sum() * (1.0 / BATCH)
        loss = recon_loss + kl
        loss.backward()
        step(lr=2e-3)
    return dec


def _slice(t, a, b):
    """Differentiable column slice [a:b] via a selection matrix."""
    cols = t.data.shape[1]
    S = np.zeros((cols, b - a))
    for j in range(a, b):
        S[j, j - a] = 1.0
    return t.matmul(Tensor(S))


def sample(dec, n, rng):
    z = rng.standard_normal((n, ZDIM))
    return dec.forward(z).data


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dec = train()
    print("VAE trained; sample shape", sample(dec, 5, rng).shape)
