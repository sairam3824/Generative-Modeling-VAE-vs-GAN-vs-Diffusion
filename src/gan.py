"""Generative Adversarial Network on the 2-D ring.

A generator maps noise z ~ N(0,I) to 2-D points; a discriminator scores
real-vs-fake. They play a minimax game trained with the non-saturating loss and
binary-cross-entropy-with-logits (numerically stable). GANs produce sharp
samples but are prone to *mode collapse* — covering only some of the 8 modes.
"""
import numpy as np

from autograd import Tensor
from common import MLP, adam, zero_grad, load

ZDIM, HID, STEPS, BATCH = 2, 64, 4000, 256


def bce_logits(logit, target):
    """Mean numerically-stable BCE with logits:
       relu(x) - x*t + log(1 + exp(-|x|)).
    `-|x|` is built from relu so it stays differentiable in `x` while keeping
    the exp argument <= 0 (no overflow). The whole expression then differentiates
    to the exact gradient sigmoid(x) - t."""
    x = logit
    t = Tensor(target)
    neg_abs = -(x.relu() + (-x).relu())            # = -|x|, differentiable & stable
    log1pexp = (neg_abs.exp() + 1.0).log()
    return (x.relu() - (x * t) + log1pexp).sum() * (1.0 / logit.data.shape[0])


def train(seed=0):
    rng = np.random.default_rng(seed)
    X = load()
    G = MLP([ZDIM, HID, HID, 2], seed=seed)
    D = MLP([2, HID, HID, 1], seed=seed + 1)
    gstep, dstep = adam(G.params()), adam(D.params())
    for s in range(STEPS):
        idx = rng.choice(len(X), BATCH, replace=False)
        real = X[idx]
        z = rng.standard_normal((BATCH, ZDIM))
        fake = G.forward(z).data                       # detached for D step
        # --- discriminator ---
        zero_grad(D.params())
        d_real = D.forward(real)
        d_fake = D.forward(fake)
        d_loss = bce_logits(d_real, np.ones((BATCH, 1))) + \
                 bce_logits(d_fake, np.zeros((BATCH, 1)))
        d_loss.backward()
        dstep(lr=1e-3)
        # --- generator (non-saturating: maximize log D(G(z))) ---
        zero_grad(G.params())
        gen = G.forward(z)
        g_loss = bce_logits(D.forward(gen), np.ones((BATCH, 1)))
        g_loss.backward()
        gstep(lr=1e-3)
    return G


def sample(G, n, rng):
    z = rng.standard_normal((n, ZDIM))
    return G.forward(z).data


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    G = train()
    print("GAN trained; sample shape", sample(G, 5, rng).shape)
