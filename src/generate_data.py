"""Sample a 2-D target distribution: a ring of 8 Gaussian modes.

A classic generative-modeling benchmark. It is low-dimensional (so we can plot
and score it exactly) yet multi-modal — which exposes the failure modes that
matter: GAN *mode collapse*, VAE *over-smoothing*, and diffusion's slower but
faithful coverage.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N, MODES, RADIUS, STD = 2400, 8, 2.0, 0.18


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    ang = 2 * np.pi * np.arange(MODES) / MODES
    centers = np.c_[RADIUS * np.cos(ang), RADIUS * np.sin(ang)]
    which = rng.integers(0, MODES, N)
    X = centers[which] + rng.standard_normal((N, 2)) * STD
    np.savetxt(DATA / "samples.csv", X, delimiter=",", header="x,y", comments="")
    print(f"Saved {N} samples from {MODES}-mode ring -> data/samples.csv")


if __name__ == "__main__":
    main()
