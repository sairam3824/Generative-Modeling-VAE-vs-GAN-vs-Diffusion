"""Train all three generators, plot their samples, and score them by energy distance."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load, energy_distance
import vae, gan, diffusion

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    rng = np.random.default_rng(0)
    X = load()
    print("training VAE ...");        dec = vae.train()
    print("training GAN ...");        G = gan.train()
    print("training diffusion ...");  net = diffusion.train()

    n = 1200
    samples = {
        "Real data": X[rng.choice(len(X), n, replace=False)],
        "VAE": vae.sample(dec, n, rng),
        "GAN": gan.sample(G, n, rng),
        "Diffusion": diffusion.sample(net, n, rng),
    }
    scores = {k: round(energy_distance(X, v, rng), 4)
              for k, v in samples.items() if k != "Real data"}

    REPORTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (name, pts) in zip(axes, samples.items()):
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.4, color="#264653")
        title = name if name == "Real data" else f"{name}  (E={scores[name]})"
        ax.set_title(title); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Generative models on an 8-mode ring (energy distance E: lower = closer to real)")
    fig.tight_layout(); fig.savefig(REPORTS / "samples.png", dpi=110)

    (REPORTS / "metrics.json").write_text(json.dumps(scores, indent=2))
    best = min(scores, key=scores.get)
    print("energy distance:", scores, "| best:", best)
    print("See reports/samples.png and reports/metrics.json")


if __name__ == "__main__":
    main()
