#!/usr/bin/env bash
# Sample the target distribution, train VAE + GAN + diffusion, compare them.
set -e
cd "$(dirname "$0")"

pip install -r requirements.txt
python3 src/generate_data.py
python3 src/compare.py
echo ""
echo "See reports/samples.png and reports/metrics.json"
