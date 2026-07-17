#!/usr/bin/env python3
"""Generate the per-neuron test-correlation histogram for the final model
(Run 10c, 0.15 Cadena cut, 523 neurons)."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

NPZ = os.path.expanduser("~/monkeyN_V1_run10c_ks4_filtered015/neuron_analysis/neuron_analysis_test.npz")
c = np.load(NPZ, allow_pickle=True)["per_neuron_corr"]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(c, bins=np.arange(0, 1.02, 0.05), color="#4C72B0",
        edgecolor="white", linewidth=0.6)
ax.axvline(c.mean(), color="#C44E52", linestyle="--", linewidth=1.5,
           label=f"mean = {c.mean():.3f}")
ax.axvline(np.median(c), color="#55A868", linestyle=":", linewidth=1.5,
           label=f"median = {np.median(c):.3f}")
ax.set_xlabel("Per-neuron test correlation $r_{\\mathrm{test}}$")
ax.set_ylabel("Number of neurons")
ax.set_xlim(0, 1)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"corr_hist_final.{ext}"), dpi=200)
print("wrote", os.path.join(FIGS, "corr_hist_final.pdf/.png"))
print(f"n={len(c)}  mean={c.mean():.3f}  median={np.median(c):.3f}  "
      f">0.3: {(c>0.3).sum()}  >0.5: {(c>0.5).sum()}  >0.7: {(c>0.7).sum()}")
