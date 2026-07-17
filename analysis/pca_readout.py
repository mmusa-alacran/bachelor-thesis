#!/usr/bin/env python3
"""Cleaner PCA of the readout vectors coloured by waveform class.

Each neuron's 1152-d readout weight vector is L2-normalised first, so the
overall gain (which tracks firing rate) is removed and any tuning-shape
structure related to waveform class can show. Axes are zoomed to the bulk.
"""
import os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = os.path.expanduser("~/celltype_probe")
sd = torch.load(os.path.expanduser("~/monkeyN_V1_run10c_ks4_filtered015/best.pt"), map_location="cpu")["model"]
feats = sd["readout.features"].numpy()[0, :, 0, :].T
d = np.load(os.path.expanduser("~/tvsd_monkeyN_V1_sua_KS4_filtered015.npz"), allow_pickle=True)
uid = d["unit_ids"]; mfr = d["responses"].mean(0)
lab = pd.read_csv(os.path.expanduser("~/waveform_labels_monkeyN.csv"))
key2 = {(int(r.electrode_id), int(r.unit_index)): r for r in lab.itertuples()}
keep, yc, yw = [], [], []
for i in range(len(uid)):
    r = key2.get((int(uid[i, 0]), int(uid[i, 1])))
    ok = r is not None and r.area == "V1" and isinstance(r.final_class, str)
    keep.append(ok); yc.append(r.final_class if ok else None); yw.append(r.width_wf_class if ok else None)
keep = np.array(keep)
feats = feats[keep]; mfr = mfr[keep]
yc = np.array([yc[i] for i in range(len(uid)) if keep[i]])
yw = np.array([yw[i] for i in range(len(uid)) if keep[i]])
n = len(yc)

# L2-normalise per neuron, then standardise columns, then PCA
fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
pca = PCA(2, random_state=0).fit(StandardScaler().fit_transform(fn))
P = pca.transform(StandardScaler().fit_transform(fn))
ev = pca.explained_variance_ratio_ * 100

def zoom(ax):
    xl = np.percentile(P[:, 0], [1, 99]); yl = np.percentile(P[:, 1], [1, 99])
    mx = (xl[1]-xl[0])*0.08; my = (yl[1]-yl[0])*0.08
    ax.set_xlim(xl[0]-mx, xl[1]+mx); ax.set_ylim(yl[0]-my, yl[1]+my)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
order = ["DOWN_wide", "DOWN_medium_shallow", "DOWN_medium_sharp",
         "DOWN_narrow_shallow", "DOWN_narrow_sharp", "UP"]
cmap = plt.cm.tab10(np.linspace(0, 1, len(order)))
for c, col in zip(order, cmap):
    m = yc == c
    axs[0].scatter(P[m, 0], P[m, 1], s=22, color=col, label=c, alpha=0.8, edgecolor="none")
axs[0].set_title("(a) waveform subtype"); axs[0].legend(fontsize=7, frameon=False, loc="best")
wc = {"narrow": "#C44E52", "medium": "#999999", "wide": "#4C72B0"}
for c, col in wc.items():
    m = yw == c
    axs[1].scatter(P[m, 0], P[m, 1], s=22, color=col, label=c, alpha=0.8, edgecolor="none")
axs[1].set_title("(b) waveform width"); axs[1].legend(fontsize=9, frameon=False, loc="best")
scv = axs[2].scatter(P[:, 0], P[:, 1], s=22, c=mfr, cmap="viridis", alpha=0.85, edgecolor="none")
axs[2].set_title("(c) mean firing rate"); fig.colorbar(scv, ax=axs[2], label="mean spike count")
for a in axs: zoom(a)
fig.suptitle(f"PCA of L2-normalised readout vectors, coloured by category (monkey N V1, n={n})")
fig.tight_layout()
fig.savefig(f"{OUT}/readout_pca.png", dpi=200)
print("wrote readout_pca.png   PC1=%.1f%% PC2=%.1f%%" % (ev[0], ev[1]))
