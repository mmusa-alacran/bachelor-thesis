#!/usr/bin/env python3
"""3D PCA and t-SNE of the readout vectors, coloured by waveform class."""
import os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT = os.path.expanduser("~/celltype_probe"); SEED = 0
sd = torch.load(os.path.expanduser("~/monkeyN_V1_run10c_ks4_filtered015/best.pt"), map_location="cpu")["model"]
feats = sd["readout.features"].numpy()[0, :, 0, :].T
d = np.load(os.path.expanduser("~/tvsd_monkeyN_V1_sua_KS4_filtered015.npz"), allow_pickle=True)
uid = d["unit_ids"]; mfr = d["responses"].mean(0)
lab = pd.read_csv(os.path.expanduser("~/waveform_labels_monkeyN.csv"))
k2 = {(int(r.electrode_id), int(r.unit_index)): r for r in lab.itertuples()}
keep, yc, yw = [], [], []
for i in range(len(uid)):
    r = k2.get((int(uid[i, 0]), int(uid[i, 1])))
    ok = r is not None and r.area == "V1" and isinstance(r.final_class, str)
    keep.append(ok); yc.append(r.final_class if ok else None); yw.append(r.width_wf_class if ok else None)
keep = np.array(keep); feats = feats[keep]; mfr = mfr[keep]
yc = np.array([yc[i] for i in range(len(uid)) if keep[i]])
yw = np.array([yw[i] for i in range(len(uid)) if keep[i]])
n = len(yc)

# L2-normalise per neuron then standardise (removes overall gain)
fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
Xs = StandardScaler().fit_transform(fn)
wc = {"narrow": "#C44E52", "medium": "#999999", "wide": "#4C72B0"}
nb = np.where(yw == "narrow", "narrow", np.where(yw == "wide", "broad/wide", "medium"))

# ---------- 3D PCA ----------
p3 = PCA(3, random_state=SEED).fit(Xs); P = p3.transform(Xs)
ev = p3.explained_variance_ratio_ * 100
fig = plt.figure(figsize=(13, 5.6))
for k, (el, az) in enumerate([(20, 35), (20, 125)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for c, col in wc.items():
        m = yw == c
        ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=16, color=col, alpha=0.8,
                   label=f"{c} (n={m.sum()})")
    ax.set_xlim(*np.percentile(P[:, 0], [2, 98]))
    ax.set_ylim(*np.percentile(P[:, 1], [2, 98]))
    ax.set_zlim(*np.percentile(P[:, 2], [2, 98]))
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)"); ax.view_init(elev=el, azim=az)
    if k == 0:
        ax.legend(fontsize=8, frameon=False)
fig.suptitle(f"3D PCA of readout vectors by waveform width "
             f"(monkey N V1, n={n}; PC1-3 = {ev[:3].sum():.1f}%)")
fig.tight_layout(); fig.savefig(f"{OUT}/readout_pca3d.png", dpi=200)
print("wrote readout_pca3d.png  PC1-3 =", np.round(ev[:3], 1), "sum", round(ev[:3].sum(), 1))

# ---------- t-SNE (perplexity 30), 3 colourings ----------
pre = PCA(30, random_state=SEED).fit_transform(Xs)
def tsne(perp):
    return TSNE(n_components=2, perplexity=perp, init="pca",
               learning_rate="auto", random_state=SEED).fit_transform(pre)
T = tsne(30)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for c, col in wc.items():
    m = yw == c
    axs[0].scatter(T[m, 0], T[m, 1], s=18, color=col, alpha=0.8, label=c)
axs[0].set_title("(a) waveform width"); axs[0].legend(fontsize=8, frameon=False)
for c, col in [("narrow", "#C44E52"), ("broad/wide", "#4C72B0"), ("medium", "#cccccc")]:
    m = nb == c
    axs[1].scatter(T[m, 0], T[m, 1], s=18, color=col, alpha=0.8, label=c)
axs[1].set_title("(b) narrow vs broad/wide"); axs[1].legend(fontsize=8, frameon=False)
sc = axs[2].scatter(T[:, 0], T[:, 1], s=18, c=mfr, cmap="viridis", alpha=0.85)
axs[2].set_title("(c) mean firing rate"); fig.colorbar(sc, ax=axs[2], label="mean spike count")
for a in axs:
    a.set_xlabel("t-SNE 1"); a.set_ylabel("t-SNE 2")
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle(f"t-SNE of readout vectors (perplexity 30, monkey N V1, n={n})")
fig.tight_layout(); fig.savefig(f"{OUT}/readout_tsne.png", dpi=200)
print("wrote readout_tsne.png")

# ---------- t-SNE perplexity sweep, coloured by width ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, perp in zip(axs, [10, 30, 50]):
    Tp = tsne(perp)
    for c, col in wc.items():
        m = yw == c
        ax.scatter(Tp[m, 0], Tp[m, 1], s=16, color=col, alpha=0.8, label=c)
    ax.set_title(f"perplexity {perp}"); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle(f"t-SNE perplexity sweep, coloured by waveform width (n={n})")
fig.tight_layout(); fig.savefig(f"{OUT}/readout_tsne_perplexity.png", dpi=200)
print("wrote readout_tsne_perplexity.png")
