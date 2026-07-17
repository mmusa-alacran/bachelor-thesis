#!/usr/bin/env python3
"""
Waveform-class decoding from the readout parameters, with error bars.

Extends probe_celltype.py, which supplies the position and shuffled-label
baselines, by adding:
  - repeated stratified cross-validation, giving a standard deviation across
    repeats rather than a single point estimate
  - balanced accuracy over stratified folds, so the class prior cannot inflate
    the score: a majority-only classifier scores exactly 1/n_classes
  - an equalized-subsample control, which fixes the class prior to uniform by
    subsampling every class to the size of the smallest
  - a PCA scatter of the readout vectors, coloured by subtype, width and
    firing rate
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

CKPT = os.path.expanduser("~/monkeyN_V1_run10c_ks4_filtered015/best.pt")
DATA = os.path.expanduser("~/tvsd_monkeyN_V1_sua_KS4_filtered015.npz")
LABELS = os.path.expanduser("~/waveform_labels_monkeyN.csv")
OUT = os.path.expanduser("~/celltype_probe")
os.makedirs(OUT, exist_ok=True)
SEED = 0
N_SPLITS, N_REPEATS = 5, 10
rng = np.random.default_rng(SEED)


def load():
    sd = torch.load(CKPT, map_location="cpu")["model"]
    feats = sd["readout.features"].numpy()[0, :, 0, :].T
    grid = sd["readout.grid"].numpy()[0, :, 0, :]
    bias = sd["readout.bias"].numpy()
    d = np.load(DATA, allow_pickle=True)
    uid = d["unit_ids"]
    mfr = d["responses"].mean(0)
    lab = pd.read_csv(LABELS)
    key2 = {(int(r.electrode_id), int(r.unit_index)): r for r in lab.itertuples()}
    keep, yc, yw = [], [], []
    for i in range(len(uid)):
        r = key2.get((int(uid[i, 0]), int(uid[i, 1])))
        ok = r is not None and r.area == "V1" and isinstance(r.final_class, str)
        keep.append(ok)
        yc.append(r.final_class if ok else None)
        yw.append(r.width_wf_class if ok else None)
    keep = np.array(keep)
    yc = np.array([yc[i] for i in range(len(uid)) if keep[i]])
    yw = np.array([yw[i] for i in range(len(uid)) if keep[i]])
    return (feats[keep], grid[keep], bias[keep], mfr[keep], yc, yw)


def pca40_logreg():
    return make_pipeline(StandardScaler(), PCA(40, random_state=SEED),
                         LogisticRegression(max_iter=4000, class_weight="balanced"))


def pca40_mlp():
    return make_pipeline(StandardScaler(), PCA(40, random_state=SEED),
                         MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                       alpha=1e-2, random_state=SEED))


def scaler_logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=4000, class_weight="balanced"))


def repeated_cv(X, y, clf):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    s = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    return s.mean(), s.std(), s


def equalized_subsample(X, y, clf, n_iter=50):
    """Subsample every class to the minority count, run 5-fold CV, repeat."""
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    accs = []
    for it in range(n_iter):
        r = np.random.default_rng(it)
        idx = np.concatenate([r.choice(np.where(y == c)[0], n_min, replace=False)
                              for c in classes])
        cv = StratifiedKFold(5, shuffle=True, random_state=it)
        s = cross_val_score(clf, X[idx], y[idx], cv=cv, scoring="balanced_accuracy")
        accs.append(s.mean())
    return float(np.mean(accs)), float(np.std(accs)), n_min


def main():
    feats, grid, bias, mfr, yc, yw = load()
    n = len(yc)
    print(f"V1 neurons: {n}")

    # binary narrow vs broad (drop medium)
    bmask = yw != "medium"
    ybin = np.where(yw[bmask] == "narrow", "narrow", "broad")

    targets = {
        "final_class (6-way)": (yc, np.ones(n, bool), 1/6),
        "width (3-way)":       (yw, np.ones(n, bool), 1/3),
        "narrow vs broad":     (ybin, bmask, 1/2),
    }
    Xall = np.hstack([feats, grid, bias.reshape(-1, 1), mfr.reshape(-1, 1)])
    methods = {
        "firing rate":            (mfr.reshape(-1, 1), scaler_logreg),
        "readout pos/extent":     (grid, scaler_logreg),
        "readout feats (lin)":    (feats, pca40_logreg),
        "readout feats (MLP)":    (feats, pca40_mlp),
        "all four (weights+spatial+bias+FR)":               (Xall, pca40_logreg),
    }

    results = {}
    lines = ["Decoding with repeated stratified CV (5-fold x 10 repeats, balanced accuracy)",
             f"V1 neurons: {n}", ""]
    for tname, (y, mask, chance) in targets.items():
        results[tname] = {"chance": chance}
        lines.append("=" * 70)
        lines.append(f"{tname}   chance={chance:.3f}   "
                     "classes: " + ", ".join(f"{k}:{v}" for k, v in
                     pd.Series(y).value_counts().items()))
        for mname, (X, clff) in methods.items():
            mu, sd, _ = repeated_cv(X[mask], y, clff())
            results[tname][mname] = (mu, sd)
            lines.append(f"  {mname:22s} {mu:.3f} +/- {sd:.3f}")
        # equalized-subsample control on readout feats (MLP)
        emu, esd, nmin = equalized_subsample(feats[mask], y, pca40_mlp())
        results[tname]["equalized feats (MLP)"] = (emu, esd)
        lines.append(f"  equalized-sample feats(MLP) {emu:.3f} +/- {esd:.3f}  "
                     f"(n/class={nmin})")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT, "probe_results_v2.txt"), "w") as f:
        f.write(report + "\n")

    # ---- figure: bars with error bars ----
    plot_methods = ["firing rate", "readout pos/extent", "readout feats (lin)",
                    "readout feats (MLP)", "all four (weights+spatial+bias+FR)"]
    colors = ["#DD8452", "#55A868", "#4C72B0", "#64B5CD", "#8172B3"]
    tnames = list(targets.keys())
    x = np.arange(len(tnames)); w = 0.15
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, m in enumerate(plot_methods):
        mus = [results[t][m][0] for t in tnames]
        sds = [results[t][m][1] for t in tnames]
        ax.bar(x + (i - 2) * w, mus, w, yerr=sds, capsize=3, label=m, color=colors[i])
    for j, t in enumerate(tnames):
        ax.hlines(results[t]["chance"], x[j] - 2.6*w, x[j] + 2.6*w,
                  color="k", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(tnames)
    ax.set_ylabel("Balanced accuracy (5x10 repeated CV)")
    ax.set_title("Waveform-class decodability from the functional model "
                 f"(monkey N V1, n={n})\nerror bars = std over repeated CV; "
                 "dotted = chance")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "celltype_decodability_errorbars.png"), dpi=200)
    print("wrote celltype_decodability_errorbars.png")

    # ---- PCA visualization ----
    # L2-normalise per neuron first (removes the overall gain, which tracks
    # firing rate) so that only tuning-shape structure is left; this matches
    # pca_readout.py and the 3D-PCA / t-SNE scripts.
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    Z = PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(fn))
    P = Z.transform(StandardScaler().fit_transform(fn))
    ev = Z.explained_variance_ratio_ * 100
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) 6-way subtype
    order = ["DOWN_wide", "DOWN_medium_shallow", "DOWN_medium_sharp",
             "DOWN_narrow_shallow", "DOWN_narrow_sharp", "UP"]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(order)))
    for c, col in zip(order, cmap):
        m = yc == c
        axs[0].scatter(P[m, 0], P[m, 1], s=18, color=col, label=c, alpha=0.8)
    axs[0].set_title("(a) waveform subtype")
    axs[0].legend(fontsize=6, frameon=False)
    # (b) width narrow/medium/wide
    wc = {"narrow": "#C44E52", "medium": "#888888", "wide": "#4C72B0"}
    for c, col in wc.items():
        m = yw == c
        axs[1].scatter(P[m, 0], P[m, 1], s=18, color=col, label=c, alpha=0.8)
    axs[1].set_title("(b) waveform width")
    axs[1].legend(fontsize=8, frameon=False)
    # (c) firing rate continuous
    sc = axs[2].scatter(P[:, 0], P[:, 1], s=18, c=mfr, cmap="viridis", alpha=0.85)
    axs[2].set_title("(c) mean firing rate")
    fig.colorbar(sc, ax=axs[2], label="mean spike count")
    for a in axs:
        a.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); a.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle(f"PCA of readout feature vectors, coloured by category "
                 f"(monkey N V1, n={n})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "readout_pca.png"), dpi=200)
    print("wrote readout_pca.png")


if __name__ == "__main__":
    main()
