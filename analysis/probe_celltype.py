#!/usr/bin/env python3
"""
Decoding probe: can the model's learned per-neuron representation predict the
waveform-defined class, and does it carry information beyond firing rate?

For each V1 neuron the model was trained on we take its readout feature-weight
vector (1152-d), its learned RF position and bias, and ask a small classifier
to recover the neuron's subtype. We compare against: a chance baseline, a
shuffled-label control, a firing-rate-only baseline, and a positive control
that classifies from the original waveform scalars.

Reads ~/waveform_labels_monkeyN.csv (from build_neuron_labels.py),
the final checkpoint, and the training .npz. Writes ~/celltype_probe/.
"""
import os
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, classification_report)

CKPT = os.path.expanduser("~/monkeyN_V1_run10c_ks4_filtered015/best.pt")
DATA = os.path.expanduser("~/tvsd_monkeyN_V1_sua_KS4_filtered015.npz")
LABELS = os.path.expanduser("~/waveform_labels_monkeyN.csv")
OUT = os.path.expanduser("~/celltype_probe")
os.makedirs(OUT, exist_ok=True)
SEED = 0
rng = np.random.default_rng(SEED)


def load_model_features():
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck["model"]
    feats = sd["readout.features"].numpy()[0, :, 0, :].T   # (n_neurons, 1152)
    grid = sd["readout.grid"].numpy()[0, :, 0, :]          # (n_neurons, 2)
    bias = sd["readout.bias"].numpy()                      # (n_neurons,)
    d = np.load(DATA, allow_pickle=True)
    uid = d["unit_ids"]                                    # (n_neurons, 2)
    mean_fr = d["responses"].mean(0)                       # per-neuron mean spike count
    return feats, grid, bias, uid, mean_fr


def evaluate(name, X, y, clf, cv, lines, n_perm=20):
    yp = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, yp)
    bacc = balanced_accuracy_score(y, yp)
    f1 = f1_score(y, yp, average="macro")
    # shuffled-label control
    perm_bacc = []
    for _ in range(n_perm):
        ys = rng.permutation(y)
        perm_bacc.append(balanced_accuracy_score(
            ys, cross_val_predict(clf, X, ys, cv=cv)))
    pb = np.mean(perm_bacc)
    lines.append(f"  {name:28s} acc={acc:.3f}  balanced_acc={bacc:.3f}  "
                 f"macroF1={f1:.3f}   (shuffled balanced_acc={pb:.3f})")
    return yp, bacc


def run_target(tag, y, feats, grid, bias, mean_fr, lines):
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"TARGET: {tag}   (n={len(y)} neurons, {len(set(y))} classes)")
    counts = pd.Series(y).value_counts()
    lines.append("  class counts: " + ", ".join(f"{k}:{v}" for k, v in counts.items()))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # chance baselines
    for strat in ("most_frequent", "stratified"):
        dum = DummyClassifier(strategy=strat, random_state=SEED)
        yp = cross_val_predict(dum, np.zeros((len(y), 1)), y, cv=cv)
        lines.append(f"  chance ({strat:13s})  acc={accuracy_score(y, yp):.3f}  "
                     f"balanced_acc={balanced_accuracy_score(y, yp):.3f}")

    # firing-rate-only baseline (confound check)
    Xfr = mean_fr.reshape(-1, 1)
    clf_fr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    evaluate("firing-rate only", Xfr, y, clf_fr, cv, lines)

    # readout position/extent only
    clf_rf = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    evaluate("readout pos/extent only", grid, y, clf_rf, cv, lines)

    # readout features: linear probe (PCA + logistic)
    clf_lin = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=SEED),
                            LogisticRegression(max_iter=4000, class_weight="balanced"))
    yp_lin, _ = evaluate("readout feats (linear)", feats, y, clf_lin, cv, lines)

    # readout features: small MLP
    clf_mlp = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=SEED),
                            MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                          alpha=1e-2, random_state=SEED))
    evaluate("readout feats (small MLP)", feats, y, clf_mlp, cv, lines)

    # all four: readout weights + spatial params + bias + firing rate
    Xall = np.hstack([feats, grid, bias.reshape(-1, 1), mean_fr.reshape(-1, 1)])
    clf_all = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=SEED),
                            LogisticRegression(max_iter=4000, class_weight="balanced"))
    evaluate("all four (weights+spatial+bias+FR)", Xall, y, clf_all, cv, lines)

    # confusion matrix for the linear readout probe
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, yp_lin, labels=labels_sorted)
    lines.append("  confusion matrix (readout linear), rows=true, cols=pred:")
    lines.append("    " + " ".join(f"{c[:10]:>10s}" for c in labels_sorted))
    for r, lab in zip(cm, labels_sorted):
        lines.append(f"    {lab[:18]:18s} " + " ".join(f"{v:10d}" for v in r))
    return yp_lin, labels_sorted


def main():
    feats, grid, bias, uid, mean_fr = load_model_features()
    lab = pd.read_csv(LABELS)
    key2row = {(int(r.electrode_id), int(r.unit_index)): r for r in lab.itertuples()}

    final_class, area, width, ampw, widw, keep = [], [], [], [], [], []
    for i in range(len(uid)):
        k = (int(uid[i, 0]), int(uid[i, 1]))
        r = key2row.get(k)
        ok = r is not None and r.area == "V1" and isinstance(r.final_class, str)
        keep.append(ok)
        final_class.append(r.final_class if ok else None)
        width.append(r.width_wf_class if ok else None)
        ampw.append(r.amp_wf_mean if ok else np.nan)
        widw.append(r.width_wf_mean if ok else np.nan)
    keep = np.array(keep)

    lines = []
    lines.append("Waveform-class decodability from the functional model (run10c 0.15-cut, monkey N V1)")
    lines.append(f"model neurons: {len(uid)} | with V1 waveform label: {keep.sum()}")

    f = feats[keep]; g = grid[keep]; b = bias[keep]; mfr = mean_fr[keep]
    yc = np.array([final_class[i] for i in range(len(uid)) if keep[i]])
    yw = np.array([width[i] for i in range(len(uid)) if keep[i]])
    amp = np.array([ampw[i] for i in range(len(uid)) if keep[i]])
    wid = np.array([widw[i] for i in range(len(uid)) if keep[i]])

    # Target 1: full 6-class final_class
    run_target("final_class (6-way waveform subtype)", yc, f, g, b, mfr, lines)

    # Target 2: width class (narrow / medium / wide)
    run_target("width_wf_class (narrow/medium/wide)", yw, f, g, b, mfr, lines)

    # Target 3: binary narrow-spiking vs broad-spiking (drop medium)
    bmask = yw != "medium"
    ybin = np.where(yw[bmask] == "narrow", "narrow_spiking", "broad_spiking")
    run_target("narrow vs broad spiking (binary)", ybin,
               f[bmask], g[bmask], b[bmask], mfr[bmask], lines)

    # Positive control: the original waveform scalars -> final_class
    lines.append("")
    lines.append("=" * 78)
    lines.append("POSITIVE CONTROL: predict final_class from the original waveform scalars "
                 "[amp_wf, width_wf]")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xw = np.column_stack([amp, wid])
    ok = ~np.isnan(Xw).any(1)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=4000, class_weight="balanced"))
    yp = cross_val_predict(clf, Xw[ok], yc[ok], cv=cv)
    lines.append(f"  acc={accuracy_score(yc[ok], yp):.3f}  "
                 f"balanced_acc={balanced_accuracy_score(yc[ok], yp):.3f}")

    report = "\n".join(lines)
    with open(os.path.join(OUT, "probe_results.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\n[done] -> {OUT}/probe_results.txt")


if __name__ == "__main__":
    main()
