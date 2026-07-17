#!/usr/bin/env python3
"""More readable visualisations of the waveform class decoding result."""
import os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

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
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------- Figure 1: cross-validated decoder score, narrow vs broad ----------
bm = yw != "medium"
yb = np.where(yw[bm] == "narrow", "narrow", "broad")
clf = make_pipeline(StandardScaler(), PCA(40, random_state=SEED),
                    LogisticRegression(max_iter=4000, class_weight="balanced"))
proba = cross_val_predict(clf, feats[bm], yb, cv=cv, method="predict_proba")
classes = clf.fit(feats[bm], yb).classes_
ni = list(classes).index("narrow")
score = proba[:, ni]               # out-of-fold P(narrow)
bacc = balanced_accuracy_score(yb, np.where(score > 0.5, "narrow", "broad"))
auc = roc_auc_score((yb == "narrow").astype(int), score)
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = np.linspace(0, 1, 200)
for c, col in [("broad", "#4C72B0"), ("narrow", "#C44E52")]:
    s = score[yb == c]
    ax.fill_between(xs, gaussian_kde(s)(xs), alpha=0.45, color=col,
                    label=f"{c} (n={ (yb==c).sum() })")
    ax.axvline(s.mean(), color=col, ls="--", lw=1)
ax.axvline(0.5, color="k", ls=":", lw=1)
ax.set_xlabel("cross-validated decoder score  P(narrow)")
ax.set_ylabel("density of neurons")
ax.set_title(f"Decoding narrow vs broad spiking from readout features\n"
             f"balanced acc = {bacc:.2f}, AUC = {auc:.2f}  (chance = 0.5)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/decoder_score_narrowbroad.png", dpi=200)
print("wrote decoder_score_narrowbroad.png  bacc=%.3f auc=%.3f" % (bacc, auc))

# ---------- Figure 2: firing rate by waveform width ----------
order = ["narrow", "medium", "wide"]; cols = ["#C44E52", "#999999", "#4C72B0"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
data = [mfr[yw == w] for w in order]
parts = ax.violinplot(data, showmedians=True, widths=0.8)
for b, c in zip(parts["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.5)
for w in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[w].set_color("k")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"{w}\n(n={(yw==w).sum()})" for w in order])
ax.set_ylabel("mean evoked spike count (40-100 ms)")
ax.set_title("Evoked firing rate by waveform width\n"
             "non-monotonic: medium-width cells highest in this evoked window")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
meds = [np.median(x) for x in data]
print("median FR by width:", dict(zip(order, [round(m, 2) for m in meds])))
fig.tight_layout(); fig.savefig(f"{OUT}/fr_by_width.png", dpi=200)
print("wrote fr_by_width.png")

# ---------- Figure 3: normalized confusion matrix, 6-way ----------
clf6 = make_pipeline(StandardScaler(), PCA(40, random_state=SEED),
                     LogisticRegression(max_iter=4000, class_weight="balanced"))
yp = cross_val_predict(clf6, feats, yc, cv=cv)
labs = ["DOWN_wide", "DOWN_medium_shallow", "DOWN_medium_sharp",
        "DOWN_narrow_shallow", "DOWN_narrow_sharp", "UP"]
cm = confusion_matrix(yc, yp, labels=labs).astype(float)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
short = [l.replace("DOWN_", "") for l in labs]
ax.set_xticks(range(6)); ax.set_xticklabels(short, rotation=40, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(short)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("6-way subtype confusion (row-normalised)\n"
             "off-diagonal spread = weak fine-subtype decoding")
fig.colorbar(im, label="fraction of true class")
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_6way.png", dpi=200)
print("wrote confusion_6way.png")
