#!/usr/bin/env python3
"""
Data-quality analysis on a preprocessed .npz (requires
`preprocess_tvsd.py --save-trials`).

A. Per-neuron split-half reliability (Spearman-Brown corrected).
   Repeats of each test image are randomly split into two halves; each half
   is averaged to give two per-image estimates which are then correlated
   across images. K splits are averaged and r_full = 2r / (1 + r) corrects
   from half- to full-N.

B. Cross-session pairwise correlation.
   For each session pair, the per-neuron mean response is computed on the
   test images shared between the two sessions, and the two
   n_neurons-dimensional vectors are correlated. High values indicate that
   the same unit ID corresponds to the same biological neuron across sessions.

Usage:
    python analyze_neuron_reliability.py \
        --data ~/tvsd_monkeyF_V1_KS4_trials.npz \
        --out  ~/monkeyF_reliability/

Outputs:
    reliability.npz           per-neuron Spearman-Brown reliability
    reliability_summary.txt   text summary, fractions above thresholds
    cross_session_corr.npz    (n_sessions, n_sessions) correlation matrix
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Preprocessed .npz (must include test_trial_responses, "
                        "test_trial_image_idx, test_trial_session_idx).")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--n-splits", type=int, default=200,
                   help="Number of random split-half iterations (default: 200)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def per_neuron_reliability(
    trial_resp: np.ndarray,
    trial_img:  np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    reliability_full   : (n_neurons,) — Spearman-Brown corrected full-N reliability
    reliability_split  : (n_neurons,) — raw split-half correlation (mean over splits)
    """
    n_trials, n_neurons = trial_resp.shape
    # group trial indices by image
    trials_of_image: dict[int, list[int]] = defaultdict(list)
    for t, im in enumerate(trial_img):
        trials_of_image[int(im)].append(t)

    images = sorted(trials_of_image.keys())
    n_images = len(images)
    # We need at least 2 trials per image to do any split-half. Drop the rest.
    images = [im for im in images if len(trials_of_image[im]) >= 2]
    print(f"  [reliability] {n_images} test images total, "
          f"{len(images)} usable (>= 2 trials)")

    corrs_per_split = np.zeros((n_splits, n_neurons), dtype=np.float64)

    for s in range(n_splits):
        # Build two per-image averages from disjoint random halves of the trials
        a = np.full((len(images), n_neurons), np.nan, dtype=np.float64)
        b = np.full((len(images), n_neurons), np.nan, dtype=np.float64)
        for i, im in enumerate(images):
            t_idx = trials_of_image[im]
            rng.shuffle(t_idx)               # in-place
            half = len(t_idx) // 2
            if half == 0:                    # 1 trial → can't split
                continue
            a[i] = trial_resp[t_idx[:half]].mean(axis=0)
            b[i] = trial_resp[t_idx[half:2*half]].mean(axis=0)
        # Pearson per neuron over images
        mask = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
        aa = a[mask]; bb = b[mask]
        aa -= aa.mean(0, keepdims=True)
        bb -= bb.mean(0, keepdims=True)
        num = (aa * bb).sum(0)
        den = np.sqrt((aa ** 2).sum(0) * (bb ** 2).sum(0)) + 1e-12
        corrs_per_split[s] = num / den

    r_split = corrs_per_split.mean(axis=0)
    # Spearman-Brown: r_full = 2*r_half / (1 + r_half). Clip near -1 to avoid blow-up.
    r_clipped = np.clip(r_split, -0.999, 0.999)
    r_full = (2.0 * r_clipped) / (1.0 + r_clipped)
    return r_full, r_split


def cross_session_correlation(
    trial_resp: np.ndarray,
    trial_img:  np.ndarray,
    trial_ses:  np.ndarray,
    sessions:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every pair of sessions (i, j), compute the Pearson correlation between
    the per-neuron mean-response vectors restricted to test images that appear
    in BOTH sessions.  Returns the matrix and the per-pair shared-image count.
    """
    session_ids = np.arange(len(sessions))
    n_sessions  = len(session_ids)
    corr_mat    = np.full((n_sessions, n_sessions), np.nan)
    nshared_mat = np.zeros((n_sessions, n_sessions), dtype=np.int32)

    # Mean response per (image, session)
    ses_img_mean: dict[tuple[int, int], np.ndarray] = {}
    for ses_id in session_ids:
        mask_s = (trial_ses == ses_id)
        if not mask_s.any():
            continue
        imgs_in_ses = np.unique(trial_img[mask_s])
        for im in imgs_in_ses:
            m = mask_s & (trial_img == im)
            ses_img_mean[(int(ses_id), int(im))] = trial_resp[m].mean(0)

    for i in session_ids:
        for j in session_ids:
            if j < i:
                continue
            shared = [im for (sid, im), _ in ses_img_mean.items() if sid == i
                      and (j, im) in ses_img_mean]
            if len(shared) < 5:              # not enough shared images
                continue
            A = np.stack([ses_img_mean[(i, im)] for im in shared]).mean(0)
            B = np.stack([ses_img_mean[(j, im)] for im in shared]).mean(0)
            A = A - A.mean()
            B = B - B.mean()
            denom = np.sqrt((A * A).sum() * (B * B).sum()) + 1e-12
            corr = (A * B).sum() / denom
            corr_mat[i, j] = corr
            corr_mat[j, i] = corr
            nshared_mat[i, j] = len(shared)
            nshared_mat[j, i] = len(shared)

    return corr_mat, nshared_mat


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[info] loading {args.data}")
    d = np.load(args.data, allow_pickle=True)
    for required in ("test_trial_responses", "test_trial_image_idx", "test_trial_session_idx"):
        if required not in d.files:
            raise SystemExit(f"ERROR: '{required}' missing from .npz. "
                             "Re-run preprocess_tvsd.py with --save-trials.")
    trial_resp = d["test_trial_responses"]
    trial_img  = d["test_trial_image_idx"]
    trial_ses  = d["test_trial_session_idx"]
    sessions   = d["sessions"] if "sessions" in d.files else np.array([])
    unit_ids   = d["unit_ids"] if "unit_ids" in d.files else None

    n_trials, n_neurons = trial_resp.shape
    print(f"[info] {n_trials} test trials, {n_neurons} neurons, "
          f"{len(set(trial_img.tolist()))} unique images, "
          f"{len(set(trial_ses.tolist()))} sessions present")

    rng = np.random.default_rng(args.seed)

    print("\n[A] per-neuron reliability (split-half + Spearman-Brown)")
    r_full, r_split = per_neuron_reliability(
        trial_resp, trial_img, n_splits=args.n_splits, rng=rng
    )

    np.savez(os.path.join(args.out, "reliability.npz"),
             r_full=r_full, r_split=r_split, unit_ids=unit_ids)

    pct = lambda x: 100.0 * x / n_neurons
    thresholds = (0.0, 0.15, 0.3, 0.5, 0.7)
    summary_lines = [
        "Per-neuron reliability (Spearman-Brown corrected split-half)",
        "=" * 60,
        f"n_neurons:      {n_neurons}",
        f"n_test_trials:  {n_trials}",
        f"n_test_images:  {len(set(trial_img.tolist()))}",
        f"n_sessions:     {len(set(trial_ses.tolist()))}",
        "",
        "Distribution of reliability:",
        f"  mean     : {np.nanmean(r_full):+.3f}",
        f"  median   : {np.nanmedian(r_full):+.3f}",
        f"  std      : {np.nanstd(r_full):+.3f}",
        f"  min/max  : {np.nanmin(r_full):+.3f} / {np.nanmax(r_full):+.3f}",
        "",
        "Fraction of neurons above threshold:",
    ]
    for t in thresholds:
        n = int(np.nansum(r_full > t))
        summary_lines.append(f"  r > {t:>4.2f}:  {n:5d} / {n_neurons}  ({pct(n):.1f}%)")

    print("\n[B] cross-session consistency")
    corr_mat, nshared = cross_session_correlation(
        trial_resp, trial_img, trial_ses,
        sessions if sessions.size else np.arange(int(trial_ses.max()) + 1)
    )
    off_diag = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    np.savez(os.path.join(args.out, "cross_session_corr.npz"),
             corr_mat=corr_mat, nshared=nshared,
             sessions=sessions)

    summary_lines += [
        "",
        "Cross-session correlation (off-diagonal, full population response vector):",
        f"  mean    : {off_diag.mean():+.3f}" if off_diag.size else "  (no shared images)",
        f"  median  : {np.median(off_diag):+.3f}" if off_diag.size else "",
        f"  min/max : {off_diag.min():+.3f} / {off_diag.max():+.3f}" if off_diag.size else "",
        f"  n_session_pairs (>= 5 shared images): {off_diag.size}",
    ]

    summary = "\n".join(summary_lines)
    with open(os.path.join(args.out, "reliability_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\n[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
