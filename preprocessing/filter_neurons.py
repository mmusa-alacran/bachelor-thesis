"""
Filter a preprocessed .npz to neurons whose test-set correlation
(from evaluate_neurons.py) is above a threshold.

Usage:
    python filter_neurons.py \
        --data      ~/tvsd_monkeyN_V1_sua.npz \
        --analysis  ~/monkeyN_V1_run7_fr_sorted/neuron_analysis/neuron_analysis_test.npz \
        --out       ~/tvsd_monkeyN_V1_sua_filtered03.npz \
        --threshold 0.3

Output keys mirror the input; `responses` (and `unit_ids` if present) are
restricted to the kept neurons. Everything else is copied verbatim.
"""

import argparse
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter .npz dataset to neurons above a test-correlation threshold."
    )
    p.add_argument("--data", required=True,
                   help="Path to preprocessed .npz (e.g. tvsd_monkeyN_V1_sua.npz)")
    p.add_argument("--analysis", required=True,
                   help="Path to neuron_analysis_test.npz from evaluate_neurons.py")
    p.add_argument("--out", required=True,
                   help="Output path for the filtered .npz")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Keep neurons with test corr >= this value (default: 0.3)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if os.path.exists(args.out):
        print(f"[warn] output file already exists and will be overwritten: {args.out}")

    print(f"[info] loading data from {args.data}")
    data = np.load(args.data, allow_pickle=True)

    print(f"[info] loading neuron analysis from {args.analysis}")
    analysis = np.load(args.analysis, allow_pickle=True)

    images    = data["images"]
    responses = data["responses"]
    per_neuron_corr = analysis["per_neuron_corr"]

    n_images, n_neurons = responses.shape
    assert len(per_neuron_corr) == n_neurons, (
        f"Mismatch: responses has {n_neurons} neurons but analysis has "
        f"{len(per_neuron_corr)} correlations. Wrong analysis file?"
    )

    keep_mask = per_neuron_corr >= args.threshold
    keep_idx  = np.where(keep_mask)[0]
    n_keep    = len(keep_idx)

    print(f"\n[info] threshold = {args.threshold:.2f}")
    print(f"[info] neurons kept:    {n_keep} / {n_neurons} "
          f"({100.0 * n_keep / n_neurons:.1f}%)")
    print(f"[info] neurons removed: {n_neurons - n_keep}")

    if n_keep == 0:
        print("[error] No neurons pass the threshold. Aborting.")
        sys.exit(1)

    corr_kept = per_neuron_corr[keep_idx]
    print(f"[info] kept corr: min={corr_kept.min():.3f}, "
          f"mean={corr_kept.mean():.3f}, max={corr_kept.max():.3f}")

    out_dict: dict = {
        "images":    images,
        "responses": responses[:, keep_idx],
    }

    for key in ("is_test", "keys"):
        if key in data:
            out_dict[key] = data[key]

    if "unit_ids" in data:
        unit_ids = data["unit_ids"]
        if unit_ids.shape[0] == n_neurons:
            out_dict["unit_ids"] = unit_ids[keep_idx]
        else:
            print(f"[warn] unit_ids shape {unit_ids.shape} does not match n_neurons={n_neurons}; "
                  "copying verbatim without filtering")
            out_dict["unit_ids"] = unit_ids

    # Keep the kept-neuron correlations and their original column indices alongside the data
    # so the filtering decision can be reproduced or undone downstream.
    out_dict["kept_neuron_corr"]     = corr_kept
    out_dict["kept_neuron_orig_idx"] = keep_idx

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    print(f"\n[done] saved filtered dataset → {out_path}")
    print(f"       responses shape: {out_dict['responses'].shape}")


if __name__ == "__main__":
    main()
