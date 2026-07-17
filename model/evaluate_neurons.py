#!/usr/bin/env python3
"""Per-neuron evaluation of a trained checkpoint on the val and test splits."""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/neuralpredictors_scaffold/src"))
from dataset import make_loaders
from models import ModelConfig, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--cut-layers", type=int, default=6)
    p.add_argument("--fine-tune", action="store_true")
    return p.parse_args()


def compute_per_neuron_correlation(predictions, targets):
    """Pearson correlation per column. Constant columns (no variance) report 0."""
    n_neurons = predictions.shape[1]
    correlations = np.zeros(n_neurons)
    for i in range(n_neurons):
        pred = predictions[:, i]
        targ = targets[:, i]
        if pred.std() < 1e-8 or targ.std() < 1e-8:
            correlations[i] = 0.0
            continue
        correlations[i] = np.corrcoef(pred, targ)[0, 1]
    return correlations


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    loaders, n_neurons = make_loaders(args.data, batch_size=args.batch_size, num_workers=0)

    print("Loading model...")
    cfg = ModelConfig(
        backbone=args.backbone,
        pretrained=args.pretrained,
        fine_tune=args.fine_tune,
        cut_layers=args.cut_layers,
    )
    model = build_model(in_shape=(3, 224, 224), outdims=n_neurons, cfg=cfg, device=device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for split_name in ["val", "test"]:
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name} set...")
        loader = loaders[split_name]

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, responses in loader:
                images = images.to(device)
                preds = model(images)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(responses.numpy())

        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        print(f"  Samples: {predictions.shape[0]}, Neurons: {predictions.shape[1]}")

        per_neuron_corr = compute_per_neuron_correlation(predictions, targets)

        print(f"\n  Per-neuron correlation ({split_name}):")
        print(f"    Mean:   {np.mean(per_neuron_corr):.4f}")
        print(f"    Median: {np.median(per_neuron_corr):.4f}")
        print(f"    Std:    {np.std(per_neuron_corr):.4f}")
        print(f"    Min:    {np.min(per_neuron_corr):.4f}")
        print(f"    Max:    {np.max(per_neuron_corr):.4f}")

        bins = [(-1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
        print(f"\n  Distribution:")
        for lo, hi in bins:
            count = np.sum((per_neuron_corr >= lo) & (per_neuron_corr < hi))
            bar = "█" * (count // 5)
            print(f"    [{lo:+.1f}, {hi:+.1f}): {count:4d} neurons  {bar}")

        sorted_idx = np.argsort(per_neuron_corr)[::-1]

        data = np.load(args.data, allow_pickle=True)
        unit_ids = data.get("unit_ids", None)

        print(f"\n  Top 20 best-predicted neurons:")
        print(f"    {'Rank':<6} {'Neuron':<8} {'Unit ID':<20} {'Corr':>8}")
        for rank, idx in enumerate(sorted_idx[:20], 1):
            uid = f"({unit_ids[idx][0]}, {unit_ids[idx][1]})" if unit_ids is not None else "?"
            print(f"    {rank:<6} {idx:<8} {uid:<20} {per_neuron_corr[idx]:>8.4f}")

        print(f"\n  Bottom 10 worst-predicted neurons:")
        for rank, idx in enumerate(sorted_idx[-10:], 1):
            uid = f"({unit_ids[idx][0]}, {unit_ids[idx][1]})" if unit_ids is not None else "?"
            print(f"    {rank:<6} {idx:<8} {uid:<20} {per_neuron_corr[idx]:>8.4f}")

        np.savez(
            os.path.join(args.out, f"neuron_analysis_{split_name}.npz"),
            per_neuron_corr=per_neuron_corr,
            predictions=predictions,
            targets=targets,
            sorted_neuron_idx=sorted_idx,
        )
        print(f"\n  Saved to {args.out}/neuron_analysis_{split_name}.npz")

    # Summary: test set is the 28-rep average and is what we report;
    # val set is single-trial and noise-limited.
    print(f"\n{'='*60}")
    print("KEY FINDINGS (test set, 28-rep average):")
    test_corr = np.load(os.path.join(args.out, "neuron_analysis_test.npz"))["per_neuron_corr"]
    val_corr  = np.load(os.path.join(args.out, "neuron_analysis_val.npz"))["per_neuron_corr"]
    n_neurons = len(test_corr)
    for thr in (0.3, 0.5, 0.7):
        n = np.sum(test_corr > thr)
        print(f"  Neurons with test corr > {thr}: {n}/{n_neurons} ({100*n/n_neurons:.1f}%)")
    print(f"  Best neuron (test): {test_corr.max():.4f}")
    print(f"  Mean val corr:      {val_corr.mean():.4f}")
    print(f"  Mean test corr:     {test_corr.mean():.4f}")


if __name__ == "__main__":
    main()
