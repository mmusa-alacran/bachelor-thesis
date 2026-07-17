"""
Dataset loader for TVSD-style .npz files.

Expected .npz contents:
  - images:    (N, C, H, W) or (N, H, W, C), uint8 or float
  - responses: (N, n_neurons), float32
  - is_test:   (N,) bool — official THINGS test images (100 images, ~28 reps each)

Split strategy:
  - Test: indices with is_test == True (100 images, 28 reps already averaged)
  - Train/Val: 90/10 of the remaining ~22,248 train images
  - Falls back to a random 80/10/10 split if `is_test` is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits_official(is_test: np.ndarray, seed: int = 0, frac_val: float = 0.1) -> Split:
    """Use the official THINGS train/test partition; split the train pool 90/10 for val."""
    test_idx = np.where(is_test)[0]
    train_pool = np.where(~is_test)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool)
    n_val = int(round(len(train_pool) * frac_val))
    val_idx = train_pool[:n_val]
    train_idx = train_pool[n_val:]

    return Split(train=train_idx, val=val_idx, test=test_idx)


def make_splits_random(n: int, seed: int = 0, frac_train: float = 0.8, frac_val: float = 0.1) -> Split:
    """Fallback: random split when is_test is not available."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return Split(train=train, val=val, test=test)


class NPZNeuralDataset(Dataset):
    """Keeps images as uint8 in RAM, converts and normalises per-sample."""
    def __init__(self, images: np.ndarray, responses: np.ndarray,
                 mean: np.ndarray, std: np.ndarray):
        # Keeping uint8 in RAM is ~4x smaller than float32 (matters for full TVSD).
        self.images = images
        self.responses = torch.from_numpy(responses.astype(np.float32))
        self.mean = torch.tensor(mean.reshape(3, 1, 1), dtype=torch.float32)
        self.std = torch.tensor(std.reshape(3, 1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.images[i].astype(np.float32)) / 255.0
        img = (img - self.mean) / self.std
        return img, self.responses[i]


def make_loaders(
    npz_path: str,
    batch_size: int = 32,
    seed: int = 0,
    num_workers: int = 0,
    frac_val: float = 0.1,
) -> Tuple[Dict[str, DataLoader], int]:
    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]
    responses = data["responses"]
    n = images.shape[0]
    n_neurons = responses.shape[1]

    # Move channels to first axis if the .npz stored them last (N, H, W, C → N, C, H, W).
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        images = np.transpose(images, (0, 3, 1, 2))

    if "is_test" in data:
        is_test = data["is_test"].astype(bool)
        splits = make_splits_official(is_test, seed=seed, frac_val=frac_val)
        n_test = is_test.sum()
        print(f"[dataset] Using official THINGS split: "
              f"{len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test "
              f"({n_test} test images with multi-rep averaging)")
    else:
        splits = make_splits_random(n, seed=seed)
        print(f"[dataset] WARNING: is_test not found, using random 80/10/10 split: "
              f"{len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test")

    # Per-channel mean/std from the training split, accumulated in chunks
    # to avoid materialising the full set as float64.
    train_indices = splits.train
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels_per_img = images.shape[2] * images.shape[3]
    total_pixels = len(train_indices) * n_pixels_per_img

    chunk_size = 1000
    for start_idx in range(0, len(train_indices), chunk_size):
        end_idx = min(start_idx + chunk_size, len(train_indices))
        chunk_idx = train_indices[start_idx:end_idx]

        chunk = images[chunk_idx].astype(np.float64) / 255.0
        channel_sum += chunk.sum(axis=(0, 2, 3))
        channel_sq_sum += (chunk ** 2).sum(axis=(0, 2, 3))

    mean = (channel_sum / total_pixels).astype(np.float32)
    var = (channel_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)

    ds_train = NPZNeuralDataset(images[splits.train], responses[splits.train], mean, std)
    ds_val = NPZNeuralDataset(images[splits.val], responses[splits.val], mean, std)
    ds_test = NPZNeuralDataset(images[splits.test], responses[splits.test], mean, std)
    del images, responses, data

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, n_neurons

