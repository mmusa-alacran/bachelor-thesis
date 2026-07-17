"""
Filter the preprocessed .npz to remove images that weren't found (all zeros).
Creates a new clean .npz with only valid image-response pairs.

Usage:
    python filter_npz.py data_samples/tvsd_monkeyF_V1.npz data_samples/tvsd_monkeyF_V1_clean.npz
"""
import sys
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "data_samples/tvsd_monkeyF_V1.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".npz", "_clean.npz")

print(f"Loading {src}...")
data = np.load(src, allow_pickle=True)
images = data["images"]
responses = data["responses"]
keys = data["keys"] if "keys" in data else None
is_test = data["is_test"] if "is_test" in data else None

# Find non-blank images (sum of pixel values > 0)
pixel_sums = images.reshape(images.shape[0], -1).sum(axis=1)
valid = pixel_sums > 0

n_total = len(images)
n_valid = valid.sum()
n_blank = n_total - n_valid
print(f"Total: {n_total}, Valid: {n_valid}, Blank: {n_blank}")

# Filter
images_clean = images[valid]
responses_clean = responses[valid]

save_dict = {"images": images_clean, "responses": responses_clean}
if keys is not None:
    save_dict["keys"] = keys[valid]
if is_test is not None:
    save_dict["is_test"] = is_test[valid]
if "electrode_ids" in data:
    save_dict["electrode_ids"] = data["electrode_ids"]

np.savez(dst, **save_dict)

import os
mb = os.path.getsize(dst) / (1024 * 1024)
print(f"\nSaved {dst} ({mb:.0f} MB)")
print(f"   images:    {images_clean.shape} ({images_clean.dtype})")
print(f"   responses: {responses_clean.shape} ({responses_clean.dtype})")
if is_test is not None:
    print(f"   Train: {(~is_test[valid]).sum()}, Test: {is_test[valid].sum()}")
