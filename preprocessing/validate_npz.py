"""
Sanity checks on the preprocessed .npz file.
Run after preprocessing finishes and you've copied the .npz locally.

Usage:
    python validate_npz.py data_samples/tvsd_monkeyF_V1.npz
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data_samples/tvsd_monkeyF_V1.npz"
data = np.load(path, allow_pickle=True)

print("=" * 60)
print("VALIDATION REPORT")
print("=" * 60)

images = data["images"]
responses = data["responses"]
is_test = data["is_test"] if "is_test" in data else None
keys = data["keys"] if "keys" in data else None

print(f"\n1. SHAPES")
print(f"   images:    {images.shape} ({images.dtype})")
print(f"   responses: {responses.shape} ({responses.dtype})")

# Check 1: Expected number of images
n_images = images.shape[0]
n_neurons = responses.shape[1]
print(f"\n2. IMAGE COUNT")
print(f"   Total images: {n_images}")
if is_test is not None:
    n_train = (~is_test).sum()
    n_test = is_test.sum()
    print(f"   Train: {n_train}, Test: {n_test}")
    if n_train < 15000:
        print(f"   WARNING: Expected ~22,248 train images, got {n_train}")
        print(f"      This could mean some sessions/arrays had missing files.")
    else:
        print(f"   OK: Train count looks reasonable (expect ~22,248)")
    if n_test < 90:
        print(f"   WARNING: Expected ~100 test images, got {n_test}")
    else:
        print(f"   OK: Test count looks good")

# Check 2: Response distributions
print(f"\n3. RESPONSE STATS (spike counts)")
print(f"   Overall: mean={responses.mean():.2f}, std={responses.std():.2f}, "
      f"min={responses.min():.1f}, max={responses.max():.1f}")
print(f"   Zero fraction: {(responses == 0).mean():.3f}")
mean_per_neuron = responses.mean(axis=0)
active_neurons = (mean_per_neuron > 0).sum()
print(f"   Active electrodes (mean > 0): {active_neurons}/{n_neurons}")
if active_neurons < n_neurons * 0.5:
    print(f"   WARNING: Less than 50% of electrodes are active!")
    print(f"      Some arrays may not have been spike-sorted.")
else:
    print(f"   OK: Majority of electrodes active")

# Check 3: Firing rates in reasonable range
window_duration = 0.200  # 50-250ms = 200ms
rates = responses / window_duration  # convert to spikes/s
print(f"\n4. FIRING RATES (spikes/s, assuming 200ms window)")
print(f"   Mean per neuron: min={rates.mean(axis=0).min():.1f}, "
      f"max={rates.mean(axis=0).max():.1f}, median={np.median(rates.mean(axis=0)):.1f}")
if rates.mean(axis=0).max() > 200:
    print(f"   WARNING: Some neurons have very high rates (>200 Hz)")
    print(f"      Could be multi-unit contamination or spike sorting artifact")
elif rates.mean(axis=0).max() < 1:
    print(f"   WARNING: Max firing rate < 1 Hz — responses seem very low")
else:
    print(f"   OK: Firing rates in typical range")

# Check 4: Images not all zeros
print(f"\n5. IMAGE VALIDITY")
zero_images = (images.reshape(n_images, -1).sum(axis=1) == 0).sum()
if zero_images > 0:
    print(f"   WARNING: {zero_images} images are all zeros (missing files?)")
else:
    print(f"   OK: No blank images")
print(f"   Image range: [{images.min():.2f}, {images.max():.2f}]")
print(f"   (Expected ~[-2.1, 2.6] for ImageNet-normalized)")

# Check 5: No NaN/Inf
print(f"\n6. DATA INTEGRITY")
img_nan = np.isnan(images).any() or np.isinf(images).any()
resp_nan = np.isnan(responses).any() or np.isinf(responses).any()
print(f"   Images NaN/Inf: {'YES' if img_nan else 'None'}")
print(f"   Responses NaN/Inf: {'YES' if resp_nan else 'None'}")

# Check 6: Response variance (dead neurons)
print(f"\n7. NEURON QUALITY")
var_per_neuron = responses.var(axis=0)
zero_var = (var_per_neuron == 0).sum()
low_var = (var_per_neuron < 0.01).sum()
print(f"   Zero-variance neurons: {zero_var}/{n_neurons}")
print(f"   Near-zero variance (<0.01): {low_var}/{n_neurons}")
if zero_var > n_neurons * 0.3:
    print(f"   WARNING: Many dead neurons — consider filtering")

print(f"\n{'=' * 60}")
print(f"Done. Review warnings above before training.")
print(f"{'=' * 60}")
