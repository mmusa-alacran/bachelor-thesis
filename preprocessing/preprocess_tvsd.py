#!/usr/bin/env python3
"""
Preprocess TVSD spike data into an .npz consumable by the training scaffold.

Defaults to Monkey N, V1 arrays 1-8, all sessions across 4 recording days.
Uses SUA: each sorted unit is one neuron, identified by (Electrode_ID,
unit_index). Within each electrode, units are sorted by the firing_rate
annotation (highest first) so the per-electrode unit_index is stable across
sessions.

Spike window: 40-100 ms post-stimulus (feedforward V1 response).
Trial and spike times are already in a common reference frame.

Example:
    python3 preprocess_tvsd.py \
        --things-dir ~/THINGS_images/object_images \
        --output ~/tvsd_monkeyN_V1_sua.npz

Output keys:
    images    (N, 3, 224, 224) uint8, raw pixel values
    responses (N, n_units)     float32, spike counts per unit
    unit_ids  (n_units, 2)     int, (Electrode_ID, unit_index)
    keys      (N,)             str, 'train_<i>' or 'test_<i>'
    is_test   (N,)             bool
"""

import argparse
import csv
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# /CSNG is the lab network mount holding the recordings.
MAINPATH = "/CSNG/Ephys_data/Macaque_data"
TVSD = f"{MAINPATH}/TVSD_data"
METADATA_TVSD = f"{TVSD}/metadata"

# Monkey N: 4 recording days, 28 sessions total.
SESSIONS_N = []
for day, max_block in [("20220111", 10), ("20220112", 11),
                        ("20220113", 4), ("20220114", 3)]:
    for b in range(1, max_block + 1):
        SESSIONS_N.append(f"macaqueN_TVSD_{day}_B{b}")

# Monkey F: 4 recording days in Jan 2024, 17 sessions total.
SESSIONS_F = []
for day, max_block in [("20240112", 4), ("20240115", 5),
                        ("20240116", 5), ("20240118", 3)]:
    for b in range(1, max_block + 1):
        SESSIONS_F.append(f"macaqueF_TVSD_{day}_B{b}")

# Arrays 3-8 are V1 and arrays 1-2 are the adjacent V2 border (per the mapping CSV
# and the per-spike Area annotation). On the collaborators' advice the well-isolated border
# units are kept, so we train on all of arrays 1-8; V1-only analyses re-filter by area.
V1_ARRAYS = list(range(1, 9))

WINDOW_START = 0.040  # 40 ms post-stimulus (start of feedforward window)
WINDOW_END = 0.100    # 100 ms post-stimulus (end of feedforward window)
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess TVSD Monkey N → .npz (SUA, 40-100ms)")
    p.add_argument("--things-dir", type=str, required=True,
                   help="Path to extracted THINGS images directory (the object_images/ folder)")
    p.add_argument("--output", type=str, default="tvsd_monkeyN_V1_sua.npz")
    p.add_argument("--window-start", type=float, default=WINDOW_START)
    p.add_argument("--window-end", type=float, default=WINDOW_END)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--use-ks4", action="store_true",
                   help="Use the KS4 spike sorting (spikes_KS4/ subdirectory) "
                        "instead of the original sorting (spikes/).")
    p.add_argument("--monkey", default="N", choices=["N", "F"],
                   help="Which animal to preprocess: N (default, Jan 2022) or F (Jan 2024).")
    p.add_argument("--save-trials", action="store_true",
                   help="Also save per-trial responses for test images, with the "
                        "session index of each trial. Needed for noise-ceiling / "
                        "reliability analysis and cross-session consistency checks.")
    return p.parse_args()


def _spike_paths(session: str, arr_id: int, use_ks4: bool) -> str:
    if use_ks4:
        return f"{TVSD}/{session}/spikes_KS4/{session}_Array{arr_id}_spikes_KS4_filtered_noWaveforms.nix"
    return f"{TVSD}/{session}/spikes/{session}_Array{arr_id}_spikes_filtered_noWaveforms.nix"


def load_image_path_mapping(csv_dir: str, things_dir: str):
    """
    Build mapping: image_key → local file path.

    CSV rows are 1-indexed (row 1 = image index 1).
    THINGS_image_path column has e.g. 'aardvark/aardvark_01b.jpg'
    """
    mapping = {}
    for split, csv_name, prefix in [
        ("train", "THINGS_train_imgs_paths.csv", "train"),
        ("test", "THINGS_test_imgs_paths.csv", "test"),
    ]:
        csv_path = os.path.join(csv_dir, csv_name)
        if not os.path.isfile(csv_path):
            print(f"  WARNING: {csv_path} not found")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row_i, row in enumerate(reader, start=1):
                things_path = row["THINGS_image_path"]
                full_path = os.path.join(things_dir, things_path)
                mapping[f"{prefix}_{row_i}"] = full_path

    return mapping


def discover_v1_units(sessions: list, v1_arrays: list, use_ks4: bool = False) -> list:
    """Scan every session and collect the union of (Electrode_ID, unit_index) on V1 arrays."""
    import neo

    all_units = set()
    n_sessions_scanned = 0

    for ses in sessions:
        for arr_id in v1_arrays:
            nix_path = _spike_paths(ses, arr_id, use_ks4)
            if not os.path.isfile(nix_path):
                continue

            io = neo.NixIO(nix_path, "ro")
            block = io.read_block()
            spike_trains = block.segments[0].spiketrains

            # Group units by electrode, collecting firing rates
            electrode_units = {}  # eid -> list of firing_rate values
            for st in spike_trains:
                eid = int(st.annotations.get("Electrode_ID", -1))
                arr = int(st.annotations.get("Array_ID", -1))
                if arr not in v1_arrays:
                    continue
                if eid not in electrode_units:
                    electrode_units[eid] = []
                electrode_units[eid].append(1)  # just count how many

            for eid, units in electrode_units.items():
                for ui in range(len(units)):
                    all_units.add((eid, ui))

            io.close()
            del block, spike_trains
            gc.collect()

        n_sessions_scanned += 1
        if n_sessions_scanned % 5 == 0:
            print(f"    Scanned {n_sessions_scanned}/{len(sessions)} sessions...")

    return sorted(all_units)


def count_spikes_per_trial_per_unit(
    nix_path: str, trial_starts: np.ndarray, unit_ids: list,
    w_start: float, w_end: float,
) -> np.ndarray:
    """
    Load a .nix file and count spikes per (trial, unit) inside [w_start, w_end].
    Within each electrode, units are sorted by firing_rate (highest first) before
    assigning unit_index so that index ordering matches `unit_ids` across sessions.
    Returns: (n_trials, n_units).
    """
    import neo

    io = neo.NixIO(nix_path, "ro")
    block = io.read_block()
    spike_trains = block.segments[0].spiketrains

    # Map unit_id -> column index
    uid_to_col = {uid: i for i, uid in enumerate(unit_ids)}

    n_trials = len(trial_starts)
    n_units = len(unit_ids)
    counts = np.zeros((n_trials, n_units), dtype=np.float32)

    # Group spike trains by electrode
    electrode_trains = {}
    for st in spike_trains:
        eid = int(st.annotations.get("Electrode_ID", -1))
        if eid not in electrode_trains:
            electrode_trains[eid] = []
        electrode_trains[eid].append(st)

    for eid, trains in electrode_trains.items():
        # Sort by firing_rate annotation (highest first) for consistent ordering
        trains_sorted = sorted(
            trains,
            key=lambda st: float(st.annotations.get("firing_rate", 0)),
            reverse=True,
        )
        for ui, st in enumerate(trains_sorted):
            uid = (eid, ui)
            if uid not in uid_to_col:
                continue
            col = uid_to_col[uid]
            times = np.array(st.times.magnitude)

            for t_i in range(n_trials):
                t0 = trial_starts[t_i] + w_start
                t1 = trial_starts[t_i] + w_end
                counts[t_i, col] += np.sum((times >= t0) & (times < t1))

    io.close()
    del block, spike_trains
    gc.collect()
    return counts


def load_and_resize_image(path: str, size: int) -> np.ndarray:
    """Load an image (.jpg or .bmp), resize to (size, size), return as (3, H, W) uint8."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
    return arr.transpose(2, 0, 1)  # (3, H, W)


def main():
    args = parse_args()
    sessions = SESSIONS_F if args.monkey == "F" else SESSIONS_N
    print(f"[config] Monkey {args.monkey}, V1 arrays {V1_ARRAYS}")
    print(f"[config] window=[{args.window_start*1000:.0f}, {args.window_end*1000:.0f}]ms, "
          f"img_size={args.img_size}, output={args.output}")
    print(f"[config] mode=SUA (individual sorted units), use_ks4={args.use_ks4}, "
          f"save_trials={args.save_trials}")
    print(f"[config] sessions: {len(sessions)}")

    print(f"\n[step 1] Building image path mapping from CSVs...")
    image_path_map = load_image_path_mapping(METADATA_TVSD, args.things_dir)
    print(f"  Mapped {len(image_path_map)} images")

    # Verify a few images exist
    n_exist = sum(1 for p in list(image_path_map.values())[:100] if os.path.isfile(p))
    print(f"  Spot check: {n_exist}/100 images found on disk")
    if n_exist < 90:
        print("  ERROR: Most images not found! Check --things-dir path.")
        sys.exit(1)

    print(f"\n[step 2] Discovering V1 units across all {len(sessions)} sessions...")
    unit_ids = discover_v1_units(sessions, V1_ARRAYS, use_ks4=args.use_ks4)
    n_units = len(unit_ids)
    print(f"  Found {n_units} unique V1 units (SUA)")

    # image_key → list of (session_idx, response_vector). The session index is
    # kept so downstream analyses can group trials by session.
    image_responses: dict[str, list[tuple[int, np.ndarray]]] = {}

    for ses_i, ses in enumerate(sessions):
        print(f"\n[step 3] Session {ses_i+1}/{len(sessions)}: {ses}")

        # Load trial metadata
        meta_csv = f"{TVSD}/{ses}/{ses}_trial_metadata.csv"
        if not os.path.isfile(meta_csv):
            print(f"  SKIP: {meta_csv} not found")
            continue

        trials = pd.read_csv(meta_csv)
        trials = trials[trials["Success"] == 1].reset_index(drop=True)
        # Column was renamed upstream: Trial_start_absolute_s → Trial_start_s
        if "Trial_start_s" in trials.columns:
            trial_starts = trials["Trial_start_s"].values
        elif "Trial_start_absolute_s" in trials.columns:
            trial_starts = trials["Trial_start_absolute_s"].values
        else:
            print(f"  SKIP: no Trial_start column found. Columns: {list(trials.columns)}")
            continue
        n_trials = len(trials)
        print(f"  Trials: {n_trials} successful")

        # Accumulate spike counts across V1 arrays
        session_counts = np.zeros((n_trials, n_units), dtype=np.float32)

        for arr_id in V1_ARRAYS:
            nix_path = _spike_paths(ses, arr_id, args.use_ks4)
            if not os.path.isfile(nix_path):
                print(f"  SKIP: Array {arr_id} not found")
                continue

            print(f"  Loading Array {arr_id}...", end="", flush=True)
            arr_counts = count_spikes_per_trial_per_unit(
                nix_path, trial_starts, unit_ids,
                args.window_start, args.window_end,
            )
            session_counts += arr_counts
            active = (arr_counts.sum(axis=0) > 0).sum()
            print(f" {active} active units, {arr_counts.sum():.0f} total spikes")

        # Store per-image responses
        for t_i in range(n_trials):
            row = trials.iloc[t_i]
            train_id = int(row["THINGS_Train_Image_Index"])
            test_id = int(row["THINGS_Test_Image_Index"])

            if train_id > 0:
                key = f"train_{train_id}"
            elif test_id > 0:
                key = f"test_{test_id}"
            else:
                continue

            if key not in image_responses:
                image_responses[key] = []
            image_responses[key].append((ses_i, session_counts[t_i]))

    print(f"\n[step 4] Averaging responses across repetitions...")
    keys = sorted(image_responses.keys())
    n_images = len(keys)
    print(f"  Total unique images: {n_images}")

    responses = np.zeros((n_images, n_units), dtype=np.float32)
    for i, key in enumerate(keys):
        reps = image_responses[key]                # list of (ses_idx, vec)
        responses[i] = np.mean([v for _, v in reps], axis=0)

    train_count = sum(1 for k in keys if k.startswith("train_"))
    test_count = sum(1 for k in keys if k.startswith("test_"))
    print(f"  Train images: {train_count}, Test images: {test_count}")
    print(f"  Active units (nonzero mean): {(responses.mean(axis=0) > 0).sum()}/{n_units}")
    print(f"  Response stats: mean={responses.mean():.3f}, std={responses.std():.3f}")

    print(f"\n[step 5] Loading {n_images} images ({args.img_size}×{args.img_size}) as uint8...")
    images = np.zeros((n_images, 3, args.img_size, args.img_size), dtype=np.uint8)

    missing = 0
    for i, key in enumerate(keys):
        if key not in image_path_map:
            missing += 1
            continue
        img_path = image_path_map[key]
        if not os.path.isfile(img_path):
            missing += 1
            continue
        images[i] = load_and_resize_image(img_path, args.img_size)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_images} loaded...")

    if missing > 0:
        print(f"  WARNING: {missing} images not found!")
    else:
        print(f"  All {n_images} images loaded successfully!")

    print(f"\n[step 6] Saving to {args.output}...")

    is_test = np.array([k.startswith("test_") for k in keys])

    save_dict = dict(
        images=images,
        responses=responses,
        is_test=is_test,
        unit_ids=np.array(unit_ids),
        keys=np.array(keys),
        sessions=np.array(sessions),         # session-index → session-name lookup
    )

    if args.save_trials:
        # Flatten test trials so they can be grouped by image OR by session downstream.
        test_idx_by_key = {k: i for i, k in enumerate(keys) if k.startswith("test_")}
        all_resp, all_img_idx, all_ses_idx = [], [], []
        for key, img_idx in test_idx_by_key.items():
            for ses_i, vec in image_responses[key]:
                all_resp.append(vec)
                all_img_idx.append(img_idx)
                all_ses_idx.append(ses_i)
        save_dict["test_trial_responses"] = np.stack(all_resp).astype(np.float32) \
                                            if all_resp else np.zeros((0, n_units), dtype=np.float32)
        save_dict["test_trial_image_idx"]   = np.asarray(all_img_idx, dtype=np.int32)
        save_dict["test_trial_session_idx"] = np.asarray(all_ses_idx, dtype=np.int32)
        print(f"  + per-trial test data: {len(all_resp)} trials over "
              f"{len(test_idx_by_key)} test images, "
              f"{len(set(all_ses_idx))} sessions")

    np.savez(args.output, **save_dict)

    file_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone! Saved {args.output} ({file_mb:.0f} MB)")
    print(f"   images:    {images.shape} ({images.dtype})")
    print(f"   responses: {responses.shape} ({responses.dtype})")
    print(f"   units:     {n_units} SUA units")
    print(f"   window:    {args.window_start*1000:.0f}-{args.window_end*1000:.0f}ms")


if __name__ == "__main__":
    main()
