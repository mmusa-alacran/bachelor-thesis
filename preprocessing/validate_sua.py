#!/usr/bin/env python3
"""
Quick validation of SUA approach on a single session + array.
Run on a host with the /CSNG network mount available:

    python3 validate_sua.py
"""

import numpy as np
import pandas as pd
import neo

TVSD = "/CSNG/Ephys_data/Macaque_data/TVSD_data"
SESSION = "macaqueF_TVSD_20240112_B1"
ARRAY = 1  # V1 array
WINDOW_START = 0.040  # 40ms
WINDOW_END = 0.100    # 100ms

nix_path = f"{TVSD}/{SESSION}/spikes/{SESSION}_Array{ARRAY}_spikes_filtered_noWaveforms.nix"
meta_csv = f"{TVSD}/{SESSION}/{SESSION}_trial_metadata.csv"

print(f"Loading: {nix_path}")
io = neo.NixIO(nix_path, "ro")
block = io.read_block()
spike_trains = block.segments[0].spiketrains

# ── 1. Show all units ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"1. UNITS IN ARRAY {ARRAY}")
print(f"{'='*60}")
print(f"Total SpikeTrain objects: {len(spike_trains)}")

# Group by electrode
electrode_units = {}
for st in spike_trains:
    eid = int(st.annotations.get("Electrode_ID", -1))
    arr = int(st.annotations.get("Array_ID", -1))
    fr = st.annotations.get("firing_rate", "?")
    if eid not in electrode_units:
        electrode_units[eid] = []
    electrode_units[eid].append({
        "firing_rate": fr,
        "n_spikes": len(st.times),
        "array": arr,
    })

print(f"Unique electrodes: {len(electrode_units)}")

multi_unit_electrodes = {e: u for e, u in electrode_units.items() if len(u) > 1}
single_unit_electrodes = {e: u for e, u in electrode_units.items() if len(u) == 1}

print(f"Electrodes with 1 unit:  {len(single_unit_electrodes)}")
print(f"Electrodes with >1 unit: {len(multi_unit_electrodes)}")

if multi_unit_electrodes:
    print(f"\nMulti-unit electrodes (showing first 5):")
    for eid in sorted(multi_unit_electrodes.keys())[:5]:
        units = multi_unit_electrodes[eid]
        print(f"  Electrode {eid}: {len(units)} units")
        for ui, u in enumerate(units):
            print(f"    unit {ui}: {u['n_spikes']} spikes, firing_rate={u['firing_rate']}")

# ── 2. Build SUA unit list ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"2. SUA UNIT IDENTIFICATION")
print(f"{'='*60}")

unit_ids = []
for eid in sorted(electrode_units.keys()):
    for ui in range(len(electrode_units[eid])):
        unit_ids.append((eid, ui))

print(f"Total SUA units: {len(unit_ids)}")
print(f"(vs MUA would be: {len(electrode_units)} electrodes)")
print(f"First 10 unit IDs: {unit_ids[:10]}")

# ── 3. Count spikes with 40-100ms window ────────────────────────────────
print(f"\n{'='*60}")
print(f"3. SPIKE COUNTING (40-100ms window)")
print(f"{'='*60}")

trials = pd.read_csv(meta_csv)
trials = trials[trials["Success"] == 1].reset_index(drop=True)
trial_starts = trials["Trial_start_absolute_s"].values
n_trials = min(50, len(trials))  # test on first 50 trials
print(f"Testing on first {n_trials} trials")

uid_to_col = {uid: i for i, uid in enumerate(unit_ids)}
counts = np.zeros((n_trials, len(unit_ids)), dtype=np.float32)

# Group spike trains by electrode for unit_index assignment
electrode_st = {}
for st in spike_trains:
    eid = int(st.annotations.get("Electrode_ID", -1))
    if eid not in electrode_st:
        electrode_st[eid] = []
    electrode_st[eid].append(st)

for eid, trains in electrode_st.items():
    for ui, st in enumerate(trains):
        uid = (eid, ui)
        if uid not in uid_to_col:
            continue
        col = uid_to_col[uid]
        times = np.array(st.times.magnitude)

        for t_i in range(n_trials):
            t0 = trial_starts[t_i] + WINDOW_START
            t1 = trial_starts[t_i] + WINDOW_END
            counts[t_i, col] += np.sum((times >= t0) & (times < t1))

active_units = (counts.sum(axis=0) > 0).sum()
print(f"Active units (>0 spikes in 50 trials): {active_units}/{len(unit_ids)}")
print(f"Mean spike count per trial per unit: {counts.mean():.3f}")
print(f"Max spike count: {counts.max():.0f}")
print(f"Fraction of zeros: {(counts == 0).mean():.3f}")

# Show per-unit stats for first 10
print(f"\nPer-unit stats (first 10):")
print(f"  {'Unit ID':<20} {'Mean':>8} {'Max':>6} {'Active trials':>15}")
for i in range(min(10, len(unit_ids))):
    uid = unit_ids[i]
    mean_c = counts[:, i].mean()
    max_c = counts[:, i].max()
    active_t = (counts[:, i] > 0).sum()
    print(f"  {str(uid):<20} {mean_c:>8.3f} {max_c:>6.0f} {active_t:>15}")

io.close()

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"OK: SUA identification works: {len(unit_ids)} units from {len(electrode_units)} electrodes")
print(f"OK: 40-100ms window produces {'reasonable' if 0 < counts.mean() < 10 else 'UNEXPECTED'} counts (mean={counts.mean():.3f})")
print(f"OK: {active_units}/{len(unit_ids)} units active")
