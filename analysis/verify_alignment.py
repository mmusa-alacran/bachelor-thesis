#!/usr/bin/env python3
"""
Alignment verification for the final (0.15-filtered) dataset.

Reproduces the r = 0.914 check: the per-neuron mean recorded response stored in
the training .npz should correlate strongly with the *independent* KS4
`firing_rate` annotation carried by the spiketrains, looked up through the same
(electrode_id, unit_index) firing-rate ranking used to build the dataset. If the
responses columns are aligned to the right units, the correlation is high; if the
neuron-to-annotation mapping is shuffled, it collapses to ~0.

Reads only /CSNG (read-only) and the .npz. Writes nothing.
"""
import os
from collections import defaultdict
import numpy as np
import neo
from scipy.stats import pearsonr, spearmanr

TVSD = "/CSNG/Ephys_data/Macaque_data/TVSD_data"
V1_ARRAYS = list(range(1, 9))
NPZ = os.path.expanduser("~/tvsd_monkeyN_V1_sua_KS4_filtered015.npz")

SESSIONS = []
for day, nb in [("20220111", 10), ("20220112", 11), ("20220113", 4), ("20220114", 3)]:
    for b in range(1, nb + 1):
        SESSIONS.append(f"macaqueN_TVSD_{day}_B{b}")


def session_fr(ses):
    """Return {(eid, ui): firing_rate} reproducing the model's ranking."""
    out = {}
    for arr in V1_ARRAYS:
        p = f"{TVSD}/{ses}/spikes_KS4/{ses}_Array{arr}_spikes_KS4_filtered_noWaveforms.nix"
        if not os.path.isfile(p):
            continue
        io = neo.NixIO(p, "ro")
        sts = io.read_block().segments[0].spiketrains
        by_e = defaultdict(list)
        for st in sts:
            by_e[int(st.annotations.get("Electrode_ID", -1))].append(st)
        for eid, trains in by_e.items():
            trains = sorted(trains, key=lambda s: float(s.annotations.get("firing_rate", 0)),
                            reverse=True)
            for ui, st in enumerate(trains):
                out[(eid, ui)] = float(st.annotations.get("firing_rate", 0))
        io.close()
    return out


def main():
    fr_acc = defaultdict(list)
    for i, ses in enumerate(SESSIONS, 1):
        fr = session_fr(ses)
        for k, v in fr.items():
            fr_acc[k].append(v)
        print(f"[{i}/{len(SESSIONS)}] {ses}: {len(fr)} units", flush=True)
    fr_mean = {k: float(np.mean(v)) for k, v in fr_acc.items()}

    d = np.load(NPZ, allow_pickle=True)
    uid = d["unit_ids"]
    mean_resp = d["responses"].mean(0)
    n = len(uid)

    fr_vec = np.full(n, np.nan)
    for i in range(n):
        k = (int(uid[i, 0]), int(uid[i, 1]))
        if k in fr_mean:
            fr_vec[i] = fr_mean[k]
    ok = ~np.isnan(fr_vec)
    print(f"\n[info] model neurons: {n}   matched to KS4 firing_rate: {ok.sum()}")

    r_p, p_p = pearsonr(mean_resp[ok], fr_vec[ok])
    r_s, p_s = spearmanr(mean_resp[ok], fr_vec[ok])
    print(f"[result] Pearson  r = {r_p:.4f}  (p = {p_p:.2e})")
    print(f"[result] Spearman r = {r_s:.4f}  (p = {p_s:.2e})")

    rng = np.random.default_rng(0)
    shuf = [pearsonr(mean_resp[ok], rng.permutation(fr_vec[ok]))[0] for _ in range(200)]
    print(f"[control] shuffled Pearson r = {np.mean(shuf):.4f} +/- {np.std(shuf):.4f}")


if __name__ == "__main__":
    main()
