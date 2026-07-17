#!/usr/bin/env python3
"""
Build a per-neuron waveform-class label table for monkey N by joining the collaborators'
waveform-subtype dataframes to the KS4 spike sorting that the model was
trained on.

Matching key: each KS4 spiketrain carries a `nix_name` annotation
(neo.spiketrain.<hash>) that is identical to the label table's `cell_name`. We
reproduce the model's (Electrode_ID, unit_index) identity by ranking the
spiketrains on each electrode by firing rate (highest = index 0), exactly as
preprocess_tvsd.py does, then look up the waveform class for that spiketrain.

A neuron appears in several sessions; we majority-vote its subtype/area across
sessions and record how consistent that vote was.

Reads the recordings from /CSNG (read-only) and the waveform-subtype dataframes
from $WAVEFORM_LABELS_DIR. Writes ~/waveform_labels_monkeyN.csv.
"""
import sys, importlib, importlib.abc, importlib.machinery, os, gc, warnings
from collections import defaultdict, Counter
warnings.simplefilter("ignore")

# numpy 2.x pickle shim (the label pickles were written under numpy 2)
class _CF(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, n, p=None, t=None):
        if n == "numpy._core" or n.startswith("numpy._core."):
            return importlib.machinery.ModuleSpec(n, self)
    def create_module(self, s):
        return importlib.import_module(s.name.replace("numpy._core", "numpy.core", 1))
    def exec_module(self, m):
        pass
sys.meta_path.insert(0, _CF())

import numpy as np, pandas as pd, neo

TVSD = "/CSNG/Ephys_data/Macaque_data/TVSD_data"

# Directory holding the per-session waveform-subtype dataframes produced by the
# collaborating group. Environment-specific: set WAVEFORM_LABELS_DIR to point at it.
LABELS_DIR = os.environ.get("WAVEFORM_LABELS_DIR", "")

V1_ARRAYS = list(range(1, 9))

SESSIONS = []
for day, nb in [("20220111", 10), ("20220112", 11), ("20220113", 4), ("20220114", 3)]:
    for b in range(1, nb + 1):
        SESSIONS.append((f"macaqueN_TVSD_{day}_B{b}",
                         f"monkeyN_all_arrays_date_{day}_B{b}.pkl"))


def session_units(ses):
    """Return {(eid, ui): nix_name} reproducing the model's ranking."""
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
                out[(eid, ui)] = str(st.annotations.get("nix_name"))
        io.close()
    return out


def main():
    if not LABELS_DIR:
        sys.exit("Set WAVEFORM_LABELS_DIR to the directory holding the waveform-subtype "
                 "dataframes (one pickle per session; see SESSIONS below for the filenames).")

    cls_votes = defaultdict(Counter)
    area_votes = defaultdict(Counter)
    width_votes = defaultdict(Counter)
    dir_votes = defaultdict(Counter)
    amp_vals = defaultdict(list)
    wid_vals = defaultdict(list)

    for i, (ses, pkl) in enumerate(SESSIONS, 1):
        pp = os.path.join(LABELS_DIR, pkl)
        if not os.path.isfile(pp):
            print(f"[{i}/{len(SESSIONS)}] {ses}: MISSING pickle {pkl}", flush=True)
            continue
        units = session_units(ses)
        df = pd.read_pickle(pp)
        cols = ["final_class", "area", "width_wf_class", "wf_direction", "amp_wf", "width_wf"]
        look = df.set_index(df["cell_name"].astype(str))[cols]
        d = {idx: look.loc[idx] for idx in look.index}
        del df, look
        gc.collect()

        n_match = 0
        for (eid, ui), nix in units.items():
            row = d.get(nix)
            if row is None:
                continue
            n_match += 1
            cls_votes[(eid, ui)][row["final_class"]] += 1
            area_votes[(eid, ui)][row["area"]] += 1
            width_votes[(eid, ui)][row["width_wf_class"]] += 1
            dir_votes[(eid, ui)][row["wf_direction"]] += 1
            try:
                amp_vals[(eid, ui)].append(float(np.asarray(row["amp_wf"])))
                wid_vals[(eid, ui)].append(float(np.asarray(row["width_wf"])))
            except Exception:
                pass
        print(f"[{i}/{len(SESSIONS)}] {ses}: {len(units)} units, {n_match} matched", flush=True)
        del d
        gc.collect()

    rows = []
    for key in sorted(cls_votes):
        eid, ui = key
        cv, av, wv, dv = cls_votes[key], area_votes[key], width_votes[key], dir_votes[key]
        cls, cls_n = cv.most_common(1)[0]
        area, _ = av.most_common(1)[0]
        width, _ = wv.most_common(1)[0]
        wdir, _ = dv.most_common(1)[0]
        nses = sum(cv.values())
        rows.append(dict(
            electrode_id=eid, unit_index=ui, n_sessions=nses,
            final_class=cls, class_consistency=cls_n / nses,
            area=area, width_wf_class=width, wf_direction=wdir,
            amp_wf_mean=np.mean(amp_vals[key]) if amp_vals[key] else np.nan,
            width_wf_mean=np.mean(wid_vals[key]) if wid_vals[key] else np.nan,
        ))
    out = pd.DataFrame(rows)
    dest = os.path.expanduser("~/waveform_labels_monkeyN.csv")
    out.to_csv(dest, index=False)
    print(f"\n[done] {len(out)} neurons -> {dest}")
    print("area distribution:\n", out["area"].value_counts())
    print("\nV1 final_class distribution:\n",
          out[out["area"] == "V1"]["final_class"].value_counts())
    print("\nmean class consistency (V1):",
          round(out[out["area"] == "V1"]["class_consistency"].mean(), 3))


if __name__ == "__main__":
    main()
