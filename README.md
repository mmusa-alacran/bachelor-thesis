# Visual representations in different neural types of neurons in primary visual cortex

Code accompanying the bachelor thesis of the same name (Charles University, Faculty of
Mathematics and Physics, 2026).

## What this project does

Cortical neurons can be sorted into classes by the shape of the extracellular action potential an
electrode records from them. Several such **waveform classes** are known in the primary visual
cortex (V1), but how they relate to what a neuron actually *computes* about an image is largely
unknown.

This pipeline attacks that question in two stages:

1. **Describe the function of each neuron.** A deep network is fitted to the spike counts that
   individual macaque V1 neurons produce in response to natural images. Once fitted, the network is
   an image-computable stand-in for those neurons: you can show it any image and read off a
   predicted response.
2. **Relate that description to the waveform classes.** The fitted model is compared against
   independently derived waveform labels for the same neurons, in three ways — decoding the class
   from the model's learned per-neuron parameters, decoding it from the model's predicted responses
   to a battery of stimuli, and measuring the model's neurons with classical physiological protocols.

The headline result is deliberately modest: the relationship is above chance and not reducible to
firing rate, but far too weak to assign a waveform class to an individual neuron.

**The model architecture is not novel.** It follows the shared-core / per-neuron-readout design
standard in this literature (Klindt et al. 2017; Lurz et al. 2021), with an ImageNet-pretrained
ConvNeXt-Tiny core (Liu et al. 2022) and a PointPooled2d readout from the
[neuralpredictors](https://github.com/sinzlab/neuralpredictors) library. What this project
contributes is the adaptation of that pipeline to a new high-density recording: the choice of
backbone and truncation depth, applying the readout regulariser during training, correcting the
validation split behaviour, and replacing the customary Poisson loss with a learnable Negative
Binomial likelihood matched to the over-dispersed spike counts.

## Data (not included)

This repository contains **code only**. The recordings are not ours to distribute.

- **Neural data**: the TVSD dataset (Papale et al. 2025, *Neuron* 113(4):539–553) — macaque V1
  responses to natural images, recorded on a 16-array, ~1024-electrode implant, spike-sorted with
  Kilosort 4. This project uses macaque N (4 recording days, 28 sessions), arrays 1–8 (V1).
- **Stimuli**: the [THINGS](https://things-initiative.org/) image database (Hebart et al. 2019).
  `preprocessing/download_things.py` fetches these.
- **Waveform labels**: produced independently within the research group from the same Kilosort 4
  sorting, and joined to model neurons by `analysis/build_neuron_labels.py`.

> **Paths are environment-specific.** These scripts were written to run against the lab's `/CSNG`
> network mount and a fixed home directory, and those paths are hard-coded. They are preserved as
> they were actually run, for the record. To reuse this pipeline elsewhere you will need to edit the
> path constants at the top of `preprocessing/preprocess_tvsd.py` and of the analysis scripts, and
> set `WAVEFORM_LABELS_DIR` to the directory holding the waveform-subtype dataframes.

## Layout

| Directory | Contents |
|---|---|
| `preprocessing/` | Turns raw Kilosort 4 NIX files into a training-ready `.npz`, and filters neurons |
| `model/` | The network, its training loop and evaluation |
| `analysis/` | Everything downstream of the fitted model: label matching, verification, decoding probes, figures |
| `scripts/` | The shell drivers used to run the training experiments end to end |

### `preprocessing/`
- **`preprocess_tvsd.py`** — the main entry point. Reads the NIX files, counts spikes in the
  40–100 ms feed-forward window, stabilises cross-session unit identity by ranking units within each
  electrode by firing rate, aligns responses to the official THINGS train/test split, and writes a
  single `.npz`.
- `download_things.py` — fetches the THINGS images.
- `filter_neurons.py` — applies the held-out predictive-correlation cut (r > 0.15) that selects the
  neurons carried into the final model and the downstream analyses.
- `filter_npz.py`, `validate_npz.py`, `validate_sua.py` — dataset subsetting and sanity checks.

### `model/`
- **`train.py`** — trains the model. `evaluate_neurons.py` scores it and writes per-neuron
  test correlations.
- `models.py`, `dataset.py`, `utils.py`, `download_convnext.py` — architecture
  assembly, data loading, helpers.
- `np_*.py` — modules **derived from the neuralpredictors library** (see
  [Third-party code](#third-party-code)). They are kept locally rather than imported because the
  library's `TransferLearningCore` probes at 64×64 and can infer the wrong `OutBatchNorm` channel
  count on some torchvision versions; these copies fix that and are the versions that produced the
  reported results. `np_measures.py` is the exception: its `Corr` and `PoissonLoss` come from
  upstream, but the `NegativeBinomialLoss` used for the reported model is original to this work.

### `analysis/`
- `build_neuron_labels.py` — joins the waveform labels onto model neurons via the firing-rate-ranked
  `(electrode, unit index)` identity.
- **`verify_alignment.py`** — checks that each column of the response matrix really belongs to the
  neuron it is labelled with, by correlating against Kilosort's own independent `firing_rate`
  annotation. Everything downstream depends on this.
- **`probe_celltype.py` / `probe_celltype_v2.py`** — the waveform-class decoding probe: decodes the
  class from each family of per-neuron model parameters, with shuffled-label, firing-rate-only and
  equalised-subsample controls. `_v2` adds repeated cross-validation for error bars.
- **`stimulus_battery.py`** — passes a stimulus battery (gratings; held-out natural images) through
  the fitted network and decodes waveform class from the *predicted responses* rather than the
  weights.
- `pca_readout.py`, `readout_pca3d_tsne.py`, `celltype_viz.py`, `make_figures.py` — dimensionality
  reduction and the thesis figures.
- `analyze_neuron_reliability.py` — per-neuron reliability statistics.

## Pipeline order

```
download_things.py                 # fetch stimuli
      |
preprocess_tvsd.py                 # NIX + THINGS  ->  dataset.npz
      |
model/train.py                     # fit the shared core + per-neuron readouts
      |
model/evaluate_neurons.py          # per-neuron test correlations
      |
filter_neurons.py                  # r > 0.15 cut  ->  final neuron set
      |
model/train.py                     # refit on the filtered set  (= the reported model)
      |
      +-- verify_alignment.py      # confirm labels sit on the right neurons
      +-- build_neuron_labels.py   # attach waveform classes
      +-- probe_celltype_v2.py     # decode class from readout parameters
      +-- stimulus_battery.py      # decode class from predicted responses
```

`scripts/run10_pipeline.sh` and `scripts/run_final_015.sh` are the drivers that were actually used;
read them for the exact arguments behind the reported runs.

## The reported model

The configuration reported in the thesis (a `config.json` is written next to every checkpoint):

| Setting | Value |
|---|---|
| Backbone | `convnext_tiny`, ImageNet-pretrained, fine-tuned |
| Truncation | after stage 6 -> 384 channels at 14x14 |
| Readout | PointPooled2d, `pool_steps=2` -> 3 scales x 384 = 1152-d per neuron |
| Loss | Negative Binomial, learnable per-neuron dispersion |
| Optimiser | AdamW, cosine schedule; backbone lr 1e-5, readout lr 1e-3 |
| Readout regulariser | `gamma_readout = 0.05` |
| Early stopping | patience 15, best-validation checkpoint |
| Neurons | 523 single units passing the r > 0.15 cut |

## Requirements

Python 3.11. See `requirements.txt`. The heavy dependencies are PyTorch (a CUDA 12.1 build was used
here), `neuralpredictors`, and `nixio` for reading the recordings. Training was run on a machine with
two RTX 3080s.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Third-party code

`model/np_point_pooled.py`, `model/np_readout_base.py`, `model/np_transfer_learning_core.py` and
parts of `model/np_measures.py` are **derived from
[neuralpredictors](https://github.com/sinzlab/neuralpredictors)**, (c) 2019 Sinz Lab, MIT licence.
Each file records its upstream source in its module docstring. `NegativeBinomialLoss` in
`np_measures.py` has no upstream counterpart and is original to this work. See
`THIRD_PARTY_NOTICES.md` for the per-file breakdown and the full upstream licence text. All other
code here is (c) 2026 Momin Musa under the MIT licence in `LICENSE`.

## Citation

Musa, M. (2026). *Visual representations in different neural types of neurons in primary visual
cortex.* Bachelor thesis, Charles University, Faculty of Mathematics and Physics.
