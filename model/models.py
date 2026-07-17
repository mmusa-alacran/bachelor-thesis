"""
Core + readout architecture:
  - TransferLearningCore: torchvision backbone (e.g. ConvNeXt) clipped after `cut_layers`
    children of `.features`, optionally pretrained and optionally fine-tuned.
  - PointPooled2d: per-neuron (x, y) readout from neuralpredictors.
  - CoreReadoutModel: forward = ELU(readout(core(x))) + 1, keeping output > 0 for Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

import torch
from torch import nn

from np_transfer_learning_core import TransferLearningCore
from np_point_pooled import PointPooled2d


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"   # any torchvision model exposing `.features`
    cut_layers: int = 6               # number of `.features` children to keep
    pretrained: bool = True
    fine_tune: bool = False           # if False, backbone is frozen (only readout trains)

    # PointPooled2d hyperparameters
    pool_steps: int = 2
    pool_kern: int = 2
    init_range: float = 0.1
    bias: bool = True


class CoreReadoutModel(nn.Module):
    """Wraps core (feature extractor) + readout (per-neuron projection)."""

    def __init__(self, core: nn.Module, readout: nn.Module):
        super().__init__()
        self.core = core
        self.readout = readout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.core(x)                # (B, C, H, W)
        y = self.readout(feats)             # (B, n_neurons)
        # ELU+1 keeps the output strictly positive, which Poisson / NB losses require.
        return torch.nn.functional.elu(y) + 1


def _infer_core_outshape(core: nn.Module, in_shape: Tuple[int, int, int], device: torch.device) -> Tuple[int, int, int]:
    """Run one dummy forward pass to recover (C, H, W) for the readout."""
    c, h, w = in_shape
    x = torch.zeros(1, c, h, w, device=device)
    with torch.no_grad():
        y = core(x)
    assert y.ndim == 4, f"Core must return feature maps (B,C,H,W). Got {tuple(y.shape)}"
    _, C, H, W = y.shape
    return (C, H, W)


def build_model(
    in_shape: Tuple[int, int, int],
    outdims: int,
    cfg: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
    mean_activity: Optional[torch.Tensor] = None,
) -> CoreReadoutModel:
    """
    in_shape:      (C, H, W) input image shape
    outdims:       number of neurons
    mean_activity: (outdims,) per-neuron mean spike count, used to initialise the
                   readout bias so predictions start at the right scale.
    """
    if cfg is None:
        cfg = ModelConfig()
    if device is None:
        device = torch.device("cpu")

    core = TransferLearningCore(
        input_channels=in_shape[0],
        tl_model_name=cfg.backbone,
        layers=cfg.cut_layers,
        pretrained=cfg.pretrained,
        fine_tune=cfg.fine_tune,
    )

    core = core.to(device)
    core_outshape = _infer_core_outshape(core, in_shape, device=device)

    # PointPooled2d expects mean_activity on CPU; .to(device) moves the bias afterwards.
    mean_act_cpu: Optional[torch.Tensor] = None
    if mean_activity is not None:
        mean_act_cpu = mean_activity.detach().cpu().float()

    readout = PointPooled2d(
        in_shape=core_outshape,
        outdims=outdims,
        pool_steps=cfg.pool_steps,
        bias=cfg.bias,
        pool_kern=cfg.pool_kern,
        init_range=cfg.init_range,
        mean_activity=mean_act_cpu,
    ).to(device)

    model = CoreReadoutModel(core=core, readout=readout).to(device)
    return model
