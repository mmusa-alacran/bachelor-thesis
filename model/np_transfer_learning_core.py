"""TransferLearningCore

Source :
  neuralpredictors/layers/cores/conv2d.py  (class TransferLearningCore)

"""

from __future__ import annotations

import logging
import warnings

import torch
import torchvision
from torch import nn

logger = logging.getLogger(__name__)


class Core(nn.Module):
    """Minimal stand-in for neuralpredictors' Core base class.

    The full project has additional helpers and regularizers. This scaffold only needs a Module.
    """

    def regularizer(self) -> torch.Tensor:
        return torch.tensor(0.0)


class TransferLearningCore(Core):
    """See module docstring."""

    def __init__(
        self,
        input_channels: int,
        tl_model_name: str,
        layers: int,
        pretrained: bool = True,
        final_batchnorm: bool = True,
        final_nonlinearity: bool = True,
        momentum: float = 0.1,
        fine_tune: bool = False,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"Ignoring input {kwargs!r} when creating {self.__class__.__name__}",
                UserWarning,
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download/load model and cut after specified layer
        # Upstream uses `pretrained=...` API. Newer torchvision uses `weights=...`.
        # We keep backward compatibility: try old API first, then weights.
        try:
            TL_model = getattr(torchvision.models, tl_model_name)(pretrained=pretrained)
        except TypeError:
            if pretrained:
                weights_enum = getattr(torchvision.models, tl_model_name.title().replace('_', ''), None)
                # If we can't resolve a weights enum cleanly, fall back to DEFAULT if available.
                weights = None
                try:
                    weights = getattr(getattr(torchvision.models, tl_model_name).Weights, "DEFAULT")
                except Exception:
                    pass
                TL_model = getattr(torchvision.models, tl_model_name)(weights=weights)
            else:
                TL_model = getattr(torchvision.models, tl_model_name)(weights=None)

        # ConvNeXt (and many others) expose features as a Sequential
        if not hasattr(TL_model, "features"):
            raise ValueError(f"torchvision model '{tl_model_name}' has no .features; pick another backbone or adapt here.")

        TL_model_clipped = nn.Sequential(*list(TL_model.features.children())[:layers])
        if len(TL_model_clipped) == 0:
            raise ValueError("Clipped model has zero layers. Increase `layers`.")
        if not isinstance(TL_model_clipped[-1], nn.Conv2d):
            warnings.warn(
                f"Final layer is of type {type(TL_model_clipped[-1])}, not nn.Conv2d",
                UserWarning,
            )

        # Fix pretrained parameters during training
        if not fine_tune:
            for param in TL_model_clipped.parameters():
                param.requires_grad = False

        # Infer output channels via a quick forward pass (works for any backbone)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels if input_channels >= 3 else 3, 64, 64)
            out = TL_model_clipped(dummy)
        self._outchannels = out.shape[1]

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module("TransferLearning", TL_model_clipped)
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self._outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # If model is designed for RGB input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1 and getattr(self.features.TransferLearning[0], "in_channels", None) == 3:
            input_ = input_.repeat(1, 3, 1, 1)
        return self.features(input_)

    def regularizer(self) -> torch.Tensor:
        return torch.tensor(0.0)

    @property
    def outchannels(self) -> int:
        """Return number of output channels of the clipped network."""
        return self._outchannels

    def initialize(self) -> None:
        logger.warning(
            "Ignoring initialization: parameters come from torchvision weights. For random weights, set pretrained=False."
        )
