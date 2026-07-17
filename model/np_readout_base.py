"""Readout base

Source:
  neuralpredictors/layers/readouts/base.py

"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
from torch.nn.modules import Module


class ConfigurationError(Exception):
    """Raised when readout configuration is inconsistent."""


class Readout(Module):
    """Base class for readouts.

    Ported from neuralpredictors to include helper methods needed by
    PointPooled2d (resolve_deprecated_gamma_readout, initialize_bias,
    apply_reduction, resolve_reduction_method).
    """

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def regularizer(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(0.0)

    # --- reduction helpers ---------------------------------------------------

    def apply_reduction(
        self, x: torch.Tensor, reduction: str = "mean", average: Optional[bool] = None
    ) -> torch.Tensor:
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(f"Reduction method '{reduction}' is not recognized.")

    def resolve_reduction_method(self, reduction: str = "mean", average: Optional[bool] = None):
        if average is not None:
            warnings.warn("Use of 'average' is deprecated. Please use `reduction` instead")
            reduction = "mean" if average else "sum"
        return reduction

    # --- deprecated gamma_readout handling -----------------------------------

    def resolve_deprecated_gamma_readout(
        self,
        feature_reg_weight: Optional[float],
        gamma_readout: Optional[float],
        default: float = 1.0,
    ) -> float:
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Use 'feature_reg_weight' instead."
            )
        if feature_reg_weight is None:
            if gamma_readout is not None:
                feature_reg_weight = gamma_readout
            else:
                feature_reg_weight = default
        return feature_reg_weight

    # --- bias initialization -------------------------------------------------

    def initialize_bias(self, mean_activity: Optional[torch.Tensor] = None) -> None:
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity
