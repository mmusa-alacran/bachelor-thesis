"""Correlation and loss measures.

Corr and PoissonLoss are derived from:
  neuralpredictors/measures/modules.py

NegativeBinomialLoss, with its learnable per-neuron dispersion, is original to
this work; it has no upstream counterpart.

"""

import logging
import warnings

import torch
import torch.distributions as d
from torch import nn

logger = logging.getLogger(__name__)


class Corr(nn.Module):
    def __init__(self, eps=1e-12, detach_target=True):
        """
        Compute correlation between the output and the target

        Args:
            eps (float, optional): Used to offset the computed variance to provide numerical stability.
                Defaults to 1e-12.
            detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
                using this as a loss to train on. Defaults to True.
        """
        self.eps = eps
        self.detach_target = detach_target
        super().__init__()

    def forward(self, output, target):
        if self.detach_target:
            target = target.detach()
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return corrs

class NegativeBinomialLoss(nn.Module):
    """Negative Binomial NLL with a learnable per-neuron dispersion.

    Uses PyTorch's NegativeBinomial(total_count=r, probs=p) parameterisation:
        mean = r * p / (1 - p)        (set equal to the model's `rate`)
        var  = mean + mean^2 / r      (over-dispersion controlled by 1/r)
    Solving for p:
        p = rate / (rate + r)
    """

    def __init__(self, n_neurons, init_log_dispersion: float = 0.0,
                 per_neuron: bool = False, avg: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.log_dispersion = nn.Parameter(
            torch.full((n_neurons,), float(init_log_dispersion))
        )
        self.per_neuron = per_neuron
        self.avg = avg
        self.eps = eps

    @property
    def dispersion(self) -> torch.Tensor:
        # softplus keeps r > 0 with smooth gradients
        return nn.functional.softplus(self.log_dispersion) + self.eps

    def forward(self, output, target):
        target = target.detach()
        rate = output.clamp(min=self.eps)
        r = self.dispersion  # (n_neurons,)
        probs = rate / (rate + r)
        dist = torch.distributions.NegativeBinomial(total_count=r, probs=probs)
        nll = -dist.log_prob(target)

        if self.per_neuron:
            nll = nll.view(-1, nll.shape[-1])
            return nll.mean(dim=0) if self.avg else nll.sum(dim=0)
        return nll.mean() if self.avg else nll.sum()


class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-08, per_neuron=False, avg=True, full_loss=False):
        """
        Computes Poisson loss between the output and target. Loss is evaluated by computing log likelihood that
        output prescribes the mean of the Poisson distribution and target is a sample from the distribution.

        Args:
            bias (float, optional): Value used to numerically stabilize evalution of the log-likelihood. This value is effecitvely added to the output during evaluation. Defaults to 1e-08.
            per_neuron (bool, optional): If set to True, the average/total Poisson loss is returned for each entry of the last dimension (assumed to be enumeration neurons) separately. Defaults to False.
            avg (bool, optional): If set to True, return mean loss. Otherwise returns the sum of loss. Defaults to True.
            full_loss (bool, optional): If set to True, compute the full loss, i.e. with Stirling correction term (not needed for optimization but needed for reporting of performance). Defaults to False.
        """
        super().__init__()
        self.bias = bias
        self.full_loss = full_loss
        self.per_neuron = per_neuron
        self.avg = avg
        if self.avg:
            warnings.warn("Poissonloss is averaged per batch. It's recommended to use `sum` instead")

    def forward(self, output, target):
        target = target.detach()
        rate = output
        loss = nn.PoissonNLLLoss(log_input=False, full=self.full_loss, eps=self.bias, reduction="none")(rate, target)

        if not self.per_neuron:
            loss = loss.mean() if self.avg else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            loss = loss.mean(dim=0) if self.avg else loss.sum(dim=0)
        assert not (torch.isnan(loss).any() or torch.isinf(loss).any()), "None or inf value encountered!"
        return loss
