from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from np_measures import PoissonLoss


@dataclass
class EvalResult:
    loss: float
    corr: float


def _epoch_corr(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12) -> float:
    """Per-neuron Pearson correlation over the full epoch, averaged across neurons.

    Aggregating all predictions before correlating is much less noisy than
    averaging per-batch correlations (a batch of 32 is a tiny sample).
    """
    delta_p = preds - preds.mean(0, keepdim=True)
    delta_t = targets - targets.mean(0, keepdim=True)
    var_p = delta_p.pow(2).mean(0)
    var_t = delta_t.pow(2).mean(0)
    corr = (delta_p * delta_t).mean(0) / ((var_p + eps) * (var_t + eps)).sqrt()
    return float(corr.mean())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gamma_readout: float = 0.0,
    clip_grad: Optional[float] = None,
    loss_fn: Optional[nn.Module] = None,
) -> EvalResult:
    """Run one training or evaluation epoch.

    optimizer:     if provided, perform a training step each batch.
    gamma_readout: L1 weight on the readout features (training only).
    clip_grad:     if provided, clip gradient L2 norm before each optimizer step.
    loss_fn:       loss module (defaults to PoissonLoss). Must already be on `device`
                   if it has learnable parameters (e.g. NegativeBinomialLoss).
    """
    is_train = optimizer is not None
    model.train(is_train)

    if loss_fn is None:
        loss_fn = PoissonLoss(avg=False)
    if any(p.requires_grad for p in loss_fn.parameters()):
        loss_fn.train(is_train)

    total_data_loss = 0.0
    n_batches = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        data_loss = loss_fn(pred, y)

        # Regulariser added to the training loss only; the reported metric is the
        # data loss alone, so values stay comparable across runs with different gamma.
        if is_train and gamma_readout > 0.0:
            loss = data_loss + gamma_readout * model.readout.regularizer()
        else:
            loss = data_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_data_loss += float(data_loss.detach().cpu())
        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())
        n_batches += 1

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    corr = _epoch_corr(preds, targets)

    return EvalResult(loss=total_data_loss / max(n_batches, 1), corr=corr)


def save_checkpoint(path: str, model: nn.Module, extra: dict | None = None) -> None:
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, map_location: str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {"model": state}
