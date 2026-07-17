from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch

from dataset import make_loaders
from models import ModelConfig, build_model
from utils import run_epoch, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Readout LR when backbone is frozen.")
    p.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone LR when fine-tuning.")
    p.add_argument("--readout-lr", type=float, default=1e-3, help="Readout LR when fine-tuning.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs without val improvement).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--fine-tune", action="store_true", help="Allow backbone training. Default: frozen.")
    p.add_argument("--cut-layers", type=int, default=6, help="Where to cut torchvision .features inside TransferLearningCore.")

    p.add_argument("--gamma-readout", type=float, default=0.01,
                   help="L1 weight on the readout feature weights. 0 disables it.")
    p.add_argument("--lr-scheduler", default="plateau", choices=["plateau", "cosine"])
    p.add_argument("--freeze-readout-lr", action="store_true",
                   help="When fine-tuning, keep the readout LR fixed at --readout-lr "
                        "while the scheduler decays the backbone LR.")
    p.add_argument("--clip-grad", type=float, default=None, help="Max L2 norm for gradient clipping. None disables it.")
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    p.add_argument("--loss", default="poisson", choices=["poisson", "nb"],
                   help="'nb' adds a learnable per-neuron dispersion (over-dispersed spike counts).")
    p.add_argument("--nb-init-log-disp", type=float, default=0.0,
                   help="Initial log_dispersion for the NB loss.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    loaders, n_neurons = make_loaders(
        args.data, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers
    )

    # Mean spike count per neuron from the training set — used to initialise the readout bias.
    mean_activity = loaders['train'].dataset.responses.mean(dim=0)
    print(f"[info] mean_activity: min={float(mean_activity.min()):.3f}, "
          f"max={float(mean_activity.max()):.3f}, "
          f"mean={float(mean_activity.mean()):.3f} spikes/trial")

    cfg = ModelConfig(
        backbone=args.backbone,
        cut_layers=args.cut_layers,
        pretrained=args.pretrained,
        fine_tune=args.fine_tune,
    )

    # Infer input shape (C,H,W) from the first training batch.
    xb, _ = next(iter(loaders['train']))
    if xb.ndim != 4:
        raise ValueError(f'Expected images as 4D tensor (N,C,H,W). Got {tuple(xb.shape)}')
    in_shape = (xb.shape[1], xb.shape[2], xb.shape[3])

    model = build_model(in_shape=in_shape, outdims=n_neurons, cfg=cfg, device=device,
                        mean_activity=mean_activity)

    # Negative binomial loss has a per-neuron dispersion parameter; it must be optimised
    # alongside the model parameters, so we collect it for the optimizer below.
    if args.loss == "nb":
        from np_measures import NegativeBinomialLoss
        loss_fn = NegativeBinomialLoss(
            n_neurons=n_neurons, init_log_dispersion=args.nb_init_log_disp, avg=False
        ).to(device)
        nb_params = list(loss_fn.parameters())
        print(f"[info] loss=nb (n_neurons={n_neurons}, init_log_disp={args.nb_init_log_disp})")
    else:
        from np_measures import PoissonLoss
        loss_fn = PoissonLoss(avg=False)
        nb_params = []
        print(f"[info] loss=poisson")

    # Differential LRs: pretrained backbone gets a slow LR, randomly initialised readout a fast one.
    # Weight decay is only applied to the backbone; the readout has its own L1 regulariser.
    if args.fine_tune:
        backbone_params = [p for p in model.core.parameters() if p.requires_grad]
        readout_params = [p for p in model.readout.parameters() if p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
            {"params": readout_params,  "lr": args.readout_lr,  "weight_decay": 0.0},
        ]
        print(f"[info] differential lr: backbone={args.backbone_lr:.1e}, readout={args.readout_lr:.1e}")
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": args.lr, "weight_decay": args.weight_decay}]
        print(f"[info] frozen backbone, readout lr={args.lr:.1e}")

    # NB dispersion is a shape/scale parameter — no weight decay; use the readout LR.
    if nb_params:
        nb_lr = args.readout_lr if args.fine_tune else args.lr
        param_groups.append({"params": nb_params, "lr": nb_lr, "weight_decay": 0.0})

    OptCls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    opt = OptCls(param_groups, weight_decay=0)  # per-group weight_decay set above
    print(f"[info] optimizer={args.optimizer}")

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=0
        )
        print(f"[info] scheduler=cosine (T_max={args.epochs})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )
        print(f"[info] scheduler=plateau (factor=0.5, patience=5)")

    freeze_readout = args.fine_tune and args.freeze_readout_lr
    print(f"[info] gamma_readout={args.gamma_readout:.3g}  "
          f"clip_grad={args.clip_grad}  "
          f"freeze_readout_lr={freeze_readout}")

    best_val = -1e9
    epochs_no_improve = 0
    best_path = os.path.join(args.out, "best.pt")

    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_cfg": asdict(cfg), "n_neurons": n_neurons}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], device, optimizer=opt,
                       gamma_readout=args.gamma_readout, clip_grad=args.clip_grad,
                       loss_fn=loss_fn)
        va = run_epoch(model, loaders["val"], device, optimizer=None,
                       loss_fn=loss_fn)
        current_lr = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] train loss={tr.loss:.4f} corr={tr.corr:.4f} | "
              f"val loss={va.loss:.4f} corr={va.corr:.4f} | lr={current_lr:.2e}")

        if args.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(va.corr)

        # Restore readout LR after the scheduler step — only the backbone should be decayed.
        if freeze_readout:
            opt.param_groups[1]['lr'] = args.readout_lr

        if va.corr > best_val:
            best_val = va.corr
            epochs_no_improve = 0
            extra = {"best_val_corr": best_val, "epoch": epoch}
            if args.loss == "nb":
                extra["loss_state"] = loss_fn.state_dict()
            save_checkpoint(best_path, model, extra=extra)
            print(f"[info] saved best: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[info] early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    last_extra = {"best_val_corr": best_val}
    if args.loss == "nb":
        last_extra["loss_state"] = loss_fn.state_dict()
    save_checkpoint(os.path.join(args.out, "last.pt"), model, extra=last_extra)
    print(f"[done] best val corr = {best_val:.4f}")


if __name__ == "__main__":
    main()