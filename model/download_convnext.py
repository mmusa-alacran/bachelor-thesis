'''
download_convnext.py
--------------------
Download pretrained ConvNeXt weights via timm and save to disk.


'''

from __future__ import annotations

import argparse
import os

import torch
import timm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--out", required=True)
    p.add_argument("--in-chans", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    m = timm.create_model(args.backbone, pretrained=True, in_chans=args.in_chans)
    torch.save(m.state_dict(), args.out)
    print(f"[done] saved {args.backbone} state_dict -> {args.out}")


if __name__ == "__main__":
    main()
