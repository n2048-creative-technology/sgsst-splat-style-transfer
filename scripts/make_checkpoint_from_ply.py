#!/usr/bin/env python3
"""Bridge script: converts an already-trained point_cloud.ply (e.g. from
the companion gaussian-splatting-pipeline repo, or any standard
gaussian-splatting train.py run) into a checkpoint .pth compatible with
SGSST's stylize.py --start_checkpoint argument.

A plain gaussian-splatting training run (train.py) produces only the
final point_cloud.ply, not a resumable training checkpoint. This script
uses SGSST's own GaussianModel.load_ply() + capture() to build an
equivalent checkpoint directly from the ply, skipping a from-scratch
re-train we don't need (the splat is already trained).

Usage:
    python make_checkpoint_from_ply.py <point_cloud.ply> <output.pth> [--sh-degree 3]
"""
import sys
import argparse
import torch
from argparse import Namespace

sys.path.insert(0, ".")  # run from the SGSST/ dir so scene/, utils/ import correctly
from scene.gaussian_model import GaussianModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply_path")
    p.add_argument("out_path")
    p.add_argument("--sh-degree", type=int, default=3)
    args = p.parse_args()

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians from {args.ply_path}")

    # training_setup() builds the Adam optimizer gaussians.capture() expects
    # to serialize via optimizer.state_dict(). Use SGSST's real
    # OptimizationParams defaults so the resulting checkpoint is loadable by
    # stylize.py's gaussians.restore(model_params, opt) with the same opt.
    from arguments import OptimizationParams
    opt_parser = argparse.ArgumentParser()
    opt_params_def = OptimizationParams(opt_parser)
    training_args = opt_parser.parse_args([])

    gaussians.training_setup(training_args)

    iteration = 7000  # matches our pipeline's trained iteration count
    torch.save((gaussians.capture(), iteration), args.out_path)
    print(f"Wrote checkpoint: {args.out_path}")


if __name__ == "__main__":
    main()
