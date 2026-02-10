# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from . import vision_transformer as vits
from .unet_autoencoder_adapter import UNetAutoencoderAdapter   # <-- NEW
from .minkunet_attention import MinkUNetSparseAttention125

logger = logging.getLogger("dinov2")

def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")

    # ---------- UNet autoencoder ----------
    if args.arch == "unet_autoencoder":
        # pull optional args with safe defaults
        in_chans   = getattr(args, "in_chans", 3)
        unet_base  = getattr(args, "unet_base", 64)
        embed_dim  = getattr(args, "embed_dim", 384)
        out_stride = getattr(args, "out_stride", 4)       # also equals patch_size for iBOT grid

        teacher = UNetAutoencoderAdapter(
            in_chans=in_chans, base=unet_base, embed_dim=embed_dim,
            out_stride=out_stride
        )
        if only_teacher:
            return teacher, teacher.embed_dim
        # student mirrors teacher (drop_path/etc. not used in this backbone)
        student = UNetAutoencoderAdapter(
            in_chans=in_chans, base=unet_base, embed_dim=embed_dim,
            out_stride=out_stride
        )
        embed_dim = student.embed_dim
        return student, teacher, embed_dim
    # ----------------------------------------------------
    if args.arch == "minkunet":
        patch_factor = getattr(args, "patch_factor", 4)
        embed_dim = 64
        teacher = MinkUNetSparseAttention125(patch_factor=patch_factor)
        if only_teacher:
            return teacher, embed_dim
        student = MinkUNetSparseAttention125(patch_factor=patch_factor)
        return student, teacher, embed_dim
    # ----------------------------------------------------
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
        return student, teacher, embed_dim

    raise NotImplementedError(f"Unknown arch '{args.arch}'")

def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


