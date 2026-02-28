# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
import sys
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def compute_feature_variance_per_batch(x_norm_patchtokens, x_norm_clstoken):
    """
    Compute feature variance statistics per batch to monitor feature diversity
    and detect potential feature collapse during training.

    Args:
        x_norm_patchtokens (torch.Tensor): Normalized patch tokens of shape (B, N, D)
            where B = batch size, N = number of patches, D = embedding dimension.
        x_norm_clstoken (torch.Tensor): Normalized CLS tokens of shape (B, D).

    Returns:
        dict: Dictionary containing variance statistics:
            - patch_var_per_sample: (B,) mean variance across patches and dims for each sample
            - patch_var_mean: scalar, average of patch_var_per_sample over the batch
            - patch_var_across_batch: (N, D) variance of each patch-feature across the batch
            - patch_var_across_batch_mean: scalar, mean of patch_var_across_batch
            - cls_var_across_batch: (D,) variance of CLS token features across the batch
            - cls_var_across_batch_mean: scalar, mean of cls_var_across_batch
    """
    # Variance of patch tokens per sample: for each sample compute var across patches
    # x_norm_patchtokens shape: (B, N, D)
    patch_var_per_sample = x_norm_patchtokens.var(dim=1).mean(dim=-1)  # (B,)
    patch_var_mean = patch_var_per_sample.mean()  # scalar

    # Variance across the batch dimension (detects collapse if near zero)
    patch_var_across_batch = x_norm_patchtokens.var(dim=0)  # (N, D)
    patch_var_across_batch_mean = patch_var_across_batch.mean()  # scalar

    # CLS token variance across the batch
    cls_var_across_batch = x_norm_clstoken.var(dim=0)  # (D,)
    cls_var_across_batch_mean = cls_var_across_batch.mean()  # scalar

    logger.info(
        f"Feature variance -- "
        f"patch (per-sample mean): {patch_var_mean.item():.6f}, "
        f"patch (across-batch mean): {patch_var_across_batch_mean.item():.6f}, "
        f"cls (across-batch mean): {cls_var_across_batch_mean.item():.6f}"
    )

    return {
        "patch_var_per_sample": patch_var_per_sample,
        "patch_var_mean": patch_var_mean,
        "patch_var_across_batch": patch_var_across_batch,
        "patch_var_across_batch_mean": patch_var_across_batch_mean,
        "cls_var_across_batch": cls_var_across_batch,
        "cls_var_across_batch_mean": cls_var_across_batch_mean,
    }


def compute_feature_covariance_per_batch(x_norm_patchtokens, x_norm_clstoken):
    """
    Compute feature covariance statistics per batch to monitor inter-feature
    correlations and detect dimensional collapse (where different feature
    dimensions become highly correlated).

    Args:
        x_norm_patchtokens (torch.Tensor): Normalized patch tokens of shape (B, N, D)
            where B = batch size, N = number of patches, D = embedding dimension.
        x_norm_clstoken (torch.Tensor): Normalized CLS tokens of shape (B, D).

    Returns:
        dict: Dictionary containing covariance statistics:
            - patch_cov_per_sample: (B, D, D) covariance matrix across patches for each sample
            - patch_cov_off_diag_mean: scalar, mean absolute off-diagonal value averaged over batch
            - patch_cov_across_batch: (D, D) covariance of patch features pooled across the batch
            - patch_cov_across_batch_off_diag_mean: scalar, mean absolute off-diagonal value
            - cls_cov_across_batch: (D, D) covariance of CLS tokens across the batch
            - cls_cov_across_batch_off_diag_mean: scalar, mean absolute off-diagonal value
    """
    B, N, D = x_norm_patchtokens.shape

    # --- Per-sample covariance of patch tokens (across patches) ---
    # Center each sample's patches: (B, N, D)
    patch_centered = x_norm_patchtokens - x_norm_patchtokens.mean(dim=1, keepdim=True)
    # Covariance: (B, D, D)  —  (B, D, N) @ (B, N, D) / (N - 1)
    patch_cov_per_sample = torch.bmm(patch_centered.transpose(1, 2), patch_centered) / max(N - 1, 1)

    # Mean absolute off-diagonal per sample, then average over batch
    eye = torch.eye(D, device=patch_cov_per_sample.device, dtype=patch_cov_per_sample.dtype).unsqueeze(0)  # (1, D, D)
    off_diag_mask = 1.0 - eye
    patch_off_diag = (patch_cov_per_sample * off_diag_mask).abs().sum(dim=(1, 2)) / max(D * (D - 1), 1)  # (B,)
    patch_cov_off_diag_mean = patch_off_diag.mean()  # scalar

    # --- Covariance of patch features pooled across the entire batch ---
    # Reshape to (B*N, D) and compute covariance across all patch tokens
    all_patches = x_norm_patchtokens.reshape(B * N, D)
    all_patches_centered = all_patches - all_patches.mean(dim=0, keepdim=True)
    patch_cov_across_batch = (all_patches_centered.T @ all_patches_centered) / max(B * N - 1, 1)  # (D, D)
    patch_cov_across_batch_off_diag_mean = (
        (patch_cov_across_batch * off_diag_mask.squeeze(0)).abs().sum() / max(D * (D - 1), 1)
    )

    # --- CLS token covariance across the batch ---
    cls_centered = x_norm_clstoken - x_norm_clstoken.mean(dim=0, keepdim=True)  # (B, D)
    cls_cov_across_batch = (cls_centered.T @ cls_centered) / max(B - 1, 1)  # (D, D)
    cls_cov_across_batch_off_diag_mean = (
        (cls_cov_across_batch * off_diag_mask.squeeze(0)).abs().sum() / max(D * (D - 1), 1)
    )

    logger.info(
        f"Feature covariance -- "
        f"patch off-diag (per-sample mean): {patch_cov_off_diag_mean.item():.6f}, "
        f"patch off-diag (across-batch): {patch_cov_across_batch_off_diag_mean.item():.6f}, "
        f"cls off-diag (across-batch): {cls_cov_across_batch_off_diag_mean.item():.6f}"
    )

    return {
        "patch_cov_per_sample": patch_cov_per_sample,
        "patch_cov_off_diag_mean": patch_cov_off_diag_mean,
        "patch_cov_across_batch": patch_cov_across_batch,
        "patch_cov_across_batch_off_diag_mean": patch_cov_across_batch_off_diag_mean,
        "cls_cov_across_batch": cls_cov_across_batch,
        "cls_cov_across_batch_off_diag_mean": cls_cov_across_batch_off_diag_mean,
    }


class BlockChunk(nn.ModuleList):
    def forward(self, x, return_attention=False):
        if return_attention:
            for i, b in enumerate(self):
                if i < len(self) - 1:
                    x = b(x)
                else:
                    return b(x, return_attention=True)
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            print(f'chunking the blocks into {block_chunks} chunks, with chunk size {chunksize}, depth: {depth}')
            # sys.exit()
            if depth !=1:
                for i in range(0, depth, chunksize):
                    # this is to keep the block index consistent if we chunk the block list
                    chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            else:
                chunked_blocks.append([nn.Identity()] + blocks_list)
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x
 
    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)
        
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        print('Forward pass through the vision transformer backbone ...', 'Input shape: ', x.shape)
        x = self.prepare_tokens_with_masks(x, masks)

        print('Prepared tokens shape: ', x.shape)

        # sys.exit()
        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)

        x_norm_clstoken = x_norm[:, 0]
        x_norm_patchtokens = x_norm[:, self.num_register_tokens + 1 :]

        # Compute feature variance per batch
        feature_variance = compute_feature_variance_per_batch(x_norm_patchtokens, x_norm_clstoken)

        # Compute feature covariance per batch
        feature_covariance = compute_feature_covariance_per_batch(x_norm_patchtokens, x_norm_clstoken)

        return {
            "x_norm_clstoken": x_norm_clstoken,
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm_patchtokens,
            "x_prenorm": x,
            "masks": masks,
            "feature_variance": feature_variance,
            "feature_covariance": feature_covariance,
        }
    
    ### Function from dinov1
    def get_last_self_attention(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
            
        x = self.prepare_tokens_with_masks(x, masks)
        ## Implementation/Suggestion to get the attention map
        # Run through model, at the last block just return the attention.
        print(f'len blocks: {len(self.blocks)}')
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                x = self.norm(x)
            else: 
                # print('Returning attention from the last block')
                x = self.norm(x)
                return blk(x, return_attention=True)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        print('Forward pass through the vision transformer ...')
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

'''
    vit_tiny will be trained from scratch and monitored. The idea is to make sure the model is trying to find the global minimum of the loss lanscape, instead of diverging or getting stuck in a bad local minimum. So we set the depth to 1, which is the smallest possible vision transformer, and we can easily visualize the loss landscape in this case. If the loss is diverging, we can try to reduce the learning rate or add more regularization. If the loss is flat, we can try to increase the learning rate or reduce regularization. If the loss is oscillating, we can try to add momentum or use a different optimizer. The goal is to find a good set of hyperparameters that allows the model to converge to a good solution.
'''
def vit_tiny(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=2, 
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        # embed_dim=1536,
        depth=24,
        num_heads=16,
        # num_heads=32,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
