# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Self-attention at small resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification

from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution
from warpconvnet.nn.functional.global_pool import global_pool

from .minkunet.blocks import (
    ConvBlock2D, ConvTrBlock2D,
    ResidualSparseBlock2D,
    BottleneckSparseAttention2D
    )

# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class MinkUNetSparseAttention(nn.Module):
    """
    U-ResNet-style sparse model following MinkUNet18 architecture.
    - Initial conv at full resolution
    - Encoder: strided convolutions + residual blocks (2 stages)
    - Bottleneck: sparse attention at smallest resolution
    - Decoder: transposed convolutions + skip connections + residual blocks (2 stages)
    - Head: dense classification layer (4 classes)
    """
    def __init__(self, *,
                 spatial_encoding: bool = True,
                 flash_attention: bool = True,
                 encoding_dim: int = 32,
                 encoding_range: float = 1.0,
                 patch_factor: int = 4,
                 **kwargs,):
        super().__init__()

        assert patch_factor in (1, 2, 4)
        self.patch_factor = patch_factor
        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0 = ConvBlock2D(1, 32, kernel_size=3, stride=1)  # [B,1,500,500] → [B,32,500,500]

        # ---- Encoder (2 stages) ----
        # Stage 1: 500×500 to 250×250
        self.conv1 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block1 = ResidualSparseBlock2D(32, 32, kernel_size=3) # Channel stays 32

        # Stage 2: 250×250 to 125×125
        self.conv2 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block2 = ResidualSparseBlock2D(32, 64, kernel_size=3) # Channel projection 32 to 64

        # ---- Bottleneck (attention at 125×125) ----
        # Global context at 125×125 resolution (15625 spatial tokens)
        self.bottleneck = BottleneckSparseAttention2D(channels=64, attn_channels=128, heads=4,
                                                      encoding=spatial_encoding, flash=flash_attention,
                                                      encoding_range=encoding_range, encoding_channels=encoding_dim)

        # ---- Decoder (2 stages, symmetric to encoder) ----
        # Stage 1: 125×125 to 250×250
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)   # Upsample
        self.block6 = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # Merge skip1, process

        # Stage 2: 250×250 to 500×500 (full resolution)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)   # Upsample
        self.block8 = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # Merge skip0, process

        # ---- Final projection + classification head ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)  # Feature refinement

        # DINO heads
        self.patch_head = nn.LayerNorm(64, eps=1e-6)
        self.cls_head = nn.Sequential(
            #  nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(64, eps=1e-6)
        )

    def forward_one(self, x, is_training=True, img_size=(500,500)):
        """
        Forward pass through the entire network.
        Input: [B,1,500,500] dense tensor
        Output: [B,4] log-probabilities (4 classes)
        """
        dino_dict = {}
        ds_img_size = (img_size[0]//self.patch_factor, img_size[1]//self.patch_factor)
        # Convert dense input image to sparse voxel representation
        xs = Voxels.from_dense(x)

        # ============ ENCODER ============

        # Initial convolution at full resolution
        out = self.conv0(xs)                    # [B,1,500,500] to [B,32,500,500]
        out_p1 = out                            # Skip connection for final decoder stage

        # Stage 1: 500×500 to 250×250
        out = self.conv1(out_p1)                # Downsample spatially
        out = self.block1(out)                  # Residual processing
        out_b1p2 = out                          # Skip connection [B,32,250,250]

        # Stage 2: 250×250 to 125×125
        out = self.conv2(out_b1p2)              # Downsample spatially
        out = self.block2(out)                  # Residual + channel projection 32 to 64
                                                # Result: [B,64,125,125]

        # ============ BOTTLENECK (Sparse Attention at 125×125) ============
        out = self.bottleneck(out)              # [B,64,125,125] -> [B,128,125,125] (attention) -> [B,64,125,125]
        pooled_out = global_pool(out, reduce='mean')
        out_feat = pooled_out.features
        dino_dict["x_norm_clstoken"] = self.cls_head(out_feat)
        #  with torch.no_grad():
        #      mask = (out_dense.abs().sum(dim=1) > 0)
        #      print("Shape : ", mask.shape)
        #      active_ratio = mask.float().mean()
        #      print("Active ratio:", active_ratio.item())
        if self.patch_factor == 4:
            d_out = out.to_dense(channel_dim=1, spatial_shape=ds_img_size).permute(0, 2, 3, 1) # [B,64,125,125] dense tensor
            B, Hp, Wp, D = d_out.shape
            patches = d_out.reshape(B, Hp * Wp, D)
            dino_dict["x_norm_patchtokens"] = self.patch_head(patches)
        #  # ============ DECODER ============
        #
        #  # Stage 1: 125×125 to 250×250
        #  out = self.convtr5(out, out_b1p2)       # Upsample, guided by skip geometry
        #  out = cat(out, out_b1p2)                # [B,64,250,250] + [B,32,250,250] = [B,96,250,250]
        #  out = self.block6(out)                  # Process to [B,64,250,250]
        #  if self.patch_factor == 2:
        #      d_out = out.to_dense(channel_dim=1, spatial_shape=ds_img_size).permute(0, 2, 3, 1) # [B,64,250,250] dense tensor
        #      B, Hp, Wp, D = d_out.shape
        #      patches = d_out.reshape(B, Hp * Wp, D)
        #      dino_dict["x_norm_patchtokens"] = self.patch_head(patches)
        #
        #  # Stage 2: 250×250 to 500×500 (full resolution)
        #  out = self.convtr7(out, out_p1)         # Upsample
        #  out = cat(out, out_p1)                  # [B,64,500,500] + [B,32,500,500] = [B,96,500,500]
        #  out = self.block8(out)                  # Process to [B,64,500,500]
        #
        #  # ============ FINAL PROJECTION + HEAD ============
        #  out = self.final(out)                   # Feature refinement [B,64,500,500]
        #
        #  # Convert to dense for classification
        #  out_dense = out.to_dense(channel_dim=1, spatial_shape=img_size)  # [B,64,500,500] dense tensor
        #  #  dino_dict["x_norm_clstoken"] = self.cls_head(out_dense)
        #  if self.patch_factor == 1:
        #      d_out = out_dense.permute(0, 2, 3, 1)
        #      B, Hp, Wp, D = d_out.shape
        #      patches = d_out.reshape(B, Hp * Wp, D)
        #      dino_dict["x_norm_patchtokens"] = self.patch_head(patches)
        if is_training:
            return dino_dict
        else:
            return x["norm_clstoken"]

    def forward(self, x: Union[Tensor, List[Tensor]], masks=None, is_training: bool = True):
        # mirrors DinoVisionTransformer.forward
        if isinstance(x, (list, tuple)):       # [global, local]
            g, l = x
            g_h, g_w = g.shape[-2], g.shape[-1]
            l_h, l_w = l.shape[-2], l.shape[-1]
            return self.forward_one(g, is_training, (g_h, g_w)), self.forward_one(l, is_training, (l_h, l_w))
        else:
            x_h, x_w = x.shape[-2], x.shape[-1]
            return self.forward_one(x, is_training, (x_h, x_w))


class MinkUNetSparseAttention125(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=True, encoding_range=125.0, **kwargs)

class MinkUNetSparseAttentionNoEnc(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)

class MinkUNetSparseAttentionNoFlash(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)

class MinkUNetSparseAttentionNoFlash125(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, encoding_range=125.0, **kwargs)

class MinkUNetSparseAttentionNoFlashEnc(MinkUNetSparseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=False, **kwargs)
