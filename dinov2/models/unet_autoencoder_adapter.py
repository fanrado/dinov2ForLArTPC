import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

# -------------------------
# UNet building blocks
# -------------------------
#Interesting that batchNorm layers do not work with FDSP for reasons I do not understand, so groupnorm it is
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, gn_pref=8):
        super().__init__()

        def _gn(c, pref=gn_pref):
            # pick a group count that divides c (try 32/16/8/4/2/1)
            for g in (pref, 32, 16, 8, 4, 2, 1):
                if c % g == 0 and g > 0:
                    return nn.GroupNorm(g, c)
            return nn.GroupNorm(1, c)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x): return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # upsample by 2, then concat skip, then DoubleConv
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle off-by-one shapes
        dy, dx = skip.size(-2) - x.size(-2), skip.size(-1) - x.size(-1)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x): return self.conv(x)

# -------------------------
# UNet encoder & decoder
# -------------------------

class UNetEncoder(nn.Module):
    """
    depth=5 -> strides: 1,2,4,8,16; channels: base,2b,4b,8b,16b
    """
    def __init__(self, in_chans=3, base=64):
        super().__init__()
        self.inc   = DoubleConv(in_chans, base)        # s=1
        self.down1 = Down(base,     base * 2)          # s=2
        self.down2 = Down(base * 2, base * 4)          # s=4
        self.down3 = Down(base * 4, base * 8)          # s=8
        self.down4 = Down(base * 8, base * 16)         # s=16

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x1 = self.inc(x)          # s1
        x2 = self.down1(x1)       # s2
        x3 = self.down2(x2)       # s4
        x4 = self.down3(x3)       # s8
        x5 = self.down4(x4)       # s16 (bottleneck)
        skips = [x4, x3, x2, x1]  # decoder order
        return x5, skips


class UNetDecoder(nn.Module):
    """
    Standard UNet up path. Returns:
     - dec_feat at a requested out_stride ∈ {16,8,4,2,1}
     - reconstruction (B, out_chans, H, W)
    """
    def __init__(self, out_chans=3, base=64, out_stride=1):
        super().__init__()
        assert out_stride in (1, 2, 4, 8, 16)
        self.out_stride = out_stride
        self.base = base

        self.up1 = Up(base * 16, base * 8)  # -> s=8
        self.up2 = Up(base * 8,  base * 4)  # -> s=4
        self.up3 = Up(base * 4,  base * 2)  # -> s=2
        self.up4 = Up(base * 2,  base)      # -> s=1
        self.outc = OutConv(base, out_chans)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]):
        x4, x3, x2, x1 = skips  # s8, s4, s2, s1
        dec_feat = None
        if self.out_stride == 16: dec_feat = bottleneck

        y = self.up1(bottleneck, x4)   # s=8
        if self.out_stride == 8:  dec_feat = y

        y = self.up2(y, x3)            # s=4
        if self.out_stride == 4:  dec_feat = y

        y = self.up3(y, x2)            # s=2
        if self.out_stride == 2:  dec_feat = y

        y = self.up4(y, x1)            # s=1
        if self.out_stride == 1:  dec_feat = y

        recon = self.outc(y)
        return dec_feat, recon

# -------------------------
# Adapter to DINO interface
# -------------------------

def _ln(d): return nn.LayerNorm(d, eps=1e-6)

class UNetAutoencoderAdapter(nn.Module):
    """
    Presents UNet encoder+decoder as a ViT-like DINO backbone:
      returns dict with:
        - "x_norm_clstoken":    (B, D)
        - "x_norm_patchtokens": (B, H'*W', D)  with H' = H / patch_size
        - optional "reconstruction": (B, C, H, W)
    """
    def __init__(self,
                 in_chans: int = 3,
                 base: int = 64,
                 embed_dim: int = 384,
                 out_stride: int = 4,
                 use_reconstruction: bool = False):
        super().__init__()
        assert out_stride in (1, 2, 4, 8, 16) # this is are magic number to work in iBOT
        # expose common attrs expected by other code
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = out_stride   # IMPORTANT: align iBOT patch grid with decoder stride

        self.in_chans = in_chans
        self.base = base
        self.use_reconstruction = use_reconstruction

        self.encoder = UNetEncoder(in_chans=in_chans, base=base)
        self.decoder = UNetDecoder(out_chans=in_chans, base=base, out_stride=out_stride)

        feat_dim = {16: base * 16, 8: base * 8, 4: base * 4, 2: base * 2, 1: base}[out_stride]
        self.proj = nn.Conv2d(feat_dim, embed_dim, kernel_size=1, bias=True)
        self.norm_patch = _ln(embed_dim)
        self.norm_cls   = _ln(embed_dim)

    @torch.no_grad()
    def _check_divisible(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        s = self.patch_size
        assert h % s == 0 and w % s == 0, f"Input {h}x{w} not divisible by stride/patch {s}"

    def _tokens_from_feat(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.proj(feat)                     # (B, D, H', W')
        B, D, Hp, Wp = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(B, Hp * Wp, D)
        patches = self.norm_patch(patches)
        cls = F.adaptive_avg_pool2d(f, 1).flatten(1)
        cls = self.norm_cls(cls)
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}

    def _forward_one(self, x: torch.Tensor, return_recon: bool):
        bneck, skips = self.encoder(x)
        dec_feat, recon = self.decoder(bneck, skips)
        out = self._tokens_from_feat(dec_feat)
        if return_recon and self.use_reconstruction:
            if recon.shape[-2:] != x.shape[-2:]:
                recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
            out["reconstruction"] = recon
        return out

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], masks=None, is_training: bool = True):
        # mirrors DinoVisionTransformer.forward
        if isinstance(x, (list, tuple)):       # [global, local]
            g, l = x
            self._check_divisible(g); self._check_divisible(l)
            return self._forward_one(g, return_recon=True), self._forward_one(l, return_recon=False)
        else:
            self._check_divisible(x)
            return self._forward_one(x, return_recon=True)

