import os, zlib, numpy as np
from glob import glob
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import sys
import h5py
import subprocess

class WireCellFramesDataset(Dataset):
    """
    ViT transformer expect 3 channels RGB-like, the input should fit this scheme
    DUNE WireCellFrames pixel-maps as 3-channel images (U,V,Z) from .gz files.
    Also we can do a single plane in 3 channels so it would be training on 1 plane instead of 3
    Returns (image, dummy_target). DINOv2 will apply its own multi-crop transform.
    """
    def __init__(self, root,
                 split=None,
                 transform=None,
                 target_transform=None,
                 **kwargs):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # -------------------------------
        # Parse optional flags from :extra=
        # -------------------------------
        # Accept comma-separated K=V pairs, e.g. extra="plane=Z,mono=mono3"
        self.plane = None       # 'U','V','Z','0','1','2' -> choose single plane; None -> use all 3
        self.mono = "mono3"     # 'mono3' (replicate single plane to 3ch) | 'mono1' (true 1ch 'L' image), mono1 is actually not implemented, it requires changing the model which I think will be done anyway, so just keep config for now
        self.rebin = 1
        extra = kwargs.get("extra", None)
        if isinstance(extra, str):
            for token in extra.split(","):
                if not token.strip():
                    continue
                k, _, v = token.partition("=")
                k = k.strip().lower()
                v = v.strip()
                if k == "plane":
                    self.plane = v.upper()  # allow 'U','V','Z' or '0','1','2'
                elif k == "mono":
                    self.mono = v.lower()   # 'mono3' or 'mono1'
                elif k == "rebin":
                    self.rebin = int(v.lower)
                    if 1500 % self.rebin != 0:
                        raise RuntimeError(f"Not able to rebin 1500 ticks by value : {self.rebin}")

        self.entries = []
        self.tot_entries = 0
        self.entries_per_file = 0
        if not os.path.isdir(root):
            raise RuntimeError(f"{root} is not a directory!")
        for gz in sorted(glob(os.path.join(root, "*rec.h5"))):
            key = os.path.splitext(os.path.basename(gz))[0].replace("rec", "")
            nentries_f = int(subprocess.run(f'h5ls {gz} | wc -l', shell=True, capture_output=True).stdout)
            self.entries_per_file = nentries_f
            self.tot_entries += nentries_f
            self.entries.append((key, nentries_f, gz))

        if not self.entries:
            raise RuntimeError(f"No .gz files found under {root}/<flavor>/event*.gz")

    def __len__(self):
        return self.tot_entries

    def _read_array(self, gz_path, entry):
        with h5py.File(gz_path, "r") as f:
            arr = np.uint8(np.array(f[f'{entry+1}/frame_rebinned_reco']))
            if arr.shape[0] != 2560:
                raise RuntimeError(f"Input frame has channel dimensions {arr.shape[0]}. Expected 2560")
            arr_u = np.resize(arr[:800,:], (960, arr.shape[1]))
            arr_v = np.resize(arr[800:1600,:], (960, arr.shape[1]))
            arr_z = arr[1600:,:]
            arr_frame = np.array([arr_u, arr_v, arr_z])
        if self.rebin > 1:
            arr_frame = arr_frame.reshape(arr_frame.shape[0], arr_frame.shape[1],
                                          arr_frame.shape[2] // self.rebin, self.rebin).sum(axis=3)
        arr_frame = np.swapaxes(arr_frame, 1, 2)
        return arr_frame  # (3, H, W), H = ticks, W = channel

    def _to_pil(self, arr3):
        """
        Convert a (3,H,W) numpy uint8 to PIL according to options:
          - plane=None: return RGB-like from [U,V,Z] (3ch)
          - plane=..., mono=mono3: replicate selected plane -> 3ch RGB-like
          - plane=..., mono=mono1: return single-channel 'L' image (requires in_chans=1 model)
        """
        if self.plane is None:
            img = np.moveaxis(arr3, 0, -1)                     # (H,W,3)
            return Image.fromarray(img, mode="RGB")
        # choose one plane
        idx_map = {"U": 0, "V": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        idx = idx_map[self.plane]
        plane = arr3[idx]                                      # (H,W)
        if self.mono == "mono1":
            return Image.fromarray(plane, mode="L")            # true 1-channel
        # default: replicate into 3 channels (Option A)
        img = np.repeat(plane[..., None], 3, axis=2)           # (H,W,3)
        return Image.fromarray(img, mode="RGB")

    def __getitem__(self, idx):
        file_idx = idx // self.entries_per_file
        _, _, gz = self.entries[file_idx]
        arr = self._read_array(gz, idx-file_idx)
        img = self._to_pil(arr)
        target = 0  # dummy label for SSL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def save(self, idx, output='frame.png'):
        img, _ = self.__getitem__(idx)
        img.save(output)
